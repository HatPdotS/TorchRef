"""SfFFT -- structure factors via FFT.

An nn.Module that reads the crystal off a :class:`~torchref.model.context.ModelContext`,
sizes its real-space grid lazily from that crystal and the resolution, builds the
electron density from atomic parameters, and transforms it to structure factors.
Usable standalone or as ``ModelFT``'s submodule.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from torchref.base.fourier import ifft
from torchref.base.reciprocal import extract_structure_factor_from_grid
from torchref.config import dtypes
from torchref.symmetry import Cell, SpaceGroup
from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.device_resolution import require_cell_dtype, resolve_device

if TYPE_CHECKING:
    from torchref.model.context import ModelContext


class SfFFT(DeviceMovementMixin, nn.Module):
    """
    Structure Factor calculator using FFT (Fast Fourier Transform).

    The engine does not own a cell or a space group: it holds the
    :class:`~torchref.model.context.ModelContext` it was given and reads
    ``ctx.cell`` and ``ctx.spacegroup`` whenever it needs them. The grid is derived
    from that crystal, ``max_res`` and ``explicit_gridsize`` on first use and kept
    until any of them changes (:attr:`grid_key`), so assigning a new cell or
    resolution to the context needs no further call.

    Parameters
    ----------
    ctx : ModelContext
        The crystallographic context to read the cell and space group from. May be
        incomplete at construction; the grid is sized once both are set.
    max_res : float, optional
        Maximum resolution for grid spacing in Angstroms. Default is 1.0.
    explicit_gridsize : tuple of int, optional
        Fixed grid dimensions ``(nx, ny, nz)``; overrides the resolution-derived size.
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is dtypes.float.
    device : torch.device, optional
        Computation device. Defaults to the cell's device when the context has one,
        else the configured default device.
    verbose : int, optional
        Verbosity level for logging. Default is 0.
    use_late_symmetry : bool, optional
        Apply symmetry in reciprocal space after the FFT when the grid permits
        exact indexing (default); otherwise symmetrise the density map first.

    Attributes
    ----------
    cell, spacegroup : Cell, SpaceGroup
        Read through from the context.
    gridsize, voxel_size : torch.Tensor or None
        Grid dimensions ``(nx, ny, nz)`` and the voxel edge vector sum, resolved on
        access; ``None`` while the context has no cell or space group. No coordinate
        grid is stored: the splats derive a voxel's Cartesian position from its
        index, so materialising one would cost ``12 * nx * ny * nz`` bytes that
        nothing reads. Call :func:`torchref.base.fourier.get_real_grid` if you
        genuinely need one.
    """

    def __init__(
        self,
        ctx: "ModelContext",
        *,
        max_res: Optional[float] = 1.0,
        explicit_gridsize: Optional[Tuple[int, int, int]] = None,
        dtype_float: torch.dtype = None,
        device: Optional[torch.device] = None,
        verbose: int = 0,
        use_late_symmetry: bool = True,
    ):
        super().__init__()
        self.ctx = ctx
        self.max_res = max_res
        self.explicit_gridsize = explicit_gridsize
        self.dtype_float = dtypes.float if dtype_float is None else dtype_float

        # One device for the module and everything it builds. An explicit
        # ``device`` moves the crystal as a whole, so cell, space group and the
        # grid buffers cannot end up split; without one the cell's device wins,
        # and an empty context falls back to the configured default.
        if device is not None:
            ctx.to(device)
        self.device = resolve_device(ctx.cell, device=device)

        self.verbose = verbose
        self.use_late_symmetry = use_late_symmetry

        # Derived from the crystal on first use. Non-persistent: rebuilt from the
        # context on restore rather than read back from a checkpoint.
        self.register_buffer("_gridsize", None, persistent=False)
        self.register_buffer("_voxel_size", None, persistent=False)
        self._late_symmetry_compatible: Optional[bool] = None
        self._grid_key = None

    # =========================================================================
    # Crystal, read through from the context
    # =========================================================================

    @property
    def cell(self) -> Optional[Cell]:
        """The context's unit cell."""
        return self.ctx.cell

    @property
    def spacegroup(self) -> Optional[SpaceGroup]:
        """The context's space group."""
        return self.ctx.spacegroup

    @property
    def fractional_matrix(self) -> Optional[torch.Tensor]:
        """Fractionalization matrix from the cell, on this module's device/dtype."""
        cell = self.ctx.cell
        if cell is None:
            return None
        # Move device first, then cast: a combined ``.to(device=cpu,
        # dtype=float64)`` from an MPS-resident cell raises because MPS
        # rejects the transient float64 view (MPS has no float64).
        return cell.fractional_matrix.to(device=self.device).to(dtype=self.dtype_float)

    @property
    def inv_fractional_matrix(self) -> Optional[torch.Tensor]:
        """Orthogonalization matrix from the cell, on this module's device/dtype."""
        cell = self.ctx.cell
        if cell is None:
            return None
        # Move device first, then cast (see ``fractional_matrix``).
        return cell.inv_fractional_matrix.to(device=self.device).to(
            dtype=self.dtype_float
        )

    # =========================================================================
    # Grid
    # =========================================================================

    @property
    def explicit_gridsize(self) -> Optional[Tuple[int, int, int]]:
        """Fixed grid dimensions, or None to size the grid from ``max_res``."""
        return self._explicit_gridsize

    @explicit_gridsize.setter
    def explicit_gridsize(self, value) -> None:
        self._explicit_gridsize = (
            None if value is None else tuple(int(x) for x in value)
        )

    @property
    def grid_key(self):
        """Everything the grid is derived from, as a hashable tuple.

        ``(cell.key, spacegroup.key, max_res, explicit_gridsize)``, or ``None``
        while the context has no cell or space group. Callers that cache anything
        grid-shaped can compare against it.
        """
        crystal = self.ctx.crystal_key
        if crystal is None:
            return None
        return (
            *crystal,
            None if self.max_res is None else float(self.max_res),
            self.explicit_gridsize,
        )

    def ensure_grid(self) -> bool:
        """Bring the grid in line with :attr:`grid_key`.

        Returns
        -------
        bool
            True when the grid was (re)built, False when it was already current or
            the context has no crystal yet.

        Raises
        ------
        RuntimeError
            If neither ``max_res`` nor ``explicit_gridsize`` is set, or the cell or
            space group disagree with this module's dtype or device.
        """
        key = self.grid_key
        if key == self._grid_key:
            return False
        if key is None:
            # The crystal went away; drop the grid derived from the old one.
            self._gridsize = None
            self._voxel_size = None
            self._late_symmetry_compatible = None
            self._grid_key = None
            return False

        cell, spacegroup = self.ctx.cell, self.ctx.spacegroup
        require_cell_dtype(cell, self.dtype_float, type(self).__name__)
        if spacegroup.matrices.dtype != self.dtype_float:
            raise RuntimeError(
                f"{type(self).__name__} was built for {self.dtype_float} but its "
                f"space group holds {spacegroup.matrices.dtype}. Rebuild the space "
                "group at the module's dtype."
            )
        if spacegroup.matrices.device != cell.data.device:
            raise RuntimeError(
                f"{type(self).__name__}: cell on {cell.data.device} but space group "
                f"on {spacegroup.matrices.device}. Move the context as a whole."
            )

        if self.explicit_gridsize is not None:
            gridsize = self.explicit_gridsize
        elif self.max_res is not None:
            gridsize = self.compute_optimal_gridsize(self.max_res)
        else:
            raise RuntimeError(
                f"{type(self).__name__} cannot size its grid: set max_res or "
                "explicit_gridsize."
            )
        shape = tuple(int(n) for n in gridsize)

        if self.verbose > 1:
            print(f"Setting up grids with max_res={self.max_res} Å")

        previous = self._gridsize
        self._gridsize = torch.tensor(shape, dtype=dtypes.int, device=self.device)

        # The step between diagonally adjacent grid points, i.e. the sum of the three
        # cell edge vectors each divided by its own sampling count. Equal to the true
        # per-axis voxel edge lengths only for an orthogonal cell; kept because that is
        # what the previous grid-differencing definition produced.
        self._voxel_size = self.fractional_matrix @ (
            1.0 / self._gridsize.to(self.dtype_float)
        )

        # Every symmetry-equivalent HKL lands on an integer grid point exactly when
        # the grid admits direct indexing, which the space group answers without
        # building an operator.
        self._late_symmetry_compatible = spacegroup.can_index_directly(shape)
        if self.verbose > 0 and self.use_late_symmetry:
            if self._late_symmetry_compatible:
                print("SfFFT: Using late symmetry (reciprocal space)")
            else:
                print(
                    "SfFFT: Late symmetry disabled - grid not compatible "
                    "(falling back to early symmetry)"
                )

        # The space group memoises its map operator and reciprocal extractor per
        # grid shape. They are keyed on the shape, so a same-shape rebuild keeps
        # them; a different shape drops them so the old operator's sampling grids
        # do not stay resident.
        if previous is not None and tuple(int(n) for n in previous.tolist()) != shape:
            spacegroup.reset_cache()

        if self.verbose > 2:
            print(f"Grid shape: {shape}")
            print(f"Voxel size: {self._voxel_size}")

        # Last, so a failure above leaves the old key in place and the next call
        # tries again.
        self._grid_key = key
        return True

    def _require_grid(self) -> None:
        """Resolve the grid, refusing to proceed without a crystal."""
        self.ensure_grid()
        if self._gridsize is None:
            raise RuntimeError(
                f"{type(self).__name__} has no crystal to size its grid: the context "
                f"{self.ctx!r} has no cell or space group. Load a structure, or pass a "
                "context whose cell and spacegroup are set."
            )

    @property
    def gridsize(self) -> Optional[torch.Tensor]:
        """Grid dimensions ``(nx, ny, nz)``, or None without a crystal."""
        self.ensure_grid()
        return self._gridsize

    @property
    def voxel_size(self) -> Optional[torch.Tensor]:
        """Voxel edge vector sum, or None without a crystal."""
        self.ensure_grid()
        return self._voxel_size

    @property
    def grid_shape(self) -> Optional[Tuple[int, int, int]]:
        """Map dimensions ``(nx, ny, nz)`` as Python ints, or None without a crystal."""
        gridsize = self.gridsize
        if gridsize is None:
            return None
        return tuple(int(n) for n in gridsize)

    @property
    def late_symmetry_compatible(self) -> Optional[bool]:
        """Whether the current grid admits reciprocal-space symmetrisation."""
        self.ensure_grid()
        return self._late_symmetry_compatible

    def compute_optimal_gridsize(self, max_res: Optional[float] = None) -> tuple:
        """
        Compute optimal grid dimensions from the context's cell and space group.

        Uses Cell.compute_grid_size() for base calculation and
        Symmetry.suggest_grid_size() for symmetry optimization.

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution in Angstroms. If None, uses self.max_res.

        Returns
        -------
        tuple of int
            Optimal grid dimensions (nx, ny, nz).

        Raises
        ------
        RuntimeError
            If the context has no cell or space group.
        """
        cell, spacegroup = self.ctx.cell, self.ctx.spacegroup
        if cell is None or spacegroup is None:
            raise RuntimeError(
                f"{type(self).__name__}: the context {self.ctx!r} has no cell or "
                "space group to size a grid from."
            )

        resolution = max_res if max_res is not None else self.max_res

        # Use Cell's method for base grid size calculation
        gridsize_initial = cell.compute_grid_size(resolution)

        if self.verbose > 1:
            print(f"Initial grid size from cell: {gridsize_initial}")

        # Optimize for symmetry and FFT-friendliness
        gridsize_optimized = spacegroup.suggest_grid_size(
            gridsize_initial, make_fft_friendly=True
        )
        if self.verbose > 1 and gridsize_optimized != gridsize_initial:
            print(
                f"Optimized grid size from {gridsize_initial} to {gridsize_optimized} "
                f"(symmetry + FFT friendly)"
            )
        return tuple(int(n) for n in gridsize_optimized)

    def setup_grid(
        self,
        *,
        max_res: Optional[float] = None,
        gridsize: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Override the grid's inputs explicitly and resolve it now.

        Parameters
        ----------
        max_res : float, optional
            New maximum resolution in Angstroms. None leaves the current value.
        gridsize : tuple of int, optional
            Fixed grid size (nx, ny, nz), kept until cleared through
            :attr:`explicit_gridsize`. None leaves the current value.
        """
        if max_res is not None:
            self.max_res = float(max_res)
        if gridsize is not None:
            self.explicit_gridsize = gridsize
        self.ensure_grid()

    # =========================================================================
    # Density Map Building Methods
    # =========================================================================

    def build_density_map(
        self,
        xyz_iso: torch.Tensor,
        adp_iso: torch.Tensor,
        occ_iso: torch.Tensor,
        A_iso: torch.Tensor,
        B_iso: torch.Tensor,
        xyz_aniso: Optional[torch.Tensor] = None,
        u_aniso: Optional[torch.Tensor] = None,
        occ_aniso: Optional[torch.Tensor] = None,
        A_aniso: Optional[torch.Tensor] = None,
        B_aniso: Optional[torch.Tensor] = None,
        apply_symmetry: bool = True,
    ) -> torch.Tensor:
        """
        Build electron density map from atomic parameters.

        Parameters
        ----------
        xyz_iso, adp_iso, occ_iso : torch.Tensor
            Isotropic atoms: coordinates ``(n_iso, 3)``, ADPs ``(n_iso,)``,
            occupancies ``(n_iso,)``.
        A_iso, B_iso : torch.Tensor
            ITC92 amplitudes / widths for the isotropic atoms, ``(n_iso, 5)``.
        xyz_aniso, u_aniso, occ_aniso : torch.Tensor, optional
            Anisotropic atoms: coordinates ``(n_aniso, 3)``, U components
            ``(n_aniso, 6)``, occupancies ``(n_aniso,)``.
        A_aniso, B_aniso : torch.Tensor, optional
            ITC92 amplitudes / widths for the anisotropic atoms, ``(n_aniso, 5)``.
        apply_symmetry : bool, optional
            If True, apply crystallographic symmetry to the map. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map with shape (nx, ny, nz).
        """
        self._require_grid()

        from torchref.base.electron_density.main import build_electron_density

        density_map = build_electron_density(
            grid_shape=self.grid_shape,
            device=self.device,
            xyz_iso=xyz_iso,
            adp_iso=adp_iso,
            occ_iso=occ_iso,
            A_iso=A_iso,
            B_iso=B_iso,
            inv_frac_matrix=self.inv_fractional_matrix,
            frac_matrix=self.fractional_matrix,
            xyz_aniso=xyz_aniso,
            u_aniso=u_aniso,
            occ_aniso=occ_aniso,
            A_aniso=A_aniso,
            B_aniso=B_aniso,
            dtype=self.dtype_float,
        )

        if apply_symmetry:
            density_map = self.ctx.spacegroup.symmetrize_map(density_map)

        return density_map

    # =========================================================================
    # Structure Factor Methods
    # =========================================================================

    def map_to_structure_factors(
        self,
        density_map: torch.Tensor,
        hkl: torch.Tensor,
        apply_symmetry: bool = True,
    ) -> torch.Tensor:
        """
        Convert density map to structure factors via FFT.

        Parameters
        ----------
        density_map : torch.Tensor
            Electron density map with shape (nx, ny, nz).
            If apply_symmetry=True, this should be a P1 density map.
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        apply_symmetry : bool, optional
            If True (default), apply symmetry in reciprocal space. If False, the
            density map is assumed to already have symmetry applied (early
            symmetry path).

        Returns
        -------
        torch.Tensor
            Complex structure factors with shape (n_reflections,).
        """
        self._require_grid()
        reciprocal_space_grid = ifft(density_map, self.ctx.cell.volume)

        if apply_symmetry:
            # Memoised on the space group per (hkl, grid shape).
            extractor = self.ctx.spacegroup.reciprocal_extractor(hkl, self.grid_shape)
            return extractor.extract_from_grid(reciprocal_space_grid)
        return extract_structure_factor_from_grid(reciprocal_space_grid, hkl)

    def compute_structure_factors(
        self,
        hkl: torch.Tensor,
        xyz_iso: torch.Tensor,
        adp_iso: torch.Tensor,
        occ_iso: torch.Tensor,
        A_iso: torch.Tensor,
        B_iso: torch.Tensor,
        xyz_aniso: Optional[torch.Tensor] = None,
        u_aniso: Optional[torch.Tensor] = None,
        occ_aniso: Optional[torch.Tensor] = None,
        A_aniso: Optional[torch.Tensor] = None,
        B_aniso: Optional[torch.Tensor] = None,
        apply_symmetry: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute structure factors from atomic parameters (end-to-end).

        Builds the density map and transforms it in one call. With
        ``use_late_symmetry`` (default) and a compatible grid, symmetry is
        applied in reciprocal space after the FFT (faster); otherwise it is
        applied to the density map before it.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        xyz_iso, adp_iso, occ_iso : torch.Tensor
            Isotropic atoms: coordinates ``(n_iso, 3)``, ADPs ``(n_iso,)``,
            occupancies ``(n_iso,)``.
        A_iso, B_iso : torch.Tensor
            ITC92 amplitudes / widths for the isotropic atoms.
        xyz_aniso, u_aniso, occ_aniso : torch.Tensor, optional
            Anisotropic atoms: coordinates, U components ``(n_aniso, 6)``,
            occupancies.
        A_aniso, B_aniso : torch.Tensor, optional
            ITC92 amplitudes / widths for the anisotropic atoms.
        apply_symmetry : bool, optional
            If True, apply crystallographic symmetry. Default is True.

        Returns
        -------
        sf : torch.Tensor
            Complex structure factors with shape (n_reflections,).
        density_map : torch.Tensor
            Electron density map with shape (nx, ny, nz).
            Note: When using late symmetry, this is the P1 map (without symmetry).
        """
        # Resolve the grid first: the late-symmetry flag belongs to the grid the
        # density is about to be built on.
        self._require_grid()
        use_late = (
            apply_symmetry and self.use_late_symmetry and self._late_symmetry_compatible
        )

        density_map = self.build_density_map(
            xyz_iso=xyz_iso,
            adp_iso=adp_iso,
            occ_iso=occ_iso,
            A_iso=A_iso,
            B_iso=B_iso,
            xyz_aniso=xyz_aniso,
            u_aniso=u_aniso,
            occ_aniso=occ_aniso,
            A_aniso=A_aniso,
            B_aniso=B_aniso,
            apply_symmetry=not use_late and apply_symmetry,  # Early symmetry
        )
        sf = self.map_to_structure_factors(
            density_map,
            hkl,
            apply_symmetry=use_late,  # Late symmetry
        )
        return sf, density_map
