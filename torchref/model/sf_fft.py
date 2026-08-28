"""SfFFT -- structure factors via FFT.

An nn.Module owning the real-space grid setup, the electron-density build from
atomic parameters, and the FFT to structure factors. Usable standalone or as
``ModelFT``'s submodule. ``FFT`` is a deprecated alias for :class:`SfFFT`.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchref.base.fourier import get_real_grid, ifft
from torchref.base.reciprocal import extract_structure_factor_from_grid
from torchref.config import dtypes, get_default_device
from torchref.symmetry import Cell, SpaceGroup
from torchref.symmetry.spacegroup import SpaceGroupLike
from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.device_resolution import resolve_device


class SfFFT(DeviceMovementMixin, nn.Module):
    """
    Structure Factor calculator using FFT (Fast Fourier Transform).

    Built from a Cell and optionally a SpaceGroup, which drive the grid. Call
    :meth:`setup_grid` before :meth:`build_density_map`; the higher-level
    :meth:`compute_structure_factors` does both.

    Parameters
    ----------
    cell : Cell
        Unit cell object containing cell parameters.
    spacegroup : SpaceGroupLike, optional
        Space group specification (string, int, or gemmi.SpaceGroup).
        If None, defaults to P1.
    max_res : float, optional
        Maximum resolution for grid spacing in Angstroms. Default is 1.5.
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is dtypes.float.
    device : torch.device, optional
        Computation device. Defaults to the configured default device
        (``get_default_device()``).
    verbose : int, optional
        Verbosity level for logging. Default is 0.

    Attributes
    ----------
    cell, spacegroup : Cell, SpaceGroup
        The unit cell and the space group as an nn.Module carrying its symmetry
        matrices and translations; ``symmetry`` is an alias for ``spacegroup``.
    gridsize, real_space_grid, voxel_size : torch.Tensor or None
        Grid dimensions ``(nx, ny, nz)``, coordinate grid ``(nx, ny, nz, 3)`` and
        voxel dimensions -- all ``None`` until :meth:`setup_grid` runs.
    """

    def __init__(
        self,
        cell: Optional[Cell] = None,
        spacegroup: SpaceGroupLike = None,
        max_res: float = 1.5,
        dtype_float: torch.dtype = None,
        device: Optional[torch.device] = None,
        verbose: int = 0,
        use_late_symmetry: bool = True,
    ):
        """
        Initialize the SfFFT module with cell and spacegroup.

        Parameters
        ----------
        cell : Cell, optional
            Unit cell object. If None, must be set later via set_cell().
        spacegroup : SpaceGroupLike, optional
            Space group specification. If None, defaults to P1.
        max_res : float, optional
            Maximum resolution for grid spacing in Angstroms. Default is 1.5.
        dtype_float : torch.dtype, optional
            Data type for floating point tensors. Default is dtypes.float.
        device : torch.device, optional
            Computation device. Default is None (uses cell's device). If Cell is also None, defaults to CPU.
        verbose : int, optional
            Verbosity level for logging. Default is 0.
        use_late_symmetry : bool, optional
            If True (default), apply symmetry in reciprocal space after FFT
            ("late symmetry") for faster structure factor calculation.
            If False, apply symmetry to density map before FFT ("early symmetry").
        """
        super().__init__()
        self.max_res = max_res
        if dtype_float is None:
            dtype_float = dtypes.float
        self.dtype_float = dtype_float

        # One device for the module and everything it builds. ``resolve_device``
        # also moves ``cell`` when an explicit ``device`` disagrees with it, so
        # the cell and the SpaceGroup below cannot end up split.
        self.device = resolve_device(cell, device=device)

        self.verbose = verbose
        self.use_late_symmetry = use_late_symmetry

        # Store cell and spacegroup
        self._cell = cell
        self._spacegroup = None

        if spacegroup is not None or cell is not None:
            # ``self.device``, not the raw ``device`` argument: the latter is
            # ``None`` on the derive-from-cell path, which would silently put
            # the symmetry matrices on the global default instead.
            self._spacegroup = SpaceGroup(
                spacegroup, dtype=dtype_float, device=self.device
            )

        # Buffers (registered during setup_grid)
        self.register_buffer("gridsize", None)
        self.register_buffer("real_space_grid", None)
        self.register_buffer("voxel_size", None)

        # Late symmetry compatibility flag (set during setup_grid)
        self._late_symmetry_compatible: Optional[bool] = None

    # =========================================================================
    # Cell and SpaceGroup properties
    # =========================================================================

    @property
    def cell(self) -> Optional[Cell]:
        """Unit cell object."""
        return self._cell

    @cell.setter
    def cell(self, value: Cell):
        """Set unit cell."""
        self._cell = value

    @property
    def spacegroup(self) -> Optional[SpaceGroup]:
        """Space group object (SpaceGroup nn.Module)."""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value: SpaceGroupLike):
        """Set space group."""
        if value is not None:
            self._spacegroup = SpaceGroup(
                value, dtype=self.dtype_float, device=self.device
            )
        else:
            self._spacegroup = None

    @property
    def symmetry(self) -> Optional[SpaceGroup]:
        """Symmetry operations handler (alias for spacegroup)."""
        return self._spacegroup

    @property
    def fractional_matrix(self) -> Optional[torch.Tensor]:
        """Get fractionalization matrix from cell, on this module's device/dtype."""
        if self._cell is not None:
            # Move device first, then cast: a combined ``.to(device=cpu,
            # dtype=float64)`` from an MPS-resident cell raises because MPS
            # rejects the transient float64 view (MPS has no float64).
            return self._cell.fractional_matrix.to(device=self.device).to(
                dtype=self.dtype_float
            )
        return None

    @property
    def inv_fractional_matrix(self) -> Optional[torch.Tensor]:
        """Get orthogonalization matrix from cell, on this module's device/dtype."""
        if self._cell is not None:
            # Move device first, then cast (see ``fractional_matrix``).
            return self._cell.inv_fractional_matrix.to(device=self.device).to(
                dtype=self.dtype_float
            )
        return None

    def set_cell_and_spacegroup(self, cell: Cell, spacegroup: SpaceGroupLike = None):
        """
        Set cell and spacegroup for this SfFFT instance.

        Parameters
        ----------
        cell : Cell
            Unit cell object.
        spacegroup : SpaceGroupLike, optional
            Space group specification.

        Notes
        -----
        Receiver wins: this module may already own grid buffers, so an incoming
        cell on another device is moved to match rather than dragging the
        module after it.
        """
        self.device = resolve_device(self, cell)
        self._cell = cell
        self.spacegroup = spacegroup

    # =========================================================================
    # Grid Setup Methods
    # =========================================================================

    def compute_optimal_gridsize(self, max_res: Optional[float] = None) -> tuple:
        """
        Compute optimal grid dimensions using the stored cell and spacegroup.

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
            If cell has not been set.
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")

        resolution = max_res if max_res is not None else self.max_res

        # Use Cell's method for base grid size calculation
        gridsize_initial = self._cell.compute_grid_size(resolution)

        if self.verbose > 1:
            print(f"Initial grid size from cell: {gridsize_initial}")

        # Optimize for symmetry and FFT-friendliness
        gridsize_optimized = self._spacegroup.suggest_grid_size(
            gridsize_initial, make_fft_friendly=True
        )
        if self.verbose > 1 and gridsize_optimized != gridsize_initial:
            print(
                f"Optimized grid size from {gridsize_initial} to {gridsize_optimized} "
                f"(symmetry + FFT friendly)"
            )
        return gridsize_optimized

    @staticmethod
    def compute_real_space_grid(
        fractional_matrix: torch.Tensor,
        gridsize: torch.Tensor,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate the real-space coordinate grid.

        Parameters
        ----------
        fractional_matrix : torch.Tensor
            Fractionalization matrix mapping Cartesian to fractional
            coordinates, with shape (3, 3).
        gridsize : torch.Tensor
            Grid dimensions (nx, ny, nz).
        device : torch.device, optional
            Target device. Defaults to the configured default device.

        Returns
        -------
        torch.Tensor
            Real-space grid with shape (nx, ny, nz, 3).
        """
        # Forward ``device`` as-is, including ``None``: ``get_real_grid`` infers
        # from ``fractional_matrix`` when no device is given, and resolving the
        # global default here would preempt that.
        return get_real_grid(
            fractional_matrix=fractional_matrix, gridsize=gridsize, device=device
        )

    def setup_grid(
        self,
        gridsize: Optional[Tuple[int, int, int]] = None,
        max_res: Optional[float] = None,
    ):
        """
        Setup the real-space grid for electron density calculation.

        This method initializes and stores the grid state for subsequent
        density map calculations. Uses the stored cell and spacegroup.

        Parameters
        ----------
        gridsize : tuple of int, optional
            Explicit grid size (nx, ny, nz). If None, computed automatically
            using Cell.compute_grid_size() and Symmetry.suggest_grid_size().
        max_res : float, optional
            Maximum resolution in Angstroms. If None, uses self.max_res.

        Raises
        ------
        RuntimeError
            If cell has not been set.
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")

        if max_res is not None:
            self.max_res = max_res

        if self.verbose > 1:
            print(f"Setting up grids with max_res={self.max_res} Å")

        # Compute or use provided grid size
        if gridsize is not None:
            self.gridsize = torch.tensor(gridsize, dtype=dtypes.int, device=self.device)
        else:
            optimal_gridsize = self.compute_optimal_gridsize(self.max_res)
            self.gridsize = torch.tensor(
                optimal_gridsize, dtype=dtypes.int, device=self.device
            )

        # Compute real space grid
        self.real_space_grid = self.compute_real_space_grid(
            self._cell.fractional_matrix, self.gridsize, self.device
        )

        # Compute voxel size
        self.voxel_size = self.real_space_grid[2, 2, 2] - self.real_space_grid[1, 1, 1]

        # Every symmetry-equivalent HKL lands on an integer grid point exactly when
        # the grid admits direct indexing, which the space group answers without
        # building an operator.
        if self._spacegroup is not None:
            self._late_symmetry_compatible = self._spacegroup.can_index_directly(
                self.real_space_grid.shape[:-1]
            )

            if self.use_late_symmetry and self._late_symmetry_compatible:
                if self.verbose > 0:
                    print(
                        "SfFFT: Using late symmetry (reciprocal space)"
                    )
            elif self.use_late_symmetry and not self._late_symmetry_compatible:
                if self.verbose > 0:
                    print(
                        "SfFFT: Late symmetry disabled - grid not compatible "
                        "(falling back to early symmetry)"
                    )
        else:
            self._late_symmetry_compatible = False

        # The grid shape changed, so the space group's cached operators are stale.
        if self._spacegroup is not None:
            self._spacegroup.reset_cache()

        if self.verbose > 2:
            print(f"Grid shape: {self.real_space_grid.shape[:-1]}")
            print(f"Voxel size: {self.voxel_size}")

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

        Calls :meth:`setup_grid` itself if no grid has been set up yet.

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
        if self.real_space_grid is None:
            self.setup_grid()

        from torchref.base.electron_density.main import build_electron_density

        density_map = build_electron_density(
            real_space_grid=self.real_space_grid,
            xyz_iso=xyz_iso,
            adp_iso=adp_iso,
            occ_iso=occ_iso,
            A_iso=A_iso,
            B_iso=B_iso,
            inv_frac_matrix=self.inv_fractional_matrix,
            frac_matrix=self.fractional_matrix,
            voxel_size=self.voxel_size,
            xyz_aniso=xyz_aniso,
            u_aniso=u_aniso,
            occ_aniso=occ_aniso,
            A_aniso=A_aniso,
            B_aniso=B_aniso,
            dtype=self.dtype_float,
        )

        # Apply symmetry if requested
        if apply_symmetry and self._spacegroup is not None:
            density_map = self._spacegroup.symmetrize_map(density_map)

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
            If True (default) and late symmetry is enabled/compatible, apply
            symmetry in reciprocal space. If False, the density map is assumed
            to already have symmetry applied (early symmetry path).

        Returns
        -------
        torch.Tensor
            Complex structure factors with shape (n_reflections,).
        """
        reciprocal_space_grid = ifft(density_map, self.cell.volume)

        # Use late symmetry if enabled, compatible, and requested
        if apply_symmetry:
            # Lazily build / reuse cached extractor (precomputed flat indices)
            grid_shape = tuple(int(x) for x in self.gridsize)
            extractor = self._spacegroup.reciprocal_extractor(hkl, grid_shape)
            return extractor.extract_from_grid(reciprocal_space_grid)
        else:
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
        # Late symmetry: build a P1 map, symmetrize in reciprocal space.
        # Early symmetry: symmetrize the density map before the FFT.
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

    # =========================================================================
    # Device Movement
    # =========================================================================

    def reset_cache(self) -> None:
        """Drop the space group's cached operators; recomputed on next use."""
        if self._spacegroup is not None:
            self._spacegroup.reset_cache()

    def copy(self) -> "SfFFT":
        """Create a deep copy of this SfFFT module.

        Returns
        -------
        SfFFT
            A new SfFFT instance with cloned cell, spacegroup, and buffers.
        """
        # Clone the cell
        new_cell = self._cell.clone() if self._cell is not None else None

        # Copy the spacegroup
        new_spacegroup = (
            self._spacegroup.copy() if self._spacegroup is not None else None
        )

        # Create new SfFFT with copied components
        new_fft = SfFFT(
            cell=new_cell,
            spacegroup=new_spacegroup,
            max_res=self.max_res,
            dtype_float=self.dtype_float,
            device=self.device,
            verbose=self.verbose,
            use_late_symmetry=self.use_late_symmetry,
        )

        return new_fft


# Backward compatibility alias — deprecated, use SfFFT directly
def FFT(*args, **kwargs):
    """Deprecated: use SfFFT instead."""
    import warnings

    warnings.warn(
        "FFT is deprecated, use SfFFT instead. "
        "FFT will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return SfFFT(*args, **kwargs)
