"""
SfFFT - Structure Factor calculation via FFT (Fast Fourier Transform).

This module provides a PyTorch nn.Module that handles:
- Grid setup for real-space electron density calculations
- Building electron density maps from atomic parameters
- FFT-based conversion to structure factors

The SfFFT class can be used standalone (stateless) or with stored grid state
(stateful), making it flexible for various use cases.

Note: FFT is provided as a backward compatibility alias for SfFFT.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from torchref.base.fourier import get_real_grid, ifft
from torchref.base.reciprocal import extract_structure_factor_from_grid
from torchref.config import dtypes

from torchref.symmetry import Cell, SpaceGroup
from torchref.symmetry.map_symmetry import MapSymmetry
from torchref.symmetry.spacegroup import SpaceGroupLike
from torchref.utils.device_mixin import DeviceMovementMixin


class SfFFT(DeviceMovementMixin, nn.Module):
    """
    Structure Factor calculator using FFT (Fast Fourier Transform).

    This module encapsulates all FFT-related functionality for computing electron
    density maps and structure factors. It is initialized with a Cell and optionally
    a SpaceGroup, which are used for grid calculations.

    Parameters
    ----------
    cell : Cell
        Unit cell object containing cell parameters.
    spacegroup : SpaceGroupLike, optional
        Space group specification (string, int, or gemmi.SpaceGroup).
        If None, defaults to P1.
    max_res : float, optional
        Maximum resolution for grid spacing in Angstroms. Default is 1.0.
    radius_angstrom : float, optional
        Radius in Angstroms for density calculation around each atom. Default is 4.0.
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is dtypes.float.
    device : torch.device, optional
        Computation device. Default is torch.device('cpu').
    verbose : int, optional
        Verbosity level for logging. Default is 0.

    Attributes
    ----------
    cell : Cell
        Unit cell object.
    spacegroup : SpaceGroup
        Space group object (SpaceGroup nn.Module with matrices and translations).
    symmetry : SpaceGroup
        Alias for spacegroup (backward compatibility).
    max_res : float
        Maximum resolution for grid spacing.
    radius_angstrom : float
        Radius for density calculation around each atom.
    gridsize : torch.Tensor or None
        Grid dimensions (nx, ny, nz) when grid is set up.
    real_space_grid : torch.Tensor or None
        Real-space coordinate grid with shape (nx, ny, nz, 3).
    voxel_size : torch.Tensor or None
        Voxel dimensions.
    map_symmetry : MapSymmetry or None
        Symmetry operator for map calculations.

    Examples
    --------
    Standalone usage::

        from torchref.symmetry import Cell
        cell = Cell([50, 60, 70, 90, 90, 90])
        sf_fft = SfFFT(cell, spacegroup='P212121', max_res=1.5)
        sf_fft.setup_grid()
        density_map = sf_fft.build_density_map(xyz, b, occ, A, B, inv_frac, frac)
        sf = sf_fft.map_to_structure_factors(density_map, hkl)

    With ModelFT (composition)::

        model = ModelFT()
        model.load_pdb('structure.pdb')
        sf = model.get_structure_factor(hkl)  # Uses internal SfFFT instance
    """

    def __init__(
        self,
        cell: Optional[Cell] = None,
        spacegroup: SpaceGroupLike = None,
        max_res: float = 1.5,
        radius_angstrom: float = 3.0,
        dtype_float: torch.dtype = dtypes.float,
        device: torch.device = torch.device("cpu"),
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
        radius_angstrom : float, optional
            Radius in Angstroms for density calculation. Default is 3.0.
        dtype_float : torch.dtype, optional
            Data type for floating point tensors. Default is dtypes.float.
        device : torch.device, optional
            Computation device. Default is torch.device('cpu').
        verbose : int, optional
            Verbosity level for logging. Default is 0.
        use_late_symmetry : bool, optional
            If True (default), apply symmetry in reciprocal space after FFT
            ("late symmetry") for faster structure factor calculation (~5x speedup).
            If False, apply symmetry to density map before FFT ("early symmetry").
        """
        super().__init__()
        self.max_res = max_res
        self.radius_angstrom = radius_angstrom
        self.dtype_float = dtype_float
        self.device = device
        self.verbose = verbose
        self.use_late_symmetry = use_late_symmetry

        # Store cell and spacegroup
        self._cell = cell
        self._spacegroup = None

        if spacegroup is not None or cell is not None:
            self._spacegroup = SpaceGroup(
                spacegroup, dtype=dtype_float, device=device
            )

        # Buffers (registered during setup_grid)
        self.register_buffer("gridsize", None)
        self.register_buffer("real_space_grid", None)
        self.register_buffer("voxel_size", None)

        # Map symmetry operator (set during setup_grid)
        self.map_symmetry: Optional[MapSymmetry] = None

        # Late symmetry compatibility flag (set during setup_grid)
        self._late_symmetry_compatible: Optional[bool] = None

        # Cached reciprocal symmetry extractor (precomputed flat indices)
        self._sym_extractor = None
        self._sym_extractor_hkl_id: Optional[int] = None

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
        """Get fractionalization matrix from cell."""
        if self._cell is not None:
            return self._cell.fractional_matrix
        return None
    
    @property
    def inv_fractional_matrix(self) -> Optional[torch.Tensor]:
        """Get orthogonalization matrix from cell."""
        if self._cell is not None:
            return self._cell.inv_fractional_matrix
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
        """
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

        from torchref.symmetry.spacegroup import suggest_grid_size

        # Use Cell's method for base grid size calculation
        gridsize_initial = self._cell.compute_grid_size(resolution)

        if self.verbose > 1:
            print(f"Initial grid size from cell: {gridsize_initial}")

        # Optimize for symmetry and FFT-friendliness
        gridsize_optimized = suggest_grid_size(
            gridsize_initial, self._spacegroup, make_fft_friendly=True
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
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Generate the real-space coordinate grid.

        Parameters
        ----------
        cell_data : torch.Tensor
            Unit cell parameters [a, b, c, alpha, beta, gamma].
        gridsize : torch.Tensor
            Grid dimensions (nx, ny, nz).
        device : torch.device, optional
            Target device. Default is CPU.

        Returns
        -------
        torch.Tensor
            Real-space grid with shape (nx, ny, nz, 3).
        """
        return get_real_grid(fractional_matrix=fractional_matrix, gridsize=gridsize, device=device)

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
            self.gridsize = torch.tensor(
                gridsize, dtype=dtypes.int, device=self.device
            )
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
        self.voxel_size = (
            self.real_space_grid[2, 2, 2] - self.real_space_grid[1, 1, 1]
        )

        # Initialize map symmetry operator if space group is set
        if self._spacegroup is not None:
            self.map_symmetry = MapSymmetry(
                space_group=self._spacegroup,
                map_shape=self.real_space_grid.shape[:-1],
                cell_params=self._cell.data,
                verbose=self.verbose,
                device=self.device,
            )

            # Check late symmetry compatibility
            self._late_symmetry_compatible = self._check_late_symmetry_compatible()

            if self.use_late_symmetry and self._late_symmetry_compatible:
                if self.verbose > 0:
                    print("SfFFT: Using late symmetry (reciprocal space) for ~5x speedup")
            elif self.use_late_symmetry and not self._late_symmetry_compatible:
                if self.verbose > 0:
                    print(
                        "SfFFT: Late symmetry disabled - grid not compatible "
                        "(falling back to early symmetry)"
                    )
        else:
            self.map_symmetry = None
            self._late_symmetry_compatible = False

        # Invalidate cached symmetry extractor (grid shape changed)
        self._sym_extractor = None
        self._sym_extractor_hkl_id = None

        if self.verbose > 2:
            print(f"Grid shape: {self.real_space_grid.shape[:-1]}")
            print(f"Voxel size: {self.voxel_size}")

    def _check_late_symmetry_compatible(self) -> bool:
        """
        Check if late symmetry can be used (all equiv HKLs land on grid).

        Late symmetry requires that symmetry-equivalent HKL indices map to
        integer grid points. This is ensured when using MapSymmetryDirect
        (direct indexing without interpolation), which is created by the
        MapSymmetry factory when the grid is compatible.

        Returns
        -------
        bool
            True if late symmetry is compatible, False otherwise.
        """
        if self.map_symmetry is None:
            return False

        # Check if using MapSymmetryDirect (not interpolation)
        # MapSymmetryDirect has the can_use_direct_indexing attribute set to True
        from torchref.symmetry.map_symmetry import MapSymmetryDirect
        return isinstance(self.map_symmetry, MapSymmetryDirect)

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

        This method requires `setup_grid()` to have been called first.

        Parameters
        ----------
        xyz_iso : torch.Tensor
            Isotropic atom coordinates with shape (n_iso, 3).
        adp_iso : torch.Tensor
            Isotropic ADPs (atomic displacement parameters) with shape (n_iso,).
        occ_iso : torch.Tensor
            Isotropic occupancies with shape (n_iso,).
        A_iso : torch.Tensor
            ITC92 A parameters for isotropic atoms with shape (n_iso, 5).
        B_iso : torch.Tensor
            ITC92 B parameters for isotropic atoms with shape (n_iso, 5).
        xyz_aniso : torch.Tensor, optional
            Anisotropic atom coordinates with shape (n_aniso, 3).
        u_aniso : torch.Tensor, optional
            Anisotropic U parameters with shape (n_aniso, 6).
        occ_aniso : torch.Tensor, optional
            Anisotropic occupancies with shape (n_aniso,).
        A_aniso : torch.Tensor, optional
            ITC92 A parameters for anisotropic atoms with shape (n_aniso, 5).
        B_aniso : torch.Tensor, optional
            ITC92 B parameters for anisotropic atoms with shape (n_aniso, 5).
        apply_symmetry : bool, optional
            If True, apply crystallographic symmetry to the map. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map with shape (nx, ny, nz).

        Raises
        ------
        RuntimeError
            If setup_grid() has not been called.
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
            radius_angstrom=self.radius_angstrom,
            voxel_size=self.voxel_size,
            xyz_aniso=xyz_aniso,
            u_aniso=u_aniso,
            occ_aniso=occ_aniso,
            A_aniso=A_aniso,
            B_aniso=B_aniso,
            dtype=self.dtype_float,
        )

        # Apply symmetry if requested
        if apply_symmetry and self.map_symmetry is not None:
            density_map = self.map_symmetry(density_map)

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
            If True and late symmetry is enabled/compatible, apply symmetry
            in reciprocal space. Default is False (assume map already has
            symmetry applied or use early symmetry path).

        Returns
        -------
        torch.Tensor
            Complex structure factors with shape (n_reflections,).
        """
        reciprocal_space_grid = ifft(density_map, self.cell.volume)

        # Use late symmetry if enabled, compatible, and requested
        if apply_symmetry:
            # Lazily build / reuse cached extractor (precomputed flat indices)
            if self._sym_extractor is None or id(hkl) != self._sym_extractor_hkl_id:
                from torchref.base.reciprocal import ReciprocalSymmetryExtractor
                grid_shape = tuple(int(x) for x in self.gridsize)
                self._sym_extractor = ReciprocalSymmetryExtractor(
                    hkl, self.spacegroup, grid_shape,
                    device=reciprocal_space_grid.device,
                )
                self._sym_extractor_hkl_id = id(hkl)
            return self._sym_extractor.extract_from_grid(reciprocal_space_grid)
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

        This is a convenience method that builds the density map and computes
        structure factors in one call.

        When use_late_symmetry=True (default) and the grid is compatible,
        symmetry is applied in reciprocal space after FFT for ~5x speedup.
        Otherwise, symmetry is applied to the density map before FFT.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        xyz_iso : torch.Tensor
            Isotropic atom coordinates with shape (n_iso, 3).
        adp_iso : torch.Tensor
            Isotropic ADPs (atomic displacement parameters) with shape (n_iso,).
        occ_iso : torch.Tensor
            Isotropic occupancies with shape (n_iso,).
        A_iso : torch.Tensor
            ITC92 A parameters for isotropic atoms.
        B_iso : torch.Tensor
            ITC92 B parameters for isotropic atoms.
        xyz_aniso : torch.Tensor, optional
            Anisotropic atom coordinates.
        u_aniso : torch.Tensor, optional
            Anisotropic U parameters.
        occ_aniso : torch.Tensor, optional
            Anisotropic occupancies.
        A_aniso : torch.Tensor, optional
            ITC92 A parameters for anisotropic atoms.
        B_aniso : torch.Tensor, optional
            ITC92 B parameters for anisotropic atoms.
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
        # Decide symmetry strategy:
        # - Late symmetry: build P1 map, apply symmetry in reciprocal space
        # - Early symmetry: apply symmetry to density map before FFT
        use_late = (
            apply_symmetry
            and self.use_late_symmetry
            and self._late_symmetry_compatible
        )

        # Build density map (with or without early symmetry)
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
        # Extract structure factors (with or without late symmetry)
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
        """Drop the cached symmetry extractor; recomputed on next use."""
        self._sym_extractor = None
        self._sym_extractor_hkl_id = None

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
        new_spacegroup = self._spacegroup.copy() if self._spacegroup is not None else None

        # Create new SfFFT with copied components
        new_fft = SfFFT(
            cell=new_cell,
            spacegroup=new_spacegroup,
            max_res=self.max_res,
            radius_angstrom=self.radius_angstrom,
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
