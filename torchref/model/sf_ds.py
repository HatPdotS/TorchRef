"""
SfDS - Structure Factor calculation via Direct Summation.

This module provides a PyTorch nn.Module that handles:
- Direct summation structure factor calculations
- Support for both isotropic and anisotropic atoms
- Crystallographic symmetry handling

The SfDS class provides an alternative to SfFFT for computing structure
factors without requiring a grid-based FFT approach.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchref.base.direct_summation import (
    Engine,
    ds_aniso,
    ds_iso,
)
from torchref.base.reciprocal import (
    get_scattering_vectors,
    reciprocal_basis_matrix,
)
from torchref.config import dtypes, get_complex_dtype, get_default_device
from torchref.symmetry import Cell, SpaceGroup
from torchref.symmetry.spacegroup import SpaceGroupLike
from torchref.utils.device_mixin import DeviceMovementMixin


class SfDS(DeviceMovementMixin, nn.Module):
    """
    Structure Factor calculator using Direct Summation.

    This module computes structure factors by directly summing atomic
    contributions without building an intermediate electron density map.
    It is initialized with a Cell and optionally a SpaceGroup.

    Includes automatic batching to handle memory constraints for large
    structures or high-resolution data.

    Parameters
    ----------
    cell : Cell, optional
        Unit cell object containing cell parameters.
    spacegroup : SpaceGroupLike, optional
        Space group specification (string, int, or gemmi.SpaceGroup).
        If None, defaults to P1.
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is dtypes.float.
    device : torch.device, optional
        Computation device. Defaults to the configured device.current.
    verbose : int, optional
        Verbosity level for logging. Default is 0.
    max_memory_gb : float, optional
        Maximum memory to use for intermediate tensors in GB. Default is 2.0.
        Set to None to disable batching.

    Attributes
    ----------
    cell : Cell
        Unit cell object.
    spacegroup : SpaceGroup
        Space group object (SpaceGroup nn.Module with matrices and translations).

    Examples
    --------
    Standalone usage::

        from torchref.symmetry import Cell
        cell = Cell([50, 60, 70, 90, 90, 90])
        sf_ds = SfDS(cell, spacegroup='P212121')
        sf, _ = sf_ds.compute_structure_factors(
            hkl, xyz_iso, adp_iso, occ_iso, A_iso, B_iso
        )

    With memory limit for large structures::

        sf_ds = SfDS(cell, spacegroup='P212121', max_memory_gb=4.0)
        sf, _ = sf_ds.compute_structure_factors(...)  # Auto-batches if needed

    Notes
    -----
    Key differences from SfFFT:
    - No grid setup required
    - No build_density_map() or map_to_structure_factors() methods
    - Computes scattering factors internally from A/B ITC92 coefficients
    - Returns (sf, None) instead of (sf, density_map)
    - Automatic batching for memory management
    """

    def __init__(
        self,
        cell: Optional[Cell] = None,
        spacegroup: SpaceGroupLike = None,
        dtype_float: torch.dtype = None,
        device: torch.device = None,
        verbose: int = 0,
        max_memory_gb: float = 2.0,
        engine: Engine = Engine.AUTO,
    ):
        """
        Initialize the SfDS module with cell and spacegroup.

        Parameters
        ----------
        cell : Cell, optional
            Unit cell object. If None, must be set later.
        spacegroup : SpaceGroupLike, optional
            Space group specification. If None, defaults to P1.
        dtype_float : torch.dtype, optional
            Data type for floating point tensors. Default is dtypes.float.
        device : torch.device, optional
            Computation device. Defaults to the configured device.current.
        verbose : int, optional
            Verbosity level for logging. Default is 0.
        max_memory_gb : float, optional
            Maximum memory for intermediate tensors in GB. Default is 2.0.
        engine : Engine, optional
            Structure-factor backend selector. ``Engine.AUTO`` (default)
            derives the backend from device/dtype/availability (Triton on
            CUDA+float32, else checkpointed eager). ``Engine.TRITON`` and
            ``Engine.EAGER`` force a path (for tests/benchmarks).
        """
        super().__init__()
        if dtype_float is None:
            dtype_float = dtypes.float
        if device is None:
            device = get_default_device()
        self.dtype_float = dtype_float
        self.device = device
        self.verbose = verbose
        self.max_memory_gb = max_memory_gb
        self.engine = engine

        # Store cell and spacegroup
        self._cell = cell
        self._spacegroup = None

        if spacegroup is not None or cell is not None:
            self._spacegroup = SpaceGroup(spacegroup, dtype=dtype_float, device=device)

        # Cache reciprocal basis matrix
        self._recB: Optional[torch.Tensor] = None

    # =========================================================================
    # Cell and SpaceGroup properties
    # =========================================================================

    @property
    def cell(self) -> Optional[Cell]:
        """Unit cell object."""
        return self._cell

    @cell.setter
    def cell(self, value: Cell):
        """Set unit cell and invalidate cached reciprocal basis matrix."""
        self._cell = value
        self._recB = None  # Invalidate cache

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
        Set cell and spacegroup for this SfDS instance.

        Parameters
        ----------
        cell : Cell
            Unit cell object.
        spacegroup : SpaceGroupLike, optional
            Space group specification.
        """
        self._cell = cell
        self._recB = None  # Invalidate cache
        self.spacegroup = spacegroup

    # =========================================================================
    # Internal helper methods
    # =========================================================================

    def _get_reciprocal_basis_matrix(self) -> torch.Tensor:
        """
        Get or compute the reciprocal basis matrix.

        Returns
        -------
        torch.Tensor
            Reciprocal basis matrix of shape (3, 3) with a*, b*, c* as rows.

        Raises
        ------
        RuntimeError
            If cell has not been set.
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")

        if self._recB is None:
            self._recB = reciprocal_basis_matrix(self._cell.data)

        return self._recB

    def _compute_scattering_factors(
        self, s: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute atomic scattering factors from ITC92 A and B coefficients.

        The scattering factor is computed as:
            f(s) = sum_i A_i * exp(-B_i * s^2 / 4)

        Parameters
        ----------
        s : torch.Tensor
            Scattering vector magnitudes of shape (N_reflections,).
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (N_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (N_atoms, 5).

        Returns
        -------
        torch.Tensor
            Atomic scattering factors of shape (N_reflections, N_atoms).
        """
        # s: (N_refl,) -> (N_refl, 1, 1)
        # A, B: (N_atoms, 5)
        s_sq = (s.reshape(-1, 1, 1) ** 2) / 4  # (N_refl, 1, 1)
        # B: (N_atoms, 5) -> (1, N_atoms, 5)
        B_expanded = B.unsqueeze(0)  # (1, N_atoms, 5)
        A_expanded = A.unsqueeze(0)  # (1, N_atoms, 5)

        # Compute exponential terms: (N_refl, N_atoms, 5)
        exp_terms = torch.exp(-B_expanded * s_sq)

        # Sum over Gaussian components: (N_refl, N_atoms)
        f = torch.sum(A_expanded * exp_terms, dim=-1)

        return f

    def _get_spacegroup_callable(self):
        """
        Create a callable for direct summation functions.

        The direct summation functions expect a callable that takes
        fractional coordinates with shape (3, N) and returns coordinates
        with shape (3, N, n_ops).

        This allows the structure factor summation to:
        1. Compute h.r for all (reflection, atom, symop) combinations
        2. Sum exp(2*pi*i*h.r) over symmetry operations for each (reflection, atom)
        3. Multiply by atom-specific factors and sum over atoms

        Returns
        -------
        callable
            Function that applies symmetry operations.
        """
        if self._spacegroup is None:
            # P1 symmetry - identity operation only
            def p1_symmetry(coords_3N):
                # coords_3N: (3, N) -> (3, N, 1)
                return coords_3N.unsqueeze(2)

            return p1_symmetry

        def apply_symmetry(coords_3N):
            # coords_3N: (3, N) -> coords_N3: (N, 3)
            coords_N3 = coords_3N.T

            # Apply symmetry: (N, 3) -> (N, 3, ops)
            transformed = self._spacegroup.apply(coords_N3)

            # Reorder to (3, N, ops)
            # (N, 3, ops) -> (3, N, ops)
            result = transformed.permute(1, 0, 2)

            return result

        return apply_symmetry

    def _cartesian_to_fractional(self, xyz_cartesian: torch.Tensor) -> torch.Tensor:
        """
        Convert Cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        xyz_cartesian : torch.Tensor
            Cartesian coordinates with shape (N, 3).

        Returns
        -------
        torch.Tensor
            Fractional coordinates with shape (N, 3).
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")

        # Use Cell's to_fractional method via inv_fractional_matrix
        # fractional = cartesian @ inv_frac_matrix.T
        return torch.matmul(xyz_cartesian, self.inv_fractional_matrix.T)

    # =========================================================================
    # Structure Factor Computation
    # =========================================================================

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
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute structure factors from atomic parameters using direct summation.

        Uses "late symmetry" approach (same as SfFFT): first computes P1 structure
        factors at symmetry-equivalent HKLs, then combines them with phase shifts.

        The symmetry formula is:
            F_sym(h) = Σ_ops exp(2πi h.t) * F_P1(R^T @ h)

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        xyz_iso : torch.Tensor
            Isotropic atom coordinates (Cartesian) with shape (n_iso, 3).
        adp_iso : torch.Tensor
            Isotropic ADPs (atomic displacement parameters) with shape (n_iso,).
        occ_iso : torch.Tensor
            Isotropic occupancies with shape (n_iso,).
        A_iso : torch.Tensor
            ITC92 A parameters for isotropic atoms with shape (n_iso, 5).
        B_iso : torch.Tensor
            ITC92 B parameters for isotropic atoms with shape (n_iso, 5).
        xyz_aniso : torch.Tensor, optional
            Anisotropic atom coordinates (Cartesian) with shape (n_aniso, 3).
        u_aniso : torch.Tensor, optional
            Anisotropic U parameters with shape (n_aniso, 6).
        occ_aniso : torch.Tensor, optional
            Anisotropic occupancies with shape (n_aniso,).
        A_aniso : torch.Tensor, optional
            ITC92 A parameters for anisotropic atoms with shape (n_aniso, 5).
        B_aniso : torch.Tensor, optional
            ITC92 B parameters for anisotropic atoms with shape (n_aniso, 5).
        apply_symmetry : bool, optional
            If True, apply crystallographic symmetry. Default is True.

        Returns
        -------
        sf : torch.Tensor
            Complex structure factors with shape (n_reflections,).
        None
            Second return value is None (for API compatibility with SfFFT).
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")

        # Cache atomic parameters for reuse
        xyz_frac_iso = (
            self._cartesian_to_fractional(xyz_iso) if len(xyz_iso) > 0 else None
        )
        xyz_frac_aniso = (
            self._cartesian_to_fractional(xyz_aniso)
            if xyz_aniso is not None and len(xyz_aniso) > 0
            else None
        )

        # No symmetry: compute F_P1 directly
        if not apply_symmetry or self._spacegroup is None:
            sf_p1 = self._compute_p1_sf(
                hkl,
                xyz_frac_iso,
                adp_iso,
                occ_iso,
                A_iso,
                B_iso,
                xyz_frac_aniso,
                u_aniso,
                occ_aniso,
                A_aniso,
                B_aniso,
            )
            return sf_p1, None

        # Apply late symmetry: F_sym(h) = Σ_ops exp(2πi h.t) * F_P1(R^T @ h)
        from torchref.base.reciprocal import (
            compute_symmetry_equivalent_hkls,
            compute_translation_phases,
        )

        n_ops = self._spacegroup.n_ops
        rotation_matrices = self._spacegroup.matrices
        translations = self._spacegroup.translations

        # Compute equivalent HKLs: (n_ops, N, 3)
        equiv_hkls = compute_symmetry_equivalent_hkls(hkl, rotation_matrices)

        # Compute translation phase shifts: (n_ops, N)
        phases = compute_translation_phases(hkl, translations)

        # Compute F_P1 at each equivalent HKL and combine
        sf_total = torch.zeros(
            hkl.shape[0], dtype=get_complex_dtype(), device=self.device
        )

        for i in range(n_ops):
            equiv_hkl_i = equiv_hkls[i].to(dtype=self.dtype_float)  # (N, 3)

            sf_p1_i = self._compute_p1_sf(
                equiv_hkl_i,
                xyz_frac_iso,
                adp_iso,
                occ_iso,
                A_iso,
                B_iso,
                xyz_frac_aniso,
                u_aniso,
                occ_aniso,
                A_aniso,
                B_aniso,
            )

            # Apply phase and accumulate
            sf_total = sf_total + phases[i] * sf_p1_i

        return sf_total, None

    def _compute_p1_sf(
        self,
        hkl: torch.Tensor,
        xyz_frac_iso: Optional[torch.Tensor],
        adp_iso: torch.Tensor,
        occ_iso: torch.Tensor,
        A_iso: torch.Tensor,
        B_iso: torch.Tensor,
        xyz_frac_aniso: Optional[torch.Tensor],
        u_aniso: Optional[torch.Tensor],
        occ_aniso: Optional[torch.Tensor],
        A_aniso: Optional[torch.Tensor],
        B_aniso: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute P1 structure factors (no symmetry expansion).

        Dispatches to the capability-selected backend (Triton on CUDA+float32,
        else checkpointed eager). The backend computes scattering factors
        internally from A/B and never materializes a large (N_refl, N_atoms)
        intermediate.
        """
        # Get reciprocal basis matrix and compute scattering vectors
        recB = self._get_reciprocal_basis_matrix()
        s_vectors = get_scattering_vectors(hkl, self._cell.data, recB)
        s = torch.norm(s_vectors, dim=1)

        sf_total = torch.zeros(
            hkl.shape[0], dtype=get_complex_dtype(), device=self.device
        )

        # Compute isotropic contribution
        if xyz_frac_iso is not None and len(xyz_frac_iso) > 0:
            sf_iso = ds_iso(
                hkl,
                s,
                xyz_frac_iso,
                occ_iso,
                adp_iso,
                A_iso,
                B_iso,
                engine=self.engine,
                max_memory_gb=self.max_memory_gb,
            )
            sf_total = sf_total + sf_iso.to(sf_total.dtype)

        # Compute anisotropic contribution
        if xyz_frac_aniso is not None and len(xyz_frac_aniso) > 0:
            sf_aniso = ds_aniso(
                hkl,
                s_vectors,
                xyz_frac_aniso,
                occ_aniso,
                u_aniso,
                A_aniso,
                B_aniso,
                engine=self.engine,
                max_memory_gb=self.max_memory_gb,
            )
            sf_total = sf_total + sf_aniso.to(sf_total.dtype)

        return sf_total

    # =========================================================================
    # Device Movement
    # =========================================================================

    def reset_cache(self) -> None:
        """Drop the cached reciprocal-basis matrix; recomputed on next use."""
        self._recB = None

    def copy(self) -> "SfDS":
        """Create a deep copy of this SfDS module.

        Returns
        -------
        SfDS
            A new SfDS instance with cloned cell and spacegroup.
        """
        # Clone the cell
        new_cell = self._cell.clone() if self._cell is not None else None

        # Copy the spacegroup
        new_spacegroup = (
            self._spacegroup.copy() if self._spacegroup is not None else None
        )

        # Create new SfDS with copied components
        new_ds = SfDS(
            cell=new_cell,
            spacegroup=new_spacegroup,
            dtype_float=self.dtype_float,
            device=self.device,
            verbose=self.verbose,
            max_memory_gb=self.max_memory_gb,
            engine=self.engine,
        )

        return new_ds
