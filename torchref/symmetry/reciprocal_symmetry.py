"""
Reciprocal space symmetry operations for structure factor grids.

This module provides efficient symmetry operations for reciprocal space data (h, k, l grids),
analogous to map_symmetry.py for real space density maps.

Key concepts:
- Miller indices transform as h' = h @ R^T under rotation R
- Systematic absences occur when translation causes destructive interference
- Centric reflections have phases restricted to 0 or π
- Friedel pairs: F(h,k,l) = F*(-h,-k,-l) for normal scattering

Main interfaces:
- ReciprocalSymmetry: Factory function for grid-based reciprocal space symmetry
- expand_reflections: Expand ReflectionData from asymmetric unit to P1
- expand_reciprocal_grid: Expand a reciprocal space grid from asymmetric unit to P1

Space groups can be specified as strings, integers (1-230), or gemmi.SpaceGroup objects.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torchref.symmetry.spacegroup import SpaceGroup, SpaceGroupLike

if TYPE_CHECKING:
    from torchref.io.datasets.reflection_data import ReflectionData


def ReciprocalSymmetry(
    space_group: SpaceGroupLike,
    grid_shape,
    dtype_float=torch.float32,
    verbose=1,
    device=torch.device("cpu"),
):
    """
    Factory function to create the appropriate ReciprocalSymmetry implementation.

    Parameters
    ----------
    space_group : str, int, or gemmi.SpaceGroup
        Space group specification (e.g., 'P21', 4, gemmi.SpaceGroup('P 21')).
    grid_shape : tuple of int
        Shape of the reciprocal space grid (nh, nk, nl).
        The grid spans from -n//2 to n//2 for each dimension.
    dtype_float : torch.dtype, default torch.float32
        Floating point precision to use.
    verbose : int, default 1
        Verbosity level (0=silent, 1=info, 2=debug).
    device : torch.device, default torch.device('cpu')
        Device to use for computation.

    Returns
    -------
    ReciprocalSymmetryGrid
        Implementation for reciprocal space grid symmetry operations.
    """
    return ReciprocalSymmetryGrid(space_group, grid_shape, dtype_float, verbose, device)


class ReciprocalSymmetryGrid(nn.Module):
    """
    Reciprocal space symmetry operations for Miller index grids.

    Handles symmetry operations in reciprocal space including:
    - Miller index transformation under symmetry operations
    - Systematic absence detection
    - Centric reflection identification
    - Friedel pair handling
    - Symmetry expansion/averaging of structure factors

    In reciprocal space, symmetry operations transform Miller indices as:
        (h', k', l') = (h, k, l) @ R^T

    where R is the rotation matrix from real space symmetry.

    Attributes
    ----------
    space_group : str
        Space group name.
    grid_shape : tuple of int
        Shape of the reciprocal space grid (nh, nk, nl).
    symmetry : SpaceGroup
        Base symmetry operations handler.
    n_ops : int
        Number of symmetry operations.

    Examples
    --------
    ::

        recip_sym = ReciprocalSymmetry('P21', grid_shape=(64, 64, 64))
        F_expanded = recip_sym(F_asym)  # Expand from asymmetric unit
        F_avg = recip_sym.symmetry_average(F_full)  # Average symmetry-related reflections
    """

    def __init__(
        self,
        space_group,
        grid_shape,
        dtype_float=torch.float32,
        verbose=1,
        device=torch.device("cpu"),
    ):
        """
        Initialize reciprocal space symmetry operator.

        Parameters
        ----------
        space_group : str
            Space group name.
        grid_shape : tuple of int
            Shape of the reciprocal space grid (nh, nk, nl).
        dtype_float : torch.dtype, default torch.float32
            Floating point precision.
        verbose : int, default 1
            Verbosity level.
        device : torch.device, default torch.device('cpu')
            Computation device.
        """
        super().__init__()
        self.dtype_float = dtype_float
        self.space_group = space_group
        self.grid_shape = tuple(grid_shape)
        self.verbose = verbose
        self.device = device

        # Get symmetry operations from base class
        self.symmetry = SpaceGroup(space_group, dtype=dtype_float, device=device)
        self.n_ops = self.symmetry.matrices.shape[0]

        # Precompute reciprocal space rotation matrices (transpose of real space)
        # In reciprocal space: h' = R^T @ h  (or equivalently h' = h @ R)
        self._setup_reciprocal_matrices()

        if self.verbose > 0:
            print(f"ReciprocalSymmetryGrid initialized for {space_group}")
            print(f"  Number of symmetry operations: {self.n_ops}")
            print(f"  Grid shape: {self.grid_shape}")

        # Setup Miller index grid
        self._setup_hkl_grid()

        # Precompute index mappings for each symmetry operation
        self._setup_symmetry_index_grids()

        # Compute systematic absences mask
        self._setup_systematic_absences()

        # Compute centric reflection mask
        self._setup_centric_reflections()

        if self.verbose > 0:
            n_absent = self.systematic_absences.sum().item()
            n_centric = self.centric_mask.sum().item()
            n_total = np.prod(self.grid_shape)
            print(f"  Systematic absences: {n_absent} ({100*n_absent/n_total:.2f}%)")
            print(f"  Centric reflections: {n_centric} ({100*n_centric/n_total:.2f}%)")

    def _setup_reciprocal_matrices(self):
        """
        Setup rotation matrices for reciprocal space.

        In reciprocal space, the transformation is h' = R^T @ h
        where R is the real-space rotation matrix.
        """
        # Transpose each rotation matrix for reciprocal space
        # Real space: r' = R @ r + t
        # Reciprocal space: h' = R^T @ h (translations cause phase shifts, not index changes)
        recip_matrices = self.symmetry.matrices.transpose(-2, -1).contiguous()
        self.register_buffer("reciprocal_matrices", recip_matrices)

    def _setup_hkl_grid(self):
        """
        Setup Miller index grid.

        Creates a grid where indices span from -n//2 to n//2 for each dimension,
        following the FFT convention (0...n//2, -n//2+1...-1).
        """
        nh, nk, nl = self.grid_shape

        # Create index arrays following FFT convention
        h = torch.fft.fftfreq(nh, d=1.0) * nh  # gives 0,1,2,...,n//2,-n//2+1,...,-1
        k = torch.fft.fftfreq(nk, d=1.0) * nk
        l = torch.fft.fftfreq(nl, d=1.0) * nl

        h = h.to(dtype=torch.int64, device=self.device)
        k = k.to(dtype=torch.int64, device=self.device)
        l = l.to(dtype=torch.int64, device=self.device)

        # Create 3D grid of Miller indices
        grid_h, grid_k, grid_l = torch.meshgrid(h, k, l, indexing="ij")
        hkl_grid = torch.stack([grid_h, grid_k, grid_l], dim=-1)

        self.register_buffer("hkl_grid", hkl_grid)

        # Also store as float for matrix operations
        hkl_grid_float = hkl_grid.to(dtype=self.dtype_float)
        self.register_buffer("hkl_grid_float", hkl_grid_float)

    def _setup_symmetry_index_grids(self):
        """
        Precompute index grids mapping each (h,k,l) to its symmetry equivalents.

        For each symmetry operation, computes where each reflection maps to
        in the grid, allowing efficient gathering/scattering operations.
        """
        nh, nk, nl = self.grid_shape
        hkl_flat = self.hkl_grid_float.reshape(-1, 3)  # (N, 3)

        index_grids_list = []
        phase_shift_grids_list = []

        for i in range(self.n_ops):
            # Transform Miller indices: h' = R^T @ h
            # Using batch matrix multiply: (N, 3) @ (3, 3) -> (N, 3)
            transformed = torch.matmul(hkl_flat, self.reciprocal_matrices[i].T)

            # Round to nearest integer (should be exact for valid symmetry ops)
            transformed_int = torch.round(transformed).to(torch.int64)

            # Compute phase shift from translation: phase = 2π * h · t
            # This phase modifies the structure factor: F(h') = F(h) * exp(2πi h·t)
            translation = self.symmetry.translations[i]
            phase_shift = 2.0 * np.pi * torch.matmul(hkl_flat, translation)
            phase_shift = phase_shift.reshape(nh, nk, nl)
            phase_shift_grids_list.append(phase_shift)

            # Convert to grid indices (wrap with periodic boundary)
            idx_h = transformed_int[:, 0] % nh
            idx_k = transformed_int[:, 1] % nk
            idx_l = transformed_int[:, 2] % nl

            # Stack indices
            index_grid = torch.stack([idx_h, idx_k, idx_l], dim=-1)
            index_grid = index_grid.reshape(nh, nk, nl, 3)
            index_grids_list.append(index_grid)

        # Stack all grids: (n_ops, nh, nk, nl, 3)
        index_grids = torch.stack(index_grids_list, dim=0)
        self.register_buffer("index_grids", index_grids)

        # Stack phase shifts: (n_ops, nh, nk, nl)
        phase_shifts = torch.stack(phase_shift_grids_list, dim=0)
        self.register_buffer("phase_shifts", phase_shifts)

    def _setup_systematic_absences(self):
        """
        Compute mask for systematic absences.

        A reflection (h,k,l) is systematically absent if for some symmetry operation
        with translation t: h' = h (maps to itself) AND h·t ≠ 0 (mod 1).

        This causes destructive interference from the translation component.
        """
        nh, nk, nl = self.grid_shape
        absences = torch.zeros(self.grid_shape, dtype=torch.bool, device=self.device)

        hkl_flat = self.hkl_grid_float.reshape(-1, 3)

        for i in range(self.n_ops):
            # Transform indices
            transformed = torch.matmul(hkl_flat, self.reciprocal_matrices[i].T)
            transformed_int = torch.round(transformed).to(torch.int64)

            # Check if h' ≡ h (reflection maps to itself)
            hkl_int = self.hkl_grid.reshape(-1, 3)
            same_reflection = (transformed_int == hkl_int).all(dim=-1)

            # Compute phase shift h·t
            translation = self.symmetry.translations[i]
            h_dot_t = torch.matmul(hkl_flat, translation)

            # Check if phase is non-integer (mod 1 ≠ 0)
            # A reflection is absent if it maps to itself with non-zero phase
            phase_mod = torch.abs(h_dot_t - torch.round(h_dot_t))
            non_zero_phase = phase_mod > 1e-6

            # Mark as absent
            absent_mask = (same_reflection & non_zero_phase).reshape(self.grid_shape)
            absences = absences | absent_mask

        self.register_buffer("systematic_absences", absences)

    def _setup_centric_reflections(self):
        """
        Compute mask for centric reflections.

        A reflection is centric if there exists a symmetry operation that maps
        h to -h (including Friedel mate consideration). Centric reflections
        have phases restricted to 0 or π.
        """
        nh, nk, nl = self.grid_shape
        centric = torch.zeros(self.grid_shape, dtype=torch.bool, device=self.device)

        hkl_flat = self.hkl_grid_float.reshape(-1, 3)

        for i in range(self.n_ops):
            # Transform indices
            transformed = torch.matmul(hkl_flat, self.reciprocal_matrices[i].T)
            transformed_int = torch.round(transformed).to(torch.int64)

            # Check if h' ≡ -h (maps to Friedel mate)
            hkl_int = self.hkl_grid.reshape(-1, 3)
            maps_to_minus_h = (transformed_int == -hkl_int).all(dim=-1)

            centric_mask = maps_to_minus_h.reshape(self.grid_shape)
            centric = centric | centric_mask

        self.register_buffer("centric_mask", centric)

    def apply_to_indices(self, hkl, operation_index=None):
        """
        Apply symmetry operation(s) to Miller indices.

        Parameters
        ----------
        hkl : torch.Tensor, shape (..., 3)
            Miller indices (h, k, l).
        operation_index : int, optional
            If specified, apply only this operation.
            If None, apply all operations.

        Returns
        -------
        torch.Tensor
            Transformed Miller indices.
            If operation_index is None: shape (n_ops, ..., 3)
            Otherwise: shape (..., 3)
        """
        hkl = hkl.to(dtype=self.dtype_float, device=self.device)
        original_shape = hkl.shape[:-1]

        if operation_index is not None:
            # Apply single operation
            R = self.reciprocal_matrices[operation_index]
            transformed = torch.matmul(hkl, R.T)
            return torch.round(transformed).to(torch.int64)
        else:
            # Apply all operations
            hkl_flat = hkl.reshape(-1, 3)  # (N, 3)
            results = []
            for i in range(self.n_ops):
                R = self.reciprocal_matrices[i]
                transformed = torch.matmul(hkl_flat, R.T)
                results.append(torch.round(transformed).to(torch.int64))

            # Stack: (n_ops, N, 3)
            stacked = torch.stack(results, dim=0)
            return stacked.reshape(self.n_ops, *original_shape, 3)

    def get_phase_shift(self, hkl, operation_index):
        """
        Get phase shift for a symmetry operation on given Miller indices.

        The phase shift is exp(2πi h·t) where t is the translation.

        Parameters
        ----------
        hkl : torch.Tensor, shape (..., 3)
            Miller indices.
        operation_index : int
            Symmetry operation index.

        Returns
        -------
        torch.Tensor
            Phase shift in radians, shape (...).
        """
        hkl = hkl.to(dtype=self.dtype_float, device=self.device)
        translation = self.symmetry.translations[operation_index]
        phase = 2.0 * np.pi * torch.matmul(hkl, translation)
        return phase

    def get_symmetry_mate(self, F_grid, operation_index):
        """
        Apply a single symmetry operation to a structure factor grid.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.
        operation_index : int
            Index of the symmetry operation (0 to n_ops-1).

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Structure factors after applying symmetry operation.
            Includes phase shift from translation component.
        """
        if operation_index < 0 or operation_index >= self.n_ops:
            raise ValueError(
                f"Operation index {operation_index} out of range [0, {self.n_ops-1}]"
            )

        if F_grid.shape != self.grid_shape:
            raise ValueError(
                f"Grid shape {F_grid.shape} doesn't match expected {self.grid_shape}"
            )

        # Get precomputed index grid for this operation
        idx_grid = self.index_grids[operation_index]  # (nh, nk, nl, 3)

        # Gather structure factors from transformed positions
        F_transformed = F_grid[idx_grid[..., 0], idx_grid[..., 1], idx_grid[..., 2]]

        # Apply phase shift from translation: F(h') = F(h) * exp(2πi h·t)
        phase = self.phase_shifts[operation_index]
        if F_grid.is_complex():
            phase_factor = torch.exp(1j * phase.to(F_grid.dtype))
            F_transformed = F_transformed * phase_factor
        # For real-valued grids (amplitudes), no phase shift needed

        return F_transformed

    def get_all_symmetry_mates(self, F_grid):
        """
        Get all symmetry-related structure factor grids.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.

        Returns
        -------
        list of torch.Tensor
            List of symmetry-related grids.
        """
        return [self.get_symmetry_mate(F_grid, i) for i in range(self.n_ops)]

    def symmetry_average(self, F_grid, weights=None):
        """
        Average structure factors over all symmetry equivalents.

        This is useful for enforcing symmetry constraints on structure factors.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.
        weights : torch.Tensor, optional
            Weights for averaging, shape (nh, nk, nl).
            If None, equal weights are used.

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Symmetry-averaged structure factors.
        """
        mates = self.get_all_symmetry_mates(F_grid)
        stacked = torch.stack(mates, dim=0)  # (n_ops, nh, nk, nl)

        if weights is not None:
            weights = weights.unsqueeze(0)  # (1, nh, nk, nl)
            stacked = stacked * weights
            return stacked.sum(dim=0) / (weights.sum() * self.n_ops)
        else:
            return stacked.mean(dim=0)

    def expand_to_p1(self, F_asym, asym_mask=None):
        """
        Expand structure factors from asymmetric unit to full P1.

        Takes structure factors defined on the asymmetric unit and
        generates the full reciprocal space by applying all symmetry
        operations.

        Parameters
        ----------
        F_asym : torch.Tensor, shape (nh, nk, nl)
            Structure factors on asymmetric unit (other positions can be zero).
        asym_mask : torch.Tensor, optional
            Boolean mask indicating asymmetric unit positions.
            If None, non-zero values are assumed to be the asymmetric unit.

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Full structure factor grid with all symmetry equivalents filled.
        """
        F_full = F_asym.clone()

        for i in range(1, self.n_ops):  # Skip identity
            F_mate = self.get_symmetry_mate(F_asym, i)

            # Only fill in positions that are zero (not yet set)
            if F_full.is_complex():
                mask = F_full.abs() < 1e-10
            else:
                mask = F_full.abs() < 1e-10

            F_full = torch.where(mask, F_mate, F_full)

        return F_full

    def apply_friedel(self, F_grid):
        """
        Apply Friedel's law: F(-h,-k,-l) = F*(h,k,l).

        For normal (non-anomalous) scattering, the structure factor
        at -h is the complex conjugate of F(h).

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Structure factors with Friedel symmetry enforced.
        """
        # Flip all indices: F(-h,-k,-l)
        F_friedel = torch.flip(F_grid, dims=[0, 1, 2])

        # Roll to handle the asymmetry at 0 index
        nh, nk, nl = self.grid_shape
        F_friedel = torch.roll(F_friedel, shifts=(1, 1, 1), dims=(0, 1, 2))

        if F_grid.is_complex():
            F_friedel = F_friedel.conj()

        # Average F(h) and F*(-h)
        return 0.5 * (F_grid + F_friedel)

    def is_systematic_absence(self, h, k, l):
        """
        Check if a reflection is systematically absent.

        Parameters
        ----------
        h, k, l : int
            Miller indices.

        Returns
        -------
        bool
            True if the reflection is systematically absent.
        """
        nh, nk, nl = self.grid_shape
        idx_h = h % nh
        idx_k = k % nk
        idx_l = l % nl
        return self.systematic_absences[idx_h, idx_k, idx_l].item()

    def is_centric(self, h, k, l):
        """
        Check if a reflection is centric.

        Parameters
        ----------
        h, k, l : int
            Miller indices.

        Returns
        -------
        bool
            True if the reflection is centric (phase restricted to 0 or π).
        """
        nh, nk, nl = self.grid_shape
        idx_h = h % nh
        idx_k = k % nk
        idx_l = l % nl
        return self.centric_mask[idx_h, idx_k, idx_l].item()

    def get_epsilon(self):
        """
        Compute epsilon (multiplicity) factors for each reflection.

        Epsilon is the number of symmetry operations that map h to itself
        (or to its Friedel mate for acentric space groups).

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Epsilon factors for each reflection.
        """
        epsilon = torch.zeros(self.grid_shape, dtype=torch.int32, device=self.device)

        hkl_flat = self.hkl_grid.reshape(-1, 3)

        for i in range(self.n_ops):
            # Get transformed indices
            idx_grid = self.index_grids[i]
            transformed_flat = idx_grid.reshape(-1, 3)

            # Check if h' ≡ h or h' ≡ -h (same reflection or Friedel)
            same = (transformed_flat == hkl_flat).all(dim=-1)

            nh, nk, nl = self.grid_shape
            # Also check Friedel (-h, -k, -l)
            hkl_neg = (-self.hkl_grid).reshape(-1, 3)
            hkl_neg[:, 0] = hkl_neg[:, 0] % nh
            hkl_neg[:, 1] = hkl_neg[:, 1] % nk
            hkl_neg[:, 2] = hkl_neg[:, 2] % nl

            friedel = (transformed_flat == hkl_neg).all(dim=-1)

            contributes = (same | friedel).reshape(self.grid_shape)
            epsilon += contributes.to(torch.int32)

        return epsilon

    def forward(self, F_grid, mode="average"):
        """
        Apply symmetry to structure factor grid.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.
        mode : str, default 'average'
            Operation mode:
            - 'average': Average over all symmetry equivalents
            - 'expand': Expand from asymmetric unit to full grid
            - 'sum': Sum all symmetry mates (for accumulation)

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Processed structure factor grid.
        """
        if mode == "average":
            return self.symmetry_average(F_grid)
        elif mode == "expand":
            return self.expand_to_p1(F_grid)
        elif mode == "sum":
            mates = self.get_all_symmetry_mates(F_grid)
            return torch.stack(mates, dim=0).sum(dim=0)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'average', 'expand', or 'sum'."
            )

    def __call__(self, F_grid, mode="average"):
        """Make the class callable."""
        return self.forward(F_grid, mode=mode)

    def cuda(self):
        """Move to GPU."""
        super().cuda()
        self.device = torch.device("cuda")
        return self

    def cpu(self):
        """Move to CPU."""
        super().cpu()
        self.device = torch.device("cpu")
        return self

    def to(self, device):
        """Move to specified device."""
        super().to(device)
        self.device = device
        return self

    def get_symmetry_info(self):
        """
        Get information about reciprocal space symmetry.

        Returns
        -------
        dict
            Dictionary with symmetry information.
        """
        return {
            "space_group": self.space_group,
            "n_operations": self.n_ops,
            "reciprocal_matrices": self.reciprocal_matrices,
            "translations": self.symmetry.translations,
            "n_systematic_absences": self.systematic_absences.sum().item(),
            "n_centric": self.centric_mask.sum().item(),
            "grid_shape": self.grid_shape,
        }

    def __repr__(self):
        return (
            f"ReciprocalSymmetryGrid(space_group='{self.space_group}', "
            f"n_ops={self.n_ops}, grid_shape={self.grid_shape})"
        )


# =============================================================================
# Standalone functions for symmetry expansion
# =============================================================================


def expand_hkl(
    hkl: torch.Tensor,
    spacegroup: SpaceGroupLike,
    include_friedel: bool = True,
    remove_absences: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand Miller indices under crystallographic symmetry.

    This is the core low-level function for HKL expansion. It takes Miller
    indices and a space group, and returns the expanded indices along with
    an index mapping and phase offsets needed to expand any associated data.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Input Miller indices (asymmetric unit).
    spacegroup : str, int, or gemmi.SpaceGroup
        Space group specification.
    include_friedel : bool, default True
        Include Friedel mates (-h, -k, -l).
    remove_absences : bool, default True
        Remove systematically absent reflections.
    device : torch.device, optional
        Computation device. If None, uses hkl's device.

    Returns
    -------
    expanded_hkl : torch.Tensor, shape (M, 3), dtype=int32
        All unique expanded Miller indices.
    orig_indices : torch.Tensor, shape (M,), dtype=int64
        Index mapping expanded → original reflection.
        Use to expand any data: ``F_expanded = F_orig[orig_indices]``
    phase_shifts : torch.Tensor, shape (M,), dtype=float32
        Phase offsets from translations (radians).
        Apply to phases: ``phase_expanded = phase_orig[orig_indices] + phase_shifts``

    Examples
    --------
    ::

        import torch
        from torchref.symmetry import expand_hkl

        hkl_asu = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.int32)
        hkl_p1, indices, phases = expand_hkl(hkl_asu, 'P21')

        # Expand amplitude data
        F_asu = torch.tensor([100.0, 80.0, 75.0])
        F_p1 = F_asu[indices]

        # Expand phase data (apply phase shifts)
        phi_asu = torch.tensor([0.0, 1.5, 2.0])
        phi_p1 = phi_asu[indices] + phases
    """
    if device is None:
        device = hkl.device

    # Get symmetry operations
    symmetry = SpaceGroup(spacegroup, dtype=torch.float32, device=device)
    n_ops = symmetry.matrices.shape[0]

    # Reciprocal space matrices (transpose of real space)
    recip_matrices = symmetry.matrices.transpose(-2, -1)
    translations = symmetry.translations

    # Convert hkl to float for matrix operations
    hkl_float = hkl.to(dtype=torch.float32, device=device)
    n_orig = len(hkl_float)

    # Apply all symmetry operations
    all_hkl = []
    all_phases = []

    for i in range(n_ops):
        # h' = h @ R^T
        hkl_transformed = torch.round(torch.matmul(hkl_float, recip_matrices[i].T)).to(
            torch.int32
        )
        # Phase shift from translation: 2π × h·t
        phase_shift = 2.0 * np.pi * torch.matmul(hkl_float, translations[i])

        all_hkl.append(hkl_transformed)
        all_phases.append(phase_shift)

    # Add Friedel mates if requested
    if include_friedel:
        for i in range(n_ops):
            all_hkl.append(-all_hkl[i])
            all_phases.append(-all_phases[i])

    # Stack all transformed hkl and phases
    hkl_expanded = torch.cat(all_hkl, dim=0)
    phases_expanded = torch.cat(all_phases, dim=0)

    # Remove duplicates - keep unique (h,k,l) tuples with index mapping
    hkl_np = hkl_expanded.cpu().numpy()
    phase_np = phases_expanded.cpu().numpy()

    # Build dictionary: key=(h,k,l), value=(first_occurrence_idx, phase)
    unique_dict = {}
    for idx, (h, phase) in enumerate(zip(hkl_np, phase_np)):
        key = tuple(h)
        if key not in unique_dict:
            unique_dict[key] = (idx, phase)

    # Extract unique data
    unique_indices = [v[0] for v in unique_dict.values()]
    unique_phases = [v[1] for v in unique_dict.values()]

    # Map back to original reflection index
    n_total_ops = n_ops * (2 if include_friedel else 1)
    orig_indices = [idx % n_orig for idx in unique_indices]

    # Build output tensors
    expanded_hkl = torch.tensor(
        [list(k) for k in unique_dict.keys()], dtype=torch.int32, device=device
    )
    phase_shifts = torch.tensor(unique_phases, dtype=torch.float32, device=device)
    orig_idx_tensor = torch.tensor(orig_indices, dtype=torch.int64, device=device)

    # Remove systematic absences if requested
    sg = SpaceGroup(spacegroup)
    is_p1 = sg.number == 1

    if remove_absences and not is_p1:
        absence_mask = _check_systematic_absences(
            expanded_hkl, symmetry.matrices, translations, device
        )
        keep_mask = ~absence_mask

        expanded_hkl = expanded_hkl[keep_mask]
        phase_shifts = phase_shifts[keep_mask]
        orig_idx_tensor = orig_idx_tensor[keep_mask]

    return expanded_hkl, orig_idx_tensor, phase_shifts


def complete_hkl(
    input_hkl: torch.Tensor,
    cell: torch.Tensor,
    spacegroup: SpaceGroupLike,
    d_min: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complete a set of Miller indices by identifying missing reflections.

    Generates all possible reflections within the resolution limit for the
    given spacegroup (removing systematic absences), then maps the input
    reflections to this complete set.

    NOTE: This does NOT expand symmetry - stays in the same spacegroup.
    Use this to identify which reflections are missing from a dataset.

    Parameters
    ----------
    input_hkl : torch.Tensor, shape (N, 3)
        Input Miller indices (may be incomplete).
    cell : torch.Tensor, shape (6,)
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str, int, or gemmi.SpaceGroup
        Space group specification.
    d_min : float
        High resolution limit in Angstroms.
    device : torch.device, optional
        Computation device. If None, uses input_hkl's device.

    Returns
    -------
    complete_hkl : torch.Tensor, shape (M, 3), dtype int32
        All possible Miller indices within resolution (minus systematic absences).
    input_indices : torch.Tensor, shape (M,), dtype int64
        Index mapping: complete → input. For present reflections, gives
        the index in input_hkl. For missing reflections, gives -1.
        Use: ``F_complete[~missing] = F_input[input_indices[~missing]]``
    missing_mask : torch.Tensor, shape (M,), dtype bool
        True where reflection is missing from input.

    Examples
    --------
    ::

        import torch
        from torchref.symmetry import complete_hkl

        # Incomplete dataset
        input_hkl = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.int32)
        cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])

        complete, indices, missing = complete_hkl(input_hkl, cell, 'P21', d_min=10.0)

        # Fill F values
        F_input = torch.tensor([100.0, 80.0])
        F_complete = torch.zeros(len(complete))
        present_mask = ~missing
        F_complete[present_mask] = F_input[indices[present_mask]]
    """
    from torchref.base.reciprocal import generate_possible_hkl

    if device is None:
        device = input_hkl.device

    # Generate all possible HKL within resolution
    all_hkl = generate_possible_hkl(cell, d_min, device=device)

    # Get symmetry operations for absence check
    symmetry = SpaceGroup(spacegroup, dtype=torch.float32, device=device)
    translations = symmetry.translations

    # Remove systematic absences
    sg = SpaceGroup(spacegroup)
    is_p1 = sg.number == 1

    if not is_p1:
        absence_mask = _check_systematic_absences(
            all_hkl, symmetry.matrices, translations, device
        )
        all_hkl = all_hkl[~absence_mask]

    # Build lookup dictionary from input hkl to indices
    input_hkl_np = input_hkl.cpu().numpy()
    input_lookup = {}
    for idx, hkl in enumerate(input_hkl_np):
        key = tuple(hkl)
        input_lookup[key] = idx

    # Match complete set to input
    all_hkl_np = all_hkl.cpu().numpy()
    n_complete = len(all_hkl)

    input_indices = torch.full((n_complete,), -1, dtype=torch.int64, device=device)
    missing_mask = torch.ones(n_complete, dtype=torch.bool, device=device)

    for i, hkl in enumerate(all_hkl_np):
        key = tuple(hkl)
        if key in input_lookup:
            input_indices[i] = input_lookup[key]
            missing_mask[i] = False

    return all_hkl, input_indices, missing_mask


def expand_reflections(
    reflection_data: "ReflectionData",
    include_friedel: bool = True,
    remove_absences: bool = True,
    verbose: int = 1,
) -> "ReflectionData":
    """
    Expand reflection data from asymmetric unit to P1 using symmetry operations.

    Takes a ReflectionData object containing reflections in the asymmetric unit
    and generates all symmetry-equivalent reflections, returning a new
    ReflectionData object with the expanded set.

    This is a high-level wrapper around expand_hkl() that handles all
    ReflectionData fields automatically.

    Parameters
    ----------
    reflection_data : ReflectionData
        Input reflection data with hkl, F, F_sigma, etc.
    include_friedel : bool, default True
        If True, also include Friedel mates (-h, -k, -l).
    remove_absences : bool, default True
        If True, remove systematically absent reflections from output.
    verbose : int, default 1
        Verbosity level.

    Returns
    -------
    ReflectionData
        New ReflectionData object with expanded reflections.
        The spacegroup is set to 'P1' since symmetry has been expanded.

    See Also
    --------
    expand_hkl : Low-level function for HKL expansion without ReflectionData.

    Examples
    --------
    ::

        from torchref.io.datasets.reflection_data import ReflectionData
        from torchref.symmetry import expand_reflections

        data = ReflectionData().load_mtz('data.mtz')
        data_p1 = expand_reflections(data)
        print(f"Expanded from {len(data)} to {len(data_p1)} reflections")
    """
    from torchref.io.datasets.reflection_data import ReflectionData as RefData

    if reflection_data.hkl is None:
        raise ValueError("ReflectionData has no Miller indices loaded")

    space_group = reflection_data.spacegroup or "P1"
    device = reflection_data.device
    n_orig = len(reflection_data.hkl)

    if verbose > 0:
        symmetry = SpaceGroup(space_group, dtype=torch.float32, device=device)
        print(f"Expanding reflections for {space_group}")
        print(f"  Original reflections: {n_orig}")
        print(f"  Symmetry operations: {symmetry.n_ops}")

    # Use low-level expand_hkl for core expansion
    hkl_expanded, orig_idx_tensor, phase_shifts = expand_hkl(
        reflection_data.hkl,
        space_group,
        include_friedel=include_friedel,
        remove_absences=remove_absences,
        device=device,
    )

    if verbose > 0:
        print(f"  After expansion: {len(hkl_expanded)} unique reflections")

    # Create new ReflectionData
    expanded = RefData(verbose=reflection_data.verbose, device=device)

    # Copy and expand data fields using orig_idx_tensor
    expanded.hkl = hkl_expanded
    expanded.cell = (
        reflection_data.cell.clone() if reflection_data.cell is not None else None
    )
    expanded.spacegroup = SpaceGroup("P1")  # Now in P1 since symmetry is expanded

    # Expand amplitude data
    if reflection_data.F is not None:
        expanded.F = reflection_data.F[orig_idx_tensor]

    if reflection_data.F_sigma is not None:
        expanded.F_sigma = reflection_data.F_sigma[orig_idx_tensor]

    if reflection_data.I is not None:
        expanded.I = reflection_data.I[orig_idx_tensor]

    if hasattr(reflection_data, "I_sigma") and reflection_data.I_sigma is not None:
        expanded.I_sigma = reflection_data.I_sigma[orig_idx_tensor]

    # Handle phases - apply phase shift from translation
    if hasattr(reflection_data, "phase") and reflection_data.phase is not None:
        expanded.phase = reflection_data.phase[orig_idx_tensor] + phase_shifts
    else:
        # Store phase shifts for potential later use
        expanded._expansion_phase_shifts = phase_shifts

    # Figure of merit - expand from original
    if hasattr(reflection_data, "fom") and reflection_data.fom is not None:
        expanded.fom = reflection_data.fom[orig_idx_tensor]

    # R-free flags - expand from original
    if reflection_data.rfree_flags is not None:
        expanded.rfree_flags = reflection_data.rfree_flags[orig_idx_tensor]

    # Recalculate resolution
    if expanded.cell is not None:
        expanded._calculate_resolution()

    # Clear bin indices (invalidated by expansion)
    expanded.bin_indices = None

    # Copy metadata
    expanded.amplitude_source = reflection_data.amplitude_source
    expanded.intensity_source = reflection_data.intensity_source
    expanded.phase_source = reflection_data.phase_source
    expanded.rfree_source = reflection_data.rfree_source

    # Track provenance
    expanded.source = reflection_data
    expanded.last_op = f"expand_to_p1(include_friedel={include_friedel})"

    return expanded


def _check_systematic_absences(
    hkl: torch.Tensor,
    matrices: torch.Tensor,
    translations: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Check which reflections are systematically absent.

    A reflection is absent if some symmetry operation maps h to h
    with a non-integer phase shift (h·t not integer).

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Miller indices to check.
    matrices : torch.Tensor, shape (n_ops, 3, 3)
        Rotation matrices.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors.
    device : torch.device
        Computation device.

    Returns
    -------
    torch.Tensor
        Boolean mask, True for systematically absent reflections.
    """
    n_refl = len(hkl)
    n_ops = matrices.shape[0]
    absent = torch.zeros(n_refl, dtype=torch.bool, device=device)

    hkl_float = hkl.to(dtype=torch.float32, device=device)
    recip_matrices = matrices.transpose(-2, -1).to(dtype=torch.float32, device=device)
    translations = translations.to(dtype=torch.float32, device=device)

    for i in range(n_ops):
        R = recip_matrices[i]
        t = translations[i]

        # Transform: h' = h @ R^T
        hkl_transformed = torch.matmul(hkl_float, R.T)
        hkl_transformed_int = torch.round(hkl_transformed).to(torch.int32)

        # Check if h' == h (maps to itself)
        same_reflection = (hkl_transformed_int == hkl).all(dim=-1)

        # Compute phase shift h·t
        h_dot_t = torch.matmul(hkl_float, t)

        # Check if phase is non-integer (indicates absence)
        phase_mod = torch.abs(h_dot_t - torch.round(h_dot_t))
        non_integer_phase = phase_mod > 1e-6

        # Absent if maps to itself with non-integer phase
        absent = absent | (same_reflection & non_integer_phase)

    return absent


def reduce_hkl(
    hkl_p1: torch.Tensor,
    spacegroup: SpaceGroupLike,
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduce P1 Miller indices to asymmetric unit of a target spacegroup.

    This is the inverse of expand_hkl(). Takes a complete set of P1 reflections
    and maps them back to the asymmetric unit of the target spacegroup. Multiple
    P1 reflections that are symmetry-equivalent merge into single ASU reflections.

    The return uses a 2D index mapping with "constant multiplicity" design:
    each ASU reflection has exactly n_ops slots (for symmetry operations),
    enabling simple vectorized aggregation.

    Parameters
    ----------
    hkl_p1 : torch.Tensor, shape (N, 3)
        Input Miller indices in P1 (complete hemisphere).
    spacegroup : str, int, or gemmi.SpaceGroup
        Target space group specification.
    include_friedel : bool, default True
        If True, also consider Friedel mates when finding ASU representative.
    device : torch.device, optional
        Computation device. If None, uses hkl_p1's device.

    Returns
    -------
    hkl_asu : torch.Tensor, shape (M, 3), dtype int32
        Unique Miller indices in the asymmetric unit.
    reduction_indices : torch.Tensor, shape (M, n_ops * (2 if include_friedel else 1)), dtype int64
        For each ASU reflection, indices into hkl_p1 for its symmetry equivalents.
        Value of -1 indicates no P1 reflection at that position.
        Use: ``F_asu = aggregate(F_p1[reduction_indices], dim=1)``
    phase_shifts : torch.Tensor, shape (M, n_ops * (2 if include_friedel else 1)), dtype float32
        Phase shifts to apply before aggregation.
        For Friedel mates, the phase is negated.

    Examples
    --------
    ::

        import torch
        from torchref.symmetry import expand_hkl, reduce_hkl

        # Start with ASU, expand to P1, then reduce back
        hkl_asu = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.int32)
        hkl_p1, exp_idx, exp_phase = expand_hkl(hkl_asu, 'P21')

        # Reduce back to ASU
        hkl_asu_back, red_idx, red_phase = reduce_hkl(hkl_p1, 'P21')

        # Aggregate F values from P1 to ASU
        F_p1 = torch.randn(len(hkl_p1))
        valid_mask = red_idx >= 0
        F_gathered = torch.where(valid_mask, F_p1[red_idx.clamp(min=0)], torch.zeros_like(F_p1[0]))
        F_asu = F_gathered.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

    Notes
    -----
    The "constant multiplicity" design means the second dimension is always
    n_ops (or 2*n_ops with Friedel), regardless of whether all equivalent
    reflections exist in hkl_p1. Missing positions are marked with -1.
    This enables efficient batch aggregation without variable-length operations.
    """
    if device is None:
        device = hkl_p1.device

    # Get symmetry operations
    symmetry = SpaceGroup(spacegroup, dtype=torch.float32, device=device)
    n_ops = symmetry.matrices.shape[0]

    # Reciprocal space matrices (transpose of real space)
    recip_matrices = symmetry.matrices.transpose(-2, -1)
    translations = symmetry.translations

    # Total number of equivalent positions per ASU reflection
    n_equiv = n_ops * (2 if include_friedel else 1)

    # Convert hkl to float for matrix operations
    hkl_float = hkl_p1.to(dtype=torch.float32, device=device)
    n_p1 = len(hkl_float)

    # Build lookup from hkl tuple to index in P1 array
    hkl_p1_np = hkl_p1.cpu().numpy()
    p1_lookup = {tuple(h): idx for idx, h in enumerate(hkl_p1_np)}

    # For each P1 reflection, find its "canonical" ASU representative
    # The canonical form is the lexicographically smallest (h, k, l) among all equivalents
    def get_canonical_hkl(hkl_single):
        """Find canonical ASU representative for a single reflection."""
        equivalents = []

        for i in range(n_ops):
            # h' = h @ R^T
            hkl_trans = torch.round(torch.matmul(hkl_single, recip_matrices[i].T)).to(
                torch.int32
            )
            equivalents.append(hkl_trans)

            if include_friedel:
                equivalents.append(-hkl_trans)

        # Stack and find lexicographically smallest
        equiv_stack = torch.stack(equivalents, dim=0)

        # Sort by (h, k, l) lexicographically
        # Convert to tuple for comparison
        equiv_np = equiv_stack.cpu().numpy()
        equiv_tuples = [tuple(e) for e in equiv_np]
        canonical = min(equiv_tuples)

        return canonical

    # Map each P1 reflection to its canonical ASU representative
    p1_to_asu = {}  # maps P1 index to canonical ASU tuple
    asu_reflections = (
        {}
    )  # maps canonical ASU tuple to list of (P1_idx, phase_shift, equiv_idx)

    for p1_idx, hkl_single in enumerate(hkl_float):
        canonical = get_canonical_hkl(hkl_single)

        p1_to_asu[p1_idx] = canonical

        if canonical not in asu_reflections:
            asu_reflections[canonical] = []

        # Find which equivalent this P1 reflection corresponds to
        for equiv_idx in range(n_ops):
            R = recip_matrices[equiv_idx]
            t = translations[equiv_idx]

            hkl_trans = torch.round(torch.matmul(hkl_single, R.T)).to(torch.int32)
            phase_shift = 2.0 * np.pi * torch.matmul(hkl_single, t)

            if tuple(hkl_trans.cpu().numpy()) == canonical:
                asu_reflections[canonical].append(
                    (p1_idx, phase_shift.item(), equiv_idx)
                )
                break

            if include_friedel:
                if tuple((-hkl_trans).cpu().numpy()) == canonical:
                    # Friedel mate: phase is negated
                    asu_reflections[canonical].append(
                        (p1_idx, -phase_shift.item(), equiv_idx + n_ops)
                    )
                    break

    # Build output tensors
    asu_list = sorted(asu_reflections.keys())
    n_asu = len(asu_list)

    hkl_asu = torch.tensor(asu_list, dtype=torch.int32, device=device)
    reduction_indices = torch.full(
        (n_asu, n_equiv), -1, dtype=torch.int64, device=device
    )
    phase_shifts = torch.zeros((n_asu, n_equiv), dtype=torch.float32, device=device)

    # Fill in the indices and phase shifts
    for asu_idx, asu_hkl in enumerate(asu_list):
        for p1_idx, phase, equiv_idx in asu_reflections[asu_hkl]:
            reduction_indices[asu_idx, equiv_idx] = p1_idx
            phase_shifts[asu_idx, equiv_idx] = phase

    return hkl_asu, reduction_indices, phase_shifts


def expand_reciprocal_grid(
    F_grid: torch.Tensor,
    space_group: str,
    mode: str = "average",
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Expand or symmetrize a reciprocal space grid using crystallographic symmetry.

    This is a convenience function that creates a ReciprocalSymmetryGrid
    and applies it to the input grid.

    Parameters
    ----------
    F_grid : torch.Tensor, shape (nh, nk, nl)
        Input structure factor grid (can be complex or real).
    space_group : str
        Space group symbol (e.g., 'P21', 'P212121').
    mode : str, default 'average'
        Operation mode:
        - 'average': Average over all symmetry equivalents (symmetrize)
        - 'expand': Expand from asymmetric unit to full grid
        - 'sum': Sum all symmetry mates
    include_friedel : bool, default True
        If True, also apply Friedel symmetry after space group symmetry.
    device : torch.device, optional
        Device for computation. If None, uses F_grid's device.

    Returns
    -------
    torch.Tensor, shape (nh, nk, nl)
        Symmetrized or expanded structure factor grid.

    Examples
    --------
    ::

        import torch
        from torchref.symmetry import expand_reciprocal_grid

        # Create a test grid with some values in asymmetric unit
        F = torch.zeros(32, 32, 32, dtype=torch.complex64)
        F[5, 3, 2] = 1.0 + 0.5j

        # Expand to full grid
        F_full = expand_reciprocal_grid(F, 'P21', mode='expand')

        # Or symmetrize an existing full grid
        F_sym = expand_reciprocal_grid(F_noisy, 'P21', mode='average')
    """
    if device is None:
        device = F_grid.device

    grid_shape = F_grid.shape
    dtype = torch.float32 if not F_grid.is_complex() else F_grid.real.dtype

    # Create symmetry handler
    recip_sym = ReciprocalSymmetryGrid(
        space_group=space_group,
        grid_shape=grid_shape,
        dtype_float=dtype,
        verbose=0,
        device=device,
    )

    # Apply symmetry operation
    F_result = recip_sym(F_grid, mode=mode)

    # Apply Friedel symmetry if requested
    if include_friedel:
        F_result = recip_sym.apply_friedel(F_result)

    return F_result
