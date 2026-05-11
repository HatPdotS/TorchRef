"""
Reciprocal space symmetry application for structure factor calculation.

This module provides an alternative to the MapSymmetry-based approach for
structure factor calculation. Instead of symmetrizing the density map in
real space ("early symmetry"), we apply symmetry directly in reciprocal space
after FFT ("late symmetry").

Late symmetry is approximately 5x faster for structure factor calculation
because it avoids the expensive map symmetrization step.

Mathematical Background
-----------------------
For a symmetry operation {R|t} (rotation R, translation t):

    F_sym(h) = sum_ops exp(2*pi*i * h.t) * F_P1(R^T @ h)

Where:
- F_sym(h) = structure factor with symmetry at Miller index h
- F_P1(h') = P1 structure factor at h' = R^T @ h
- exp(2*pi*i * h.t) = translation phase shift

Since R is an integer matrix (crystallographic), R^T @ h gives integer indices
that can be extracted directly from the reciprocal space grid.

Terminology
-----------
- Early symmetry: Apply symmetry to density map before FFT (MapSymmetry approach)
- Late symmetry: Apply symmetry in reciprocal space after FFT (this module)
"""

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from torchref.utils.autograd_ops import gather_with_index_add

from .grid_operations import extract_structure_factor_from_grid

if TYPE_CHECKING:
    from torchref.symmetry.spacegroup import SpaceGroup


def compute_symmetry_equivalent_hkls(
    hkl: torch.Tensor,
    rotation_matrices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute symmetry-equivalent HKLs: h' = R^T @ h for each operation.

    In reciprocal space, Miller indices transform as h' = h @ R (or equivalently
    h' = R^T @ h when treating h as a column vector).

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Miller indices.
    rotation_matrices : torch.Tensor, shape (n_ops, 3, 3)
        Real-space rotation matrices. These are transposed internally
        for reciprocal space transformation.

    Returns
    -------
    torch.Tensor, shape (n_ops, N, 3)
        Equivalent HKLs for each symmetry operation.
    """
    device = hkl.device
    dtype = torch.float32

    # Ensure correct types
    hkl_float = hkl.to(dtype=dtype, device=device)  # (N, 3)

    # For reciprocal space: F_sym(h) = sum_ops exp(2πi h·t) * F_P1(R^T @ h)
    # With row vector convention: h' = h @ R equals (R^T @ h) in column notation
    # So we use rotation_matrices directly (no transpose)
    rot_matrices = rotation_matrices.to(dtype=dtype, device=device)  # (n_ops, 3, 3)

    n_ops = rot_matrices.shape[0]

    # Compute h' = h @ R for each operation
    # hkl_float: (N, 3) -> (1, N, 3)
    # rot_matrices: (n_ops, 3, 3)
    # Result: (n_ops, N, 3)
    hkl_expanded = hkl_float.unsqueeze(0).expand(n_ops, -1, -1)  # (n_ops, N, 3)

    # Batch matrix multiply: (n_ops, N, 3) @ (n_ops, 3, 3) -> (n_ops, N, 3)
    equiv_hkl = torch.bmm(hkl_expanded, rot_matrices)

    # Round to nearest integer (should be exact for valid crystallographic ops)
    equiv_hkl = torch.round(equiv_hkl).to(torch.int64)

    return equiv_hkl


def compute_translation_phases(
    hkl: torch.Tensor,
    translations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute phase shifts: exp(2*pi*i * h.t) for each operation.

    The translation component of a symmetry operation causes a phase shift
    in the structure factor.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Miller indices.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors in fractional coordinates.

    Returns
    -------
    torch.Tensor, shape (n_ops, N)
        Complex phase factors exp(2*pi*i * h.t).
    """
    device = hkl.device
    dtype = torch.float32

    # Ensure correct types
    hkl_float = hkl.to(dtype=dtype, device=device)  # (N, 3)
    translations = translations.to(dtype=dtype, device=device)  # (n_ops, 3)

    # Compute h.t for each operation
    # hkl_float: (N, 3), translations: (n_ops, 3)
    # We want: (n_ops, N) where each entry is h.t
    # h.t = hkl_float @ translations.T -> (N, n_ops)
    # Then transpose to get (n_ops, N)
    h_dot_t = torch.matmul(hkl_float, translations.T).T  # (n_ops, N)

    # Phase factor: exp(2*pi*i * h.t)
    phase = 2.0 * np.pi * h_dot_t
    phase_factor = torch.exp(1j * phase.to(torch.float32))

    return phase_factor  # (n_ops, N) complex


def extract_structure_factors_with_symmetry(
    reciprocal_grid: torch.Tensor,
    hkl: torch.Tensor,
    rotation_matrices: torch.Tensor,
    translations: torch.Tensor,
) -> torch.Tensor:
    """
    Extract structure factors with symmetry applied in reciprocal space.

    This is the main function that replaces the MapSymmetry approach for
    structure factor extraction. Instead of symmetrizing the density map
    and then extracting F(hkl), we extract F at all symmetry-equivalent
    positions and sum with phases.

    Parameters
    ----------
    reciprocal_grid : torch.Tensor, shape (Nx, Ny, Nz)
        Complex reciprocal space grid from FFT of P1 density map.
    hkl : torch.Tensor, shape (N, 3)
        Target Miller indices.
    rotation_matrices : torch.Tensor, shape (n_ops, 3, 3)
        Real-space rotation matrices from symmetry operations.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors from symmetry operations.

    Returns
    -------
    torch.Tensor, shape (N,)
        Complex structure factors with symmetry applied.
    """
    device = reciprocal_grid.device
    Nx, Ny, Nz = reciprocal_grid.shape

    # Move everything to the same device
    hkl = hkl.to(device=device)
    rotation_matrices = rotation_matrices.to(device=device)
    translations = translations.to(device=device)

    n_ops = rotation_matrices.shape[0]
    N = hkl.shape[0]

    # 1. Compute equivalent HKLs: (n_ops, N, 3)
    equiv_hkls = compute_symmetry_equivalent_hkls(hkl, rotation_matrices)

    # 2. Vectorized extraction: single flat gather for all symops at once.
    # Use gather_with_index_add so the gradient back into reciprocal_grid
    # is a single index_add_ (no radix-sort + dedup scatter).
    flat_indices = _equiv_hkls_to_flat_indices(equiv_hkls, Nx, Ny, Nz)
    f_all = gather_with_index_add(
        reciprocal_grid.reshape(-1), flat_indices,
    )  # (n_ops * N,)
    f_p1 = f_all.view(n_ops, N)

    # 3. Compute phase shifts: (n_ops, N)
    phases = compute_translation_phases(hkl, translations)

    # 4. Apply phases and sum: (N,)
    f_sym = (f_p1 * phases).sum(dim=0)

    return f_sym


def _equiv_hkls_to_flat_indices(
    equiv_hkls: torch.Tensor, Nx: int, Ny: int, Nz: int,
) -> torch.Tensor:
    """Convert (n_ops, N, 3) equiv HKLs to flat linear grid indices."""
    all_hkl = equiv_hkls.reshape(-1, 3)  # (n_ops*N, 3)
    hi = torch.remainder(all_hkl[:, 0], Nx)
    ki = torch.remainder(all_hkl[:, 1], Ny)
    li = torch.remainder(all_hkl[:, 2], Nz)
    return (hi * (Ny * Nz) + ki * Nz + li).to(torch.int64)


class ReciprocalSymmetryExtractor:
    """
    Class-based interface for reciprocal space symmetry extraction.

    This provides a more efficient interface when computing structure factors
    multiple times with the same symmetry and HKLs (e.g., during refinement).
    Precomputes equivalent HKLs, phase factors, and flat grid indices so that
    each call reduces to a single gather + multiply + sum (~3 GPU kernels
    instead of ~28).

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Target Miller indices.
    symmetry : SpaceGroup
        SpaceGroup object containing rotation matrices and translations.
    grid_shape : tuple of int
        Reciprocal grid dimensions (Nx, Ny, Nz).
    device : torch.device, optional
        Device for computation.

    Examples
    --------
    >>> extractor = ReciprocalSymmetryExtractor(hkl, symmetry, grid_shape=(209, 86, 67))
    >>> f_calc = extractor.extract_from_grid(reciprocal_grid)
    """

    def __init__(
        self,
        hkl: torch.Tensor,
        symmetry: "SpaceGroup",
        grid_shape: tuple,
        device: Optional[torch.device] = None,
    ):
        self.device = device or hkl.device
        self.hkl = hkl.to(device=self.device)
        self.symmetry = symmetry
        self.n_ops = symmetry.n_ops
        self.N = len(hkl)
        self.grid_shape = grid_shape

        # Precompute equivalent HKLs
        self.equiv_hkls = compute_symmetry_equivalent_hkls(
            self.hkl,
            symmetry.matrices.to(device=self.device),
        )  # (n_ops, N, 3)

        # Precompute phase factors
        self.phases = compute_translation_phases(
            self.hkl,
            symmetry.translations.to(device=self.device),
        )  # (n_ops, N) complex

        # Precompute flat linear indices for single-gather extraction
        Nx, Ny, Nz = grid_shape
        self._flat_indices = _equiv_hkls_to_flat_indices(
            self.equiv_hkls, Nx, Ny, Nz,
        )  # (n_ops * N,) int64

    def __call__(self, density_map: torch.Tensor) -> torch.Tensor:
        """
        Compute structure factors from P1 density map.

        Parameters
        ----------
        density_map : torch.Tensor, shape (Nx, Ny, Nz)
            P1 electron density map (NO symmetry applied).

        Returns
        -------
        torch.Tensor, shape (N,)
            Complex structure factors with symmetry applied.
        """
        return self.extract(density_map)

    def extract(self, density_map: torch.Tensor) -> torch.Tensor:
        """
        Extract structure factors from P1 density map.

        Parameters
        ----------
        density_map : torch.Tensor, shape (Nx, Ny, Nz)
            P1 electron density map (NO symmetry applied).

        Returns
        -------
        torch.Tensor, shape (N,)
            Complex structure factors with symmetry applied.
        """
        # FFT the density map
        from torchref.base.fourier.fft import ifft
        reciprocal_grid = ifft(density_map)

        # Extract from grid
        return self.extract_from_grid(reciprocal_grid)

    def extract_from_grid(self, reciprocal_grid: torch.Tensor) -> torch.Tensor:
        """
        Extract structure factors from precomputed reciprocal grid.

        Uses precomputed flat indices for a single vectorized gather,
        avoiding per-symop Python loops and kernel launches.

        Parameters
        ----------
        reciprocal_grid : torch.Tensor, shape (Nx, Ny, Nz)
            Complex reciprocal space grid from FFT.

        Returns
        -------
        torch.Tensor, shape (N,)
            Complex structure factors with symmetry applied.
        """
        # Gather via custom autograd op so the backward is a single
        # ``index_add_`` (atomic scatter, no radix sort + dedup).
        f_all = gather_with_index_add(
            reciprocal_grid.reshape(-1), self._flat_indices,
        )  # (n_ops * N,)
        f_sym = (f_all.view(self.n_ops, self.N) * self.phases).sum(dim=0)
        return f_sym

    def to(self, device: torch.device) -> "ReciprocalSymmetryExtractor":
        """Move extractor to specified device."""
        self.device = device
        self.hkl = self.hkl.to(device=device)
        self.equiv_hkls = self.equiv_hkls.to(device=device)
        self.phases = self.phases.to(device=device)
        self._flat_indices = self._flat_indices.to(device=device)
        return self
