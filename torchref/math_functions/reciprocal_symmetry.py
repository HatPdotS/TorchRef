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

from torchref.math_functions.math_torch import (
    extract_structure_factor_from_grid,
    ifft,
)

if TYPE_CHECKING:
    from torchref.symmetry.symmetry import Symmetry


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

    # Reciprocal space matrices: R_recip = R^T
    # For h' = R^T @ h, we compute h @ R (row vector convention)
    recip_matrices = rotation_matrices.transpose(-2, -1).to(
        dtype=dtype, device=device
    )  # (n_ops, 3, 3)

    n_ops = recip_matrices.shape[0]

    # Compute h' = h @ R for each operation
    # hkl_float: (N, 3) -> (1, N, 3)
    # recip_matrices: (n_ops, 3, 3)
    # Result: (n_ops, N, 3)
    hkl_expanded = hkl_float.unsqueeze(0).expand(n_ops, -1, -1)  # (n_ops, N, 3)

    # Batch matrix multiply: (n_ops, N, 3) @ (n_ops, 3, 3) -> (n_ops, N, 3)
    equiv_hkl = torch.bmm(hkl_expanded, recip_matrices)

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

    # Move everything to the same device
    hkl = hkl.to(device=device)
    rotation_matrices = rotation_matrices.to(device=device)
    translations = translations.to(device=device)

    n_ops = rotation_matrices.shape[0]

    # 1. Compute equivalent HKLs: (n_ops, N, 3)
    equiv_hkls = compute_symmetry_equivalent_hkls(hkl, rotation_matrices)

    # 2. Extract F_P1 at each equivalent HKL: (n_ops, N)
    # We need to extract from the grid for each operation
    f_p1_list = []
    for i in range(n_ops):
        f_i = extract_structure_factor_from_grid(reciprocal_grid, equiv_hkls[i])
        f_p1_list.append(f_i)

    f_p1 = torch.stack(f_p1_list, dim=0)  # (n_ops, N)

    # 3. Compute phase shifts: (n_ops, N)
    phases = compute_translation_phases(hkl, translations)

    # 4. Apply phases and sum: (N,)
    f_sym = (f_p1 * phases).sum(dim=0)

    return f_sym


class ReciprocalSymmetryExtractor:
    """
    Class-based interface for reciprocal space symmetry extraction.

    This provides a more efficient interface when computing structure factors
    multiple times with the same symmetry and HKLs (e.g., during refinement).
    Precomputes equivalent HKLs and phase factors.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Target Miller indices.
    symmetry : Symmetry
        Symmetry object containing rotation matrices and translations.
    device : torch.device, optional
        Device for computation.

    Examples
    --------
    >>> extractor = ReciprocalSymmetryExtractor(hkl, symmetry)
    >>> f_calc = extractor(density_map)  # Compute structure factors
    """

    def __init__(
        self,
        hkl: torch.Tensor,
        symmetry: "Symmetry",
        device: Optional[torch.device] = None,
    ):
        self.device = device or hkl.device
        self.hkl = hkl.to(device=self.device)
        self.symmetry = symmetry
        self.n_ops = symmetry.n_ops
        self.N = len(hkl)

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
        reciprocal_grid = ifft(density_map)

        # Extract from grid
        return self.extract_from_grid(reciprocal_grid)

    def extract_from_grid(self, reciprocal_grid: torch.Tensor) -> torch.Tensor:
        """
        Extract structure factors from precomputed reciprocal grid.

        Use this when you already have the FFT result.

        Parameters
        ----------
        reciprocal_grid : torch.Tensor, shape (Nx, Ny, Nz)
            Complex reciprocal space grid from FFT.

        Returns
        -------
        torch.Tensor, shape (N,)
            Complex structure factors with symmetry applied.
        """
        # Extract F at all equivalent HKLs
        f_p1_list = []
        for i in range(self.n_ops):
            f_i = extract_structure_factor_from_grid(
                reciprocal_grid, self.equiv_hkls[i]
            )
            f_p1_list.append(f_i)

        f_p1 = torch.stack(f_p1_list, dim=0)  # (n_ops, N)

        # Apply phases and sum
        f_sym = (f_p1 * self.phases).sum(dim=0)

        return f_sym

    def to(self, device: torch.device) -> "ReciprocalSymmetryExtractor":
        """Move extractor to specified device."""
        self.device = device
        self.hkl = self.hkl.to(device=device)
        self.equiv_hkls = self.equiv_hkls.to(device=device)
        self.phases = self.phases.to(device=device)
        return self
