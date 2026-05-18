"""
Restrict the Patterson function to within a sphere of radius Ω (molecule
diameter) so the rotation function sees only intramolecular vectors.

This is the operational equivalent of Phaser's χ_Ω weighting (LERF1, §2.1.3):
the sphere of integration excludes intermolecular Patterson peaks that come
from cross-vectors between symmetry-related copies in the crystal — those are
the dominant contamination of |F_obs|² on real data and are why the bare
Crowther-style rotation function (|E|² · |E|² overlap) fails on real F_obs.

Implementation
--------------
- Place |F|² values onto a regular 3-D reciprocal-space grid of a large cubic
  cell (radius ≥ Ω, so the sphere fits inside).
- 3-D FFT → real-space Patterson in Cartesian coordinates of the cubic cell.
- Apply spherical mask: zero values at |r| > Ω.
- IFFT → sphere-restricted Patterson coefficients on the cubic grid.
- Extract values at the original real-cell HKL positions via trilinear
  interpolation.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from torchref.symmetry.cell import Cell
from torchref.base.reciprocal.interpolation import (
    interpolate_structure_factor_from_grid,
)


def _make_cubic_grid(F_sq_real: torch.Tensor,
                     hkl_real: torch.Tensor,
                     real_cell: Cell,
                     cubic_side: float,
                     gridsize: int) -> Tuple[torch.Tensor, Cell]:
    """
    Place |F|² values onto a (gridsize, gridsize, gridsize) cubic-cell
    reciprocal grid in FFT layout. Multiple real-HKL points falling into the
    same cubic-grid cell are accumulated.

    Returns
    -------
    grid : torch.Tensor (real), shape (N, N, N)
        Cubic reciprocal grid with FFT layout (DC at index 0, negative HKL
        wrapped to high indices).
    cubic_cell : Cell
    """
    device = F_sq_real.device
    dtype = F_sq_real.dtype
    N = int(gridsize)
    cubic_cell = Cell(
        [cubic_side, cubic_side, cubic_side, 90.0, 90.0, 90.0],
        dtype=torch.float32, device=device,
    )
    # h_real → Cartesian s
    rec_real = real_cell.reciprocal_basis_matrix.to(device).to(dtype)
    s = hkl_real.to(dtype) @ rec_real  # (M, 3)
    # Cartesian s → cubic-cell HKL (orthogonal cubic, so rec_basis = (1/a)·I → inv = a·I)
    h_cubic_f = s * cubic_side  # (M, 3), real float
    h_idx = torch.round(h_cubic_f).long()  # (M, 3), nearest integer cubic HKL
    # Wrap into [0, N) (negative HKL → high indices)
    h_idx = h_idx % N
    # Accumulate values into the grid
    flat_idx = (h_idx[:, 0] * N + h_idx[:, 1]) * N + h_idx[:, 2]
    grid_flat = torch.zeros(N * N * N, dtype=dtype, device=device)
    grid_flat.index_add_(0, flat_idx, F_sq_real)
    grid = grid_flat.view(N, N, N)
    return grid, cubic_cell


def _sphere_mask(N: int, cubic_side: float, omega: float,
                 dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build a sphere indicator mask on an (N, N, N) real-space grid covering the
    cubic cell (side = cubic_side). The mask is 1 inside |r| < omega, 0 outside.
    The grid layout is FFT-compatible: index 0 = origin, indices > N/2 wrap to
    negative coordinates.
    """
    idx = torch.arange(N, device=device)
    # Voxel offsets, wrapped to [-N/2, N/2)
    offsets = torch.where(idx < (N + 1) // 2, idx, idx - N).to(dtype) * (cubic_side / N)
    rx, ry, rz = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
    r2 = rx ** 2 + ry ** 2 + rz ** 2
    return (r2 < omega ** 2).to(dtype)


def restrict_to_sphere(
    F_squared: torch.Tensor,
    hkl_real: torch.Tensor,
    real_cell: Cell,
    omega_A: float,
    cubic_side_A: float | None = None,
    gridsize: int | None = None,
    max_res_A: float = 3.0,
) -> Tuple[torch.Tensor, Cell, torch.Tensor]:
    """
    Sphere-restrict a Patterson-like coefficient set defined at the real-cell
    HKL positions.

    Parameters
    ----------
    F_squared : torch.Tensor (real), shape (M,)
        Values to filter (e.g. (|E|² − 1) per shell, or |F|² with origin removed).
    hkl_real : torch.Tensor, shape (M, 3)
        Miller indices in the real (crystal) cell.
    real_cell : Cell
    omega_A : float
        Sphere-of-integration radius (Å). Typically the molecule's bounding-box
        radius — vectors longer than 2·omega are intermolecular and get filtered out.
    cubic_side_A : float, optional
        Cubic cell side (Å) for the FFT. Default: 4 · omega_A (so the Patterson
        sphere fits comfortably with padding to avoid wraparound aliasing).
    gridsize : int, optional
        Grid size (one dim). Default: 2 · ceil(cubic_side / max_res_A).
    max_res_A : float, default 3.0
        Resolution limit (Å) for the cubic grid spacing.

    Returns
    -------
    filtered_values : torch.Tensor (real), shape (M,)
        Sphere-restricted Patterson coefficients at the SAME real-cell HKL set.
    cubic_cell : Cell
    cubic_grid : torch.Tensor (real complex-real), shape (N, N, N)
        Final sphere-restricted reciprocal grid (kept for caller introspection).
    """
    if cubic_side_A is None:
        cubic_side_A = 4.0 * omega_A
    if gridsize is None:
        gridsize = 2 * int(math.ceil(cubic_side_A / max_res_A))
        # Round to even for cleaner FFT
        if gridsize % 2:
            gridsize += 1

    grid_recip, cubic_cell = _make_cubic_grid(
        F_squared, hkl_real, real_cell, cubic_side_A, gridsize,
    )
    # FFT to real space (the grid is real-valued)
    patterson_real = torch.fft.fftn(grid_recip, dim=(0, 1, 2)).real
    mask = _sphere_mask(gridsize, cubic_side_A, omega_A,
                        dtype=patterson_real.dtype, device=patterson_real.device)
    patterson_real_masked = patterson_real * mask
    # IFFT back (will be approximately real for a real Patterson)
    grid_recip_filtered = torch.fft.ifftn(
        patterson_real_masked, dim=(0, 1, 2)
    ).real

    # Extract values at original real-cell HKL positions via trilinear
    # interpolation. Note: interpolate_structure_factor_from_grid expects a
    # complex grid for `interpolate_amplitude=False` and a complex or real grid
    # for `interpolate_amplitude=True`. Our grid is real — wrap in complex for
    # the utility.
    rec_real = real_cell.reciprocal_basis_matrix.to(F_squared.device).to(F_squared.dtype)
    s = hkl_real.to(F_squared.dtype) @ rec_real
    h_cubic_f = s * cubic_side_A  # cubic-cell float HKL
    grid_complex = grid_recip_filtered.to(torch.complex64)
    filtered = interpolate_structure_factor_from_grid(
        grid_complex, h_cubic_f, interpolate_amplitude=False,
    ).real.to(F_squared.dtype)
    return filtered, cubic_cell, grid_recip_filtered
