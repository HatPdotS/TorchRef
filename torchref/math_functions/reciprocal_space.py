"""
Reciprocal space utilities for crystallographic calculations.

This module provides functions for working with Miller indices and
reciprocal space, including generating all possible HKL indices within
a resolution limit.
"""

from typing import Optional

import torch

from torchref.math_functions.math_torch import (
    get_d_spacing,
    reciprocal_basis_matrix,
)


def generate_possible_hkl(
    cell: torch.Tensor, d_min: float, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate all possible Miller indices within a resolution limit.

    Creates a complete set of (h, k, l) indices where the d-spacing
    is greater than or equal to d_min.

    Parameters
    ----------
    cell : torch.Tensor, shape (6,)
        Unit cell parameters [a, b, c, alpha, beta, gamma] in Angstroms and degrees.
    d_min : float
        High resolution limit in Angstroms (minimum d-spacing).
    device : torch.device, optional
        Device for computation. If None, uses cell's device.

    Returns
    -------
    torch.Tensor, shape (M, 3), dtype int32
        All Miller indices with d-spacing >= d_min.

    Examples
    --------
    ::

        import torch
        cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
        hkl = generate_possible_hkl(cell, d_min=2.0)
        print(f"Generated {len(hkl)} reflections")
    """
    if device is None:
        device = cell.device

    cell = cell.to(device)

    # Compute reciprocal basis to get a*, b*, c* lengths
    recB = reciprocal_basis_matrix(cell)
    a_star = torch.linalg.norm(recB[0])
    b_star = torch.linalg.norm(recB[1])
    c_star = torch.linalg.norm(recB[2])

    # Maximum h, k, l values (conservative estimate)
    # d = 1/|s| where s = h*a* + k*b* + l*c*
    # For d >= d_min, we need |s| <= 1/d_min
    # Conservative bound: h_max = ceil(1 / (d_min * a*))
    s_max = 1.0 / d_min
    h_max = int(torch.ceil(s_max / a_star).item())
    k_max = int(torch.ceil(s_max / b_star).item())
    l_max = int(torch.ceil(s_max / c_star).item())

    # Generate all combinations of h, k, l
    # Include negative values for completeness
    h_range = torch.arange(-h_max, h_max + 1, device=device, dtype=torch.int32)
    k_range = torch.arange(-k_max, k_max + 1, device=device, dtype=torch.int32)
    l_range = torch.arange(-l_max, l_max + 1, device=device, dtype=torch.int32)

    # Create meshgrid of all combinations
    hh, kk, ll = torch.meshgrid(h_range, k_range, l_range, indexing="ij")
    hkl_all = torch.stack([hh.flatten(), kk.flatten(), ll.flatten()], dim=1)

    # Remove (0, 0, 0) - not a valid reflection
    not_origin = (hkl_all != 0).any(dim=1)
    hkl_all = hkl_all[not_origin]

    # Filter by resolution
    d_spacing = get_d_spacing(hkl_all.float(), cell, recB=recB)
    valid_res = d_spacing >= d_min
    hkl_valid = hkl_all[valid_res]

    return hkl_valid


def compute_d_spacing_batch(
    hkl: torch.Tensor, cell: torch.Tensor, recB: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute d-spacing for a batch of Miller indices.

    Wrapper around get_d_spacing for convenience.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Miller indices.
    cell : torch.Tensor, shape (6,)
        Unit cell parameters.
    recB : torch.Tensor, optional
        Pre-computed reciprocal basis matrix.

    Returns
    -------
    torch.Tensor, shape (N,)
        D-spacing values in Angstroms.
    """
    return get_d_spacing(hkl, cell, recB=recB)
