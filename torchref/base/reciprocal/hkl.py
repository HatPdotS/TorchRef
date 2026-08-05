"""Miller index generation and d-spacing calculation."""

from typing import Optional

import torch

from torchref.config import dtypes
from .basis import reciprocal_basis_matrix, get_scattering_vectors


def get_d_spacing(hkl: torch.Tensor, cell: torch.Tensor, recB=None):
    """
    Calculate d-spacing from Miller indices.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N, 3).
    cell : torch.Tensor
        Cell parameters [a, b, c, alpha, beta, gamma].
    recB : torch.Tensor, optional
        Pre-computed reciprocal basis matrix of shape (3, 3).

    Returns
    -------
    torch.Tensor
        D-spacing values of shape (N,) in Angstroms.
    """
    s = get_scattering_vectors(hkl, cell, recB)
    d_spacing = 1.0 / torch.linalg.norm(s, axis=1)
    return d_spacing


def compute_d_spacing_batch(
    hkl: torch.Tensor, cell: torch.Tensor, recB: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Alias for :func:`get_d_spacing`, which is already batched."""
    return get_d_spacing(hkl, cell, recB=recB)


def generate_possible_hkl(
    cell: torch.Tensor, d_min: float, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate every Miller index with d-spacing >= ``d_min``.

    Covers the full sphere (both Friedel halves, ``(0,0,0)`` excluded); no
    space-group symmetry is applied, so systematic absences are still present.

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
    torch.Tensor, shape (M, 3)
        Miller indices at integer dtype ``dtypes.int`` (int32 by default).
    """
    if device is None:
        device = cell.device

    cell = cell.to(device)

    recB = reciprocal_basis_matrix(cell)
    a_star = torch.linalg.norm(recB[0])
    b_star = torch.linalg.norm(recB[1])
    c_star = torch.linalg.norm(recB[2])

    # Per-axis bound ceil(s_max / a*) over-covers the sphere; the resolution
    # filter at the end trims it back.
    s_max = 1.0 / d_min
    h_max = int(torch.ceil(s_max / a_star).item())
    k_max = int(torch.ceil(s_max / b_star).item())
    l_max = int(torch.ceil(s_max / c_star).item())

    h_range = torch.arange(-h_max, h_max + 1, device=device, dtype=dtypes.int)
    k_range = torch.arange(-k_max, k_max + 1, device=device, dtype=dtypes.int)
    l_range = torch.arange(-l_max, l_max + 1, device=device, dtype=dtypes.int)

    hh, kk, ll = torch.meshgrid(h_range, k_range, l_range, indexing="ij")
    hkl_all = torch.stack([hh.flatten(), kk.flatten(), ll.flatten()], dim=1)

    # (0, 0, 0) is not a reflection.
    not_origin = (hkl_all != 0).any(dim=1)
    hkl_all = hkl_all[not_origin]

    d_spacing = get_d_spacing(hkl_all.float(), cell, recB=recB)
    valid_res = d_spacing >= d_min
    hkl_valid = hkl_all[valid_res]

    return hkl_valid
