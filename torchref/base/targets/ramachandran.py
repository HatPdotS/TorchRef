"""Ramachandran restraint via bilinear interpolation on a precomputed NLL surface."""

import torch

from ._common import torsions_from_xyz
from ._dispatch import use_triton


def _ramachandran_math_eager(
    xyz: torch.Tensor,
    phi_idx: torch.Tensor,
    psi_idx: torch.Tensor,
    nll_surfaces: torch.Tensor,
    surface_type: torch.Tensor,
) -> torch.Tensor:
    phi_deg = -torsions_from_xyz(xyz, phi_idx)
    psi_deg = -torsions_from_xyz(xyz, psi_idx)

    phi_idx_grid = (phi_deg + 180.0) % 360.0
    psi_idx_grid = (psi_deg + 180.0) % 360.0

    phi_lo = phi_idx_grid.detach().floor().long() % 360
    phi_hi = (phi_lo + 1) % 360
    psi_lo = psi_idx_grid.detach().floor().long() % 360
    psi_hi = (psi_lo + 1) % 360
    phi_frac = phi_idx_grid - phi_idx_grid.detach().floor()
    psi_frac = psi_idx_grid - psi_idx_grid.detach().floor()

    s = surface_type
    v00 = nll_surfaces[s, phi_lo, psi_lo]
    v01 = nll_surfaces[s, phi_lo, psi_hi]
    v10 = nll_surfaces[s, phi_hi, psi_lo]
    v11 = nll_surfaces[s, phi_hi, psi_hi]

    nll = (
        (1 - phi_frac) * (1 - psi_frac) * v00
        + (1 - phi_frac) * psi_frac * v01
        + phi_frac * (1 - psi_frac) * v10
        + phi_frac * psi_frac * v11
    )
    return nll.sum()


def ramachandran_math(
    xyz: torch.Tensor,
    phi_idx: torch.Tensor,
    psi_idx: torch.Tensor,
    nll_surfaces: torch.Tensor,
    surface_type: torch.Tensor,
) -> torch.Tensor:
    """Ramachandran bilinear-interpolated NLL.

    Mirrors ``RamachandranTarget.forward``.

    Dispatches to
    :func:`torchref.base.targets.triton.ramachandran_math_triton` on
    CUDA float32 (~10× faster fwd+bw on A100). Falls back to eager
    otherwise.

    Parameters
    ----------
    xyz : torch.Tensor
        (N_atoms, 3) Cartesian coordinates.
    phi_idx, psi_idx : torch.Tensor
        (N, 4) atom indices for the two backbone dihedrals.
    nll_surfaces : torch.Tensor
        (n_surface_types, 360, 360) precomputed NLL = -log P(φ, ψ | type).
    surface_type : torch.Tensor
        (N,) integer type per residue.
    """
    if use_triton(xyz):
        from .triton.ramachandran import ramachandran_math_triton
        return ramachandran_math_triton(
            xyz, phi_idx, psi_idx, nll_surfaces, surface_type,
        )
    return _ramachandran_math_eager(
        xyz, phi_idx, psi_idx, nll_surfaces, surface_type,
    )
