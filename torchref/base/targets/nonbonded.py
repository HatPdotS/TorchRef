"""Non-bonded (VDW) heavy-heavy repulsion NLL — prolsq mode with symmetry mates."""

from typing import Optional

import torch

from ._common import LOG_2PI
from ._dispatch import use_triton


def _nonbonded_heavy_math_eager(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    min_distances: torch.Tensor,
    symop_indices: Optional[torch.Tensor],
    cell_offsets: Optional[torch.Tensor],
    symop_matrices: Optional[torch.Tensor],
    symop_translations: Optional[torch.Tensor],
    fractional_matrix: torch.Tensor,
    inv_fractional_matrix: torch.Tensor,
    c_rep: torch.Tensor,
    r_exp: torch.Tensor,
    buffer: float,
    sigma_vdw: torch.Tensor,
) -> torch.Tensor:
    pos1 = xyz[indices[:, 0]]

    has_symmetry = (
        symop_indices is not None
        and symop_indices.numel() > 0
        and not bool((symop_indices == 0).all())
    )

    if not has_symmetry:
        pos2 = xyz[indices[:, 1]]
    else:
        mate_source = xyz[indices[:, 1]]
        frac = mate_source @ inv_fractional_matrix.T
        R = symop_matrices[symop_indices].to(frac.dtype)
        t = symop_translations[symop_indices].to(frac.dtype)
        offsets = cell_offsets.to(frac.dtype)
        frac_transformed = torch.bmm(R, frac.unsqueeze(-1)).squeeze(-1) + t + offsets
        pos2 = frac_transformed @ fractional_matrix.T

    diff = pos2 - pos1
    actual_distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    violations = torch.clamp(min_distances + buffer - actual_distances, min=0.0)
    shape_energy = c_rep * (violations ** r_exp)
    per_pair_const = torch.log(sigma_vdw) + 0.5 * LOG_2PI
    return shape_energy.sum() + per_pair_const * violations.shape[0]


def nonbonded_heavy_math(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    min_distances: torch.Tensor,
    symop_indices: Optional[torch.Tensor],
    cell_offsets: Optional[torch.Tensor],
    symop_matrices: Optional[torch.Tensor],
    symop_translations: Optional[torch.Tensor],
    fractional_matrix: torch.Tensor,
    inv_fractional_matrix: torch.Tensor,
    c_rep: torch.Tensor,
    r_exp: torch.Tensor,
    buffer: float,
    sigma_vdw: torch.Tensor,
) -> torch.Tensor:
    """Heavy-heavy VDW prolsq repulsion NLL.

    Matches the prolsq branch of ``NonBondedTarget.forward`` plus the symmetry-aware
    gather of ``NonBondedTarget._compute_positions``. The H-VDW term ``NonBondedHTarget``
    adds is **not** included here. Dispatches to
    :func:`torchref.base.targets.triton.nonbonded_heavy_math_triton` on CUDA float32
    (the gain coming mostly from the analytic backward), eager otherwise.

    Parameters
    ----------
    xyz : torch.Tensor
        (N_atoms, 3) Cartesian coordinates of the ASU.
    indices : torch.Tensor
        (N, 2) per-pair atom indices.
    min_distances : torch.Tensor
        (N,) VDW threshold per pair.
    symop_indices : torch.Tensor, optional
        (N,) symmetry-operator index per pair; 0 = identity.
    cell_offsets : torch.Tensor, optional
        (N, 3) fractional cell offsets per pair.
    symop_matrices, symop_translations : torch.Tensor, optional
        (n_symops, 3, 3) and (n_symops, 3) — the symmetry operator table.
    fractional_matrix, inv_fractional_matrix : torch.Tensor
        ``cell.fractional_matrix`` and its inverse (3, 3).
    c_rep, r_exp, sigma_vdw : torch.Tensor
        Scalar repulsion coefficient, exponent, and effective tolerance.
    buffer : float
        Distance buffer in Å.
    """
    if use_triton(xyz):
        from .triton.nonbonded import nonbonded_heavy_math_triton
        return nonbonded_heavy_math_triton(
            xyz, indices, min_distances,
            symop_indices, cell_offsets,
            symop_matrices, symop_translations,
            fractional_matrix, inv_fractional_matrix,
            c_rep, r_exp, buffer, sigma_vdw,
        )
    return _nonbonded_heavy_math_eager(
        xyz, indices, min_distances,
        symop_indices, cell_offsets,
        symop_matrices, symop_translations,
        fractional_matrix, inv_fractional_matrix,
        c_rep, r_exp, buffer, sigma_vdw,
    )
