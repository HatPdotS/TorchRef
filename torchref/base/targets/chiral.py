"""Chiral-volume restraint NLL."""

import torch

from ._common import LOG_2PI
from ._dispatch import use_triton


def _chiral_math_eager(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    ideal_volumes: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    p_c = xyz[indices[:, 0]]
    v1 = xyz[indices[:, 1]] - p_c
    v2 = xyz[indices[:, 2]] - p_c
    v3 = xyz[indices[:, 3]] - p_c

    volumes = torch.sum(v1 * torch.cross(v2, v3, dim=-1), dim=-1)

    achiral_mask = ideal_volumes == 0
    effective_ideal = torch.where(
        achiral_mask, torch.full_like(ideal_volumes, 2.5), ideal_volumes,
    )
    effective_volumes = torch.where(achiral_mask, torch.abs(volumes), volumes)
    deviations = effective_volumes - effective_ideal

    nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * LOG_2PI
    return nll.sum()


def chiral_math(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    ideal_volumes: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Chiral volume NLL (matches ``ChiralTarget.forward``).

    For each chiral center, computes the signed tetrahedral volume
    ``V = v1 . (v2 x v3)`` where ``vi = xyz[i] - xyz[center]``. Achiral
    centers (``ideal_volumes == 0``) are restrained on ``|V|`` against 2.5.

    Dispatches to :func:`torchref.base.targets.triton.chiral_math_triton`
    on CUDA float32 (~4× faster fwd+bw on A100). Falls back to eager
    otherwise.

    Parameters
    ----------
    xyz : torch.Tensor
        (N_atoms, 3) Cartesian coordinates.
    indices : torch.Tensor
        (N, 4) integer indices ``[center, a1, a2, a3]``.
    ideal_volumes : torch.Tensor
        (N,) target signed volumes.
    sigmas : torch.Tensor
        (N,) standard deviations.
    """
    if use_triton(xyz):
        from .triton.chiral import chiral_math_triton
        return chiral_math_triton(xyz, indices, ideal_volumes, sigmas)
    return _chiral_math_eager(xyz, indices, ideal_volumes, sigmas)
