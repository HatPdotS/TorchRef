"""Bond-angle restraint NLL."""

import torch

from ._common import COS_CLAMP, EPS, LOG_2PI
from ._dispatch import use_triton


def _angle_math_eager(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    references_rad: torch.Tensor,
    sigmas_rad: torch.Tensor,
) -> torch.Tensor:
    pos_a = xyz[idx[:, 0]]
    pos_b = xyz[idx[:, 1]]
    pos_c = xyz[idx[:, 2]]
    v1 = pos_a - pos_b
    v2 = pos_c - pos_b
    # Floor the norms (coincident atoms) and clamp cos to (-1, 1); both keep
    # the ``acos`` backward (-1/sqrt(1-cos**2)) finite at collinearity so the
    # eager gradient matches the Triton kernel instead of returning NaN.
    n1 = torch.linalg.norm(v1, dim=-1).clamp_min(EPS)
    n2 = torch.linalg.norm(v2, dim=-1).clamp_min(EPS)
    cos_angle = torch.clamp(
        torch.sum(v1 * v2, dim=-1) / (n1 * n2),
        -COS_CLAMP,
        COS_CLAMP,
    )
    angles_rad = torch.acos(cos_angle)
    deviations = angles_rad - references_rad
    nll = 0.5 * (deviations / sigmas_rad) ** 2 + torch.log(sigmas_rad) + 0.5 * LOG_2PI
    return nll.sum()


def angle_math(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    references_rad: torch.Tensor,
    sigmas_rad: torch.Tensor,
) -> torch.Tensor:
    """Angle NLL: gather, compute angle, Gaussian NLL, sum.

    Dispatches to :func:`torchref.base.targets.triton.angle_math_triton`
    on CUDA float32. Falls back to eager otherwise.

    Parameters
    ----------
    xyz : torch.Tensor
        (N_atoms, 3) Cartesian coordinates.
    idx : torch.Tensor
        (N, 3) integer indices [a, b, c] with ``b`` as the vertex.
    references_rad : torch.Tensor
        (N,) target angles in radians.
    sigmas_rad : torch.Tensor
        (N,) standard deviations in radians.
    """
    if use_triton(xyz):
        from .triton.angle import angle_math_triton

        return angle_math_triton(xyz, idx, references_rad, sigmas_rad)
    return _angle_math_eager(xyz, idx, references_rad, sigmas_rad)
