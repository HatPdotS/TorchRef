"""Bond-length restraint NLL."""

import torch

from ._common import EPS, LOG_2PI
from ._dispatch import use_triton


def _bond_math_eager(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    references: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    pos1 = xyz[idx[:, 0]]
    pos2 = xyz[idx[:, 1]]
    # ``sqrt(sumsq + EPS)`` instead of ``norm`` so the d/dxyz = u/||u||
    # backward is finite at zero distance (coincident atoms) — matches the
    # Triton kernel's ``d_safe`` guard rather than returning NaN.
    diff = pos2 - pos1
    bond_lengths = (diff.pow(2).sum(dim=-1) + EPS).sqrt()
    deviations = bond_lengths - references
    nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * LOG_2PI
    return nll.sum()


def bond_math(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    references: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Bond NLL: gather, distance, Gaussian NLL, sum.

    Mirrors ``BondTarget.forward`` (``geometry/bond``) including the
    bond-length computation from ``Restraints.bond_lengths``.

    On CUDA float32 inputs this dispatches to
    :func:`torchref.base.targets.triton.bond_math_triton` (~2.5× faster
    fwd+bw on A100). All other inputs use the eager implementation.

    Parameters
    ----------
    xyz : torch.Tensor
        (N_atoms, 3) Cartesian coordinates.
    idx : torch.Tensor
        (N, 2) integer indices into ``xyz``.
    references : torch.Tensor
        (N,) target bond lengths.
    sigmas : torch.Tensor
        (N,) standard deviations.
    """
    if use_triton(xyz):
        from .triton.bond import bond_math_triton

        return bond_math_triton(xyz, idx, references, sigmas)
    return _bond_math_eager(xyz, idx, references, sigmas)
