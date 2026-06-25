"""ADP (B-factor) restraint NLLs: similarity, KL-divergence, locality."""

import torch

from ._common import LOG_2PI
from ._dispatch import use_triton


def _adp_simu_math_eager(
    b: torch.Tensor,
    pair_indices: torch.Tensor,
    simu_sigma: torch.Tensor,
) -> torch.Tensor:
    diffs = b[pair_indices[:, 0]] - b[pair_indices[:, 1]]
    nll = (
        0.5 * (diffs / simu_sigma) ** 2
        + torch.log(simu_sigma)
        + 0.5 * LOG_2PI
    )
    return nll.sum()


def adp_simu_math(
    b: torch.Tensor,
    pair_indices: torch.Tensor,
    simu_sigma: torch.Tensor,
) -> torch.Tensor:
    """ADP similarity (SIMU) NLL on bonded-atom B-factor differences.

    Dispatches to
    :func:`torchref.base.targets.triton.adp_simu_math_triton` on CUDA
    float32 (~1.6× faster fwd+bw on A100). Falls back to eager
    otherwise.

    Parameters
    ----------
    b : torch.Tensor
        (N_atoms,) B-factors.
    pair_indices : torch.Tensor
        (N, 2) bonded-atom pairs to compare.
    simu_sigma : torch.Tensor
        Scalar sigma on the difference (a buffer in the target).
    """
    if use_triton(b):
        from .triton.adp_simu import adp_simu_math_triton
        return adp_simu_math_triton(b, pair_indices, simu_sigma)
    return _adp_simu_math_eager(b, pair_indices, simu_sigma)

