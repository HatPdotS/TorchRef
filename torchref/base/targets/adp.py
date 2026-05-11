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


def adp_kl_math(
    log_adp: torch.Tensor,
    target_log_std: float = 0.2,
) -> torch.Tensor:
    """KL divergence regularizer on log(B).

    Mirrors ``Model.adp_kl_divergence_loss``: KL between an empirical Gaussian
    (mean fixed to current ``mean(log_adp)`` detached, std = ``std(log_adp)``)
    and a target Gaussian with the same mean but fixed std.
    """
    sigma_data = torch.std(log_adp)
    log_sigma_ratio = torch.log(
        torch.tensor(target_log_std, device=log_adp.device, dtype=log_adp.dtype)
        / sigma_data
    )
    variance_ratio = (sigma_data ** 2) / (2 * target_log_std ** 2)
    return log_sigma_ratio + variance_ratio - 0.5


def adp_locality_math(
    b: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
) -> torch.Tensor:
    """ADP locality NLL: weighted MSE on log(B) differences with KNN.

    Mirrors ``ADPLocalityTarget.forward``. Neighbor list construction is the
    target's bookkeeping and is not included here.
    """
    log_adp = torch.log(b.clamp(min=1e-3))
    neighbor_log_adp = log_adp[neighbor_indices]
    diff = log_adp.unsqueeze(1) - neighbor_log_adp
    weights = 1.0 / (neighbor_distances + 1e-6)
    weighted_sq_diff = weights * (diff / 0.5) ** 2
    return weighted_sq_diff.sum()
