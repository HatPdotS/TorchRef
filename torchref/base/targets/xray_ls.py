"""Least-squares X-ray loss math.

Caller is responsible for producing the post-``XrayTarget.get_data``
tensors (F_obs, F_calc, sigma, mask).
"""

import torch

from ._dispatch import use_triton


def _ls_xray_loss_math_eager(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    weighting: str = "sigma",
) -> torch.Tensor:
    F_calc_amp = torch.abs(F_calc)
    diff = F_obs - F_calc_amp

    if weighting == "sigma":
        eps = torch.median(sigma) * 1e-1
        sigma_safe = torch.clamp(sigma, min=eps)
        weights = 1.0 / (sigma_safe ** 2)
    elif weighting == "unit":
        weights = torch.ones_like(F_obs)
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting}")

    loss = 0.5 * weights * (diff ** 2)
    return (loss * mask).sum()


def ls_xray_loss_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    weighting: str = "sigma",
) -> torch.Tensor:
    """Weighted least-squares loss on already-scaled amplitudes.

    Dispatches to :func:`torchref.base.targets.triton.xray_ls.ls_xray_loss_math_triton`
    on CUDA float32; falls back to the eager implementation otherwise.
    """
    if use_triton(F_calc, F_obs, sigma):
        from .triton.xray_ls import ls_xray_loss_math_triton
        return ls_xray_loss_math_triton(F_obs, F_calc, sigma, mask, weighting=weighting)
    return _ls_xray_loss_math_eager(F_obs, F_calc, sigma, mask, weighting=weighting)
