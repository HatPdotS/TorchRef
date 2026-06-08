"""Gaussian X-ray loss math.

Caller is responsible for producing the post-``XrayTarget.get_data``
tensors (F_obs, F_calc, sigma, mask).

The public entry point ``gaussian_xray_loss_math`` dispatches to the
Triton kernel on CUDA float32 and to the eager implementation
otherwise, matching the pattern used by ``bond_math``, ``angle_math``,
etc.
"""

import numpy as np
import torch

from ._dispatch import use_triton


def _gaussian_xray_loss_math_eager(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    F_calc_amp = torch.abs(F_calc)
    diff = F_obs - F_calc_amp

    eps = torch.median(sigma) * 1e-1
    sigma_safe = torch.clamp(sigma, min=eps)

    log_2pi = torch.log(
        torch.tensor(2.0 * np.pi, device=sigma.device, dtype=sigma.dtype)
    )
    nll = 0.5 * (diff ** 2) / (sigma_safe ** 2) + torch.log(sigma_safe) + 0.5 * log_2pi
    return (nll * mask).sum()


def gaussian_xray_loss_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Gaussian NLL on already-scaled amplitudes.

    ``mask`` defaults to all reflections (``None``); compact inputs need no mask.
    Dispatches to :func:`torchref.base.targets.triton.xray_gaussian.gaussian_xray_loss_math_triton`
    on CUDA float32 inputs; falls back to the eager implementation otherwise.
    """
    if mask is None:
        mask = torch.ones(F_obs.shape[0], dtype=torch.bool, device=F_obs.device)
    if use_triton(F_calc, F_obs, sigma):
        from .triton.xray_gaussian import gaussian_xray_loss_math_triton
        return gaussian_xray_loss_math_triton(F_obs, F_calc, sigma, mask)
    return _gaussian_xray_loss_math_eager(F_obs, F_calc, sigma, mask)
