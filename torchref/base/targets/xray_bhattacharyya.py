"""Bhattacharyya overlap X-ray loss math.

Mirrors the loss computation in :class:`BhattacharyyaXrayTarget.forward` after
``sigma_m`` has been computed in a no-grad block (so ``sigma_m`` is treated
as a constant input here).
"""

import torch

from ._dispatch import use_triton


def _bhattacharyya_xray_loss_math_eager(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma_d: torch.Tensor,
    sigma_m: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    F_calc_amp = torch.abs(F_calc)
    eps = 1e-6
    sigma_d_safe = torch.clamp(sigma_d, min=eps)
    sigma_m_safe = torch.clamp(sigma_m, min=eps)

    var_d = sigma_d_safe ** 2
    var_m = sigma_m_safe ** 2
    var_sum = var_d + var_m

    diff = F_obs - F_calc_amp
    l_mean = (diff ** 2) / (4.0 * var_sum)
    l_var = 0.5 * torch.log(var_sum / (2.0 * sigma_d_safe * sigma_m_safe))

    return ((l_mean + l_var) * mask).sum()


def bhattacharyya_xray_loss_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma_d: torch.Tensor,
    sigma_m: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Bhattacharyya overlap loss between data and model Gaussians.

    L_h = (F_obs - |F_calc|)^2 / (4 * (sigma_d^2 + sigma_m^2))
        + 0.5 * log((sigma_d^2 + sigma_m^2) / (2 * sigma_d * sigma_m))

    ``mask`` defaults to all reflections (``None``); compact inputs need no mask.
    Dispatches to :func:`torchref.base.targets.triton.xray_bhattacharyya.bhattacharyya_xray_loss_math_triton`
    on CUDA float32; falls back to the eager implementation otherwise.
    """
    if mask is None:
        mask = torch.ones(F_obs.shape[0], dtype=torch.bool, device=F_obs.device)
    if use_triton(F_calc, F_obs, sigma_d, sigma_m):
        from .triton.xray_bhattacharyya import bhattacharyya_xray_loss_math_triton
        return bhattacharyya_xray_loss_math_triton(F_obs, F_calc, sigma_d, sigma_m, mask)
    return _bhattacharyya_xray_loss_math_eager(F_obs, F_calc, sigma_d, sigma_m, mask)
