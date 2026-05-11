"""Least-squares X-ray loss math.

Mirrors :class:`LeastSquaresXrayTarget.forward`.
"""

import torch


def ls_xray_loss_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    weighting: str = "sigma",
) -> torch.Tensor:
    """Weighted least-squares loss on already-scaled amplitudes."""
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
