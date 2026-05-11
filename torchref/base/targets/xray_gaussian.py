"""Gaussian X-ray loss math.

Mirrors :class:`GaussianXrayTarget.forward`. The caller is responsible for
producing the post-:meth:`get_data` tensors.
"""

import numpy as np
import torch


def gaussian_xray_loss_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood on already-scaled amplitudes.

    Matches ``GaussianXrayTarget.forward`` lines 37-51.
    """
    F_calc_amp = torch.abs(F_calc)
    diff = F_obs - F_calc_amp

    eps = torch.median(sigma) * 1e-1
    sigma_safe = torch.clamp(sigma, min=eps)

    log_2pi = torch.log(
        torch.tensor(2.0 * np.pi, device=sigma.device, dtype=sigma.dtype)
    )
    nll = 0.5 * (diff ** 2) / (sigma_safe ** 2) + torch.log(sigma_safe) + 0.5 * log_2pi
    return (nll * mask).sum()
