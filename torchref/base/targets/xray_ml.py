"""Rice maximum-likelihood X-ray loss math.

Mirrors the Rice-distribution loss in
``torchref/refinement/targets/xray/rice.py`` verbatim. The
caller is responsible for everything ``XrayTarget.get_data`` does:
unpacking ``ReflectionData``, running the ``Scaler`` forward to produce
``|F_calc|`` from a complex ``f_calc``, and building the work/free mask.
"""

import numpy as np
import torch

from ._dispatch import use_triton


def _ml_xray_loss_math_eager(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    centric_flags: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.ones_like(F_obs)
    beta = sigma ** 2
    epsilon = torch.ones_like(F_obs)

    if centric_flags is None:
        centric_flags = torch.zeros_like(F_obs, dtype=torch.bool)

    F_calc_amp = torch.abs(F_calc)

    eb = epsilon * beta
    eb = torch.clamp(eb, min=1e-6)

    term1 = -torch.log(2 * F_obs / eb + 1e-12)
    term2 = (F_obs ** 2) / eb
    term3 = (alpha * F_calc_amp) ** 2 / eb

    arg_bessel = 2 * alpha * F_obs * F_calc_amp / eb
    arg_bessel = torch.clamp(arg_bessel, max=1e6)
    term4 = -(torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel)

    loss_acentric = term1 + term2 + term3 + term4

    term1_c = -0.5 * torch.log(2 / (np.pi * eb) + 1e-12)
    term2_c = (F_obs ** 2) / (2 * eb)
    term3_c = (alpha * F_calc_amp) ** 2 / (2 * eb)
    term4_c = -(alpha * F_obs * F_calc_amp) / eb

    arg_exp = -2 * alpha * F_obs * F_calc_amp / eb
    arg_exp_safe = torch.clamp(arg_exp, min=-80.0, max=80.0)
    term5_c = -torch.log((1 + torch.exp(arg_exp_safe)) / 2 + 1e-12)

    loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

    loss = torch.where(centric_flags, loss_centric, loss_acentric)
    loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))

    return (loss * mask).sum()


def ml_xray_loss_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    centric_flags: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Maximum-likelihood X-ray loss on already-scaled amplitudes.

    Matches ``RiceXrayTarget.forward``.

    Dispatches to :func:`torchref.base.targets.triton.xray_ml.ml_xray_loss_math_triton`
    on CUDA float32; falls back to the eager implementation otherwise.

    Parameters
    ----------
    F_obs : torch.Tensor
        (N,) observed amplitudes (zeros outside ``mask``).
    F_calc : torch.Tensor
        (N,) scaled calculated amplitudes (already real-valued, zeros
        outside ``mask``).
    sigma : torch.Tensor
        (N,) per-reflection sigma (ones outside ``mask``).
    centric_flags : torch.Tensor or None
        (N,) bool, True for centric reflections. ``None`` is treated as all-acentric.
    mask : torch.Tensor
        (N,) bool work-set mask applied to the final sum.
    """
    if mask is None:
        mask = torch.ones(F_obs.shape[0], dtype=torch.bool, device=F_obs.device)
    if use_triton(F_calc, F_obs, sigma):
        from .triton.xray_ml import ml_xray_loss_math_triton
        return ml_xray_loss_math_triton(F_obs, F_calc, sigma, centric_flags, mask)
    return _ml_xray_loss_math_eager(F_obs, F_calc, sigma, centric_flags, mask)
