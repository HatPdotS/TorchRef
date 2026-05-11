"""Triton kernels for the Bhattacharyya X-ray target.

Matches :func:`torchref.base.targets.xray_bhattacharyya.bhattacharyya_xray_loss_math`
to within float32 precision. ``sigma_m`` enters as a constant input (the
eager target builds it under ``no_grad``).

The math is per-reflection; the kernel reduces nothing — we ``.sum()`` the
per-reflection tensor on host so the autograd glue stays trivial.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_EPS = 1e-6


@triton.jit
def _bhatt_fwd_kernel(
    F_obs_ptr,        # (N,)
    F_calc_ptr,       # (N,) real non-negative amplitudes
    sigma_d_ptr,      # (N,)
    sigma_m_ptr,      # (N,)
    mask_ptr,         # (N,) bool
    out_ptr,          # (N,) per-reflection loss
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    sd = tl.load(sigma_d_ptr + offs, mask=valid, other=1.0)
    sm = tl.load(sigma_m_ptr + offs, mask=valid, other=1.0)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    sd_s = tl.where(sd < EPS, EPS, sd)
    sm_s = tl.where(sm < EPS, EPS, sm)
    var_d = sd_s * sd_s
    var_m = sm_s * sm_s
    var_sum = var_d + var_m

    diff = F_obs - F_calc
    l_mean = (diff * diff) / (4.0 * var_sum)
    l_var = 0.5 * tl.log(var_sum / (2.0 * sd_s * sm_s))

    loss = (l_mean + l_var) * m
    tl.store(out_ptr + offs, loss, mask=valid)


@triton.jit
def _bhatt_bwd_kernel(
    F_obs_ptr,
    F_calc_ptr,
    sigma_d_ptr,
    sigma_m_ptr,
    mask_ptr,
    grad_out,        # scalar
    dF_calc_ptr,     # (N,) — written directly (not atomic, one program per range)
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    sd = tl.load(sigma_d_ptr + offs, mask=valid, other=1.0)
    sm = tl.load(sigma_m_ptr + offs, mask=valid, other=1.0)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    sd_s = tl.where(sd < EPS, EPS, sd)
    sm_s = tl.where(sm < EPS, EPS, sm)
    var_sum = sd_s * sd_s + sm_s * sm_s

    diff = F_obs - F_calc
    # dL_h / dF_calc = -diff / (2 * var_sum) (only l_mean depends on F_calc)
    dl = -diff / (2.0 * var_sum)
    g = grad_out * dl * m
    tl.store(dF_calc_ptr + offs, g, mask=valid)


class _BhattXrayMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F_obs, F_calc, sigma_d, sigma_m, mask):
        assert F_calc.is_cuda and F_calc.dtype == torch.float32
        N = F_calc.shape[0]
        # Ensure contiguous, float32 inputs (loads assume stride 1).
        F_obs = F_obs.contiguous()
        F_calc = F_calc.contiguous()
        sigma_d = sigma_d.contiguous()
        sigma_m = sigma_m.contiguous()
        mask_u8 = mask.to(torch.uint8).contiguous()

        out = torch.empty(N, dtype=F_calc.dtype, device=F_calc.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _bhatt_fwd_kernel[grid](
            F_obs, F_calc, sigma_d, sigma_m, mask_u8, out,
            N=N, EPS=_EPS, BLOCK=BLOCK,
        )
        ctx.save_for_backward(F_obs, F_calc, sigma_d, sigma_m, mask_u8)
        return out.sum()

    @staticmethod
    def backward(ctx, grad_out):
        F_obs, F_calc, sigma_d, sigma_m, mask_u8 = ctx.saved_tensors
        N = F_calc.shape[0]
        dF_calc = torch.empty_like(F_calc)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _bhatt_bwd_kernel[grid](
            F_obs, F_calc, sigma_d, sigma_m, mask_u8,
            float(grad_out.item()), dF_calc,
            N=N, EPS=_EPS, BLOCK=BLOCK,
        )
        return None, dF_calc, None, None, None


def bhattacharyya_xray_loss_math_triton(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma_d: torch.Tensor,
    sigma_m: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Triton-backed Bhattacharyya overlap loss.

    Drop-in replacement for
    :func:`torchref.base.targets.xray_bhattacharyya.bhattacharyya_xray_loss_math`.
    """
    return _BhattXrayMathTriton.apply(F_obs, F_calc, sigma_d, sigma_m, mask)
