"""Triton kernels for the Least-Squares X-ray target.

Supports two weighting modes: 'sigma' (1/sigma^2 weights, with the same
median-floor as the eager target) and 'unit' (all weights = 1).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ls_fwd_kernel(
    F_obs_ptr,
    F_calc_ptr,
    sigma_ptr,
    mask_ptr,
    sigma_floor,        # scalar
    USE_UNIT_W: tl.constexpr,
    out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    diff = F_obs - F_calc
    if USE_UNIT_W:
        w = 1.0
    else:
        sig = tl.load(sigma_ptr + offs, mask=valid, other=1.0)
        sig_safe = tl.where(sig < sigma_floor, sigma_floor, sig)
        w = 1.0 / (sig_safe * sig_safe)

    loss = 0.5 * w * diff * diff
    tl.store(out_ptr + offs, loss * m, mask=valid)


@triton.jit
def _ls_bwd_kernel(
    F_obs_ptr,
    F_calc_ptr,
    sigma_ptr,
    mask_ptr,
    sigma_floor,
    grad_out,
    USE_UNIT_W: tl.constexpr,
    dF_calc_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    diff = F_obs - F_calc
    if USE_UNIT_W:
        w = 1.0
    else:
        sig = tl.load(sigma_ptr + offs, mask=valid, other=1.0)
        sig_safe = tl.where(sig < sigma_floor, sigma_floor, sig)
        w = 1.0 / (sig_safe * sig_safe)

    # dL_h / dF_calc = -w * diff
    g = grad_out * (-w) * diff * m
    tl.store(dF_calc_ptr + offs, g, mask=valid)


class _LSXrayMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F_obs, F_calc, sigma, mask, weighting):
        assert F_calc.is_cuda and F_calc.dtype == torch.float32
        if weighting not in ("sigma", "unit"):
            raise ValueError(f"Unknown weighting scheme: {weighting}")
        use_unit = weighting == "unit"

        N = F_calc.shape[0]
        F_obs = F_obs.contiguous()
        F_calc = F_calc.contiguous()
        sigma = sigma.contiguous()
        mask_u8 = mask.to(torch.uint8).contiguous()
        sigma_floor = 0.0 if use_unit else float((torch.median(sigma) * 0.1).item())

        out = torch.empty(N, dtype=F_calc.dtype, device=F_calc.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _ls_fwd_kernel[grid](
            F_obs, F_calc, sigma, mask_u8, sigma_floor,
            USE_UNIT_W=use_unit, out_ptr=out, N=N, BLOCK=BLOCK,
        )
        ctx.save_for_backward(F_obs, F_calc, sigma, mask_u8)
        ctx.sigma_floor = sigma_floor
        ctx.use_unit = use_unit
        return out.sum()

    @staticmethod
    def backward(ctx, grad_out):
        F_obs, F_calc, sigma, mask_u8 = ctx.saved_tensors
        N = F_calc.shape[0]
        dF_calc = torch.empty_like(F_calc)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _ls_bwd_kernel[grid](
            F_obs, F_calc, sigma, mask_u8, ctx.sigma_floor,
            float(grad_out.item()),
            USE_UNIT_W=ctx.use_unit, dF_calc_ptr=dF_calc,
            N=N, BLOCK=BLOCK,
        )
        return None, dF_calc, None, None, None


def ls_xray_loss_math_triton(F_obs, F_calc, sigma, mask, weighting="sigma"):
    """Triton-backed least-squares X-ray loss."""
    return _LSXrayMathTriton.apply(F_obs, F_calc, sigma, mask, weighting)
