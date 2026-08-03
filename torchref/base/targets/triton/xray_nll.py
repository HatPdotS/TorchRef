"""Triton kernels for the ``nll`` X-ray target (Gaussian at sigma_obs**2).

Fused: the ``median(sigma)*1e-1`` clamp is computed inside the kernel. The eager
counterpart is :func:`torchref.base.targets.xray_likelihoods.nll_math` composed with
``amplitude_var_from_sigma_obs``; ``xray_nll.nll_sigma_obs_math`` chooses between them.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG_2PI = float(math.log(2.0 * math.pi))


@triton.jit
def _gauss_fwd_kernel(
    F_obs_ptr,
    F_calc_ptr,
    sigma_ptr,
    mask_ptr,
    sigma_floor_ptr,    # 0-D tensor = median(sigma) * 1e-1, loaded in-kernel
    log_2pi,            # scalar (compile-time constant — same every call)
    out_ptr,            # (N,)
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    sigma_floor = tl.load(sigma_floor_ptr)

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    sig = tl.load(sigma_ptr + offs, mask=valid, other=1.0)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    sig_safe = tl.where(sig < sigma_floor, sigma_floor, sig)
    diff = F_obs - F_calc
    inv_s = 1.0 / sig_safe
    nll = 0.5 * (diff * inv_s) * (diff * inv_s) + tl.log(sig_safe) + 0.5 * log_2pi
    tl.store(out_ptr + offs, nll * m, mask=valid)


@triton.jit
def _gauss_bwd_kernel(
    F_obs_ptr,
    F_calc_ptr,
    sigma_ptr,
    mask_ptr,
    sigma_floor_ptr,    # 0-D tensor
    grad_out_ptr,       # 0-D tensor
    dF_calc_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    sigma_floor = tl.load(sigma_floor_ptr)
    grad_out = tl.load(grad_out_ptr)

    F_obs = tl.load(F_obs_ptr + offs, mask=valid, other=0.0)
    F_calc = tl.load(F_calc_ptr + offs, mask=valid, other=0.0)
    sig = tl.load(sigma_ptr + offs, mask=valid, other=1.0)
    m = tl.load(mask_ptr + offs, mask=valid, other=0).to(tl.float32)

    sig_safe = tl.where(sig < sigma_floor, sigma_floor, sig)
    diff = F_obs - F_calc
    inv_s2 = 1.0 / (sig_safe * sig_safe)
    # dNLL_h / dF_calc = -diff / sigma^2
    g = grad_out * (-diff) * inv_s2 * m
    tl.store(dF_calc_ptr + offs, g, mask=valid)


class _GaussXrayMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F_obs, F_calc, sigma, mask):
        assert F_calc.is_cuda and F_calc.dtype == torch.float32
        N = F_calc.shape[0]
        F_obs = F_obs.contiguous()
        F_calc = F_calc.contiguous()
        sigma = sigma.contiguous()
        mask_u8 = mask.to(torch.uint8).contiguous()

        # Match eager: floor = median(sigma) * 1e-1. Keep as 0-D device
        # tensor — no .item() host sync.
        sigma_floor_t = (torch.median(sigma) * 0.1).to(F_calc.dtype)

        out = torch.empty(N, dtype=F_calc.dtype, device=F_calc.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _gauss_fwd_kernel[grid](
            F_obs, F_calc, sigma, mask_u8, sigma_floor_t, _LOG_2PI, out,
            N=N, BLOCK=BLOCK,
        )
        ctx.save_for_backward(F_obs, F_calc, sigma, mask_u8, sigma_floor_t)
        return out.sum()

    @staticmethod
    def backward(ctx, grad_out):
        F_obs, F_calc, sigma, mask_u8, sigma_floor_t = ctx.saved_tensors
        N = F_calc.shape[0]
        dF_calc = torch.empty_like(F_calc)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _gauss_bwd_kernel[grid](
            F_obs, F_calc, sigma, mask_u8, sigma_floor_t,
            grad_out, dF_calc,
            N=N, BLOCK=BLOCK,
        )
        return None, dF_calc, None, None


def nll_sigma_obs_math_triton(F_obs, F_calc, sigma, mask):
    """Triton-backed Gaussian amplitude NLL at ``var = clamp(sigma)**2``."""
    return _GaussXrayMathTriton.apply(F_obs, F_calc, sigma, mask)
