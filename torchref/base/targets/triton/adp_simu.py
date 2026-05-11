"""Triton forward + Triton backward for the ADP-similarity (SIMU) target.

The math is trivial — gather two B-factors, subtract, Gaussian NLL — but
this target showed the widest math/target gap (~0.23 forward) in
benchmarking, so it's a clean win to tritonize.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG_2PI = float(math.log(2.0 * math.pi))


@triton.jit
def _adp_simu_fwd_kernel(
    b_ptr,         # (N_atoms,)
    idx_ptr,       # (N, 2)
    sigma,         # scalar (passed by value)
    log_sigma,     # scalar -- precomputed log(sigma)
    out_ptr,       # (N,)
    N: tl.constexpr,
    LOG_2PI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    i = tl.load(idx_ptr + offs * 2 + 0, mask=mask, other=0)
    j = tl.load(idx_ptr + offs * 2 + 1, mask=mask, other=0)
    bi = tl.load(b_ptr + i, mask=mask, other=0.0)
    bj = tl.load(b_ptr + j, mask=mask, other=0.0)
    diff = bi - bj
    nll = 0.5 * (diff / sigma) * (diff / sigma) + log_sigma + 0.5 * LOG_2PI
    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _adp_simu_bwd_kernel(
    b_ptr,
    idx_ptr,
    sigma,
    grad_out,      # scalar
    db_ptr,        # (N_atoms,)
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    i = tl.load(idx_ptr + offs * 2 + 0, mask=mask, other=0)
    j = tl.load(idx_ptr + offs * 2 + 1, mask=mask, other=0)
    bi = tl.load(b_ptr + i, mask=mask, other=0.0)
    bj = tl.load(b_ptr + j, mask=mask, other=0.0)
    diff = bi - bj
    g = grad_out * diff / (sigma * sigma)

    tl.atomic_add(db_ptr + i, g, mask=mask)
    tl.atomic_add(db_ptr + j, -g, mask=mask)


class _ADPSimuMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, pair_indices, simu_sigma):
        assert b.is_cuda and b.dtype == torch.float32
        N = pair_indices.shape[0]
        sigma = float(simu_sigma.item())
        log_sigma = float(torch.log(simu_sigma).item())
        nll = torch.empty(N, dtype=b.dtype, device=b.device)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _adp_simu_fwd_kernel[grid](
            b, pair_indices, sigma, log_sigma, nll,
            N=N, LOG_2PI=_LOG_2PI, BLOCK=BLOCK,
        )
        ctx.save_for_backward(b, pair_indices, simu_sigma)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        b, pair_indices, simu_sigma = ctx.saved_tensors
        N = pair_indices.shape[0]
        db = torch.zeros_like(b)
        sigma = float(simu_sigma.item())
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _adp_simu_bwd_kernel[grid](
            b, pair_indices, sigma, grad_out.item(), db,
            N=N, BLOCK=BLOCK,
        )
        return db, None, None


def adp_simu_math_triton(b, pair_indices, simu_sigma):
    """Triton-backed ADP similarity (SIMU) Gaussian NLL.

    Drop-in replacement for :func:`torchref.base.targets.adp.adp_simu_math`.
    """
    return _ADPSimuMathTriton.apply(b, pair_indices, simu_sigma)
