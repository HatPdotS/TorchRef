"""Triton forward + Triton backward for the ADP-similarity (SIMU) target.

The math is trivial — gather two B-factors, subtract, Gaussian NLL — but
this target showed the widest gap between the pure-math function and the
full eager target in benchmarking (the math step was ~0.23 of the eager
forward time), so it's a clean win to tritonize.

All scalar parameters (``sigma``, ``log_sigma``, ``grad_out``) are
passed as 0-D device tensors and ``tl.load``ed in-kernel — no
``.item()`` host syncs.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG_2PI = float(math.log(2.0 * math.pi))


@triton.jit
def _adp_simu_fwd_kernel(
    b_ptr,             # (N_atoms,)
    idx_ptr,           # (N, 2)
    sigma_ptr,         # 0-D tensor
    log_sigma_ptr,     # 0-D tensor — precomputed log(sigma)
    out_ptr,           # (N,)
    N: tl.constexpr,
    LOG_2PI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    sigma = tl.load(sigma_ptr)
    log_sigma = tl.load(log_sigma_ptr)

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
    sigma_ptr,         # 0-D tensor
    grad_out_ptr,      # 0-D tensor
    db_ptr,            # (N_atoms,)
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    sigma = tl.load(sigma_ptr)
    grad_out = tl.load(grad_out_ptr)

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
        # ``simu_sigma`` should already be on ``b.device`` (registered as
        # a buffer on the target module which gets moved with the model).
        # Skip the .to() if so — calling .to() creates a fresh tensor
        # each call (forbidden during CUDA Graph capture: counts as a
        # host→device sync setup op). Fall back to .to() only if the
        # caller passed a CPU tensor by mistake.
        if simu_sigma.device == b.device and simu_sigma.dtype == b.dtype:
            sigma_t = simu_sigma if simu_sigma.is_contiguous() else simu_sigma.contiguous()
        else:
            sigma_t = simu_sigma.to(device=b.device, dtype=b.dtype).contiguous()
        log_sigma_t = torch.log(sigma_t)
        nll = torch.empty(N, dtype=b.dtype, device=b.device)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _adp_simu_fwd_kernel[grid](
            b, pair_indices, sigma_t, log_sigma_t, nll,
            N=N, LOG_2PI=_LOG_2PI, BLOCK=BLOCK,
        )
        ctx.save_for_backward(b, pair_indices, sigma_t)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        b, pair_indices, sigma_t = ctx.saved_tensors
        N = pair_indices.shape[0]
        db = torch.zeros_like(b)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _adp_simu_bwd_kernel[grid](
            b, pair_indices, sigma_t, grad_out, db,
            N=N, BLOCK=BLOCK,
        )
        return db, None, None


def adp_simu_math_triton(b, pair_indices, simu_sigma):
    """Triton-backed ADP similarity (SIMU) Gaussian NLL.

    Drop-in replacement for :func:`torchref.base.targets.adp.adp_simu_math`.
    """
    return _ADPSimuMathTriton.apply(b, pair_indices, simu_sigma)
