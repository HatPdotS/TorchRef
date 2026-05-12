"""Triton forward + backward for the chiral-volume target."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_LOG_2PI = float(math.log(2.0 * math.pi))


@triton.jit
def _chiral_nll_fwd_kernel(
    xyz_ptr,
    idx_ptr,        # (N, 4) [center, a1, a2, a3]
    ideal_ptr,      # (N,)
    sig_ptr,        # (N,)
    out_ptr,        # (N,)
    N: tl.constexpr,
    LOG_2PI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    c0 = tl.load(idx_ptr + offs * 4 + 0, mask=mask, other=0)
    i1 = tl.load(idx_ptr + offs * 4 + 1, mask=mask, other=0)
    i2 = tl.load(idx_ptr + offs * 4 + 2, mask=mask, other=0)
    i3 = tl.load(idx_ptr + offs * 4 + 3, mask=mask, other=0)

    pcx = tl.load(xyz_ptr + c0 * 3 + 0, mask=mask, other=0.0)
    pcy = tl.load(xyz_ptr + c0 * 3 + 1, mask=mask, other=0.0)
    pcz = tl.load(xyz_ptr + c0 * 3 + 2, mask=mask, other=0.0)

    v1x = tl.load(xyz_ptr + i1 * 3 + 0, mask=mask, other=0.0) - pcx
    v1y = tl.load(xyz_ptr + i1 * 3 + 1, mask=mask, other=0.0) - pcy
    v1z = tl.load(xyz_ptr + i1 * 3 + 2, mask=mask, other=0.0) - pcz
    v2x = tl.load(xyz_ptr + i2 * 3 + 0, mask=mask, other=0.0) - pcx
    v2y = tl.load(xyz_ptr + i2 * 3 + 1, mask=mask, other=0.0) - pcy
    v2z = tl.load(xyz_ptr + i2 * 3 + 2, mask=mask, other=0.0) - pcz
    v3x = tl.load(xyz_ptr + i3 * 3 + 0, mask=mask, other=0.0) - pcx
    v3y = tl.load(xyz_ptr + i3 * 3 + 1, mask=mask, other=0.0) - pcy
    v3z = tl.load(xyz_ptr + i3 * 3 + 2, mask=mask, other=0.0) - pcz

    # cross23 = v2 x v3
    cx = v2y * v3z - v2z * v3y
    cy = v2z * v3x - v2x * v3z
    cz = v2x * v3y - v2y * v3x
    vol = v1x * cx + v1y * cy + v1z * cz

    ideal = tl.load(ideal_ptr + offs, mask=mask, other=0.0)
    sig = tl.load(sig_ptr + offs, mask=mask, other=1.0)

    achiral = ideal == 0.0
    eff_ideal = tl.where(achiral, 2.5, ideal)
    eff_vol = tl.where(achiral, libdevice.abs(vol), vol)
    dev = eff_vol - eff_ideal

    nll = 0.5 * (dev / sig) * (dev / sig) + tl.log(sig) + 0.5 * LOG_2PI
    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _chiral_nll_bwd_kernel(
    xyz_ptr,
    idx_ptr,
    ideal_ptr,
    sig_ptr,
    grad_out_ptr,  # 0-D tensor — loaded in-kernel (no host .item() sync)
    dxyz_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gradient of the chiral NLL.

    V = v1·(v2 × v3), vi = pi − pc. Then
        ∂V/∂p1 = v2 × v3
        ∂V/∂p2 = v3 × v1
        ∂V/∂p3 = v1 × v2
        ∂V/∂pc = −(∂V/∂p1 + ∂V/∂p2 + ∂V/∂p3)

    eff_vol = |V|   if achiral else V    ⇒ ∂eff_vol/∂V = sign(V) or 1
    dNLL/d(eff_vol) = (eff_vol − eff_ideal) / σ²

    ``grad_out_ptr`` is a 0-D tensor pointer; we load it once per block
    so the host doesn't need a ``.item()`` synchronize.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    grad_out = tl.load(grad_out_ptr)

    c0 = tl.load(idx_ptr + offs * 4 + 0, mask=mask, other=0)
    i1 = tl.load(idx_ptr + offs * 4 + 1, mask=mask, other=0)
    i2 = tl.load(idx_ptr + offs * 4 + 2, mask=mask, other=0)
    i3 = tl.load(idx_ptr + offs * 4 + 3, mask=mask, other=0)

    pcx = tl.load(xyz_ptr + c0 * 3 + 0, mask=mask, other=0.0)
    pcy = tl.load(xyz_ptr + c0 * 3 + 1, mask=mask, other=0.0)
    pcz = tl.load(xyz_ptr + c0 * 3 + 2, mask=mask, other=0.0)

    v1x = tl.load(xyz_ptr + i1 * 3 + 0, mask=mask, other=0.0) - pcx
    v1y = tl.load(xyz_ptr + i1 * 3 + 1, mask=mask, other=0.0) - pcy
    v1z = tl.load(xyz_ptr + i1 * 3 + 2, mask=mask, other=0.0) - pcz
    v2x = tl.load(xyz_ptr + i2 * 3 + 0, mask=mask, other=0.0) - pcx
    v2y = tl.load(xyz_ptr + i2 * 3 + 1, mask=mask, other=0.0) - pcy
    v2z = tl.load(xyz_ptr + i2 * 3 + 2, mask=mask, other=0.0) - pcz
    v3x = tl.load(xyz_ptr + i3 * 3 + 0, mask=mask, other=0.0) - pcx
    v3y = tl.load(xyz_ptr + i3 * 3 + 1, mask=mask, other=0.0) - pcy
    v3z = tl.load(xyz_ptr + i3 * 3 + 2, mask=mask, other=0.0) - pcz

    cx23 = v2y * v3z - v2z * v3y
    cy23 = v2z * v3x - v2x * v3z
    cz23 = v2x * v3y - v2y * v3x
    vol = v1x * cx23 + v1y * cy23 + v1z * cz23

    ideal = tl.load(ideal_ptr + offs, mask=mask, other=0.0)
    sig = tl.load(sig_ptr + offs, mask=mask, other=1.0)

    achiral = ideal == 0.0
    eff_ideal = tl.where(achiral, 2.5, ideal)
    eff_vol = tl.where(achiral, libdevice.abs(vol), vol)
    dev = eff_vol - eff_ideal

    # d eff_vol / dV  ∈ {sign(vol), 1.0}
    sign_v = tl.where(vol >= 0.0, 1.0, -1.0)
    dEff_dV = tl.where(achiral, sign_v, 1.0)
    coef = grad_out * dev / (sig * sig) * dEff_dV  # = dL/dV per chiral center

    # ∂V/∂p1 = v2 x v3 = (cx23, cy23, cz23)
    g1x = coef * cx23; g1y = coef * cy23; g1z = coef * cz23
    # ∂V/∂p2 = v3 x v1
    g2x = coef * (v3y * v1z - v3z * v1y)
    g2y = coef * (v3z * v1x - v3x * v1z)
    g2z = coef * (v3x * v1y - v3y * v1x)
    # ∂V/∂p3 = v1 x v2
    g3x = coef * (v1y * v2z - v1z * v2y)
    g3y = coef * (v1z * v2x - v1x * v2z)
    g3z = coef * (v1x * v2y - v1y * v2x)

    tl.atomic_add(dxyz_ptr + i1 * 3 + 0, g1x, mask=mask)
    tl.atomic_add(dxyz_ptr + i1 * 3 + 1, g1y, mask=mask)
    tl.atomic_add(dxyz_ptr + i1 * 3 + 2, g1z, mask=mask)
    tl.atomic_add(dxyz_ptr + i2 * 3 + 0, g2x, mask=mask)
    tl.atomic_add(dxyz_ptr + i2 * 3 + 1, g2y, mask=mask)
    tl.atomic_add(dxyz_ptr + i2 * 3 + 2, g2z, mask=mask)
    tl.atomic_add(dxyz_ptr + i3 * 3 + 0, g3x, mask=mask)
    tl.atomic_add(dxyz_ptr + i3 * 3 + 1, g3y, mask=mask)
    tl.atomic_add(dxyz_ptr + i3 * 3 + 2, g3z, mask=mask)
    # center: -(g1 + g2 + g3)
    tl.atomic_add(dxyz_ptr + c0 * 3 + 0, -(g1x + g2x + g3x), mask=mask)
    tl.atomic_add(dxyz_ptr + c0 * 3 + 1, -(g1y + g2y + g3y), mask=mask)
    tl.atomic_add(dxyz_ptr + c0 * 3 + 2, -(g1z + g2z + g3z), mask=mask)


class _ChiralMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, indices, ideal_volumes, sigmas):
        assert xyz.is_cuda and xyz.dtype == torch.float32
        N = indices.shape[0]
        nll = torch.empty(N, dtype=xyz.dtype, device=xyz.device)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _chiral_nll_fwd_kernel[grid](
            xyz, indices, ideal_volumes, sigmas, nll,
            N=N, LOG_2PI=_LOG_2PI, BLOCK=BLOCK,
        )
        ctx.save_for_backward(xyz, indices, ideal_volumes, sigmas)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        xyz, idx, ideal, sigs = ctx.saved_tensors
        N = idx.shape[0]
        dxyz = torch.zeros_like(xyz)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _chiral_nll_bwd_kernel[grid](
            xyz, idx, ideal, sigs, grad_out, dxyz,
            N=N, BLOCK=BLOCK,
        )
        return dxyz, None, None, None


def chiral_math_triton(xyz, indices, ideal_volumes, sigmas):
    """Triton-backed chiral-volume Gaussian NLL with analytic backward."""
    return _ChiralMathTriton.apply(xyz, indices, ideal_volumes, sigmas)
