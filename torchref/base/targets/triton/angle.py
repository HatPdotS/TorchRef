"""Triton forward + backward for the bond-angle Gaussian-NLL target."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_LOG_2PI = float(math.log(2.0 * math.pi))
_EPS = 1e-8


@triton.jit
def _angle_nll_fwd_kernel(
    xyz_ptr,
    idx_ptr,
    ref_ptr,
    sig_ptr,
    out_ptr,
    N: tl.constexpr,
    LOG_2PI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(idx_ptr + offs * 3 + 0, mask=mask, other=0)
    b = tl.load(idx_ptr + offs * 3 + 1, mask=mask, other=0)
    c = tl.load(idx_ptr + offs * 3 + 2, mask=mask, other=0)

    ax = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    ay = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    az = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    bx = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    by = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    bz = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    cx = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    cy = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    cz = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)

    v1x = ax - bx; v1y = ay - by; v1z = az - bz
    v2x = cx - bx; v2y = cy - by; v2z = cz - bz

    n1 = tl.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    n2 = tl.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)
    dot = v1x * v2x + v1y * v2y + v1z * v2z
    cos_t = dot / (n1 * n2)
    cos_t = tl.where(cos_t > 1.0, 1.0, cos_t)
    cos_t = tl.where(cos_t < -1.0, -1.0, cos_t)
    theta = libdevice.acos(cos_t)

    ref = tl.load(ref_ptr + offs, mask=mask, other=0.0)
    sig = tl.load(sig_ptr + offs, mask=mask, other=1.0)
    dev = theta - ref
    nll = 0.5 * (dev / sig) * (dev / sig) + tl.log(sig) + 0.5 * LOG_2PI

    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _angle_nll_bwd_kernel(
    xyz_ptr,
    idx_ptr,
    ref_ptr,
    sig_ptr,
    grad_out,
    dxyz_ptr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Analytic gradient of the angle NLL.

    With v1 = a - b, v2 = c - b, n1 = |v1|, n2 = |v2|,
    cos θ = (v1·v2)/(n1 n2), θ = acos(cos θ):

        ∂NLL/∂θ = (θ − ref) / σ²
        ∂θ/∂(cos θ) = −1 / sin θ
        ∂(cos θ)/∂v1 = v2/(n1 n2) − (cos θ / n1²) v1
        ∂(cos θ)/∂v2 = v1/(n1 n2) − (cos θ / n2²) v2

    Then ∂NLL/∂a = ∂NLL/∂v1, ∂NLL/∂c = ∂NLL/∂v2,
        ∂NLL/∂b = −(∂NLL/∂a + ∂NLL/∂c). Multiplied by grad_out.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(idx_ptr + offs * 3 + 0, mask=mask, other=0)
    b = tl.load(idx_ptr + offs * 3 + 1, mask=mask, other=0)
    c = tl.load(idx_ptr + offs * 3 + 2, mask=mask, other=0)

    ax = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    ay = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    az = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    bx = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    by = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    bz = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    cx = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    cy = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    cz = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)

    v1x = ax - bx; v1y = ay - by; v1z = az - bz
    v2x = cx - bx; v2y = cy - by; v2z = cz - bz
    n1_sq = v1x * v1x + v1y * v1y + v1z * v1z
    n2_sq = v2x * v2x + v2y * v2y + v2z * v2z
    n1 = tl.sqrt(n1_sq)
    n2 = tl.sqrt(n2_sq)
    dot = v1x * v2x + v1y * v2y + v1z * v2z
    cos_t = dot / (n1 * n2)
    cos_t = tl.where(cos_t > 1.0, 1.0, cos_t)
    cos_t = tl.where(cos_t < -1.0, -1.0, cos_t)
    sin_t = tl.sqrt(tl.maximum(1.0 - cos_t * cos_t, EPS))
    theta = libdevice.acos(cos_t)

    ref = tl.load(ref_ptr + offs, mask=mask, other=0.0)
    sig = tl.load(sig_ptr + offs, mask=mask, other=1.0)
    dev = theta - ref

    dnll_dtheta = dev / (sig * sig)
    coef = -dnll_dtheta / sin_t * grad_out

    inv_n1n2 = 1.0 / (n1 * n2)
    ct_over_n1sq = cos_t / n1_sq
    ct_over_n2sq = cos_t / n2_sq

    # ∂NLL/∂v1 = coef * (v2 * inv_n1n2 - v1 * ct_over_n1sq)
    dv1x = coef * (v2x * inv_n1n2 - v1x * ct_over_n1sq)
    dv1y = coef * (v2y * inv_n1n2 - v1y * ct_over_n1sq)
    dv1z = coef * (v2z * inv_n1n2 - v1z * ct_over_n1sq)
    dv2x = coef * (v1x * inv_n1n2 - v2x * ct_over_n2sq)
    dv2y = coef * (v1y * inv_n1n2 - v2y * ct_over_n2sq)
    dv2z = coef * (v1z * inv_n1n2 - v2z * ct_over_n2sq)

    # Scatter: a += dv1, c += dv2, b -= (dv1 + dv2)
    tl.atomic_add(dxyz_ptr + a * 3 + 0, dv1x, mask=mask)
    tl.atomic_add(dxyz_ptr + a * 3 + 1, dv1y, mask=mask)
    tl.atomic_add(dxyz_ptr + a * 3 + 2, dv1z, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 0, dv2x, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 1, dv2y, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 2, dv2z, mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 0, -(dv1x + dv2x), mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 1, -(dv1y + dv2y), mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 2, -(dv1z + dv2z), mask=mask)


class _AngleMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, idx, references_rad, sigmas_rad):
        assert xyz.is_cuda and xyz.dtype == torch.float32
        N = idx.shape[0]
        nll = torch.empty(N, dtype=xyz.dtype, device=xyz.device)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _angle_nll_fwd_kernel[grid](
            xyz, idx, references_rad, sigmas_rad, nll,
            N=N, LOG_2PI=_LOG_2PI, BLOCK=BLOCK,
        )
        ctx.save_for_backward(xyz, idx, references_rad, sigmas_rad)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        xyz, idx, refs, sigs = ctx.saved_tensors
        N = idx.shape[0]
        dxyz = torch.zeros_like(xyz)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _angle_nll_bwd_kernel[grid](
            xyz, idx, refs, sigs, float(grad_out.item()), dxyz,
            N=N, EPS=_EPS, BLOCK=BLOCK,
        )
        return dxyz, None, None, None


def angle_math_triton(xyz, idx, references_rad, sigmas_rad):
    """Triton-backed bond-angle Gaussian NLL with analytic backward.

    Drop-in replacement for :func:`torchref.base.targets.angle.angle_math`.
    """
    return _AngleMathTriton.apply(xyz, idx, references_rad, sigmas_rad)
