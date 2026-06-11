"""Custom Triton kernels for direct-summation structure factors.

Two ``torch.autograd.Function``s (isotropic + anisotropic) computing the **P1**
structure factor ``F(h) = Σ_atoms c_j(h)·[cos φ + i·sin φ]`` with
``φ = 2π(h·r_frac)`` and ``c_j = occ_j·f_j(s)·DW_j``.

Design (per the plan):

- **Forward** grids over *blocks of reflections*. Each program keeps the
  ``Fr/Fi`` accumulators for its reflection block resident in registers and
  **loops over atoms**, adding each atom's contribution to all reflections in
  the block. The ``N_atom × N_hkl`` array is never materialized in DRAM.
- **Backward** grids over *atoms* (one atom per program). Each program holds
  its atom's gradient accumulators in registers, **loops over reflections**,
  and **recomputes** ``c_j``/``φ``/trig from scratch (no per-pair intermediate
  is saved). Each atom is owned by one program, so there are no atomics.
- ``save_for_backward`` keeps only the small inputs (O(R)+O(N)), never O(R·N).

float32 only (the dispatch routes float64 / non-CUDA elsewhere). Inputs are
passed as per-component 1-D contiguous columns so loads are coalesced.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# Triton @jit functions can only read globals declared as ``tl.constexpr``.
TWO_PI = tl.constexpr(2.0 * math.pi)
NEG_TWO_PI_SQ = tl.constexpr(-2.0 * (math.pi**2))

# sm_75 (T4)-safe block sizes; tunable.
BLOCK_H = 128  # reflections per forward program / backward reflection chunk


# ===========================================================================
# Isotropic
# ===========================================================================
@triton.jit
def _ds_iso_fwd_kernel(
    hx_ptr, hy_ptr, hz_ptr, s_ptr,
    rx_ptr, ry_ptr, rz_ptr, occ_ptr, adp_ptr, A_ptr, B_ptr,
    Fr_ptr, Fi_ptr,
    N, R,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = offs < R

    hx = tl.load(hx_ptr + offs, mask=valid, other=0.0)
    hy = tl.load(hy_ptr + offs, mask=valid, other=0.0)
    hz = tl.load(hz_ptr + offs, mask=valid, other=0.0)
    s = tl.load(s_ptr + offs, mask=valid, other=0.0)
    s2q = s * s * 0.25

    Fr = tl.zeros([BLOCK_H], dtype=tl.float32)
    Fi = tl.zeros([BLOCK_H], dtype=tl.float32)

    for j in range(0, N):
        rx = tl.load(rx_ptr + j)
        ry = tl.load(ry_ptr + j)
        rz = tl.load(rz_ptr + j)
        occ_j = tl.load(occ_ptr + j)
        adp_j = tl.load(adp_ptr + j)
        f = tl.zeros([BLOCK_H], dtype=tl.float32)
        for g in range(0, 5):
            Ag = tl.load(A_ptr + j * 5 + g)
            Bg = tl.load(B_ptr + j * 5 + g)
            f += Ag * tl.exp(-Bg * s2q)
        dw = tl.exp(-adp_j * s2q)
        c = occ_j * f * dw
        phi = TWO_PI * (hx * rx + hy * ry + hz * rz)
        Fr += c * tl.cos(phi)
        Fi += c * tl.sin(phi)

    tl.store(Fr_ptr + offs, Fr, mask=valid)
    tl.store(Fi_ptr + offs, Fi, mask=valid)


@triton.jit
def _ds_iso_bwd_kernel(
    hx_ptr, hy_ptr, hz_ptr, s_ptr,
    rx_ptr, ry_ptr, rz_ptr, occ_ptr, adp_ptr, A_ptr, B_ptr,
    gFr_ptr, gFi_ptr,
    dx_ptr, dy_ptr, dz_ptr, docc_ptr, dadp_ptr,
    N, R,
    BLOCK_H: tl.constexpr,
):
    j = tl.program_id(0)  # one atom per program

    rx = tl.load(rx_ptr + j)
    ry = tl.load(ry_ptr + j)
    rz = tl.load(rz_ptr + j)
    occ_j = tl.load(occ_ptr + j)
    adp_j = tl.load(adp_ptr + j)

    gx = 0.0
    gy = 0.0
    gz = 0.0
    gocc = 0.0
    gadp = 0.0

    for h0 in range(0, R, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        valid = offs < R
        hx = tl.load(hx_ptr + offs, mask=valid, other=0.0)
        hy = tl.load(hy_ptr + offs, mask=valid, other=0.0)
        hz = tl.load(hz_ptr + offs, mask=valid, other=0.0)
        s = tl.load(s_ptr + offs, mask=valid, other=0.0)
        # invalid lanes carry gFr=gFi=0 -> zero contribution, no extra masking
        gFr = tl.load(gFr_ptr + offs, mask=valid, other=0.0)
        gFi = tl.load(gFi_ptr + offs, mask=valid, other=0.0)
        s2q = s * s * 0.25

        f = tl.zeros([BLOCK_H], dtype=tl.float32)
        for g in range(0, 5):
            Ag = tl.load(A_ptr + j * 5 + g)
            Bg = tl.load(B_ptr + j * 5 + g)
            f += Ag * tl.exp(-Bg * s2q)
        dw = tl.exp(-adp_j * s2q)
        gj = f * dw          # ∂F/∂occ weight
        c = occ_j * gj
        phi = TWO_PI * (hx * rx + hy * ry + hz * rz)
        cosp = tl.cos(phi)
        sinp = tl.sin(phi)
        base = gFr * cosp + gFi * sinp           # for occ / adp
        w = (-gFr * sinp + gFi * cosp) * c       # for xyz (× 2π·h)

        gx += tl.sum(TWO_PI * hx * w, axis=0)
        gy += tl.sum(TWO_PI * hy * w, axis=0)
        gz += tl.sum(TWO_PI * hz * w, axis=0)
        gocc += tl.sum(gj * base, axis=0)
        gadp += tl.sum(c * (-s2q) * base, axis=0)

    tl.store(dx_ptr + j, gx)
    tl.store(dy_ptr + j, gy)
    tl.store(dz_ptr + j, gz)
    tl.store(docc_ptr + j, gocc)
    tl.store(dadp_ptr + j, gadp)


# ===========================================================================
# Anisotropic
# ===========================================================================
@triton.jit
def _ds_aniso_fwd_kernel(
    hx_ptr, hy_ptr, hz_ptr, sx_ptr, sy_ptr, sz_ptr,
    rx_ptr, ry_ptr, rz_ptr, occ_ptr, U_ptr, A_ptr, B_ptr,
    Fr_ptr, Fi_ptr,
    N, R,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = offs < R

    hx = tl.load(hx_ptr + offs, mask=valid, other=0.0)
    hy = tl.load(hy_ptr + offs, mask=valid, other=0.0)
    hz = tl.load(hz_ptr + offs, mask=valid, other=0.0)
    sx = tl.load(sx_ptr + offs, mask=valid, other=0.0)
    sy = tl.load(sy_ptr + offs, mask=valid, other=0.0)
    sz = tl.load(sz_ptr + offs, mask=valid, other=0.0)
    s2q = (sx * sx + sy * sy + sz * sz) * 0.25

    Fr = tl.zeros([BLOCK_H], dtype=tl.float32)
    Fi = tl.zeros([BLOCK_H], dtype=tl.float32)

    for j in range(0, N):
        rx = tl.load(rx_ptr + j)
        ry = tl.load(ry_ptr + j)
        rz = tl.load(rz_ptr + j)
        occ_j = tl.load(occ_ptr + j)
        u0 = tl.load(U_ptr + j * 6 + 0)
        u1 = tl.load(U_ptr + j * 6 + 1)
        u2 = tl.load(U_ptr + j * 6 + 2)
        u3 = tl.load(U_ptr + j * 6 + 3)
        u4 = tl.load(U_ptr + j * 6 + 4)
        u5 = tl.load(U_ptr + j * 6 + 5)
        f = tl.zeros([BLOCK_H], dtype=tl.float32)
        for g in range(0, 5):
            Ag = tl.load(A_ptr + j * 5 + g)
            Bg = tl.load(B_ptr + j * 5 + g)
            f += Ag * tl.exp(-Bg * s2q)
        sUs = (
            u0 * sx * sx + u1 * sy * sy + u2 * sz * sz
            + 2.0 * (u3 * sx * sy + u4 * sx * sz + u5 * sy * sz)
        )
        dw = tl.exp(NEG_TWO_PI_SQ * sUs)
        c = occ_j * f * dw
        phi = TWO_PI * (hx * rx + hy * ry + hz * rz)
        Fr += c * tl.cos(phi)
        Fi += c * tl.sin(phi)

    tl.store(Fr_ptr + offs, Fr, mask=valid)
    tl.store(Fi_ptr + offs, Fi, mask=valid)


@triton.jit
def _ds_aniso_bwd_kernel(
    hx_ptr, hy_ptr, hz_ptr, sx_ptr, sy_ptr, sz_ptr,
    rx_ptr, ry_ptr, rz_ptr, occ_ptr, U_ptr, A_ptr, B_ptr,
    gFr_ptr, gFi_ptr,
    dx_ptr, dy_ptr, dz_ptr, docc_ptr, dU_ptr,
    N, R,
    BLOCK_H: tl.constexpr,
):
    j = tl.program_id(0)

    rx = tl.load(rx_ptr + j)
    ry = tl.load(ry_ptr + j)
    rz = tl.load(rz_ptr + j)
    occ_j = tl.load(occ_ptr + j)
    u0 = tl.load(U_ptr + j * 6 + 0)
    u1 = tl.load(U_ptr + j * 6 + 1)
    u2 = tl.load(U_ptr + j * 6 + 2)
    u3 = tl.load(U_ptr + j * 6 + 3)
    u4 = tl.load(U_ptr + j * 6 + 4)
    u5 = tl.load(U_ptr + j * 6 + 5)

    gx = 0.0
    gy = 0.0
    gz = 0.0
    gocc = 0.0
    gu0 = 0.0
    gu1 = 0.0
    gu2 = 0.0
    gu3 = 0.0
    gu4 = 0.0
    gu5 = 0.0

    for h0 in range(0, R, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        valid = offs < R
        hx = tl.load(hx_ptr + offs, mask=valid, other=0.0)
        hy = tl.load(hy_ptr + offs, mask=valid, other=0.0)
        hz = tl.load(hz_ptr + offs, mask=valid, other=0.0)
        sx = tl.load(sx_ptr + offs, mask=valid, other=0.0)
        sy = tl.load(sy_ptr + offs, mask=valid, other=0.0)
        sz = tl.load(sz_ptr + offs, mask=valid, other=0.0)
        gFr = tl.load(gFr_ptr + offs, mask=valid, other=0.0)
        gFi = tl.load(gFi_ptr + offs, mask=valid, other=0.0)
        s2q = (sx * sx + sy * sy + sz * sz) * 0.25

        f = tl.zeros([BLOCK_H], dtype=tl.float32)
        for g in range(0, 5):
            Ag = tl.load(A_ptr + j * 5 + g)
            Bg = tl.load(B_ptr + j * 5 + g)
            f += Ag * tl.exp(-Bg * s2q)
        sUs = (
            u0 * sx * sx + u1 * sy * sy + u2 * sz * sz
            + 2.0 * (u3 * sx * sy + u4 * sx * sz + u5 * sy * sz)
        )
        dw = tl.exp(NEG_TWO_PI_SQ * sUs)
        gj = f * dw
        c = occ_j * gj
        phi = TWO_PI * (hx * rx + hy * ry + hz * rz)
        cosp = tl.cos(phi)
        sinp = tl.sin(phi)
        base = gFr * cosp + gFi * sinp
        w = (-gFr * sinp + gFi * cosp) * c

        gx += tl.sum(TWO_PI * hx * w, axis=0)
        gy += tl.sum(TWO_PI * hy * w, axis=0)
        gz += tl.sum(TWO_PI * hz * w, axis=0)
        gocc += tl.sum(gj * base, axis=0)
        # dU_k = Σ c·(−2π²)·m_k·base ; m=[sx²,sy²,sz²,2sxsy,2sxsz,2sysz]
        cb = c * NEG_TWO_PI_SQ * base
        gu0 += tl.sum(cb * (sx * sx), axis=0)
        gu1 += tl.sum(cb * (sy * sy), axis=0)
        gu2 += tl.sum(cb * (sz * sz), axis=0)
        gu3 += tl.sum(cb * (2.0 * sx * sy), axis=0)
        gu4 += tl.sum(cb * (2.0 * sx * sz), axis=0)
        gu5 += tl.sum(cb * (2.0 * sy * sz), axis=0)

    tl.store(dx_ptr + j, gx)
    tl.store(dy_ptr + j, gy)
    tl.store(dz_ptr + j, gz)
    tl.store(docc_ptr + j, gocc)
    tl.store(dU_ptr + j * 6 + 0, gu0)
    tl.store(dU_ptr + j * 6 + 1, gu1)
    tl.store(dU_ptr + j * 6 + 2, gu2)
    tl.store(dU_ptr + j * 6 + 3, gu3)
    tl.store(dU_ptr + j * 6 + 4, gu4)
    tl.store(dU_ptr + j * 6 + 5, gu5)


# ===========================================================================
# autograd.Function wrappers
# ===========================================================================
def _cols_f32(t):
    """Return contiguous float32 per-component 1-D columns of an (M, k) tensor."""
    t = t.to(torch.float32)
    return [t[:, i].contiguous() for i in range(t.shape[1])]


class _DSIsoTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hkl, s, xyz, occ, adp, A, B):
        assert hkl.is_cuda, "Triton DS kernels require CUDA tensors"
        hx, hy, hz = _cols_f32(hkl)
        rx, ry, rz = _cols_f32(xyz)
        s_ = s.to(torch.float32).contiguous()
        occ_ = occ.to(torch.float32).contiguous()
        adp_ = adp.to(torch.float32).contiguous()
        A_ = A.to(torch.float32).contiguous()
        B_ = B.to(torch.float32).contiguous()
        R = hkl.shape[0]
        N = xyz.shape[0]
        dev = xyz.device
        Fr = torch.empty(R, dtype=torch.float32, device=dev)
        Fi = torch.empty(R, dtype=torch.float32, device=dev)
        grid = (triton.cdiv(R, BLOCK_H),)
        _ds_iso_fwd_kernel[grid](
            hx, hy, hz, s_, rx, ry, rz, occ_, adp_, A_, B_,
            Fr, Fi, N, R, BLOCK_H=BLOCK_H,
        )
        ctx.save_for_backward(hx, hy, hz, s_, rx, ry, rz, occ_, adp_, A_, B_)
        ctx.N = N
        ctx.R = R
        ctx.param_dtype = xyz.dtype
        return torch.complex(Fr, Fi)

    @staticmethod
    def backward(ctx, grad_F):
        hx, hy, hz, s_, rx, ry, rz, occ_, adp_, A_, B_ = ctx.saved_tensors
        N, R = ctx.N, ctx.R
        dev = rx.device
        gFr = grad_F.real.to(torch.float32).contiguous()
        gFi = grad_F.imag.to(torch.float32).contiguous()
        dx = torch.zeros(N, dtype=torch.float32, device=dev)
        dy = torch.zeros(N, dtype=torch.float32, device=dev)
        dz = torch.zeros(N, dtype=torch.float32, device=dev)
        docc = torch.zeros(N, dtype=torch.float32, device=dev)
        dadp = torch.zeros(N, dtype=torch.float32, device=dev)
        grid = (N,)
        _ds_iso_bwd_kernel[grid](
            hx, hy, hz, s_, rx, ry, rz, occ_, adp_, A_, B_,
            gFr, gFi, dx, dy, dz, docc, dadp, N, R, BLOCK_H=BLOCK_H,
        )
        dxyz = torch.stack([dx, dy, dz], dim=1).to(ctx.param_dtype)
        # order: hkl, s, xyz, occ, adp, A, B
        return (None, None, dxyz, docc.to(ctx.param_dtype),
                dadp.to(ctx.param_dtype), None, None)


class _DSAnisoTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hkl, s_vec, xyz, occ, U, A, B):
        assert hkl.is_cuda, "Triton DS kernels require CUDA tensors"
        hx, hy, hz = _cols_f32(hkl)
        sx, sy, sz = _cols_f32(s_vec)
        rx, ry, rz = _cols_f32(xyz)
        occ_ = occ.to(torch.float32).contiguous()
        U_ = U.to(torch.float32).contiguous()
        A_ = A.to(torch.float32).contiguous()
        B_ = B.to(torch.float32).contiguous()
        R = hkl.shape[0]
        N = xyz.shape[0]
        dev = xyz.device
        Fr = torch.empty(R, dtype=torch.float32, device=dev)
        Fi = torch.empty(R, dtype=torch.float32, device=dev)
        grid = (triton.cdiv(R, BLOCK_H),)
        _ds_aniso_fwd_kernel[grid](
            hx, hy, hz, sx, sy, sz, rx, ry, rz, occ_, U_, A_, B_,
            Fr, Fi, N, R, BLOCK_H=BLOCK_H,
        )
        ctx.save_for_backward(hx, hy, hz, sx, sy, sz, rx, ry, rz, occ_, U_, A_, B_)
        ctx.N = N
        ctx.R = R
        ctx.param_dtype = xyz.dtype
        return torch.complex(Fr, Fi)

    @staticmethod
    def backward(ctx, grad_F):
        (hx, hy, hz, sx, sy, sz, rx, ry, rz, occ_, U_, A_, B_) = ctx.saved_tensors
        N, R = ctx.N, ctx.R
        dev = rx.device
        gFr = grad_F.real.to(torch.float32).contiguous()
        gFi = grad_F.imag.to(torch.float32).contiguous()
        dx = torch.zeros(N, dtype=torch.float32, device=dev)
        dy = torch.zeros(N, dtype=torch.float32, device=dev)
        dz = torch.zeros(N, dtype=torch.float32, device=dev)
        docc = torch.zeros(N, dtype=torch.float32, device=dev)
        dU = torch.zeros(N, 6, dtype=torch.float32, device=dev)
        grid = (N,)
        _ds_aniso_bwd_kernel[grid](
            hx, hy, hz, sx, sy, sz, rx, ry, rz, occ_, U_, A_, B_,
            gFr, gFi, dx, dy, dz, docc, dU, N, R, BLOCK_H=BLOCK_H,
        )
        dxyz = torch.stack([dx, dy, dz], dim=1).to(ctx.param_dtype)
        # order: hkl, s_vec, xyz, occ, U, A, B
        return (None, None, dxyz, docc.to(ctx.param_dtype),
                dU.to(ctx.param_dtype), None, None)


def ds_iso_triton(hkl, s, xyz_frac, occ, adp, A, B):
    """Isotropic P1 structure factors via the custom Triton kernels."""
    return _DSIsoTriton.apply(hkl, s, xyz_frac, occ, adp, A, B)


def ds_aniso_triton(hkl, s_vec, xyz_frac, occ, U, A, B):
    """Anisotropic P1 structure factors via the custom Triton kernels."""
    return _DSAnisoTriton.apply(hkl, s_vec, xyz_frac, occ, U, A, B)
