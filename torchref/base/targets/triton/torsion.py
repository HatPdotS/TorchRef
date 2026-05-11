"""Triton forward + analytic backward for the omega cis/trans torsion target."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..torsion import torsion_omega_math as _omega_eager
from ._dihedral import dihedral_and_grad


_LOG_2PI = float(math.log(2.0 * math.pi))
_DEG2RAD = float(math.pi / 180.0)


@triton.jit
def _omega_nll_fwd_kernel(
    xyz_ptr,
    idx_ptr,             # (N, 4)
    sig_deg_ptr,         # (N,)
    is_proline_ptr,      # (N,) bool/i1
    out_ptr,             # (N,)
    w_cis_proline: tl.constexpr,
    w_cis_general: tl.constexpr,
    N: tl.constexpr,
    LOG_2PI: tl.constexpr,
    DEG2RAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(idx_ptr + offs * 4 + 3, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    p2x = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    p2y = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    p2z = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    p3x = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    p3y = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    p3z = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    p4x = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    p4y = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    p4z = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)

    (omega, _F1x, _F1y, _F1z, _F2x, _F2y, _F2z,
     _F3x, _F3y, _F3z, _F4x, _F4y, _F4z) = dihedral_and_grad(
        p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z,
    )

    sig_deg = tl.load(sig_deg_ptr + offs, mask=mask, other=1.0)
    sig_rad = sig_deg * DEG2RAD
    kappa = 1.0 / (sig_rad * sig_rad)
    kappa = tl.minimum(tl.maximum(kappa, 1e-3), 1e4)
    # log(I_0(κ)) — Triton libdevice has cyl_bessel_i0 but not i0e. Use
    # the direct value for moderate κ and the higher-order asymptotic for
    # large κ to avoid overflow. The 1/(8κ) term keeps the asymptotic
    # within float32 noise of the exact log I_0 at the switch point.
    asym = (
        kappa
        - 0.5 * tl.log(2.0 * 3.141592653589793 * kappa)
        + 1.0 / (8.0 * kappa)
    )
    log_i0_kappa = tl.where(
        kappa < 30.0,
        tl.log(libdevice.cyl_bessel_i0(kappa)),
        asym,
    )
    log_norm = LOG_2PI + log_i0_kappa

    is_pro = tl.load(is_proline_ptr + offs, mask=mask, other=0).to(tl.int1)
    w_cis = tl.where(is_pro, w_cis_proline, w_cis_general)
    w_trans = 1.0 - w_cis

    cos_o = tl.cos(omega)
    log_p_trans = tl.log(w_trans) - kappa * cos_o
    log_p_cis   = tl.log(w_cis)   + kappa * cos_o
    m = tl.maximum(log_p_trans, log_p_cis)
    log_mixture = m + tl.log(tl.exp(log_p_trans - m) + tl.exp(log_p_cis - m))

    nll = log_norm - log_mixture
    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _omega_nll_bwd_kernel(
    xyz_ptr,
    idx_ptr,
    sig_deg_ptr,
    is_proline_ptr,
    grad_out,
    dxyz_ptr,
    w_cis_proline: tl.constexpr,
    w_cis_general: tl.constexpr,
    N: tl.constexpr,
    DEG2RAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(idx_ptr + offs * 4 + 3, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    p2x = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    p2y = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    p2z = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    p3x = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    p3y = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    p3z = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    p4x = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    p4y = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    p4z = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)

    (omega, F1x, F1y, F1z, F2x, F2y, F2z,
     F3x, F3y, F3z, F4x, F4y, F4z) = dihedral_and_grad(
        p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z,
    )

    sig_deg = tl.load(sig_deg_ptr + offs, mask=mask, other=1.0)
    sig_rad = sig_deg * DEG2RAD
    kappa = 1.0 / (sig_rad * sig_rad)
    kappa = tl.minimum(tl.maximum(kappa, 1e-3), 1e4)

    is_pro = tl.load(is_proline_ptr + offs, mask=mask, other=0).to(tl.int1)
    w_cis = tl.where(is_pro, w_cis_proline, w_cis_general)
    w_trans = 1.0 - w_cis

    cos_o = tl.cos(omega)
    sin_o = tl.sin(omega)
    log_p_trans = tl.log(w_trans) - kappa * cos_o
    log_p_cis   = tl.log(w_cis)   + kappa * cos_o
    m = tl.maximum(log_p_trans, log_p_cis)
    log_mix = m + tl.log(tl.exp(log_p_trans - m) + tl.exp(log_p_cis - m))
    s_t = tl.exp(log_p_trans - log_mix)
    s_c = tl.exp(log_p_cis   - log_mix)
    # dNLL/dω = κ sin(ω) (s_c − s_t)
    coef = grad_out * kappa * sin_o * (s_c - s_t)

    g1x = coef * F1x; g1y = coef * F1y; g1z = coef * F1z
    g2x = coef * F2x; g2y = coef * F2y; g2z = coef * F2z
    g3x = coef * F3x; g3y = coef * F3y; g3z = coef * F3z
    g4x = coef * F4x; g4y = coef * F4y; g4z = coef * F4z

    tl.atomic_add(dxyz_ptr + a * 3 + 0, g1x, mask=mask)
    tl.atomic_add(dxyz_ptr + a * 3 + 1, g1y, mask=mask)
    tl.atomic_add(dxyz_ptr + a * 3 + 2, g1z, mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 0, g2x, mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 1, g2y, mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 2, g2z, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 0, g3x, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 1, g3y, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 2, g3z, mask=mask)
    tl.atomic_add(dxyz_ptr + d * 3 + 0, g4x, mask=mask)
    tl.atomic_add(dxyz_ptr + d * 3 + 1, g4y, mask=mask)
    tl.atomic_add(dxyz_ptr + d * 3 + 2, g4z, mask=mask)


class _TorsionOmegaMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, idx, sigmas_deg, is_proline,
                w_cis_proline: float, w_cis_general: float):
        assert xyz.is_cuda and xyz.dtype == torch.float32
        N = idx.shape[0]
        # is_proline may be bool; the kernel loads as int1 via .to(tl.int1)
        is_pro_u8 = is_proline.to(torch.uint8).contiguous()
        nll = torch.empty(N, dtype=xyz.dtype, device=xyz.device)
        BLOCK = 128
        grid = (triton.cdiv(N, BLOCK),)
        _omega_nll_fwd_kernel[grid](
            xyz, idx, sigmas_deg, is_pro_u8, nll,
            w_cis_proline=float(w_cis_proline),
            w_cis_general=float(w_cis_general),
            N=N, LOG_2PI=_LOG_2PI, DEG2RAD=_DEG2RAD, BLOCK=BLOCK,
        )
        ctx.save_for_backward(xyz, idx, sigmas_deg, is_pro_u8)
        ctx.w_cis_proline = float(w_cis_proline)
        ctx.w_cis_general = float(w_cis_general)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        xyz, idx, sigs, is_pro_u8 = ctx.saved_tensors
        N = idx.shape[0]
        dxyz = torch.zeros_like(xyz)
        BLOCK = 128
        grid = (triton.cdiv(N, BLOCK),)
        _omega_nll_bwd_kernel[grid](
            xyz, idx, sigs, is_pro_u8, float(grad_out.item()), dxyz,
            w_cis_proline=ctx.w_cis_proline,
            w_cis_general=ctx.w_cis_general,
            N=N, DEG2RAD=_DEG2RAD, BLOCK=BLOCK,
        )
        return dxyz, None, None, None, None, None


def torsion_omega_math_triton(xyz, idx, sigmas_deg, is_proline,
                              w_cis_proline=0.05, w_cis_general=0.0005):
    """Triton-backed omega cis/trans mixture NLL with analytic backward."""
    return _TorsionOmegaMathTriton.apply(
        xyz, idx, sigmas_deg, is_proline,
        float(w_cis_proline), float(w_cis_general),
    )


# ---------------------------------------------------------------------------
# Unimodal torsion (intra-residue + disulfide): dihedral + periodic wrap +
# von Mises NLL.  The wrap handles arbitrary n-fold rotational symmetry by
# picking the equivalent angle with smallest absolute deviation.
# ---------------------------------------------------------------------------


_TWO_PI = float(2.0 * math.pi)
_PI = float(math.pi)


@triton.jit
def _torsion_uni_fwd_kernel(
    xyz_ptr,
    idx_ptr,             # (N, 4)
    ref_deg_ptr,         # (N,)
    sig_deg_ptr,         # (N,)
    period_ptr,          # (N,) int (clamped >= 1)
    out_ptr,             # (N,)
    N: tl.constexpr,
    MAX_PERIOD: tl.constexpr,
    LOG_2PI: tl.constexpr,
    DEG2RAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(idx_ptr + offs * 4 + 3, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    p2x = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    p2y = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    p2z = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    p3x = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    p3y = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    p3z = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    p4x = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    p4y = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    p4z = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)

    (omega, _F1x, _F1y, _F1z, _F2x, _F2y, _F2z,
     _F3x, _F3y, _F3z, _F4x, _F4y, _F4z) = dihedral_and_grad(
        p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z,
    )
    # omega is in radians (atan2 output)
    ref_rad = tl.load(ref_deg_ptr + offs, mask=mask, other=0.0) * DEG2RAD
    diff = omega - ref_rad

    period = tl.load(period_ptr + offs, mask=mask, other=1)
    period_f = period.to(tl.float32)
    step = (2.0 * 3.141592653589793) / period_f

    # Pick equivalent k * step + diff with smallest |wrap(.)| over k in [0, period-1].
    LARGE = 1e30
    best_dev = tl.zeros_like(omega)
    best_abs = tl.full(omega.shape, LARGE, tl.float32)
    for k in tl.static_range(MAX_PERIOD):
        valid = k < period
        cand = diff + k * step
        # wrap to [-π, π]
        cand = cand - (2.0 * 3.141592653589793) * libdevice.round(cand / (2.0 * 3.141592653589793))
        a_cand = tl.abs(cand)
        is_better = valid & (a_cand < best_abs)
        best_dev = tl.where(is_better, cand, best_dev)
        best_abs = tl.where(is_better, a_cand, best_abs)

    sig_deg = tl.load(sig_deg_ptr + offs, mask=mask, other=1.0)
    sig_rad = sig_deg * DEG2RAD
    kappa = 1.0 / (sig_rad * sig_rad)
    kappa = tl.minimum(tl.maximum(kappa, 1e-3), 1e4)
    asym = (
        kappa - 0.5 * tl.log(2.0 * 3.141592653589793 * kappa)
        + 1.0 / (8.0 * kappa)
    )
    log_i0_kappa = tl.where(
        kappa < 30.0,
        tl.log(libdevice.cyl_bessel_i0(kappa)),
        asym,
    )
    log_prob = kappa * tl.cos(best_dev) - log_i0_kappa - LOG_2PI
    nll = -log_prob
    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _torsion_uni_bwd_kernel(
    xyz_ptr,
    idx_ptr,
    ref_deg_ptr,
    sig_deg_ptr,
    period_ptr,
    grad_out,
    dxyz_ptr,
    N: tl.constexpr,
    MAX_PERIOD: tl.constexpr,
    DEG2RAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(idx_ptr + offs * 4 + 3, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    p2x = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    p2y = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    p2z = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    p3x = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    p3y = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    p3z = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    p4x = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    p4y = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    p4z = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)

    (omega, F1x, F1y, F1z, F2x, F2y, F2z,
     F3x, F3y, F3z, F4x, F4y, F4z) = dihedral_and_grad(
        p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p4x, p4y, p4z,
    )
    ref_rad = tl.load(ref_deg_ptr + offs, mask=mask, other=0.0) * DEG2RAD
    diff = omega - ref_rad
    period = tl.load(period_ptr + offs, mask=mask, other=1)
    period_f = period.to(tl.float32)
    step = (2.0 * 3.141592653589793) / period_f

    LARGE = 1e30
    best_dev = tl.zeros_like(omega)
    best_abs = tl.full(omega.shape, LARGE, tl.float32)
    for k in tl.static_range(MAX_PERIOD):
        valid = k < period
        cand = diff + k * step
        cand = cand - (2.0 * 3.141592653589793) * libdevice.round(cand / (2.0 * 3.141592653589793))
        a_cand = tl.abs(cand)
        is_better = valid & (a_cand < best_abs)
        best_dev = tl.where(is_better, cand, best_dev)
        best_abs = tl.where(is_better, a_cand, best_abs)

    sig_deg = tl.load(sig_deg_ptr + offs, mask=mask, other=1.0)
    sig_rad = sig_deg * DEG2RAD
    kappa = 1.0 / (sig_rad * sig_rad)
    kappa = tl.minimum(tl.maximum(kappa, 1e-3), 1e4)
    # dNLL/d(best_dev) = κ · sin(best_dev). d(best_dev)/d(diff) = 1 (the
    # selected branch passes through up to the [-π, π] wrap, which has
    # derivative 1 a.e.). d(diff)/d(ω) = 1, so dNLL/d(ω) = κ · sin(best_dev).
    coef = grad_out * kappa * tl.sin(best_dev)

    g1x = coef * F1x; g1y = coef * F1y; g1z = coef * F1z
    g2x = coef * F2x; g2y = coef * F2y; g2z = coef * F2z
    g3x = coef * F3x; g3y = coef * F3y; g3z = coef * F3z
    g4x = coef * F4x; g4y = coef * F4y; g4z = coef * F4z

    tl.atomic_add(dxyz_ptr + a * 3 + 0, g1x, mask=mask)
    tl.atomic_add(dxyz_ptr + a * 3 + 1, g1y, mask=mask)
    tl.atomic_add(dxyz_ptr + a * 3 + 2, g1z, mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 0, g2x, mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 1, g2y, mask=mask)
    tl.atomic_add(dxyz_ptr + b * 3 + 2, g2z, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 0, g3x, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 1, g3y, mask=mask)
    tl.atomic_add(dxyz_ptr + c * 3 + 2, g3z, mask=mask)
    tl.atomic_add(dxyz_ptr + d * 3 + 0, g4x, mask=mask)
    tl.atomic_add(dxyz_ptr + d * 3 + 1, g4y, mask=mask)
    tl.atomic_add(dxyz_ptr + d * 3 + 2, g4z, mask=mask)


class _TorsionUnimodalMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, idx, references_deg, sigmas_deg, periods):
        assert xyz.is_cuda and xyz.dtype == torch.float32
        N = idx.shape[0]
        # max_period is a constexpr (compile-time). 6 is the realistic
        # upper bound for protein restraints (n-fold ring symmetries).
        max_period = int(periods.max().item())
        max_period = max(max_period, 1)
        periods_i32 = periods.clamp(min=1).to(torch.int32).contiguous()
        nll = torch.empty(N, dtype=xyz.dtype, device=xyz.device)
        BLOCK = 128
        grid = (triton.cdiv(N, BLOCK),)
        _torsion_uni_fwd_kernel[grid](
            xyz, idx, references_deg, sigmas_deg, periods_i32, nll,
            N=N, MAX_PERIOD=max_period,
            LOG_2PI=_LOG_2PI, DEG2RAD=_DEG2RAD, BLOCK=BLOCK,
        )
        ctx.save_for_backward(xyz, idx, references_deg, sigmas_deg, periods_i32)
        ctx.max_period = max_period
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        xyz, idx, refs, sigs, periods_i32 = ctx.saved_tensors
        N = idx.shape[0]
        dxyz = torch.zeros_like(xyz)
        BLOCK = 128
        grid = (triton.cdiv(N, BLOCK),)
        _torsion_uni_bwd_kernel[grid](
            xyz, idx, refs, sigs, periods_i32, float(grad_out.item()), dxyz,
            N=N, MAX_PERIOD=ctx.max_period,
            DEG2RAD=_DEG2RAD, BLOCK=BLOCK,
        )
        return dxyz, None, None, None, None


def torsion_unimodal_full_math_triton(xyz, idx, references_deg, sigmas_deg, periods):
    """Triton-backed full unimodal torsion NLL (dihedral + periodic wrap +
    von Mises NLL + sum), with analytic backward through the dihedral
    formula. Inputs match ``Restraints.restraints['torsion']['all']``:

    Parameters
    ----------
    xyz : (N_atoms, 3) float32 CUDA
    idx : (N, 4) int64 atom indices
    references_deg : (N,) float32 — target angles in degrees
    sigmas_deg : (N,) float32 — sigmas in degrees
    periods : (N,) int — n-fold periodicity (≥ 1)
    """
    return _TorsionUnimodalMathTriton.apply(
        xyz, idx, references_deg, sigmas_deg, periods,
    )
