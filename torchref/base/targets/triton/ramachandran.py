"""Triton forward + analytic backward for the Ramachandran NLL target.

The forward fuses two dihedral computes (φ, ψ) and bilinear interpolation
on the (n_surfaces, 360, 360) NLL surface into one kernel. The backward
chains:

    d(NLL)/d(phi_frac)  = (1-ψf)(v10−v00) + ψf·(v11−v01)
    d(NLL)/d(psi_frac)  = (1-φf)(v01−v00) + φf·(v11−v10)
    d(phi_frac)/d(phi_deg) = 1
    d(phi_deg)/d(positions) = −(180/π) · F_dihedral

(the leading minus is because the eager target uses ``-torsions_from_xyz``;
F_dihedral comes from :mod:`_dihedral`.) Same for ψ.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._dihedral import dihedral_and_grad


_RAD2DEG = float(180.0 / math.pi)


@triton.jit
def _rama_nll_fwd_kernel(
    xyz_ptr,
    phi_idx_ptr,        # (N, 4)
    psi_idx_ptr,        # (N, 4)
    surfaces_ptr,       # (n_surface, 360, 360)
    surface_type_ptr,   # (N,) int32
    out_ptr,            # (N,)
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(phi_idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(phi_idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(phi_idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(phi_idx_ptr + offs * 4 + 3, mask=mask, other=0)
    pax = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    pay = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    paz = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    pbx = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    pby = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    pbz = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    pcx = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    pcy = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    pcz = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    pdx = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    pdy = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    pdz = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)
    (phi_rad, _F1x, _F1y, _F1z, _F2x, _F2y, _F2z,
     _F3x, _F3y, _F3z, _F4x, _F4y, _F4z) = dihedral_and_grad(
        pax, pay, paz, pbx, pby, pbz, pcx, pcy, pcz, pdx, pdy, pdz,
    )
    phi_deg = -phi_rad * (180.0 / 3.141592653589793)

    a = tl.load(psi_idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(psi_idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(psi_idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(psi_idx_ptr + offs * 4 + 3, mask=mask, other=0)
    pax = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    pay = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    paz = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    pbx = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    pby = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    pbz = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    pcx = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    pcy = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    pcz = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    pdx = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    pdy = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    pdz = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)
    (psi_rad, _F1x, _F1y, _F1z, _F2x, _F2y, _F2z,
     _F3x, _F3y, _F3z, _F4x, _F4y, _F4z) = dihedral_and_grad(
        pax, pay, paz, pbx, pby, pbz, pcx, pcy, pcz, pdx, pdy, pdz,
    )
    psi_deg = -psi_rad * (180.0 / 3.141592653589793)

    phi_g = (phi_deg + 180.0) % 360.0
    psi_g = (psi_deg + 180.0) % 360.0
    phi_floor = tl.math.floor(phi_g)
    psi_floor = tl.math.floor(psi_g)
    phi_lo = (phi_floor.to(tl.int32)) % 360
    psi_lo = (psi_floor.to(tl.int32)) % 360
    phi_hi = (phi_lo + 1) % 360
    psi_hi = (psi_lo + 1) % 360
    phi_frac = phi_g - phi_floor
    psi_frac = psi_g - psi_floor

    s = tl.load(surface_type_ptr + offs, mask=mask, other=0)
    base = s * (360 * 360)
    v00 = tl.load(surfaces_ptr + base + phi_lo * 360 + psi_lo, mask=mask, other=0.0)
    v01 = tl.load(surfaces_ptr + base + phi_lo * 360 + psi_hi, mask=mask, other=0.0)
    v10 = tl.load(surfaces_ptr + base + phi_hi * 360 + psi_lo, mask=mask, other=0.0)
    v11 = tl.load(surfaces_ptr + base + phi_hi * 360 + psi_hi, mask=mask, other=0.0)
    nll = ((1.0 - phi_frac) * (1.0 - psi_frac) * v00
           + (1.0 - phi_frac) * psi_frac * v01
           + phi_frac * (1.0 - psi_frac) * v10
           + phi_frac * psi_frac * v11)
    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _rama_nll_bwd_kernel(
    xyz_ptr,
    phi_idx_ptr,
    psi_idx_ptr,
    surfaces_ptr,
    surface_type_ptr,
    grad_out_ptr,  # 0-D tensor — loaded in-kernel (no host .item() sync)
    dxyz_ptr,
    N: tl.constexpr,
    RAD2DEG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    grad_out = tl.load(grad_out_ptr)

    # --- phi: dihedral + gradient ---
    a = tl.load(phi_idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(phi_idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(phi_idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(phi_idx_ptr + offs * 4 + 3, mask=mask, other=0)
    pax = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    pay = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    paz = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    pbx = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    pby = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    pbz = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    pcx = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    pcy = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    pcz = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    pdx = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    pdy = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    pdz = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)
    (phi_rad, PF1x, PF1y, PF1z, PF2x, PF2y, PF2z,
     PF3x, PF3y, PF3z, PF4x, PF4y, PF4z) = dihedral_and_grad(
        pax, pay, paz, pbx, pby, pbz, pcx, pcy, pcz, pdx, pdy, pdz,
    )
    phi_a = a; phi_b = b; phi_c = c; phi_d = d
    phi_deg = -phi_rad * RAD2DEG

    # --- psi ---
    a = tl.load(psi_idx_ptr + offs * 4 + 0, mask=mask, other=0)
    b = tl.load(psi_idx_ptr + offs * 4 + 1, mask=mask, other=0)
    c = tl.load(psi_idx_ptr + offs * 4 + 2, mask=mask, other=0)
    d = tl.load(psi_idx_ptr + offs * 4 + 3, mask=mask, other=0)
    pax = tl.load(xyz_ptr + a * 3 + 0, mask=mask, other=0.0)
    pay = tl.load(xyz_ptr + a * 3 + 1, mask=mask, other=0.0)
    paz = tl.load(xyz_ptr + a * 3 + 2, mask=mask, other=0.0)
    pbx = tl.load(xyz_ptr + b * 3 + 0, mask=mask, other=0.0)
    pby = tl.load(xyz_ptr + b * 3 + 1, mask=mask, other=0.0)
    pbz = tl.load(xyz_ptr + b * 3 + 2, mask=mask, other=0.0)
    pcx = tl.load(xyz_ptr + c * 3 + 0, mask=mask, other=0.0)
    pcy = tl.load(xyz_ptr + c * 3 + 1, mask=mask, other=0.0)
    pcz = tl.load(xyz_ptr + c * 3 + 2, mask=mask, other=0.0)
    pdx = tl.load(xyz_ptr + d * 3 + 0, mask=mask, other=0.0)
    pdy = tl.load(xyz_ptr + d * 3 + 1, mask=mask, other=0.0)
    pdz = tl.load(xyz_ptr + d * 3 + 2, mask=mask, other=0.0)
    (psi_rad, SF1x, SF1y, SF1z, SF2x, SF2y, SF2z,
     SF3x, SF3y, SF3z, SF4x, SF4y, SF4z) = dihedral_and_grad(
        pax, pay, paz, pbx, pby, pbz, pcx, pcy, pcz, pdx, pdy, pdz,
    )
    psi_a = a; psi_b = b; psi_c = c; psi_d = d
    psi_deg = -psi_rad * RAD2DEG

    # --- bilinear: gather corner values, compute fractional gradients ---
    phi_g = (phi_deg + 180.0) % 360.0
    psi_g = (psi_deg + 180.0) % 360.0
    phi_floor = tl.math.floor(phi_g)
    psi_floor = tl.math.floor(psi_g)
    phi_lo = (phi_floor.to(tl.int32)) % 360
    psi_lo = (psi_floor.to(tl.int32)) % 360
    phi_hi = (phi_lo + 1) % 360
    psi_hi = (psi_lo + 1) % 360
    phi_frac = phi_g - phi_floor
    psi_frac = psi_g - psi_floor

    s = tl.load(surface_type_ptr + offs, mask=mask, other=0)
    base = s * (360 * 360)
    v00 = tl.load(surfaces_ptr + base + phi_lo * 360 + psi_lo, mask=mask, other=0.0)
    v01 = tl.load(surfaces_ptr + base + phi_lo * 360 + psi_hi, mask=mask, other=0.0)
    v10 = tl.load(surfaces_ptr + base + phi_hi * 360 + psi_lo, mask=mask, other=0.0)
    v11 = tl.load(surfaces_ptr + base + phi_hi * 360 + psi_hi, mask=mask, other=0.0)

    dNLL_dphi_frac = (1.0 - psi_frac) * (v10 - v00) + psi_frac * (v11 - v01)
    dNLL_dpsi_frac = (1.0 - phi_frac) * (v01 - v00) + phi_frac * (v11 - v10)

    # phi_g = (phi_deg + 180) % 360 → dphi_g/dphi_deg = 1 a.e.
    # phi_frac = phi_g - floor(phi_g.detach()) → dphi_frac/dphi_g = 1
    # phi_deg = -phi_rad · RAD2DEG → dphi_deg/dphi_rad = -RAD2DEG
    # So dNLL/dphi_rad = -RAD2DEG · dNLL_dphi_frac
    coef_phi = grad_out * (-RAD2DEG) * dNLL_dphi_frac
    coef_psi = grad_out * (-RAD2DEG) * dNLL_dpsi_frac

    # Scatter phi forces
    tl.atomic_add(dxyz_ptr + phi_a * 3 + 0, coef_phi * PF1x, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_a * 3 + 1, coef_phi * PF1y, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_a * 3 + 2, coef_phi * PF1z, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_b * 3 + 0, coef_phi * PF2x, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_b * 3 + 1, coef_phi * PF2y, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_b * 3 + 2, coef_phi * PF2z, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_c * 3 + 0, coef_phi * PF3x, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_c * 3 + 1, coef_phi * PF3y, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_c * 3 + 2, coef_phi * PF3z, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_d * 3 + 0, coef_phi * PF4x, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_d * 3 + 1, coef_phi * PF4y, mask=mask)
    tl.atomic_add(dxyz_ptr + phi_d * 3 + 2, coef_phi * PF4z, mask=mask)
    # Scatter psi forces
    tl.atomic_add(dxyz_ptr + psi_a * 3 + 0, coef_psi * SF1x, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_a * 3 + 1, coef_psi * SF1y, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_a * 3 + 2, coef_psi * SF1z, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_b * 3 + 0, coef_psi * SF2x, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_b * 3 + 1, coef_psi * SF2y, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_b * 3 + 2, coef_psi * SF2z, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_c * 3 + 0, coef_psi * SF3x, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_c * 3 + 1, coef_psi * SF3y, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_c * 3 + 2, coef_psi * SF3z, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_d * 3 + 0, coef_psi * SF4x, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_d * 3 + 1, coef_psi * SF4y, mask=mask)
    tl.atomic_add(dxyz_ptr + psi_d * 3 + 2, coef_psi * SF4z, mask=mask)


class _RamachandranMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, phi_idx, psi_idx, nll_surfaces, surface_type):
        assert xyz.is_cuda and xyz.dtype == torch.float32
        s32 = surface_type.to(torch.int32).contiguous()
        surfaces_c = nll_surfaces.contiguous()
        N = phi_idx.shape[0]
        nll = torch.empty(N, dtype=xyz.dtype, device=xyz.device)
        BLOCK = 64
        grid = (triton.cdiv(N, BLOCK),)
        _rama_nll_fwd_kernel[grid](
            xyz, phi_idx, psi_idx, surfaces_c, s32, nll,
            N=N, BLOCK=BLOCK,
        )
        ctx.save_for_backward(xyz, phi_idx, psi_idx, surfaces_c, s32)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        xyz, phi_idx, psi_idx, surfaces, s32 = ctx.saved_tensors
        N = phi_idx.shape[0]
        dxyz = torch.zeros_like(xyz)
        BLOCK = 64
        grid = (triton.cdiv(N, BLOCK),)
        _rama_nll_bwd_kernel[grid](
            xyz, phi_idx, psi_idx, surfaces, s32,
            grad_out, dxyz,
            N=N, RAD2DEG=_RAD2DEG, BLOCK=BLOCK,
        )
        return dxyz, None, None, None, None


def ramachandran_math_triton(xyz, phi_idx, psi_idx, nll_surfaces, surface_type):
    """Triton-backed Ramachandran bilinear-interp NLL with analytic backward."""
    return _RamachandranMathTriton.apply(xyz, phi_idx, psi_idx, nll_surfaces, surface_type)
