"""Triton forward + analytic backward for riding-hydrogen placement.

One launch each way, where the eager helper (``_place_h_jit`` in
:mod:`torchref.restraints.hydrogen_topology`) fuses only the forward and leaves its
backward to run op-by-op through autograd -- ~100 launches at 3k hydrogens, which
dominates the non-bonded backward. The math mirrors ``_place_h_jit`` exactly:

    pp        = xyz[parent_idx]
    nb_pos[i] = xyz[nb_idx[i]]
    v[i]      = (nb_pos[i] − pp) · valid[i]
    s         = Σ_i v[i]
    base      = −s / (|s| + ε)

    cross12        = v1 × v2
    cross_norm     = |cross12|
    perp1_cross    = cross12 / (cross_norm + ε)
    cardinal       = ê_argmin|base|     (detached — no grad)
    perp1_ortho    = (base × cardinal) / |base × cardinal|
    perp1          = perp1_cross  if cross_norm > 1e-6  else  perp1_ortho
    perp2          = base × perp1

    direction = c0·base + c1·perp1 + c2·perp2
    H         = pp + L · direction
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_EPS = 1e-8


@triton.jit
def _placeh_fwd_kernel(
    xyz_ptr,            # (N_heavy, 3)
    parent_ptr,         # (N_h,) int
    nb_idx_ptr,         # (N_h, 4) int (clamped to >=0)
    nb_valid_ptr,       # (N_h, 4) float (1.0 / 0.0)
    coeff_ptr,          # (N_h, 3) float
    blen_ptr,           # (N_h,) float
    out_ptr,            # (N_h, 3) float
    N_H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N_H

    pi = tl.load(parent_ptr + offs, mask=m, other=0)
    ppx = tl.load(xyz_ptr + pi * 3 + 0, mask=m, other=0.0)
    ppy = tl.load(xyz_ptr + pi * 3 + 1, mask=m, other=0.0)
    ppz = tl.load(xyz_ptr + pi * 3 + 2, mask=m, other=0.0)

    # Neighbour 0..3 — accumulate s and stash v1 and v2 for the cross product.
    nb0 = tl.load(nb_idx_ptr + offs * 4 + 0, mask=m, other=0)
    nb1 = tl.load(nb_idx_ptr + offs * 4 + 1, mask=m, other=0)
    nb2 = tl.load(nb_idx_ptr + offs * 4 + 2, mask=m, other=0)
    nb3 = tl.load(nb_idx_ptr + offs * 4 + 3, mask=m, other=0)
    vd0 = tl.load(nb_valid_ptr + offs * 4 + 0, mask=m, other=0.0)
    vd1 = tl.load(nb_valid_ptr + offs * 4 + 1, mask=m, other=0.0)
    vd2 = tl.load(nb_valid_ptr + offs * 4 + 2, mask=m, other=0.0)
    vd3 = tl.load(nb_valid_ptr + offs * 4 + 3, mask=m, other=0.0)

    v0x = (tl.load(xyz_ptr + nb0 * 3 + 0, mask=m, other=0.0) - ppx) * vd0
    v0y = (tl.load(xyz_ptr + nb0 * 3 + 1, mask=m, other=0.0) - ppy) * vd0
    v0z = (tl.load(xyz_ptr + nb0 * 3 + 2, mask=m, other=0.0) - ppz) * vd0
    v1x = (tl.load(xyz_ptr + nb1 * 3 + 0, mask=m, other=0.0) - ppx) * vd1
    v1y = (tl.load(xyz_ptr + nb1 * 3 + 1, mask=m, other=0.0) - ppy) * vd1
    v1z = (tl.load(xyz_ptr + nb1 * 3 + 2, mask=m, other=0.0) - ppz) * vd1
    v2x = (tl.load(xyz_ptr + nb2 * 3 + 0, mask=m, other=0.0) - ppx) * vd2
    v2y = (tl.load(xyz_ptr + nb2 * 3 + 1, mask=m, other=0.0) - ppy) * vd2
    v2z = (tl.load(xyz_ptr + nb2 * 3 + 2, mask=m, other=0.0) - ppz) * vd2
    v3x = (tl.load(xyz_ptr + nb3 * 3 + 0, mask=m, other=0.0) - ppx) * vd3
    v3y = (tl.load(xyz_ptr + nb3 * 3 + 1, mask=m, other=0.0) - ppy) * vd3
    v3z = (tl.load(xyz_ptr + nb3 * 3 + 2, mask=m, other=0.0) - ppz) * vd3

    sx = v0x + v1x + v2x + v3x
    sy = v0y + v1y + v2y + v3y
    sz = v0z + v1z + v2z + v3z
    s_norm = tl.sqrt(sx * sx + sy * sy + sz * sz)
    bx = -sx / (s_norm + EPS)
    by = -sy / (s_norm + EPS)
    bz = -sz / (s_norm + EPS)

    # cross12 = v0 × v1 (the eager helper uses indices [:,0,:] and [:,1,:])
    cx = v0y * v1z - v0z * v1y
    cy = v0z * v1x - v0x * v1z
    cz = v0x * v1y - v0y * v1x
    c_norm = tl.sqrt(cx * cx + cy * cy + cz * cz)
    has_cross = c_norm > 1e-6
    p1cx = cx / (c_norm + EPS)
    p1cy = cy / (c_norm + EPS)
    p1cz = cz / (c_norm + EPS)

    # cardinal axis: one-hot at argmin |base|. Detached in eager (argmin
    # has no usable gradient anyway), so we compute it with comparisons
    # instead of a scatter.
    abx = tl.abs(bx); aby = tl.abs(by); abz = tl.abs(bz)
    sel_x = (abx <= aby) & (abx <= abz)
    sel_y = (~sel_x) & (aby <= abz)
    # else z
    car_x = tl.where(sel_x, 1.0, 0.0)
    car_y = tl.where(sel_y, 1.0, 0.0)
    car_z = tl.where(~(sel_x | sel_y), 1.0, 0.0)

    # perp1_ortho = normalize(base × cardinal)
    rox = by * car_z - bz * car_y
    roy = bz * car_x - bx * car_z
    roz = bx * car_y - by * car_x
    r_norm = tl.sqrt(rox * rox + roy * roy + roz * roz)
    p1ox = rox / (r_norm + EPS)
    p1oy = roy / (r_norm + EPS)
    p1oz = roz / (r_norm + EPS)

    p1x_ = tl.where(has_cross, p1cx, p1ox)
    p1y_ = tl.where(has_cross, p1cy, p1oy)
    p1z_ = tl.where(has_cross, p1cz, p1oz)

    # perp2 = base × perp1
    p2x_ = by * p1z_ - bz * p1y_
    p2y_ = bz * p1x_ - bx * p1z_
    p2z_ = bx * p1y_ - by * p1x_

    c0 = tl.load(coeff_ptr + offs * 3 + 0, mask=m, other=0.0)
    c1 = tl.load(coeff_ptr + offs * 3 + 1, mask=m, other=0.0)
    c2 = tl.load(coeff_ptr + offs * 3 + 2, mask=m, other=0.0)
    L  = tl.load(blen_ptr + offs, mask=m, other=0.0)

    dxv = c0 * bx + c1 * p1x_ + c2 * p2x_
    dyv = c0 * by + c1 * p1y_ + c2 * p2y_
    dzv = c0 * bz + c1 * p1z_ + c2 * p2z_

    hx = ppx + L * dxv
    hy = ppy + L * dyv
    hz = ppz + L * dzv

    tl.store(out_ptr + offs * 3 + 0, hx, mask=m)
    tl.store(out_ptr + offs * 3 + 1, hy, mask=m)
    tl.store(out_ptr + offs * 3 + 2, hz, mask=m)


@triton.jit
def _placeh_bwd_kernel(
    xyz_ptr,
    parent_ptr,
    nb_idx_ptr,
    nb_valid_ptr,
    coeff_ptr,
    blen_ptr,
    grad_h_ptr,         # (N_h, 3) — d(loss)/d(xyz_h)
    dxyz_ptr,           # (N_heavy, 3) — accumulator
    N_H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N_H

    pi = tl.load(parent_ptr + offs, mask=m, other=0)
    ppx = tl.load(xyz_ptr + pi * 3 + 0, mask=m, other=0.0)
    ppy = tl.load(xyz_ptr + pi * 3 + 1, mask=m, other=0.0)
    ppz = tl.load(xyz_ptr + pi * 3 + 2, mask=m, other=0.0)

    nb0 = tl.load(nb_idx_ptr + offs * 4 + 0, mask=m, other=0)
    nb1 = tl.load(nb_idx_ptr + offs * 4 + 1, mask=m, other=0)
    nb2 = tl.load(nb_idx_ptr + offs * 4 + 2, mask=m, other=0)
    nb3 = tl.load(nb_idx_ptr + offs * 4 + 3, mask=m, other=0)
    vd0 = tl.load(nb_valid_ptr + offs * 4 + 0, mask=m, other=0.0)
    vd1 = tl.load(nb_valid_ptr + offs * 4 + 1, mask=m, other=0.0)
    vd2 = tl.load(nb_valid_ptr + offs * 4 + 2, mask=m, other=0.0)
    vd3 = tl.load(nb_valid_ptr + offs * 4 + 3, mask=m, other=0.0)

    v0x = (tl.load(xyz_ptr + nb0 * 3 + 0, mask=m, other=0.0) - ppx) * vd0
    v0y = (tl.load(xyz_ptr + nb0 * 3 + 1, mask=m, other=0.0) - ppy) * vd0
    v0z = (tl.load(xyz_ptr + nb0 * 3 + 2, mask=m, other=0.0) - ppz) * vd0
    v1x = (tl.load(xyz_ptr + nb1 * 3 + 0, mask=m, other=0.0) - ppx) * vd1
    v1y = (tl.load(xyz_ptr + nb1 * 3 + 1, mask=m, other=0.0) - ppy) * vd1
    v1z = (tl.load(xyz_ptr + nb1 * 3 + 2, mask=m, other=0.0) - ppz) * vd1
    v2x = (tl.load(xyz_ptr + nb2 * 3 + 0, mask=m, other=0.0) - ppx) * vd2
    v2y = (tl.load(xyz_ptr + nb2 * 3 + 1, mask=m, other=0.0) - ppy) * vd2
    v2z = (tl.load(xyz_ptr + nb2 * 3 + 2, mask=m, other=0.0) - ppz) * vd2
    v3x = (tl.load(xyz_ptr + nb3 * 3 + 0, mask=m, other=0.0) - ppx) * vd3
    v3y = (tl.load(xyz_ptr + nb3 * 3 + 1, mask=m, other=0.0) - ppy) * vd3
    v3z = (tl.load(xyz_ptr + nb3 * 3 + 2, mask=m, other=0.0) - ppz) * vd3

    sx = v0x + v1x + v2x + v3x
    sy = v0y + v1y + v2y + v3y
    sz = v0z + v1z + v2z + v3z
    s_norm = tl.sqrt(sx * sx + sy * sy + sz * sz)
    inv_sn = 1.0 / (s_norm + EPS)
    bx = -sx * inv_sn
    by = -sy * inv_sn
    bz = -sz * inv_sn

    cx = v0y * v1z - v0z * v1y
    cy = v0z * v1x - v0x * v1z
    cz = v0x * v1y - v0y * v1x
    c_norm = tl.sqrt(cx * cx + cy * cy + cz * cz)
    inv_cn = 1.0 / (c_norm + EPS)
    has_cross = c_norm > 1e-6
    p1cx = cx * inv_cn
    p1cy = cy * inv_cn
    p1cz = cz * inv_cn

    abx = tl.abs(bx); aby = tl.abs(by); abz = tl.abs(bz)
    sel_x = (abx <= aby) & (abx <= abz)
    sel_y = (~sel_x) & (aby <= abz)
    car_x = tl.where(sel_x, 1.0, 0.0)
    car_y = tl.where(sel_y, 1.0, 0.0)
    car_z = tl.where(~(sel_x | sel_y), 1.0, 0.0)
    rox = by * car_z - bz * car_y
    roy = bz * car_x - bx * car_z
    roz = bx * car_y - by * car_x
    r_norm = tl.sqrt(rox * rox + roy * roy + roz * roz)
    inv_rn = 1.0 / (r_norm + EPS)
    p1ox = rox * inv_rn
    p1oy = roy * inv_rn
    p1oz = roz * inv_rn

    p1x_ = tl.where(has_cross, p1cx, p1ox)
    p1y_ = tl.where(has_cross, p1cy, p1oy)
    p1z_ = tl.where(has_cross, p1cz, p1oz)

    p2x_ = by * p1z_ - bz * p1y_
    p2y_ = bz * p1x_ - bx * p1z_
    p2z_ = bx * p1y_ - by * p1x_

    c0 = tl.load(coeff_ptr + offs * 3 + 0, mask=m, other=0.0)
    c1 = tl.load(coeff_ptr + offs * 3 + 1, mask=m, other=0.0)
    c2 = tl.load(coeff_ptr + offs * 3 + 2, mask=m, other=0.0)
    L  = tl.load(blen_ptr + offs, mask=m, other=0.0)

    ghx = tl.load(grad_h_ptr + offs * 3 + 0, mask=m, other=0.0)
    ghy = tl.load(grad_h_ptr + offs * 3 + 1, mask=m, other=0.0)
    ghz = tl.load(grad_h_ptr + offs * 3 + 2, mask=m, other=0.0)

    # G_pp_direct = ghx,ghy,ghz   (from H = pp + ...)
    # G_dir = L * gh              (from H = pp + L*dir)
    Gdx = L * ghx; Gdy = L * ghy; Gdz = L * ghz

    # dir = c0*base + c1*perp1 + c2*perp2
    Gb_x = c0 * Gdx; Gb_y = c0 * Gdy; Gb_z = c0 * Gdz
    Gp1x = c1 * Gdx; Gp1y = c1 * Gdy; Gp1z = c1 * Gdz
    Gp2x = c2 * Gdx; Gp2y = c2 * Gdy; Gp2z = c2 * Gdz

    # perp2 = base × perp1  →  G_base += perp1 × G_perp2 ; G_perp1 += G_perp2 × base
    Gb_x += p1y_ * Gp2z - p1z_ * Gp2y
    Gb_y += p1z_ * Gp2x - p1x_ * Gp2z
    Gb_z += p1x_ * Gp2y - p1y_ * Gp2x
    Gp1x += Gp2y * bz - Gp2z * by
    Gp1y += Gp2z * bx - Gp2x * bz
    Gp1z += Gp2x * by - Gp2y * bx

    # Split G_perp1 by the where branch.
    Gp1cx = tl.where(has_cross, Gp1x, 0.0)
    Gp1cy = tl.where(has_cross, Gp1y, 0.0)
    Gp1cz = tl.where(has_cross, Gp1z, 0.0)
    Gp1ox = tl.where(has_cross, 0.0, Gp1x)
    Gp1oy = tl.where(has_cross, 0.0, Gp1y)
    Gp1oz = tl.where(has_cross, 0.0, Gp1z)

    # perp1_cross = cross12 / (|cross12| + eps).
    # G_cross12 = (G − (G·perp1_cross) perp1_cross) / (|cross12| + eps)
    dot_pc = Gp1cx * p1cx + Gp1cy * p1cy + Gp1cz * p1cz
    Gcx = (Gp1cx - dot_pc * p1cx) * inv_cn
    Gcy = (Gp1cy - dot_pc * p1cy) * inv_cn
    Gcz = (Gp1cz - dot_pc * p1cz) * inv_cn

    # cross12 = v0 × v1
    Gv0x = v1y * Gcz - v1z * Gcy
    Gv0y = v1z * Gcx - v1x * Gcz
    Gv0z = v1x * Gcy - v1y * Gcx
    Gv1x = Gcy * v0z - Gcz * v0y
    Gv1y = Gcz * v0x - Gcx * v0z
    Gv1z = Gcx * v0y - Gcy * v0x

    # perp1_ortho = normalise(base × cardinal). Cardinal detached.
    dot_po = Gp1ox * p1ox + Gp1oy * p1oy + Gp1oz * p1oz
    Grox = (Gp1ox - dot_po * p1ox) * inv_rn
    Groy = (Gp1oy - dot_po * p1oy) * inv_rn
    Groz = (Gp1oz - dot_po * p1oz) * inv_rn
    # perp1_ortho_raw = base × cardinal  →  G_base += cardinal × G_raw
    Gb_x += car_y * Groz - car_z * Groy
    Gb_y += car_z * Grox - car_x * Groz
    Gb_z += car_x * Groy - car_y * Grox

    # base = -s / (|s| + eps)
    dot_sb = Gb_x * bx + Gb_y * by + Gb_z * bz
    Gsx = -(Gb_x - dot_sb * bx) * inv_sn
    Gsy = -(Gb_y - dot_sb * by) * inv_sn
    Gsz = -(Gb_z - dot_sb * bz) * inv_sn

    # s = Σ_i v[i] → G_v[i] += G_s
    # plus the cross-product contributions on v0, v1
    Gv0x += Gsx; Gv0y += Gsy; Gv0z += Gsz
    Gv1x += Gsx; Gv1y += Gsy; Gv1z += Gsz
    Gv2x = Gsx;  Gv2y = Gsy;  Gv2z = Gsz
    Gv3x = Gsx;  Gv3y = Gsy;  Gv3z = Gsz

    # v[i] = (nb_pos[i] − pp) · valid[i]
    # G_nb_pos[i] = valid[i] · G_v[i]
    # G_pp_extra = − Σ_i valid[i] · G_v[i]
    Gnb0x = vd0 * Gv0x; Gnb0y = vd0 * Gv0y; Gnb0z = vd0 * Gv0z
    Gnb1x = vd1 * Gv1x; Gnb1y = vd1 * Gv1y; Gnb1z = vd1 * Gv1z
    Gnb2x = vd2 * Gv2x; Gnb2y = vd2 * Gv2y; Gnb2z = vd2 * Gv2z
    Gnb3x = vd3 * Gv3x; Gnb3y = vd3 * Gv3y; Gnb3z = vd3 * Gv3z

    Gpx = ghx - (Gnb0x + Gnb1x + Gnb2x + Gnb3x)
    Gpy = ghy - (Gnb0y + Gnb1y + Gnb2y + Gnb3y)
    Gpz = ghz - (Gnb0z + Gnb1z + Gnb2z + Gnb3z)

    # Scatter to xyz_heavy
    tl.atomic_add(dxyz_ptr + pi * 3 + 0, Gpx, mask=m)
    tl.atomic_add(dxyz_ptr + pi * 3 + 1, Gpy, mask=m)
    tl.atomic_add(dxyz_ptr + pi * 3 + 2, Gpz, mask=m)
    tl.atomic_add(dxyz_ptr + nb0 * 3 + 0, Gnb0x, mask=m)
    tl.atomic_add(dxyz_ptr + nb0 * 3 + 1, Gnb0y, mask=m)
    tl.atomic_add(dxyz_ptr + nb0 * 3 + 2, Gnb0z, mask=m)
    tl.atomic_add(dxyz_ptr + nb1 * 3 + 0, Gnb1x, mask=m)
    tl.atomic_add(dxyz_ptr + nb1 * 3 + 1, Gnb1y, mask=m)
    tl.atomic_add(dxyz_ptr + nb1 * 3 + 2, Gnb1z, mask=m)
    tl.atomic_add(dxyz_ptr + nb2 * 3 + 0, Gnb2x, mask=m)
    tl.atomic_add(dxyz_ptr + nb2 * 3 + 1, Gnb2y, mask=m)
    tl.atomic_add(dxyz_ptr + nb2 * 3 + 2, Gnb2z, mask=m)
    tl.atomic_add(dxyz_ptr + nb3 * 3 + 0, Gnb3x, mask=m)
    tl.atomic_add(dxyz_ptr + nb3 * 3 + 1, Gnb3y, mask=m)
    tl.atomic_add(dxyz_ptr + nb3 * 3 + 2, Gnb3z, mask=m)


class _PlaceHydrogensTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_heavy, parent_idx, nb_idx_clamped, nb_valid, coeffs, bond_length):
        assert xyz_heavy.is_cuda and xyz_heavy.dtype == torch.float32
        N_h = parent_idx.shape[0]
        # nb_valid was (N_h, 4, 1); flatten to (N_h, 4)
        if nb_valid.dim() == 3:
            nb_valid = nb_valid.squeeze(-1)
        # bond_length was (N_h, 1); flatten to (N_h,)
        if bond_length.dim() == 2:
            bond_length = bond_length.squeeze(-1)
        out = torch.empty(N_h, 3, dtype=xyz_heavy.dtype, device=xyz_heavy.device)
        BLOCK = 128
        grid = (triton.cdiv(N_h, BLOCK),)
        _placeh_fwd_kernel[grid](
            xyz_heavy, parent_idx, nb_idx_clamped, nb_valid, coeffs, bond_length, out,
            N_H=N_h, EPS=_EPS, BLOCK=BLOCK,
        )
        ctx.save_for_backward(xyz_heavy, parent_idx, nb_idx_clamped, nb_valid, coeffs, bond_length)
        ctx.n_heavy = xyz_heavy.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_h):
        xyz_heavy, parent_idx, nb_idx_clamped, nb_valid, coeffs, bond_length = ctx.saved_tensors
        N_h = parent_idx.shape[0]
        dxyz = torch.zeros_like(xyz_heavy)
        grad_h = grad_h.contiguous()
        BLOCK = 128
        grid = (triton.cdiv(N_h, BLOCK),)
        _placeh_bwd_kernel[grid](
            xyz_heavy, parent_idx, nb_idx_clamped, nb_valid, coeffs, bond_length,
            grad_h, dxyz,
            N_H=N_h, EPS=_EPS, BLOCK=BLOCK,
        )
        return dxyz, None, None, None, None, None


def place_riding_hydrogens_triton(
    xyz_heavy: torch.Tensor,
    parent_idx: torch.Tensor,
    nb_idx_clamped: torch.Tensor,
    nb_valid: torch.Tensor,
    coeffs: torch.Tensor,
    bond_length: torch.Tensor,
) -> torch.Tensor:
    """Triton-backed riding-hydrogen placement.

    Drop-in replacement for ``_place_h_jit`` (forward) plus an analytic
    Triton backward that avoids autograd's op-by-op traversal.
    """
    return _PlaceHydrogensTriton.apply(
        xyz_heavy, parent_idx, nb_idx_clamped, nb_valid, coeffs, bond_length,
    )
