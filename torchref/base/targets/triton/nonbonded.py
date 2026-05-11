"""Triton forward + analytic backward for the non-bonded prolsq VDW target.

The forward fuses the gather + (optional) cartesian symmetry transform +
prolsq shape energy + per-pair constant into one kernel. The backward
chains through:

    v(d)  = max(0, d_vdw + buf − d)
    E(v)  = c_rep · v^r_exp      (when v > 0)
    dE/dv = c_rep · r_exp · v^(r_exp − 1)
    dv/dd = −1                    (when v > 0, else 0)
    dd/dpos1 = −diff/d            dd/dpos2 = +diff/d
    grad_pos2_source = R_cartᵀ · grad_pos2   (with symmetry)

Cartesian symmetry transforms (M·R·M⁻¹ and M·t) are precomputed once on
the host, so the kernel only does a 3×3 matvec (forward) and 3×3
transposed matvec (backward) per pair.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG_2PI = float(math.log(2.0 * math.pi))


@triton.jit
def _nb_fwd_kernel(
    xyz_ptr,                # (N_atoms, 3)
    idx_ptr,                # (N_pairs, 2)
    min_d_ptr,              # (N_pairs,)
    symop_idx_ptr,          # (N_pairs,) int32
    cart_sym_mat_ptr,       # (n_symops, 3, 3) -- M R M^-1
    cart_sym_off_ptr,       # (n_symops, 3)    -- M t
    cell_off_cart_ptr,      # (N_pairs, 3)     -- M @ cell_offsets
    out_ptr,                # (N_pairs,)
    has_symmetry: tl.constexpr,
    N: tl.constexpr,
    BUFFER: tl.constexpr,
    C_REP: tl.constexpr,
    R_EXP: tl.constexpr,
    LOG_SIG_PLUS_HALF_LOG_2PI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    i = tl.load(idx_ptr + offs * 2 + 0, mask=mask, other=0)
    j = tl.load(idx_ptr + offs * 2 + 1, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
    msx = tl.load(xyz_ptr + j * 3 + 0, mask=mask, other=0.0)
    msy = tl.load(xyz_ptr + j * 3 + 1, mask=mask, other=0.0)
    msz = tl.load(xyz_ptr + j * 3 + 2, mask=mask, other=0.0)

    if has_symmetry:
        s = tl.load(symop_idx_ptr + offs, mask=mask, other=0)
        M00 = tl.load(cart_sym_mat_ptr + s * 9 + 0, mask=mask, other=1.0)
        M01 = tl.load(cart_sym_mat_ptr + s * 9 + 1, mask=mask, other=0.0)
        M02 = tl.load(cart_sym_mat_ptr + s * 9 + 2, mask=mask, other=0.0)
        M10 = tl.load(cart_sym_mat_ptr + s * 9 + 3, mask=mask, other=0.0)
        M11 = tl.load(cart_sym_mat_ptr + s * 9 + 4, mask=mask, other=1.0)
        M12 = tl.load(cart_sym_mat_ptr + s * 9 + 5, mask=mask, other=0.0)
        M20 = tl.load(cart_sym_mat_ptr + s * 9 + 6, mask=mask, other=0.0)
        M21 = tl.load(cart_sym_mat_ptr + s * 9 + 7, mask=mask, other=0.0)
        M22 = tl.load(cart_sym_mat_ptr + s * 9 + 8, mask=mask, other=1.0)
        ox = tl.load(cart_sym_off_ptr + s * 3 + 0, mask=mask, other=0.0)
        oy = tl.load(cart_sym_off_ptr + s * 3 + 1, mask=mask, other=0.0)
        oz = tl.load(cart_sym_off_ptr + s * 3 + 2, mask=mask, other=0.0)
        cox = tl.load(cell_off_cart_ptr + offs * 3 + 0, mask=mask, other=0.0)
        coy = tl.load(cell_off_cart_ptr + offs * 3 + 1, mask=mask, other=0.0)
        coz = tl.load(cell_off_cart_ptr + offs * 3 + 2, mask=mask, other=0.0)
        p2x = M00 * msx + M01 * msy + M02 * msz + ox + cox
        p2y = M10 * msx + M11 * msy + M12 * msz + oy + coy
        p2z = M20 * msx + M21 * msy + M22 * msz + oz + coz
    else:
        p2x = msx; p2y = msy; p2z = msz

    dx = p2x - p1x; dy = p2y - p1y; dz = p2z - p1z
    d = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)
    md = tl.load(min_d_ptr + offs, mask=mask, other=0.0)
    v = md + BUFFER - d
    v = tl.where(v > 0.0, v, 0.0)
    e = C_REP * tl.exp(R_EXP * tl.log(v + 1e-30)) * tl.where(v > 0.0, 1.0, 0.0)
    nll = e + LOG_SIG_PLUS_HALF_LOG_2PI
    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _nb_bwd_kernel(
    xyz_ptr,
    idx_ptr,
    min_d_ptr,
    symop_idx_ptr,
    cart_sym_mat_ptr,
    cart_sym_off_ptr,
    cell_off_cart_ptr,
    grad_out,                 # scalar
    dxyz_ptr,                 # (N_atoms, 3)
    has_symmetry: tl.constexpr,
    N: tl.constexpr,
    BUFFER: tl.constexpr,
    C_REP: tl.constexpr,
    R_EXP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Analytic backward for the prolsq nonbonded NLL.

    For each pair: recompute v and d, then
        coef = grad_out · c_rep · r_exp · v^(r_exp−1)   when v > 0, else 0
        grad_pos1 = -coef · (diff/d)
        grad_pos2 = +coef · (diff/d)
        grad_pos2_source = R_cartᵀ · grad_pos2     (with symmetry; identity otherwise)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    i = tl.load(idx_ptr + offs * 2 + 0, mask=mask, other=0)
    j = tl.load(idx_ptr + offs * 2 + 1, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
    msx = tl.load(xyz_ptr + j * 3 + 0, mask=mask, other=0.0)
    msy = tl.load(xyz_ptr + j * 3 + 1, mask=mask, other=0.0)
    msz = tl.load(xyz_ptr + j * 3 + 2, mask=mask, other=0.0)

    if has_symmetry:
        s = tl.load(symop_idx_ptr + offs, mask=mask, other=0)
        M00 = tl.load(cart_sym_mat_ptr + s * 9 + 0, mask=mask, other=1.0)
        M01 = tl.load(cart_sym_mat_ptr + s * 9 + 1, mask=mask, other=0.0)
        M02 = tl.load(cart_sym_mat_ptr + s * 9 + 2, mask=mask, other=0.0)
        M10 = tl.load(cart_sym_mat_ptr + s * 9 + 3, mask=mask, other=0.0)
        M11 = tl.load(cart_sym_mat_ptr + s * 9 + 4, mask=mask, other=1.0)
        M12 = tl.load(cart_sym_mat_ptr + s * 9 + 5, mask=mask, other=0.0)
        M20 = tl.load(cart_sym_mat_ptr + s * 9 + 6, mask=mask, other=0.0)
        M21 = tl.load(cart_sym_mat_ptr + s * 9 + 7, mask=mask, other=0.0)
        M22 = tl.load(cart_sym_mat_ptr + s * 9 + 8, mask=mask, other=1.0)
        ox = tl.load(cart_sym_off_ptr + s * 3 + 0, mask=mask, other=0.0)
        oy = tl.load(cart_sym_off_ptr + s * 3 + 1, mask=mask, other=0.0)
        oz = tl.load(cart_sym_off_ptr + s * 3 + 2, mask=mask, other=0.0)
        cox = tl.load(cell_off_cart_ptr + offs * 3 + 0, mask=mask, other=0.0)
        coy = tl.load(cell_off_cart_ptr + offs * 3 + 1, mask=mask, other=0.0)
        coz = tl.load(cell_off_cart_ptr + offs * 3 + 2, mask=mask, other=0.0)
        p2x = M00 * msx + M01 * msy + M02 * msz + ox + cox
        p2y = M10 * msx + M11 * msy + M12 * msz + oy + coy
        p2z = M20 * msx + M21 * msy + M22 * msz + oz + coz
    else:
        # dummies so the compiler is happy when has_symmetry=False
        M00 = 1.0; M01 = 0.0; M02 = 0.0
        M10 = 0.0; M11 = 1.0; M12 = 0.0
        M20 = 0.0; M21 = 0.0; M22 = 1.0
        p2x = msx; p2y = msy; p2z = msz

    dx = p2x - p1x; dy = p2y - p1y; dz = p2z - p1z
    d = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)
    md = tl.load(min_d_ptr + offs, mask=mask, other=0.0)
    v = md + BUFFER - d
    active = v > 0.0
    v_safe = tl.where(active, v, 1.0)  # avoid log(0)
    # dE/dv = C_REP * R_EXP * v^(R_EXP - 1)
    dEdv = C_REP * R_EXP * tl.exp((R_EXP - 1.0) * tl.log(v_safe))
    dEdv = tl.where(active, dEdv, 0.0)
    coef = grad_out * dEdv / d  # multiplier on (dx, dy, dz)

    # dE/dpos1 = -coef * (diff)    (because dd/dpos1 = -diff/d and dv/dd = -1)
    # dE/dpos2 = +coef * (diff)
    g2x = coef * dx
    g2y = coef * dy
    g2z = coef * dz

    if has_symmetry:
        # grad_pos2_source = R_cart^T @ grad_pos2
        gsx = M00 * g2x + M10 * g2y + M20 * g2z
        gsy = M01 * g2x + M11 * g2y + M21 * g2z
        gsz = M02 * g2x + M12 * g2y + M22 * g2z
    else:
        gsx = g2x; gsy = g2y; gsz = g2z

    # ∂E/∂pos1 = +coef · diff  (since dd/dpos1 = -diff/d and dv/dd = -1)
    # ∂E/∂pos2 = -coef · diff
    # pos2 = R_cart · pos2_source + offset  ⇒  ∂E/∂pos2_source = R_cartᵀ · ∂E/∂pos2
    tl.atomic_add(dxyz_ptr + j * 3 + 0, -gsx, mask=mask)
    tl.atomic_add(dxyz_ptr + j * 3 + 1, -gsy, mask=mask)
    tl.atomic_add(dxyz_ptr + j * 3 + 2, -gsz, mask=mask)
    tl.atomic_add(dxyz_ptr + i * 3 + 0,  g2x, mask=mask)
    tl.atomic_add(dxyz_ptr + i * 3 + 1,  g2y, mask=mask)
    tl.atomic_add(dxyz_ptr + i * 3 + 2,  g2z, mask=mask)


def _build_cartesian_symops(symop_matrices, symop_translations,
                            fractional_matrix, inv_fractional_matrix):
    """Precompute the cartesian rotation+translation per symop: M·R·M⁻¹, M·t."""
    M = fractional_matrix
    Minv = inv_fractional_matrix
    cart_mat = torch.einsum("ij,sjk,kl->sil",
                            M, symop_matrices.to(M.dtype), Minv)
    cart_off = symop_translations.to(M.dtype) @ M.T
    return cart_mat.contiguous(), cart_off.contiguous()


class _NonbondedHeavyMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, indices, min_distances,
                symop_indices, cell_offsets,
                symop_matrices, symop_translations,
                fractional_matrix, inv_fractional_matrix,
                c_rep, r_exp, buffer_, sigma_vdw):
        assert xyz.is_cuda and xyz.dtype == torch.float32
        N = indices.shape[0]
        nll = torch.empty(N, dtype=xyz.dtype, device=xyz.device)

        has_sym = (
            symop_indices is not None
            and symop_indices.numel() > 0
            and not bool((symop_indices == 0).all())
        )
        if has_sym:
            cart_mat, cart_off = _build_cartesian_symops(
                symop_matrices, symop_translations,
                fractional_matrix, inv_fractional_matrix,
            )
            cell_off_cart = (cell_offsets.to(xyz.dtype)
                             @ fractional_matrix.T).contiguous()
            symop_i32 = symop_indices.to(torch.int32).contiguous()
        else:
            cart_mat = torch.zeros(1, 3, 3, device=xyz.device, dtype=xyz.dtype)
            cart_off = torch.zeros(1, 3, device=xyz.device, dtype=xyz.dtype)
            cell_off_cart = torch.zeros(N, 3, device=xyz.device, dtype=xyz.dtype)
            symop_i32 = torch.zeros(N, device=xyz.device, dtype=torch.int32)

        log_sig = float(torch.log(sigma_vdw).item()) + 0.5 * _LOG_2PI
        c_rep_f = float(c_rep.item())
        r_exp_f = float(r_exp.item())
        buf_f = float(buffer_)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _nb_fwd_kernel[grid](
            xyz, indices, min_distances, symop_i32, cart_mat, cart_off,
            cell_off_cart, nll,
            has_symmetry=bool(has_sym),
            N=N, BUFFER=buf_f, C_REP=c_rep_f, R_EXP=r_exp_f,
            LOG_SIG_PLUS_HALF_LOG_2PI=log_sig, BLOCK=BLOCK,
        )

        ctx.save_for_backward(
            xyz, indices, min_distances,
            symop_i32, cart_mat, cart_off, cell_off_cart,
        )
        ctx.c_rep = c_rep_f
        ctx.r_exp = r_exp_f
        ctx.buffer = buf_f
        ctx.has_sym = bool(has_sym)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        (xyz, indices, min_distances, symop_i32,
         cart_mat, cart_off, cell_off_cart) = ctx.saved_tensors
        N = indices.shape[0]
        dxyz = torch.zeros_like(xyz)
        BLOCK = 256
        grid = (triton.cdiv(N, BLOCK),)
        _nb_bwd_kernel[grid](
            xyz, indices, min_distances, symop_i32,
            cart_mat, cart_off, cell_off_cart,
            float(grad_out.item()), dxyz,
            has_symmetry=ctx.has_sym,
            N=N, BUFFER=ctx.buffer, C_REP=ctx.c_rep, R_EXP=ctx.r_exp,
            BLOCK=BLOCK,
        )
        return (dxyz,) + (None,) * 12


def nonbonded_heavy_math_triton(
    xyz, indices, min_distances,
    symop_indices, cell_offsets,
    symop_matrices, symop_translations,
    fractional_matrix, inv_fractional_matrix,
    c_rep, r_exp, buffer_, sigma_vdw,
):
    """Triton-backed heavy-heavy VDW prolsq NLL with analytic backward."""
    return _NonbondedHeavyMathTriton.apply(
        xyz, indices, min_distances,
        symop_indices, cell_offsets,
        symop_matrices, symop_translations,
        fractional_matrix, inv_fractional_matrix,
        c_rep, r_exp, buffer_, sigma_vdw,
    )
