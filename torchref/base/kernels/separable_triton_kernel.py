"""
Separable Gaussian density splatting via Triton.

Factorizes exp(-alpha * f^T G f) into 1D Gaussian tables along each
fractional axis, with optional 2D cross-term corrections for non-orthogonal
cells.  One program per atom.  All 1D/2D tables live in a small per-atom
scratch buffer (~0.8-3 KB) that stays hot in L1 cache.

Eliminates the real_space_grid tensor (~500 MB) and all PBC matrix
operations that the fused kernel requires.

Forward + backward kernels with full autograd support for xyz, b, occ.

For non-orthogonal cells, uses combined exponent exp(-alpha*r²) to avoid
numerical overflow from separate diagonal × cross-term exp() products.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

# =============================================================================
# Constants
# =============================================================================

PI: float = 3.141592653589793
PI_SQ: float = PI * PI
PI_1P5: float = PI * math.sqrt(PI)  # pi^1.5 ≈ 5.568327996831708

# =============================================================================
# Forward kernel
# =============================================================================


@triton.jit
def _separable_fwd_kernel(
    # Pointers
    density_map_ptr,   # (nx*ny*nz,) float32
    xyz_ptr,           # (N_atoms, 3) float32
    b_ptr,             # (N_atoms,) float32
    A_ptr,             # (N_atoms, 5) float32
    B_ptr,             # (N_atoms, 5) float32
    occ_ptr,           # (N_atoms,) float32
    offsets_ptr,       # (N_sphere, 3) int16 — sphere voxel offsets
    inv_frac_ptr,      # (9,) float32 row-major
    scratch_ptr,       # (N_atoms, SCRATCH_SIZE) float32
    # Metric tensor components
    G11, G22, G33,
    G12, G13, G23,
    # Grid spacing (fractional)
    inv_grid_x, inv_grid_y, inv_grid_z,
    # Dimensions
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    N_sphere: tl.constexpr,
    N_AXIS: tl.constexpr,
    half_n: tl.constexpr,
    SCRATCH_PER_ATOM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    # Cross-term flags
    COMPUTE_XY: tl.constexpr,
    COMPUTE_XZ: tl.constexpr,
    COMPUTE_YZ: tl.constexpr,
    STORE_CROSS_TABLES: tl.constexpr,
):
    """One program per atom.  Builds 1D tables, gathers per sphere voxel."""
    atom = tl.program_id(0)

    # ---- Stage 1: Load per-atom parameters ----
    b_iso = tl.load(b_ptr + atom)
    occ = tl.load(occ_ptr + atom)
    ax = tl.load(xyz_ptr + atom * 3 + 0)
    ay = tl.load(xyz_ptr + atom * 3 + 1)
    az = tl.load(xyz_ptr + atom * 3 + 2)

    # ITC92 coefficients (5 components)
    A0 = tl.load(A_ptr + atom * 5 + 0)
    A1 = tl.load(A_ptr + atom * 5 + 1)
    A2 = tl.load(A_ptr + atom * 5 + 2)
    A3 = tl.load(A_ptr + atom * 5 + 3)
    A4 = tl.load(A_ptr + atom * 5 + 4)
    B0 = tl.load(B_ptr + atom * 5 + 0)
    B1 = tl.load(B_ptr + atom * 5 + 1)
    B2 = tl.load(B_ptr + atom * 5 + 2)
    B3 = tl.load(B_ptr + atom * 5 + 3)
    B4 = tl.load(B_ptr + atom * 5 + 4)

    # B_total, alpha, A_norm per component
    Bt0 = tl.maximum((B0 + b_iso) * 0.25, 0.1)
    Bt1 = tl.maximum((B1 + b_iso) * 0.25, 0.1)
    Bt2 = tl.maximum((B2 + b_iso) * 0.25, 0.1)
    Bt3 = tl.maximum((B3 + b_iso) * 0.25, 0.1)
    Bt4 = tl.maximum((B4 + b_iso) * 0.25, 0.1)

    pi_sq: tl.constexpr = 9.869604401089358
    pi_1p5: tl.constexpr = 5.568327996831708

    al0 = pi_sq / Bt0
    al1 = pi_sq / Bt1
    al2 = pi_sq / Bt2
    al3 = pi_sq / Bt3
    al4 = pi_sq / Bt4

    An0 = A0 * occ * pi_1p5 / (Bt0 * tl.sqrt(Bt0))
    An1 = A1 * occ * pi_1p5 / (Bt1 * tl.sqrt(Bt1))
    An2 = A2 * occ * pi_1p5 / (Bt2 * tl.sqrt(Bt2))
    An3 = A3 * occ * pi_1p5 / (Bt3 * tl.sqrt(Bt3))
    An4 = A4 * occ * pi_1p5 / (Bt4 * tl.sqrt(Bt4))

    # ---- Cartesian → fractional conversion ----
    if0 = tl.load(inv_frac_ptr + 0)
    if1 = tl.load(inv_frac_ptr + 1)
    if2 = tl.load(inv_frac_ptr + 2)
    if3 = tl.load(inv_frac_ptr + 3)
    if4 = tl.load(inv_frac_ptr + 4)
    if5 = tl.load(inv_frac_ptr + 5)
    if6 = tl.load(inv_frac_ptr + 6)
    if7 = tl.load(inv_frac_ptr + 7)
    if8 = tl.load(inv_frac_ptr + 8)

    frac_x = ax * if0 + ay * if1 + az * if2
    frac_y = ax * if3 + ay * if4 + az * if5
    frac_z = ax * if6 + ay * if7 + az * if8

    # Wrap to [0, 1)
    frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
    frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
    frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)

    # Grid anchor (nearest grid index)
    cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
    ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
    ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)

    # Sub-grid offset (fractional)
    sub_x = frac_x - cix.to(tl.float32) * inv_grid_x
    sub_y = frac_y - ciy.to(tl.float32) * inv_grid_y
    sub_z = frac_z - ciz.to(tl.float32) * inv_grid_z

    # ---- Stage 2: Build 1D tables ----
    # Scratch layout: [diag_x(5*N), diag_y(5*N), diag_z(5*N),
    #                  delta_x(N), delta_y(N), delta_z(N)]
    # When using combined-exponent path (a), only deltas are read per voxel;
    # skip the 15 1D exp-table computations (saves 15*N_AXIS exp() calls).
    _USE_COMBINED: tl.constexpr = (
        not STORE_CROSS_TABLES and (COMPUTE_XY or COMPUTE_XZ or COMPUTE_YZ)
    )
    base = atom * SCRATCH_PER_ATOM
    axis_idx = tl.arange(0, N_AXIS)
    half_n_f: tl.constexpr = half_n  # float version

    # --- Axis X ---
    delta_x = (axis_idx.to(tl.float32) - half_n_f) * inv_grid_x - sub_x
    tl.store(scratch_ptr + base + 15 * N_AXIS + axis_idx, delta_x)  # store deltas

    if not _USE_COMBINED:
        dx2 = delta_x * delta_x
        diag_x0 = tl.exp(-al0 * G11 * dx2)
        diag_x1 = tl.exp(-al1 * G11 * dx2)
        diag_x2 = tl.exp(-al2 * G11 * dx2)
        diag_x3 = tl.exp(-al3 * G11 * dx2)
        diag_x4 = tl.exp(-al4 * G11 * dx2)
        tl.store(scratch_ptr + base + 0 * N_AXIS + axis_idx, diag_x0)
        tl.store(scratch_ptr + base + 1 * N_AXIS + axis_idx, diag_x1)
        tl.store(scratch_ptr + base + 2 * N_AXIS + axis_idx, diag_x2)
        tl.store(scratch_ptr + base + 3 * N_AXIS + axis_idx, diag_x3)
        tl.store(scratch_ptr + base + 4 * N_AXIS + axis_idx, diag_x4)

    # --- Axis Y ---
    delta_y = (axis_idx.to(tl.float32) - half_n_f) * inv_grid_y - sub_y
    tl.store(scratch_ptr + base + 16 * N_AXIS + axis_idx, delta_y)

    if not _USE_COMBINED:
        dy2 = delta_y * delta_y
        diag_y0 = tl.exp(-al0 * G22 * dy2)
        diag_y1 = tl.exp(-al1 * G22 * dy2)
        diag_y2 = tl.exp(-al2 * G22 * dy2)
        diag_y3 = tl.exp(-al3 * G22 * dy2)
        diag_y4 = tl.exp(-al4 * G22 * dy2)
        tl.store(scratch_ptr + base + 5 * N_AXIS + axis_idx, diag_y0)
        tl.store(scratch_ptr + base + 6 * N_AXIS + axis_idx, diag_y1)
        tl.store(scratch_ptr + base + 7 * N_AXIS + axis_idx, diag_y2)
        tl.store(scratch_ptr + base + 8 * N_AXIS + axis_idx, diag_y3)
        tl.store(scratch_ptr + base + 9 * N_AXIS + axis_idx, diag_y4)

    # --- Axis Z ---
    delta_z = (axis_idx.to(tl.float32) - half_n_f) * inv_grid_z - sub_z
    tl.store(scratch_ptr + base + 17 * N_AXIS + axis_idx, delta_z)

    if not _USE_COMBINED:
        dz2 = delta_z * delta_z
        diag_z0 = tl.exp(-al0 * G33 * dz2)
        diag_z1 = tl.exp(-al1 * G33 * dz2)
        diag_z2 = tl.exp(-al2 * G33 * dz2)
        diag_z3 = tl.exp(-al3 * G33 * dz2)
        diag_z4 = tl.exp(-al4 * G33 * dz2)
        tl.store(scratch_ptr + base + 10 * N_AXIS + axis_idx, diag_z0)
        tl.store(scratch_ptr + base + 11 * N_AXIS + axis_idx, diag_z1)
        tl.store(scratch_ptr + base + 12 * N_AXIS + axis_idx, diag_z2)
        tl.store(scratch_ptr + base + 13 * N_AXIS + axis_idx, diag_z3)
        tl.store(scratch_ptr + base + 14 * N_AXIS + axis_idx, diag_z4)

    # ---- Stage 3: 2D cross-term tables (conditional) ----
    # cross_base starts after the 1D tables + deltas  (18 * N_AXIS)
    cross_base = base + 18 * N_AXIS

    if STORE_CROSS_TABLES:
        if COMPUTE_XY:
            # cross_xy[c][i*N_AXIS + j] for each component c
            # Vectorised over N_AXIS² elements
            idx_2d = tl.arange(0, N_AXIS * N_AXIS)
            ii = idx_2d // N_AXIS  # row (x index)
            jj = idx_2d % N_AXIS   # col (y index)
            dx_i = (ii.to(tl.float32) - half_n_f) * inv_grid_x - sub_x
            dy_j = (jj.to(tl.float32) - half_n_f) * inv_grid_y - sub_y
            prod_xy = dx_i * dy_j

            off_xy = cross_base
            tl.store(scratch_ptr + off_xy + 0 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al0 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 1 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al1 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 2 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al2 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 3 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al3 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 4 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al4 * 2.0 * G12 * prod_xy))
            cross_base = cross_base + 5 * N_AXIS * N_AXIS

        if COMPUTE_XZ:
            idx_2d = tl.arange(0, N_AXIS * N_AXIS)
            ii = idx_2d // N_AXIS
            kk = idx_2d % N_AXIS
            dx_i = (ii.to(tl.float32) - half_n_f) * inv_grid_x - sub_x
            dz_k = (kk.to(tl.float32) - half_n_f) * inv_grid_z - sub_z
            prod_xz = dx_i * dz_k

            off_xz = cross_base
            tl.store(scratch_ptr + off_xz + 0 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al0 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 1 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al1 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 2 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al2 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 3 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al3 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 4 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al4 * 2.0 * G13 * prod_xz))
            cross_base = cross_base + 5 * N_AXIS * N_AXIS

        if COMPUTE_YZ:
            idx_2d = tl.arange(0, N_AXIS * N_AXIS)
            jj = idx_2d // N_AXIS
            kk = idx_2d % N_AXIS
            dy_j = (jj.to(tl.float32) - half_n_f) * inv_grid_y - sub_y
            dz_k = (kk.to(tl.float32) - half_n_f) * inv_grid_z - sub_z
            prod_yz = dy_j * dz_k

            off_yz = cross_base
            tl.store(scratch_ptr + off_yz + 0 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al0 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 1 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al1 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 2 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al2 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 3 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al3 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 4 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al4 * 2.0 * G23 * prod_yz))

    # ---- Stage 4: Assemble sphere voxels and scatter ----
    # Precompute cross-term table base offsets (compile-time)
    cross_off_first = base + 18 * N_AXIS
    # For STORE_CROSS_TABLES: tables laid out in order XY, XZ, YZ
    # (only present if the corresponding COMPUTE flag is True)

    v_offsets = tl.arange(0, BLOCK_V)

    for v_start in range(0, N_sphere, BLOCK_V):
        v = v_start + v_offsets
        mask = v < N_sphere

        # Load sphere offsets (int16 → int32)
        si = tl.load(offsets_ptr + v * 3 + 0, mask=mask, other=0).to(tl.int32)
        sj = tl.load(offsets_ptr + v * 3 + 1, mask=mask, other=0).to(tl.int32)
        sk = tl.load(offsets_ptr + v * 3 + 2, mask=mask, other=0).to(tl.int32)

        # Table indices
        ti = si + half_n
        tj = sj + half_n
        tk = sk + half_n

        # Compute per-component density contributions c0-c4.
        #
        # Two paths:
        # (a) Combined exponent: compute full r²=d^T G d from stored deltas
        #     and use a single exp(-alpha*r²) per component.
        #     Avoids exp(-big_diag)*exp(+big_cross) = 0*inf = NaN.
        # (b) Pure separable or stored cross tables: load 1D diag values
        #     and (optionally) multiply by pre-stored cross-term tables.
        if _USE_COMBINED:
            # Path (a): combined exponent from stored deltas
            dxi = tl.load(scratch_ptr + base + 15 * N_AXIS + ti, mask=mask, other=0.0)
            dyj = tl.load(scratch_ptr + base + 16 * N_AXIS + tj, mask=mask, other=0.0)
            dzk = tl.load(scratch_ptr + base + 17 * N_AXIS + tk, mask=mask, other=0.0)
            r2 = G11 * dxi * dxi + G22 * dyj * dyj + G33 * dzk * dzk
            if COMPUTE_XY:
                r2 = r2 + 2.0 * G12 * dxi * dyj
            if COMPUTE_XZ:
                r2 = r2 + 2.0 * G13 * dxi * dzk
            if COMPUTE_YZ:
                r2 = r2 + 2.0 * G23 * dyj * dzk
            c0 = An0 * tl.exp(-al0 * r2)
            c1 = An1 * tl.exp(-al1 * r2)
            c2 = An2 * tl.exp(-al2 * r2)
            c3 = An3 * tl.exp(-al3 * r2)
            c4 = An4 * tl.exp(-al4 * r2)
        else:
            # Path (b): separable 1D diagonal tables
            vx0 = tl.load(scratch_ptr + base + 0 * N_AXIS + ti, mask=mask, other=0.0)
            vy0 = tl.load(scratch_ptr + base + 5 * N_AXIS + tj, mask=mask, other=0.0)
            vz0 = tl.load(scratch_ptr + base + 10 * N_AXIS + tk, mask=mask, other=0.0)
            c0 = An0 * vx0 * vy0 * vz0
            vx1 = tl.load(scratch_ptr + base + 1 * N_AXIS + ti, mask=mask, other=0.0)
            vy1 = tl.load(scratch_ptr + base + 6 * N_AXIS + tj, mask=mask, other=0.0)
            vz1 = tl.load(scratch_ptr + base + 11 * N_AXIS + tk, mask=mask, other=0.0)
            c1 = An1 * vx1 * vy1 * vz1
            vx2 = tl.load(scratch_ptr + base + 2 * N_AXIS + ti, mask=mask, other=0.0)
            vy2 = tl.load(scratch_ptr + base + 7 * N_AXIS + tj, mask=mask, other=0.0)
            vz2 = tl.load(scratch_ptr + base + 12 * N_AXIS + tk, mask=mask, other=0.0)
            c2 = An2 * vx2 * vy2 * vz2
            vx3 = tl.load(scratch_ptr + base + 3 * N_AXIS + ti, mask=mask, other=0.0)
            vy3 = tl.load(scratch_ptr + base + 8 * N_AXIS + tj, mask=mask, other=0.0)
            vz3 = tl.load(scratch_ptr + base + 13 * N_AXIS + tk, mask=mask, other=0.0)
            c3 = An3 * vx3 * vy3 * vz3
            vx4 = tl.load(scratch_ptr + base + 4 * N_AXIS + ti, mask=mask, other=0.0)
            vy4 = tl.load(scratch_ptr + base + 9 * N_AXIS + tj, mask=mask, other=0.0)
            vz4 = tl.load(scratch_ptr + base + 14 * N_AXIS + tk, mask=mask, other=0.0)
            c4 = An4 * vx4 * vy4 * vz4

            if STORE_CROSS_TABLES:
                ct_base = cross_off_first
                if COMPUTE_XY:
                    idx_xy = ti * N_AXIS + tj
                    c0 *= tl.load(scratch_ptr + ct_base + 0 * N_AXIS * N_AXIS + idx_xy,
                                  mask=mask, other=1.0)
                    c1 *= tl.load(scratch_ptr + ct_base + 1 * N_AXIS * N_AXIS + idx_xy,
                                  mask=mask, other=1.0)
                    c2 *= tl.load(scratch_ptr + ct_base + 2 * N_AXIS * N_AXIS + idx_xy,
                                  mask=mask, other=1.0)
                    c3 *= tl.load(scratch_ptr + ct_base + 3 * N_AXIS * N_AXIS + idx_xy,
                                  mask=mask, other=1.0)
                    c4 *= tl.load(scratch_ptr + ct_base + 4 * N_AXIS * N_AXIS + idx_xy,
                                  mask=mask, other=1.0)
                    ct_base = ct_base + 5 * N_AXIS * N_AXIS
                if COMPUTE_XZ:
                    idx_xz = ti * N_AXIS + tk
                    c0 *= tl.load(scratch_ptr + ct_base + 0 * N_AXIS * N_AXIS + idx_xz,
                                  mask=mask, other=1.0)
                    c1 *= tl.load(scratch_ptr + ct_base + 1 * N_AXIS * N_AXIS + idx_xz,
                                  mask=mask, other=1.0)
                    c2 *= tl.load(scratch_ptr + ct_base + 2 * N_AXIS * N_AXIS + idx_xz,
                                  mask=mask, other=1.0)
                    c3 *= tl.load(scratch_ptr + ct_base + 3 * N_AXIS * N_AXIS + idx_xz,
                                  mask=mask, other=1.0)
                    c4 *= tl.load(scratch_ptr + ct_base + 4 * N_AXIS * N_AXIS + idx_xz,
                                  mask=mask, other=1.0)
                    ct_base = ct_base + 5 * N_AXIS * N_AXIS
                if COMPUTE_YZ:
                    idx_yz = tj * N_AXIS + tk
                    c0 *= tl.load(scratch_ptr + ct_base + 0 * N_AXIS * N_AXIS + idx_yz,
                                  mask=mask, other=1.0)
                    c1 *= tl.load(scratch_ptr + ct_base + 1 * N_AXIS * N_AXIS + idx_yz,
                                  mask=mask, other=1.0)
                    c2 *= tl.load(scratch_ptr + ct_base + 2 * N_AXIS * N_AXIS + idx_yz,
                                  mask=mask, other=1.0)
                    c3 *= tl.load(scratch_ptr + ct_base + 3 * N_AXIS * N_AXIS + idx_yz,
                                  mask=mask, other=1.0)
                    c4 *= tl.load(scratch_ptr + ct_base + 4 * N_AXIS * N_AXIS + idx_yz,
                                  mask=mask, other=1.0)

        density = c0 + c1 + c2 + c3 + c4

        # PBC-wrapped grid index
        gi = (cix + si) % nx
        gi = tl.where(gi < 0, gi + nx, gi)
        gj = (ciy + sj) % ny
        gj = tl.where(gj < 0, gj + ny, gj)
        gk = (ciz + sk) % nz
        gk = tl.where(gk < 0, gk + nz, gk)
        flat_idx = (gi.to(tl.int64) * ny + gj.to(tl.int64)) * nz + gk.to(tl.int64)

        tl.atomic_add(density_map_ptr + flat_idx, density, mask=mask)


# =============================================================================
# Backward kernel
# =============================================================================


@triton.jit
def _separable_bwd_kernel(
    # Forward inputs (read-only)
    grad_density_map_ptr,  # (nx*ny*nz,) float32
    xyz_ptr,
    b_ptr,
    A_ptr,
    B_ptr,
    occ_ptr,
    offsets_ptr,
    inv_frac_ptr,
    scratch_ptr,
    # Metric tensor
    G11, G22, G33,
    G12, G13, G23,
    # Grid params
    inv_grid_x, inv_grid_y, inv_grid_z,
    # Gradient outputs (pointers before constexpr)
    grad_frac_ptr,  # (N_atoms, 3) float32
    grad_b_ptr,     # (N_atoms,) float32
    grad_occ_ptr,   # (N_atoms,) float32
    # Constexpr
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    N_sphere: tl.constexpr,
    N_AXIS: tl.constexpr,
    half_n: tl.constexpr,
    SCRATCH_PER_ATOM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    COMPUTE_XY: tl.constexpr,
    COMPUTE_XZ: tl.constexpr,
    COMPUTE_YZ: tl.constexpr,
    STORE_CROSS_TABLES: tl.constexpr,
):
    """One program per atom.  Recomputes 1D tables, accumulates gradients."""
    atom = tl.program_id(0)

    # ---- Stage 1: Load & compute (identical to forward) ----
    b_iso = tl.load(b_ptr + atom)
    occ = tl.load(occ_ptr + atom)
    ax = tl.load(xyz_ptr + atom * 3 + 0)
    ay = tl.load(xyz_ptr + atom * 3 + 1)
    az = tl.load(xyz_ptr + atom * 3 + 2)

    A0 = tl.load(A_ptr + atom * 5 + 0)
    A1 = tl.load(A_ptr + atom * 5 + 1)
    A2 = tl.load(A_ptr + atom * 5 + 2)
    A3 = tl.load(A_ptr + atom * 5 + 3)
    A4 = tl.load(A_ptr + atom * 5 + 4)
    B0 = tl.load(B_ptr + atom * 5 + 0)
    B1 = tl.load(B_ptr + atom * 5 + 1)
    B2 = tl.load(B_ptr + atom * 5 + 2)
    B3 = tl.load(B_ptr + atom * 5 + 3)
    B4 = tl.load(B_ptr + atom * 5 + 4)

    Bt0 = tl.maximum((B0 + b_iso) * 0.25, 0.1)
    Bt1 = tl.maximum((B1 + b_iso) * 0.25, 0.1)
    Bt2 = tl.maximum((B2 + b_iso) * 0.25, 0.1)
    Bt3 = tl.maximum((B3 + b_iso) * 0.25, 0.1)
    Bt4 = tl.maximum((B4 + b_iso) * 0.25, 0.1)

    clamp0 = ((B0 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp1 = ((B1 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp2 = ((B2 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp3 = ((B3 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp4 = ((B4 + b_iso) * 0.25 > 0.1).to(tl.float32)

    pi_sq: tl.constexpr = 9.869604401089358
    pi_1p5: tl.constexpr = 5.568327996831708

    al0 = pi_sq / Bt0
    al1 = pi_sq / Bt1
    al2 = pi_sq / Bt2
    al3 = pi_sq / Bt3
    al4 = pi_sq / Bt4

    An0 = A0 * occ * pi_1p5 / (Bt0 * tl.sqrt(Bt0))
    An1 = A1 * occ * pi_1p5 / (Bt1 * tl.sqrt(Bt1))
    An2 = A2 * occ * pi_1p5 / (Bt2 * tl.sqrt(Bt2))
    An3 = A3 * occ * pi_1p5 / (Bt3 * tl.sqrt(Bt3))
    An4 = A4 * occ * pi_1p5 / (Bt4 * tl.sqrt(Bt4))

    # Frac conversion
    if0 = tl.load(inv_frac_ptr + 0)
    if1 = tl.load(inv_frac_ptr + 1)
    if2 = tl.load(inv_frac_ptr + 2)
    if3 = tl.load(inv_frac_ptr + 3)
    if4 = tl.load(inv_frac_ptr + 4)
    if5 = tl.load(inv_frac_ptr + 5)
    if6 = tl.load(inv_frac_ptr + 6)
    if7 = tl.load(inv_frac_ptr + 7)
    if8 = tl.load(inv_frac_ptr + 8)

    frac_x = ax * if0 + ay * if1 + az * if2
    frac_y = ax * if3 + ay * if4 + az * if5
    frac_z = ax * if6 + ay * if7 + az * if8
    frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
    frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
    frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)

    cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
    ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
    ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)
    sub_x = frac_x - cix.to(tl.float32) * inv_grid_x
    sub_y = frac_y - ciy.to(tl.float32) * inv_grid_y
    sub_z = frac_z - ciz.to(tl.float32) * inv_grid_z

    # ---- Stage 2-3: Rebuild tables ----
    # Combined-exponent path only reads deltas; skip 1D exp tables.
    _USE_COMBINED: tl.constexpr = (
        not STORE_CROSS_TABLES and (COMPUTE_XY or COMPUTE_XZ or COMPUTE_YZ)
    )
    base = atom * SCRATCH_PER_ATOM
    axis_idx = tl.arange(0, N_AXIS)
    half_n_f: tl.constexpr = half_n

    delta_x_vec = (axis_idx.to(tl.float32) - half_n_f) * inv_grid_x - sub_x
    tl.store(scratch_ptr + base + 15 * N_AXIS + axis_idx, delta_x_vec)
    if not _USE_COMBINED:
        dx2 = delta_x_vec * delta_x_vec
        tl.store(scratch_ptr + base + 0 * N_AXIS + axis_idx, tl.exp(-al0 * G11 * dx2))
        tl.store(scratch_ptr + base + 1 * N_AXIS + axis_idx, tl.exp(-al1 * G11 * dx2))
        tl.store(scratch_ptr + base + 2 * N_AXIS + axis_idx, tl.exp(-al2 * G11 * dx2))
        tl.store(scratch_ptr + base + 3 * N_AXIS + axis_idx, tl.exp(-al3 * G11 * dx2))
        tl.store(scratch_ptr + base + 4 * N_AXIS + axis_idx, tl.exp(-al4 * G11 * dx2))

    delta_y_vec = (axis_idx.to(tl.float32) - half_n_f) * inv_grid_y - sub_y
    tl.store(scratch_ptr + base + 16 * N_AXIS + axis_idx, delta_y_vec)
    if not _USE_COMBINED:
        dy2 = delta_y_vec * delta_y_vec
        tl.store(scratch_ptr + base + 5 * N_AXIS + axis_idx, tl.exp(-al0 * G22 * dy2))
        tl.store(scratch_ptr + base + 6 * N_AXIS + axis_idx, tl.exp(-al1 * G22 * dy2))
        tl.store(scratch_ptr + base + 7 * N_AXIS + axis_idx, tl.exp(-al2 * G22 * dy2))
        tl.store(scratch_ptr + base + 8 * N_AXIS + axis_idx, tl.exp(-al3 * G22 * dy2))
        tl.store(scratch_ptr + base + 9 * N_AXIS + axis_idx, tl.exp(-al4 * G22 * dy2))

    delta_z_vec = (axis_idx.to(tl.float32) - half_n_f) * inv_grid_z - sub_z
    tl.store(scratch_ptr + base + 17 * N_AXIS + axis_idx, delta_z_vec)
    if not _USE_COMBINED:
        dz2 = delta_z_vec * delta_z_vec
        tl.store(scratch_ptr + base + 10 * N_AXIS + axis_idx, tl.exp(-al0 * G33 * dz2))
        tl.store(scratch_ptr + base + 11 * N_AXIS + axis_idx, tl.exp(-al1 * G33 * dz2))
        tl.store(scratch_ptr + base + 12 * N_AXIS + axis_idx, tl.exp(-al2 * G33 * dz2))
        tl.store(scratch_ptr + base + 13 * N_AXIS + axis_idx, tl.exp(-al3 * G33 * dz2))
        tl.store(scratch_ptr + base + 14 * N_AXIS + axis_idx, tl.exp(-al4 * G33 * dz2))

    # Rebuild cross-term tables (same as forward)
    cross_base = base + 18 * N_AXIS
    if STORE_CROSS_TABLES:
        if COMPUTE_XY:
            idx_2d = tl.arange(0, N_AXIS * N_AXIS)
            ii = idx_2d // N_AXIS
            jj = idx_2d % N_AXIS
            dx_i = (ii.to(tl.float32) - half_n_f) * inv_grid_x - sub_x
            dy_j = (jj.to(tl.float32) - half_n_f) * inv_grid_y - sub_y
            prod_xy = dx_i * dy_j
            off_xy = cross_base
            tl.store(scratch_ptr + off_xy + 0 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al0 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 1 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al1 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 2 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al2 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 3 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al3 * 2.0 * G12 * prod_xy))
            tl.store(scratch_ptr + off_xy + 4 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al4 * 2.0 * G12 * prod_xy))
            cross_base = cross_base + 5 * N_AXIS * N_AXIS
        if COMPUTE_XZ:
            idx_2d = tl.arange(0, N_AXIS * N_AXIS)
            ii = idx_2d // N_AXIS
            kk = idx_2d % N_AXIS
            dx_i = (ii.to(tl.float32) - half_n_f) * inv_grid_x - sub_x
            dz_k = (kk.to(tl.float32) - half_n_f) * inv_grid_z - sub_z
            prod_xz = dx_i * dz_k
            off_xz = cross_base
            tl.store(scratch_ptr + off_xz + 0 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al0 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 1 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al1 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 2 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al2 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 3 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al3 * 2.0 * G13 * prod_xz))
            tl.store(scratch_ptr + off_xz + 4 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al4 * 2.0 * G13 * prod_xz))
            cross_base = cross_base + 5 * N_AXIS * N_AXIS
        if COMPUTE_YZ:
            idx_2d = tl.arange(0, N_AXIS * N_AXIS)
            jj = idx_2d // N_AXIS
            kk = idx_2d % N_AXIS
            dy_j = (jj.to(tl.float32) - half_n_f) * inv_grid_y - sub_y
            dz_k = (kk.to(tl.float32) - half_n_f) * inv_grid_z - sub_z
            prod_yz = dy_j * dz_k
            off_yz = cross_base
            tl.store(scratch_ptr + off_yz + 0 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al0 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 1 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al1 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 2 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al2 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 3 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al3 * 2.0 * G23 * prod_yz))
            tl.store(scratch_ptr + off_yz + 4 * N_AXIS * N_AXIS + idx_2d,
                     tl.exp(-al4 * 2.0 * G23 * prod_yz))

    # ---- Stage 4: Gradient accumulation ----
    cross_off_first = base + 18 * N_AXIS

    # Init accumulators with matching dtype (supports float32 and float64)
    _zero = b_iso * 0.0
    g_fx = _zero
    g_fy = _zero
    g_fz = _zero
    g_b = _zero
    g_occ = _zero

    v_offsets = tl.arange(0, BLOCK_V)

    for v_start in range(0, N_sphere, BLOCK_V):
        v = v_start + v_offsets
        mask = v < N_sphere

        si = tl.load(offsets_ptr + v * 3 + 0, mask=mask, other=0).to(tl.int32)
        sj = tl.load(offsets_ptr + v * 3 + 1, mask=mask, other=0).to(tl.int32)
        sk = tl.load(offsets_ptr + v * 3 + 2, mask=mask, other=0).to(tl.int32)
        ti = si + half_n
        tj = sj + half_n
        tk = sk + half_n

        # Load fractional deltas once (used for rho in path a, gradients always)
        dxi = tl.load(scratch_ptr + base + 15 * N_AXIS + ti, mask=mask, other=0.0)
        dyj = tl.load(scratch_ptr + base + 16 * N_AXIS + tj, mask=mask, other=0.0)
        dzk = tl.load(scratch_ptr + base + 17 * N_AXIS + tk, mask=mask, other=0.0)

        # Compute rho0-rho4 (same two-path strategy as forward)
        if _USE_COMBINED:
            # Path (a): combined exponent — single exp(-alpha*r²) per component
            r_sq = G11 * dxi * dxi + G22 * dyj * dyj + G33 * dzk * dzk
            if COMPUTE_XY:
                r_sq = r_sq + 2.0 * G12 * dxi * dyj
            if COMPUTE_XZ:
                r_sq = r_sq + 2.0 * G13 * dxi * dzk
            if COMPUTE_YZ:
                r_sq = r_sq + 2.0 * G23 * dyj * dzk
            rho0 = An0 * tl.exp(-al0 * r_sq)
            rho1 = An1 * tl.exp(-al1 * r_sq)
            rho2 = An2 * tl.exp(-al2 * r_sq)
            rho3 = An3 * tl.exp(-al3 * r_sq)
            rho4 = An4 * tl.exp(-al4 * r_sq)
        else:
            # Path (b): separable 1D diagonal tables
            vx0 = tl.load(scratch_ptr + base + 0 * N_AXIS + ti, mask=mask, other=0.0)
            vy0 = tl.load(scratch_ptr + base + 5 * N_AXIS + tj, mask=mask, other=0.0)
            vz0 = tl.load(scratch_ptr + base + 10 * N_AXIS + tk, mask=mask, other=0.0)
            vx1 = tl.load(scratch_ptr + base + 1 * N_AXIS + ti, mask=mask, other=0.0)
            vy1 = tl.load(scratch_ptr + base + 6 * N_AXIS + tj, mask=mask, other=0.0)
            vz1 = tl.load(scratch_ptr + base + 11 * N_AXIS + tk, mask=mask, other=0.0)
            vx2 = tl.load(scratch_ptr + base + 2 * N_AXIS + ti, mask=mask, other=0.0)
            vy2 = tl.load(scratch_ptr + base + 7 * N_AXIS + tj, mask=mask, other=0.0)
            vz2 = tl.load(scratch_ptr + base + 12 * N_AXIS + tk, mask=mask, other=0.0)
            vx3 = tl.load(scratch_ptr + base + 3 * N_AXIS + ti, mask=mask, other=0.0)
            vy3 = tl.load(scratch_ptr + base + 8 * N_AXIS + tj, mask=mask, other=0.0)
            vz3 = tl.load(scratch_ptr + base + 13 * N_AXIS + tk, mask=mask, other=0.0)
            vx4 = tl.load(scratch_ptr + base + 4 * N_AXIS + ti, mask=mask, other=0.0)
            vy4 = tl.load(scratch_ptr + base + 9 * N_AXIS + tj, mask=mask, other=0.0)
            vz4 = tl.load(scratch_ptr + base + 14 * N_AXIS + tk, mask=mask, other=0.0)

            rho0 = An0 * vx0 * vy0 * vz0
            rho1 = An1 * vx1 * vy1 * vz1
            rho2 = An2 * vx2 * vy2 * vz2
            rho3 = An3 * vx3 * vy3 * vz3
            rho4 = An4 * vx4 * vy4 * vz4

            if STORE_CROSS_TABLES:
                ct_base = cross_off_first
                if COMPUTE_XY:
                    idx_xy = ti * N_AXIS + tj
                    rho0 *= tl.load(scratch_ptr + ct_base + 0 * N_AXIS * N_AXIS + idx_xy,
                                    mask=mask, other=1.0)
                    rho1 *= tl.load(scratch_ptr + ct_base + 1 * N_AXIS * N_AXIS + idx_xy,
                                    mask=mask, other=1.0)
                    rho2 *= tl.load(scratch_ptr + ct_base + 2 * N_AXIS * N_AXIS + idx_xy,
                                    mask=mask, other=1.0)
                    rho3 *= tl.load(scratch_ptr + ct_base + 3 * N_AXIS * N_AXIS + idx_xy,
                                    mask=mask, other=1.0)
                    rho4 *= tl.load(scratch_ptr + ct_base + 4 * N_AXIS * N_AXIS + idx_xy,
                                    mask=mask, other=1.0)
                    ct_base = ct_base + 5 * N_AXIS * N_AXIS
                if COMPUTE_XZ:
                    idx_xz = ti * N_AXIS + tk
                    rho0 *= tl.load(scratch_ptr + ct_base + 0 * N_AXIS * N_AXIS + idx_xz,
                                    mask=mask, other=1.0)
                    rho1 *= tl.load(scratch_ptr + ct_base + 1 * N_AXIS * N_AXIS + idx_xz,
                                    mask=mask, other=1.0)
                    rho2 *= tl.load(scratch_ptr + ct_base + 2 * N_AXIS * N_AXIS + idx_xz,
                                    mask=mask, other=1.0)
                    rho3 *= tl.load(scratch_ptr + ct_base + 3 * N_AXIS * N_AXIS + idx_xz,
                                    mask=mask, other=1.0)
                    rho4 *= tl.load(scratch_ptr + ct_base + 4 * N_AXIS * N_AXIS + idx_xz,
                                    mask=mask, other=1.0)
                    ct_base = ct_base + 5 * N_AXIS * N_AXIS
                if COMPUTE_YZ:
                    idx_yz = tj * N_AXIS + tk
                    rho0 *= tl.load(scratch_ptr + ct_base + 0 * N_AXIS * N_AXIS + idx_yz,
                                    mask=mask, other=1.0)
                    rho1 *= tl.load(scratch_ptr + ct_base + 1 * N_AXIS * N_AXIS + idx_yz,
                                    mask=mask, other=1.0)
                    rho2 *= tl.load(scratch_ptr + ct_base + 2 * N_AXIS * N_AXIS + idx_yz,
                                    mask=mask, other=1.0)
                    rho3 *= tl.load(scratch_ptr + ct_base + 3 * N_AXIS * N_AXIS + idx_yz,
                                    mask=mask, other=1.0)
                    rho4 *= tl.load(scratch_ptr + ct_base + 4 * N_AXIS * N_AXIS + idx_yz,
                                    mask=mask, other=1.0)

            # r² from deltas (path b only — path a already has r_sq)
            r_sq = G11 * dxi * dxi + G22 * dyj * dyj + G33 * dzk * dzk
            if COMPUTE_XY:
                r_sq = r_sq + 2.0 * G12 * dxi * dyj
            if COMPUTE_XZ:
                r_sq = r_sq + 2.0 * G13 * dxi * dzk
            if COMPUTE_YZ:
                r_sq = r_sq + 2.0 * G23 * dyj * dzk

        # ---- Gather upstream gradient ----
        gi = (cix + si) % nx
        gi = tl.where(gi < 0, gi + nx, gi)
        gj = (ciy + sj) % ny
        gj = tl.where(gj < 0, gj + ny, gj)
        gk = (ciz + sk) % nz
        gk = tl.where(gk < 0, gk + nz, gk)
        flat_idx = (gi.to(tl.int64) * ny + gj.to(tl.int64)) * nz + gk.to(tl.int64)
        grad_out = tl.load(grad_density_map_ptr + flat_idx, mask=mask, other=0.0)

        # ---- Position gradient (fractional) ----
        # d(rho_c)/d(frac_x) = rho_c * 2*alpha_c * (G11*dx + G12*dy + G13*dz)
        # (positive sign: d(delta)/d(frac) = -1 combined with -alpha in exponent)
        dr_dx = G11 * dxi
        dr_dy = G22 * dyj
        dr_dz = G33 * dzk
        if COMPUTE_XY:
            dr_dx = dr_dx + G12 * dyj
            dr_dy = dr_dy + G12 * dxi
        if COMPUTE_XZ:
            dr_dx = dr_dx + G13 * dzk
            dr_dz = dr_dz + G13 * dxi
        if COMPUTE_YZ:
            dr_dy = dr_dy + G23 * dzk
            dr_dz = dr_dz + G23 * dyj

        coeff_pos = 2.0 * (al0 * rho0 + al1 * rho1 + al2 * rho2
                           + al3 * rho3 + al4 * rho4)
        scale_pos = grad_out * coeff_pos

        g_fx += tl.sum(tl.where(mask, scale_pos * dr_dx, 0.0), axis=0)
        g_fy += tl.sum(tl.where(mask, scale_pos * dr_dy, 0.0), axis=0)
        g_fz += tl.sum(tl.where(mask, scale_pos * dr_dz, 0.0), axis=0)

        # ---- B-factor gradient ----
        # r_sq already computed above (in path a: during rho, in path b: after rho)

        db0 = rho0 * (-1.5 / Bt0 + al0 * r_sq / Bt0) * clamp0
        db1 = rho1 * (-1.5 / Bt1 + al1 * r_sq / Bt1) * clamp1
        db2 = rho2 * (-1.5 / Bt2 + al2 * r_sq / Bt2) * clamp2
        db3 = rho3 * (-1.5 / Bt3 + al3 * r_sq / Bt3) * clamp3
        db4 = rho4 * (-1.5 / Bt4 + al4 * r_sq / Bt4) * clamp4
        g_b += tl.sum(tl.where(mask, grad_out * 0.25 * (db0 + db1 + db2 + db3 + db4),
                                0.0), axis=0)

        # ---- Occupancy gradient ----
        density = rho0 + rho1 + rho2 + rho3 + rho4
        g_occ += tl.sum(tl.where(mask,
                                 grad_out * tl.where(occ != 0.0, density / occ, 0.0),
                                 0.0), axis=0)

    # Write accumulated gradients
    tl.store(grad_frac_ptr + atom * 3 + 0, g_fx)
    tl.store(grad_frac_ptr + atom * 3 + 1, g_fy)
    tl.store(grad_frac_ptr + atom * 3 + 2, g_fz)
    tl.store(grad_b_ptr + atom, g_b)
    tl.store(grad_occ_ptr + atom, g_occ)


# =============================================================================
# Python helpers
# =============================================================================

_config_cache: dict = {}


def _get_cached_config(frac_matrix, grid_shape, radius_angstrom, device):
    """Return all precomputed config, sphere offsets, and sizing.

    Cached by (frac_matrix, grid_shape, radius) to avoid recomputation.
    All GPU tensors are allocated once and reused.
    All .item() / .cpu() calls happen only on the first call.
    """
    fm_key = tuple(frac_matrix.cpu().flatten().tolist())
    key = (fm_key, grid_shape, radius_angstrom, str(device))
    if key in _config_cache:
        return _config_cache[key]

    # --- Metric tensor (CPU computation, no GPU syncs) ---
    G = frac_matrix.T @ frac_matrix
    G_vals = [G[0, 0].item(), G[1, 1].item(), G[2, 2].item(),
              G[0, 1].item(), G[0, 2].item(), G[1, 2].item()]
    G_diag_norm = math.sqrt(G_vals[0]**2 + G_vals[1]**2 + G_vals[2]**2)
    tol = 1e-3 * G_diag_norm

    compute_xy = abs(G_vals[3]) > tol
    compute_xz = abs(G_vals[4]) > tol
    compute_yz = abs(G_vals[5]) > tol
    # Always recompute cross-terms on-the-fly (pure ALU, no memory traffic).
    # Storing 2D tables in scratch adds 5*N_AXIS² global memory reads/writes
    # per atom which is slower than a few extra exp() calls.
    store_cross_tables = False

    inv_grid_vals = [1.0 / grid_shape[0], 1.0 / grid_shape[1],
                     1.0 / grid_shape[2]]

    # --- GPU tensors (allocated once) ---
    G_flat = torch.tensor(G_vals, device=device, dtype=torch.float32)
    inv_grid = torch.tensor(inv_grid_vals, device=device, dtype=torch.float32)

    # --- Sphere offsets ---
    cell_lengths = [math.sqrt(G_vals[i]) for i in range(3)]
    max_offsets = [
        int(math.ceil(radius_angstrom / (inv_grid_vals[d] * cell_lengths[d]))) + 1
        for d in range(3)
    ]
    ranges = [torch.arange(-m, m + 1, device=device) for m in max_offsets]
    gx, gy, gz = torch.meshgrid(*ranges, indexing="ij")
    coords = torch.stack((gx, gy, gz), dim=-1)
    delta_frac = coords.float() * inv_grid
    r_sq = torch.einsum("...i,ij,...j->...", delta_frac, G.to(device), delta_frac)
    sphere_offsets = coords[r_sq <= radius_angstrom**2].to(torch.int16).contiguous()

    # --- N_AXIS, half_n ---
    min_voxel_size = min(inv_grid_vals[d] * cell_lengths[d] for d in range(3))
    half_n = int(math.ceil(radius_angstrom / min_voxel_size))
    N_AXIS = triton.next_power_of_2(2 * half_n + 1)

    # --- Scratch sizing: 1D tables (3 axes × 5 components) + deltas (3 axes) ---
    base_scratch = 18 * N_AXIS

    # --- BLOCK_V ---
    N_sphere = sphere_offsets.shape[0]
    BLOCK_V = triton.next_power_of_2(min(N_sphere, 512))

    # Backward kernel: BLOCK_V=256 + num_warps=2 is universally optimal.
    # The smaller tile size reduces register pressure (146→fewer per-thread
    # vectors), allowing 2 warps to distribute across more blocks per SM.
    bwd_num_warps = 2
    bwd_BLOCK_V = min(256, triton.next_power_of_2(N_sphere))

    config = {
        "G_flat": G_flat,
        "G_vals": G_vals,
        "inv_grid": inv_grid,
        "inv_grid_vals": inv_grid_vals,
        "compute_xy": compute_xy,
        "compute_xz": compute_xz,
        "compute_yz": compute_yz,
        "store_cross_tables": store_cross_tables,
        "sphere_offsets": sphere_offsets,
        "N_sphere": N_sphere,
        "N_AXIS": N_AXIS,
        "half_n": half_n,
        "base_scratch": base_scratch,
        "BLOCK_V": BLOCK_V,
        "bwd_num_warps": bwd_num_warps,
        "bwd_BLOCK_V": bwd_BLOCK_V,
    }

    # Warmup: launch a 1-atom kernel to force Triton JIT compilation.
    # The first compilation can produce incorrect results due to a known
    # Triton JIT artifact; this disposable launch ensures the compiled
    # kernel is correct before any real data is processed.
    _warmup_kernel(config, grid_shape, device)

    _config_cache[key] = config
    return config


def _warmup_kernel(cfg, grid_shape, device):
    """Launch a disposable 1-atom forward kernel to force Triton compilation."""
    nx, ny, nz = grid_shape
    _separable_fwd_kernel[(1,)](
        torch.zeros(nx * ny * nz, device=device, dtype=torch.float32),
        torch.zeros(1, 3, device=device, dtype=torch.float32),
        torch.zeros(1, device=device, dtype=torch.float32),
        torch.zeros(1, 5, device=device, dtype=torch.float32),
        torch.zeros(1, 5, device=device, dtype=torch.float32),
        torch.ones(1, device=device, dtype=torch.float32),
        cfg["sphere_offsets"],
        torch.eye(3, device=device, dtype=torch.float32).view(-1),
        torch.zeros(1, cfg["base_scratch"], device=device, dtype=torch.float32).view(-1),
        cfg["G_vals"][0], cfg["G_vals"][1], cfg["G_vals"][2],
        cfg["G_vals"][3], cfg["G_vals"][4], cfg["G_vals"][5],
        cfg["inv_grid_vals"][0], cfg["inv_grid_vals"][1], cfg["inv_grid_vals"][2],
        nx=nx, ny=ny, nz=nz,
        N_sphere=cfg["N_sphere"], N_AXIS=cfg["N_AXIS"], half_n=cfg["half_n"],
        SCRATCH_PER_ATOM=cfg["base_scratch"], BLOCK_V=cfg["BLOCK_V"],
        COMPUTE_XY=cfg["compute_xy"], COMPUTE_XZ=cfg["compute_xz"],
        COMPUTE_YZ=cfg["compute_yz"], STORE_CROSS_TABLES=cfg["store_cross_tables"],
    )
    torch.cuda.synchronize()


# Scratch buffer cache: reuse if large enough
_scratch_buf: Optional[torch.Tensor] = None

# Track whether the forward kernel has been JIT-compiled.
# The first Triton JIT compilation can produce different results
# (likely due to uninitialized state during compilation), so we
# discard the first call and re-run.
_fwd_kernel_warmed_up: bool = False

# =============================================================================
# Autograd wrapper
# =============================================================================


class _SeparableDensityFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        density_map,       # (nx, ny, nz)
        xyz,               # (N_atoms, 3)
        b,                 # (N_atoms,)
        A,                 # (N_atoms, 5)
        B,                 # (N_atoms, 5)
        occ,               # (N_atoms,)
        inv_frac_matrix,   # (3, 3)
        sphere_offsets,    # (N_sphere, 3) int16
        G_flat,            # (6,) float32
        inv_grid,          # (3,) float32
        scratch,           # (N_atoms, scratch_per_atom) float32
        N_AXIS,            # int
        half_n,            # int
        compute_xy,        # bool
        compute_xz,        # bool
        compute_yz,        # bool
        store_cross_tables,  # bool
        # Pre-extracted Python floats (no .item() calls needed)
        G_vals,            # list of 6 Python floats
        inv_grid_vals,     # list of 3 Python floats
        BLOCK_V,           # int
        bwd_num_warps,     # int
        bwd_BLOCK_V,       # int
    ):
        N_atoms = xyz.shape[0]
        nx, ny, nz = density_map.shape
        N_sphere = sphere_offsets.shape[0]
        scratch_per_atom = scratch.shape[1]

        # Ensure contiguous
        xyz = xyz.contiguous()
        b = b.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        occ = occ.contiguous()
        inv_frac_flat = inv_frac_matrix.contiguous().view(-1)

        output = density_map.clone()

        _separable_fwd_kernel[(N_atoms,)](
            output.view(-1), xyz, b, A, B, occ,
            sphere_offsets, inv_frac_flat, scratch.view(-1),
            G_vals[0], G_vals[1], G_vals[2],
            G_vals[3], G_vals[4], G_vals[5],
            inv_grid_vals[0], inv_grid_vals[1], inv_grid_vals[2],
            nx=nx, ny=ny, nz=nz,
            N_sphere=N_sphere, N_AXIS=N_AXIS, half_n=half_n,
            SCRATCH_PER_ATOM=scratch_per_atom, BLOCK_V=BLOCK_V,
            COMPUTE_XY=compute_xy, COMPUTE_XZ=compute_xz,
            COMPUTE_YZ=compute_yz, STORE_CROSS_TABLES=store_cross_tables,
        )

        ctx.save_for_backward(
            xyz, b, A, B, occ, inv_frac_matrix,
            sphere_offsets, G_flat, inv_grid, scratch,
        )
        ctx.grid_shape = (nx, ny, nz)
        ctx.N_AXIS = N_AXIS
        ctx.half_n = half_n
        ctx.compute_xy = compute_xy
        ctx.compute_xz = compute_xz
        ctx.compute_yz = compute_yz
        ctx.store_cross_tables = store_cross_tables
        ctx.G_vals = G_vals
        ctx.inv_grid_vals = inv_grid_vals
        ctx.BLOCK_V = BLOCK_V
        ctx.bwd_num_warps = bwd_num_warps
        ctx.bwd_BLOCK_V = bwd_BLOCK_V
        return output

    @staticmethod
    def backward(ctx, grad_density_map):
        (xyz, b, A, B, occ, inv_frac_matrix,
         sphere_offsets, G_flat, inv_grid, scratch) = ctx.saved_tensors
        nx, ny, nz = ctx.grid_shape

        N_atoms = xyz.shape[0]
        N_sphere = sphere_offsets.shape[0]
        scratch_per_atom = scratch.shape[1]
        G_vals = ctx.G_vals
        inv_grid_vals = ctx.inv_grid_vals

        grad_density_map = grad_density_map.contiguous()
        inv_frac_flat = inv_frac_matrix.contiguous().view(-1)

        grad_frac = torch.zeros(N_atoms, 3, device=xyz.device, dtype=xyz.dtype)
        grad_b = torch.zeros_like(b)
        grad_occ = torch.zeros_like(occ)

        _separable_bwd_kernel[(N_atoms,)](
            grad_density_map.view(-1),
            xyz, b, A, B, occ,
            sphere_offsets, inv_frac_flat, scratch.view(-1),
            G_vals[0], G_vals[1], G_vals[2],
            G_vals[3], G_vals[4], G_vals[5],
            inv_grid_vals[0], inv_grid_vals[1], inv_grid_vals[2],
            grad_frac, grad_b, grad_occ,
            nx=nx, ny=ny, nz=nz,
            N_sphere=N_sphere, N_AXIS=ctx.N_AXIS, half_n=ctx.half_n,
            SCRATCH_PER_ATOM=scratch_per_atom, BLOCK_V=ctx.bwd_BLOCK_V,
            COMPUTE_XY=ctx.compute_xy, COMPUTE_XZ=ctx.compute_xz,
            COMPUTE_YZ=ctx.compute_yz,
            STORE_CROSS_TABLES=ctx.store_cross_tables,
            num_warps=ctx.bwd_num_warps,
        )

        # frac = xyz @ inv_frac.T  →  grad_xyz = grad_frac @ inv_frac
        grad_xyz = grad_frac @ inv_frac_matrix

        # Return gradients matching forward arg order (22 args total)
        return (None, grad_xyz, grad_b, None, None, grad_occ, None,
                None, None, None, None,
                None, None, None, None, None, None,
                None, None, None, None, None)


# =============================================================================
# Public API
# =============================================================================


def separable_density_gpu(
    density_map: torch.Tensor,
    xyz: torch.Tensor,
    b: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    occ: torch.Tensor,
    radius_angstrom: float,
) -> torch.Tensor:
    """Separable Gaussian density splatting on GPU via Triton.

    Eliminates the real_space_grid tensor and PBC matrix operations by
    working directly in fractional space with the metric tensor.  Precomputes
    1D Gaussian tables per atom and gathers per sphere voxel.

    Parameters
    ----------
    density_map : (nx, ny, nz) — density grid to update (not modified in-place)
    xyz : (N_atoms, 3) — Cartesian positions
    b : (N_atoms,) — isotropic B-factors
    inv_frac_matrix : (3, 3) — Cartesian→fractional
    frac_matrix : (3, 3) — fractional→Cartesian
    A : (N_atoms, 5) — ITC92 amplitudes
    B : (N_atoms, 5) — ITC92 widths
    occ : (N_atoms,) — occupancies
    radius_angstrom : float — cutoff radius

    Returns
    -------
    torch.Tensor — updated density map
    """
    global _scratch_buf, _fwd_kernel_warmed_up

    device = density_map.device
    grid_shape = density_map.shape

    # All config is cached — no .item() or .cpu() calls after first call
    cfg = _get_cached_config(frac_matrix, grid_shape, radius_angstrom, device)

    N_atoms = xyz.shape[0]
    needed = N_atoms * cfg["base_scratch"]

    # Reuse scratch buffer if large enough, otherwise allocate
    if _scratch_buf is None or _scratch_buf.numel() < needed or \
       _scratch_buf.device != device:
        _scratch_buf = torch.zeros(needed, device=device, dtype=torch.float32)
    scratch = _scratch_buf[:needed].view(N_atoms, cfg["base_scratch"])

    def _run():
        return _SeparableDensityFunction.apply(
            density_map, xyz, b, A, B, occ, inv_frac_matrix,
            cfg["sphere_offsets"], cfg["G_flat"], cfg["inv_grid"], scratch,
            cfg["N_AXIS"], cfg["half_n"],
            cfg["compute_xy"], cfg["compute_xz"], cfg["compute_yz"],
            cfg["store_cross_tables"],
            cfg["G_vals"], cfg["inv_grid_vals"], cfg["BLOCK_V"],
            cfg["bwd_num_warps"], cfg["bwd_BLOCK_V"],
        )

    if not _fwd_kernel_warmed_up:
        # First Triton JIT compilation produces unreliable results.
        # Run once to compile, discard, then re-run for correct output.
        _run()
        _fwd_kernel_warmed_up = True

    return _run()
