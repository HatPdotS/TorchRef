"""Variable-radius GPU density-splatting kernels (Triton).

One program per atom (``grid=(n_atoms,)``); each iterates the cubic per-axis voxel box
(sized in-kernel from ``r2cut`` and the inverse-cell metric), **decoding each voxel offset
arithmetically** from the lane index, and truncates to the per-atom sphere
(``r2 <= r2cut``). So every atom is splatted at its own ``N_sigma * sigma_eff`` radius
with no host-built work plan or offset buffer. This is the production CUDA path, in float32
and float64: every kernel computes in the dtype of its inputs (``DT``), except the voxel-index
decode, which stays float32 because its exactness bound is a float32 one.

A batched call (``(B, nx, ny, nz)`` map plus ``batch_offsets``, see
:mod:`~torchref.base.electron_density.batching`) passes each atom's member id; the kernel
then offsets its map pointer by ``member * nx * ny * nz`` (in int64), and nothing else
changes. ``HAS_BATCH`` is a compile-time flag, so an unbatched launch compiles the load
away.

Every kernel follows the truncation contract stated in ``cpu/sphere_splat.py``, so results
agree with the portable splat to float noise. Isotropic kernels carry the scalar ADP ``b``;
anisotropic ones carry the 6-component ``U`` and evaluate ``q = w^T Minv w`` with
``M = (B_g*I + 8*pi^2*U)/4`` inverted in-kernel. Backward accumulates per-atom grads with
``atomic_add``, masking out-of-sphere voxels exactly as the forward does so they
contribute neither density nor gradient.

The backward is differentiable once more: under ``create_graph=True`` it is recorded as
``_SplatGrad``, whose backward runs a second-order kernel per splat. That kernel scatters
the tangent map and gathers each atom's Hessian-vector product in one pass over the same
box, so a second derivative stays on the GPU. A third derivative raises.
"""

from __future__ import annotations

import torch

from torchref.base.electron_density.batching import check_batch, member_index
from torchref.base.targets._dispatch import first_order_only

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - guarded so CPU-only import works
    _HAVE_TRITON = False


# Fixed per-pass launch configs. Discovered via @triton.autotune (which picked
# these independently for fwd vs bwd) but baked in deterministically: runtime
# autotune mis-tunes on a shared/throttling GPU partition. Forward likes wide
# blocks; the reduction-heavy backward likes a smaller block + fewer warps.
# Re-tune if porting to different hardware.
FWD_BLOCK_V, FWD_NUM_WARPS = 512, 8
BWD_BLOCK_V, BWD_NUM_WARPS = 256, 4


if _HAVE_TRITON:
    # Triton kernels may only read globals declared as tl.constexpr(...).
    PI_1P5 = tl.constexpr(5.568327996831708)  # pi**1.5
    PI_SQ = tl.constexpr(9.869604401089358)  # pi**2
    TWO_PI_SQ = tl.constexpr(19.739208802178716)  # 2*pi**2  (anisotropic M term)

    @triton.jit
    def _sym3_inv(a, b, c, d, e, f):
        """Inverse (6 unique comps) + det of symmetric M=[[a,d,e],[d,b,f],[e,f,c]]."""
        det = a * (b * c - f * f) - d * (d * c - e * f) + e * (d * f - e * b)
        inv_det = 1.0 / det
        mi00 = (b * c - f * f) * inv_det
        mi11 = (a * c - e * e) * inv_det
        mi22 = (a * b - d * d) * inv_det
        mi01 = (e * f - d * c) * inv_det
        mi02 = (d * f - e * b) * inv_det
        mi12 = (d * e - a * f) * inv_det
        return mi00, mi11, mi22, mi01, mi02, mi12, det

    @triton.jit
    def _sym3_sandwich(p00, p11, p22, p01, p02, p12, m00, m11, m22, m01, m02, m12):
        """``tr(P M)`` and the 6 unique comps of ``P M P``, both P and M symmetric."""
        c00 = p00 * m00 + p01 * m01 + p02 * m02
        c01 = p00 * m01 + p01 * m11 + p02 * m12
        c02 = p00 * m02 + p01 * m12 + p02 * m22
        c10 = p01 * m00 + p11 * m01 + p12 * m02
        c11 = p01 * m01 + p11 * m11 + p12 * m12
        c12 = p01 * m02 + p11 * m12 + p12 * m22
        c20 = p02 * m00 + p12 * m01 + p22 * m02
        c21 = p02 * m01 + p12 * m11 + p22 * m12
        c22 = p02 * m02 + p12 * m12 + p22 * m22
        return (
            c00 + c11 + c22,
            c00 * p00 + c01 * p01 + c02 * p02,
            c10 * p01 + c11 * p11 + c12 * p12,
            c20 * p02 + c21 * p12 + c22 * p22,
            c00 * p01 + c01 * p11 + c02 * p12,
            c00 * p02 + c01 * p12 + c02 * p22,
            c10 * p02 + c11 * p12 + c12 * p22,
        )

    @triton.jit
    def _wq_grid_fwd_kernel(
        density_map_ptr,
        member_ptr,
        xyz_ptr, b_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr,
        inv_frac_ptr, frac_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr, HAS_BATCH: tl.constexpr,
    ):
        """One program per atom. Iterates the cubic box [-bh,bh]^3, decoding each
        voxel offset from the lane index, and truncates to r2 <= r2cut. Voxel
        Cartesian coord is computed analytically from the wrapped index."""
        if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
        if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
        if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)
        f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
        f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
        f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)
        # voxel step vectors (frac columns / n) + inv-frac row norms (loop-invariant)
        uax = f0 / nx; uay = f3 / nx; uaz = f6 / nx
        ubx = f1 / ny; uby = f4 / ny; ubz = f7 / ny
        ucx = f2 / nz; ucy = f5 / nz; ucz = f8 / nz
        inva = tl.sqrt(if0 * if0 + if1 * if1 + if2 * if2)
        invb = tl.sqrt(if3 * if3 + if4 * if4 + if5 * if5)
        invc = tl.sqrt(if6 * if6 + if7 * if7 + if8 * if8)
        v_lane = tl.arange(0, BLOCK_V)

        atom = tl.program_id(0)
        DT = xyz_ptr.dtype.element_ty
        dm_ptr = density_map_ptr
        if HAS_BATCH:
            dm_ptr += tl.load(member_ptr + atom).to(tl.int64) * (nx * ny * nz)
        r2cut = tl.load(r2cut_ptr + atom)
        r = tl.sqrt(r2cut)
        # per-axis bounding box of the cartesian r-sphere in index space (triclinic-correct:
        # max|off_a| = n_a * r * ||inv_frac_row_a||). The sphere mask culls the corners.
        bhx = tl.extra.cuda.libdevice.ceil(r * nx * inva).to(tl.int32)
        bhy = tl.extra.cuda.libdevice.ceil(r * ny * invb).to(tl.int32)
        bhz = tl.extra.cuda.libdevice.ceil(r * nz * invc).to(tl.int32)
        sx_ = 2 * bhx + 1; sy_ = 2 * bhy + 1; sz_ = 2 * bhz + 1
        syz = sy_ * sz_
        n = sx_ * syz
        # Float reciprocals for the decode, avoiding the integer divide. Decoding at the
        # half index is exact for every box side up to 123 voxels with a reciprocal up to
        # 2 ulp off (175 if correctly rounded), checked exhaustively in float32 emulation.
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)

        b_iso = tl.load(b_ptr + atom)
        occ = tl.load(occ_ptr + atom)
        ax = tl.load(xyz_ptr + atom * 3 + 0)
        ay = tl.load(xyz_ptr + atom * 3 + 1)
        az = tl.load(xyz_ptr + atom * 3 + 2)
        A0 = tl.load(A_ptr + atom * 5 + 0); A1 = tl.load(A_ptr + atom * 5 + 1)
        A2 = tl.load(A_ptr + atom * 5 + 2); A3 = tl.load(A_ptr + atom * 5 + 3)
        A4 = tl.load(A_ptr + atom * 5 + 4)
        B0 = tl.load(B_ptr + atom * 5 + 0); B1 = tl.load(B_ptr + atom * 5 + 1)
        B2 = tl.load(B_ptr + atom * 5 + 2); B3 = tl.load(B_ptr + atom * 5 + 3)
        B4 = tl.load(B_ptr + atom * 5 + 4)

        Bt0 = tl.maximum((B0 + b_iso) * 0.25, 0.1)
        Bt1 = tl.maximum((B1 + b_iso) * 0.25, 0.1)
        Bt2 = tl.maximum((B2 + b_iso) * 0.25, 0.1)
        Bt3 = tl.maximum((B3 + b_iso) * 0.25, 0.1)
        Bt4 = tl.maximum((B4 + b_iso) * 0.25, 0.1)
        An0 = A0 * occ * PI_1P5 / (Bt0 * tl.sqrt(Bt0))
        An1 = A1 * occ * PI_1P5 / (Bt1 * tl.sqrt(Bt1))
        An2 = A2 * occ * PI_1P5 / (Bt2 * tl.sqrt(Bt2))
        An3 = A3 * occ * PI_1P5 / (Bt3 * tl.sqrt(Bt3))
        An4 = A4 * occ * PI_1P5 / (Bt4 * tl.sqrt(Bt4))

        frac_x = ax * if0 + ay * if1 + az * if2
        frac_y = ax * if3 + ay * if4 + az * if5
        frac_z = ax * if6 + ay * if7 + az * if8
        frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
        frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
        frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)
        cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
        ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
        ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)
        # atom sub-voxel residual: w0 = frac @ (frac_wrapped - ci/n) (min-image displacement base)
        rx = frac_x - cix.to(DT) / nx
        ry = frac_y - ciy.to(DT) / ny
        rz = frac_z - ciz.to(DT) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = ((v.to(tl.float32) + 0.5) * inv_syz).to(tl.int32)  # floor via trunc
            rem = v - ix * syz
            iy = ((rem.to(tl.float32) + 0.5) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(DT); ofyf = off_y.to(DT); ofzf = off_z.to(DT)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # Write index, wrapped fully (the box may span the cell more than once) as
            # wrap does in the CPU kernel; % takes the dividend's sign, hence the add.
            vix = (cix + off_x) % nx; vix = vix + tl.where(vix < 0, nx, 0)
            viy = (ciy + off_y) % ny; viy = viy + tl.where(viy < 0, ny, 0)
            viz = (ciz + off_z) % nz; viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)
            density = (
                An0 * tl.exp(-PI_SQ * r2 / Bt0)
                + An1 * tl.exp(-PI_SQ * r2 / Bt1)
                + An2 * tl.exp(-PI_SQ * r2 / Bt2)
                + An3 * tl.exp(-PI_SQ * r2 / Bt3)
                + An4 * tl.exp(-PI_SQ * r2 / Bt4)
            )
            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            tl.atomic_add(dm_ptr + dm_flat, density, mask=wmask)
            v_start += BLOCK_V

    @triton.jit
    def _wq_grid_bwd_kernel(
        grad_density_map_ptr,
        member_ptr,
        xyz_ptr, b_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr,
        inv_frac_ptr, frac_ptr,
        grad_xyz_ptr, grad_b_ptr, grad_occ_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr, HAS_BATCH: tl.constexpr,
    ):
        """One program per atom backward; grads atomic_add into per-atom buffers.
        Out-of-sphere voxels (r2 > r2cut) are masked exactly as in the forward."""
        if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
        if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
        if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)
        f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
        f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
        f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)
        # voxel step vectors (frac columns / n) + inv-frac row norms (loop-invariant)
        uax = f0 / nx; uay = f3 / nx; uaz = f6 / nx
        ubx = f1 / ny; uby = f4 / ny; ubz = f7 / ny
        ucx = f2 / nz; ucy = f5 / nz; ucz = f8 / nz
        inva = tl.sqrt(if0 * if0 + if1 * if1 + if2 * if2)
        invb = tl.sqrt(if3 * if3 + if4 * if4 + if5 * if5)
        invc = tl.sqrt(if6 * if6 + if7 * if7 + if8 * if8)
        v_lane = tl.arange(0, BLOCK_V)

        atom = tl.program_id(0)
        DT = xyz_ptr.dtype.element_ty
        dm_ptr = grad_density_map_ptr
        if HAS_BATCH:
            dm_ptr += tl.load(member_ptr + atom).to(tl.int64) * (nx * ny * nz)
        r2cut = tl.load(r2cut_ptr + atom)
        r = tl.sqrt(r2cut)
        # per-axis bounding box of the cartesian r-sphere in index space (triclinic-correct:
        # max|off_a| = n_a * r * ||inv_frac_row_a||). The sphere mask culls the corners.
        bhx = tl.extra.cuda.libdevice.ceil(r * nx * inva).to(tl.int32)
        bhy = tl.extra.cuda.libdevice.ceil(r * ny * invb).to(tl.int32)
        bhz = tl.extra.cuda.libdevice.ceil(r * nz * invc).to(tl.int32)
        sx_ = 2 * bhx + 1; sy_ = 2 * bhy + 1; sz_ = 2 * bhz + 1
        syz = sy_ * sz_
        n = sx_ * syz
        # Float reciprocals for the decode, avoiding the integer divide. Decoding at the
        # half index is exact for every box side up to 123 voxels with a reciprocal up to
        # 2 ulp off (175 if correctly rounded), checked exhaustively in float32 emulation.
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)

        b_iso = tl.load(b_ptr + atom)
        occ = tl.load(occ_ptr + atom)
        ax = tl.load(xyz_ptr + atom * 3 + 0)
        ay = tl.load(xyz_ptr + atom * 3 + 1)
        az = tl.load(xyz_ptr + atom * 3 + 2)
        A0 = tl.load(A_ptr + atom * 5 + 0); A1 = tl.load(A_ptr + atom * 5 + 1)
        A2 = tl.load(A_ptr + atom * 5 + 2); A3 = tl.load(A_ptr + atom * 5 + 3)
        A4 = tl.load(A_ptr + atom * 5 + 4)
        B0 = tl.load(B_ptr + atom * 5 + 0); B1 = tl.load(B_ptr + atom * 5 + 1)
        B2 = tl.load(B_ptr + atom * 5 + 2); B3 = tl.load(B_ptr + atom * 5 + 3)
        B4 = tl.load(B_ptr + atom * 5 + 4)

        Bt0 = tl.maximum((B0 + b_iso) * 0.25, 0.1)
        Bt1 = tl.maximum((B1 + b_iso) * 0.25, 0.1)
        Bt2 = tl.maximum((B2 + b_iso) * 0.25, 0.1)
        Bt3 = tl.maximum((B3 + b_iso) * 0.25, 0.1)
        Bt4 = tl.maximum((B4 + b_iso) * 0.25, 0.1)
        clamp0 = ((B0 + b_iso) * 0.25 > 0.1).to(DT)
        clamp1 = ((B1 + b_iso) * 0.25 > 0.1).to(DT)
        clamp2 = ((B2 + b_iso) * 0.25 > 0.1).to(DT)
        clamp3 = ((B3 + b_iso) * 0.25 > 0.1).to(DT)
        clamp4 = ((B4 + b_iso) * 0.25 > 0.1).to(DT)
        # Occupancy-free amplitudes: the density is linear in occ, so its gradient is
        # the occ = 1 density, and occ scales only the other gradients.
        An0 = A0 * PI_1P5 / (Bt0 * tl.sqrt(Bt0))
        An1 = A1 * PI_1P5 / (Bt1 * tl.sqrt(Bt1))
        An2 = A2 * PI_1P5 / (Bt2 * tl.sqrt(Bt2))
        An3 = A3 * PI_1P5 / (Bt3 * tl.sqrt(Bt3))
        An4 = A4 * PI_1P5 / (Bt4 * tl.sqrt(Bt4))

        frac_x = ax * if0 + ay * if1 + az * if2
        frac_y = ax * if3 + ay * if4 + az * if5
        frac_z = ax * if6 + ay * if7 + az * if8
        frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
        frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
        frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)
        cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
        ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
        ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)
        # atom sub-voxel residual: w0 = frac @ (frac_wrapped - ci/n) (min-image displacement base)
        rx = frac_x - cix.to(DT) / nx
        ry = frac_y - ciy.to(DT) / ny
        rz = frac_z - ciz.to(DT) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        # Per-lane partial sums, reduced once after the loop. A scalar 0.0 start would be
        # a float32 loop-carried value, which a float64 sum cannot be assigned to.
        g_ax = tl.zeros([BLOCK_V], dtype=DT); g_ay = tl.zeros([BLOCK_V], dtype=DT)
        g_az = tl.zeros([BLOCK_V], dtype=DT); g_b = tl.zeros([BLOCK_V], dtype=DT)
        g_occ = tl.zeros([BLOCK_V], dtype=DT)
        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = ((v.to(tl.float32) + 0.5) * inv_syz).to(tl.int32)  # floor via trunc
            rem = v - ix * syz
            iy = ((rem.to(tl.float32) + 0.5) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(DT); ofyf = off_y.to(DT); ofzf = off_z.to(DT)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # Write index, wrapped fully (the box may span the cell more than once) as
            # wrap does in the CPU kernel; % takes the dividend's sign, hence the add.
            vix = (cix + off_x) % nx; vix = vix + tl.where(vix < 0, nx, 0)
            viy = (ciy + off_y) % ny; viy = viy + tl.where(viy < 0, ny, 0)
            viz = (ciz + off_z) % nz; viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)

            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            grad_out = tl.load(dm_ptr + dm_flat, mask=vmask, other=0.0)
            e0 = tl.exp(-PI_SQ * r2 / Bt0); e1 = tl.exp(-PI_SQ * r2 / Bt1)
            e2 = tl.exp(-PI_SQ * r2 / Bt2); e3 = tl.exp(-PI_SQ * r2 / Bt3)
            e4 = tl.exp(-PI_SQ * r2 / Bt4)
            coeff_xyz = (
                An0 * e0 / Bt0 + An1 * e1 / Bt1 + An2 * e2 / Bt2
                + An3 * e3 / Bt3 + An4 * e4 / Bt4
            )
            scale_xyz = grad_out * 2.0 * PI_SQ * coeff_xyz
            g_ax += tl.where(wmask, scale_xyz * wx, 0.0)
            g_ay += tl.where(wmask, scale_xyz * wy, 0.0)
            g_az += tl.where(wmask, scale_xyz * wz, 0.0)
            db0 = An0 * e0 * (-1.5 / Bt0 + PI_SQ * r2 / (Bt0 * Bt0)) * clamp0
            db1 = An1 * e1 * (-1.5 / Bt1 + PI_SQ * r2 / (Bt1 * Bt1)) * clamp1
            db2 = An2 * e2 * (-1.5 / Bt2 + PI_SQ * r2 / (Bt2 * Bt2)) * clamp2
            db3 = An3 * e3 * (-1.5 / Bt3 + PI_SQ * r2 / (Bt3 * Bt3)) * clamp3
            db4 = An4 * e4 * (-1.5 / Bt4 + PI_SQ * r2 / (Bt4 * Bt4)) * clamp4
            g_b += tl.where(wmask, grad_out * 0.25 * (db0 + db1 + db2 + db3 + db4), 0.0)
            dens = An0 * e0 + An1 * e1 + An2 * e2 + An3 * e3 + An4 * e4
            g_occ += tl.where(wmask, grad_out * dens, 0.0)
            v_start += BLOCK_V

        tl.atomic_add(grad_xyz_ptr + atom * 3 + 0, occ * tl.sum(g_ax, axis=0))
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 1, occ * tl.sum(g_ay, axis=0))
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 2, occ * tl.sum(g_az, axis=0))
        tl.atomic_add(grad_b_ptr + atom, occ * tl.sum(g_b, axis=0))
        tl.atomic_add(grad_occ_ptr + atom, tl.sum(g_occ, axis=0))

    # ====================================================================
    # ANISOTROPIC variable-radius kernels: same skeleton, scalar b replaced
    # by the 6-component U, density = An_g*exp(-pi^2 * w^T Minv w) with
    # M_g = (B_g*I + 8*pi^2*U)/4. (Math from the reference aniso kernel.)
    # ====================================================================
    @triton.jit
    def _wq_grid_aniso_fwd_kernel(
        density_map_ptr,
        member_ptr,
        xyz_ptr, u_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr,
        inv_frac_ptr, frac_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr, HAS_BATCH: tl.constexpr,
    ):
        """Per-Gaussian M = (B_g*I + 8*pi^2*U)/4 is built + inverted ONCE per atom
        (manual 5-fold unroll: Triton @jit has no lists) before the voxel loop;
        the loop decodes the cubic box, truncates to r2<=r2cut, and evaluates
        density += An_g*exp(-pi^2 * w^T Minv w)."""
        if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
        if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
        if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)
        f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
        f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
        f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)
        # voxel step vectors (frac columns / n) + inv-frac row norms (loop-invariant)
        uax = f0 / nx; uay = f3 / nx; uaz = f6 / nx
        ubx = f1 / ny; uby = f4 / ny; ubz = f7 / ny
        ucx = f2 / nz; ucy = f5 / nz; ucz = f8 / nz
        inva = tl.sqrt(if0 * if0 + if1 * if1 + if2 * if2)
        invb = tl.sqrt(if3 * if3 + if4 * if4 + if5 * if5)
        invc = tl.sqrt(if6 * if6 + if7 * if7 + if8 * if8)
        v_lane = tl.arange(0, BLOCK_V)

        atom = tl.program_id(0)
        DT = xyz_ptr.dtype.element_ty
        dm_ptr = density_map_ptr
        if HAS_BATCH:
            dm_ptr += tl.load(member_ptr + atom).to(tl.int64) * (nx * ny * nz)
        r2cut = tl.load(r2cut_ptr + atom)
        r = tl.sqrt(r2cut)
        # per-axis bounding box of the cartesian r-sphere in index space (triclinic-correct:
        # max|off_a| = n_a * r * ||inv_frac_row_a||). The sphere mask culls the corners.
        bhx = tl.extra.cuda.libdevice.ceil(r * nx * inva).to(tl.int32)
        bhy = tl.extra.cuda.libdevice.ceil(r * ny * invb).to(tl.int32)
        bhz = tl.extra.cuda.libdevice.ceil(r * nz * invc).to(tl.int32)
        sx_ = 2 * bhx + 1; sy_ = 2 * bhy + 1; sz_ = 2 * bhz + 1
        syz = sy_ * sz_
        n = sx_ * syz
        # Float reciprocals for the decode, avoiding the integer divide. Decoding at the
        # half index is exact for every box side up to 123 voxels with a reciprocal up to
        # 2 ulp off (175 if correctly rounded), checked exhaustively in float32 emulation.
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)

        occ = tl.load(occ_ptr + atom)
        ax = tl.load(xyz_ptr + atom * 3 + 0)
        ay = tl.load(xyz_ptr + atom * 3 + 1)
        az = tl.load(xyz_ptr + atom * 3 + 2)
        u11 = tl.load(u_ptr + atom * 6 + 0); u22 = tl.load(u_ptr + atom * 6 + 1)
        u33 = tl.load(u_ptr + atom * 6 + 2); u12 = tl.load(u_ptr + atom * 6 + 3)
        u13 = tl.load(u_ptr + atom * 6 + 4); u23 = tl.load(u_ptr + atom * 6 + 5)
        dd = TWO_PI_SQ * u12; ee = TWO_PI_SQ * u13; ff = TWO_PI_SQ * u23
        du = TWO_PI_SQ * u11; dv = TWO_PI_SQ * u22; dw = TWO_PI_SQ * u33

        p00_0, p11_0, p22_0, p01_0, p02_0, p12_0, dt0 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 0) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 0) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 0) * 0.25 + dw, dd, ee, ff)
        p00_1, p11_1, p22_1, p01_1, p02_1, p12_1, dt1 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 1) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 1) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 1) * 0.25 + dw, dd, ee, ff)
        p00_2, p11_2, p22_2, p01_2, p02_2, p12_2, dt2 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 2) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 2) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 2) * 0.25 + dw, dd, ee, ff)
        p00_3, p11_3, p22_3, p01_3, p02_3, p12_3, dt3 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 3) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 3) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 3) * 0.25 + dw, dd, ee, ff)
        p00_4, p11_4, p22_4, p01_4, p02_4, p12_4, dt4 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 4) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 4) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 4) * 0.25 + dw, dd, ee, ff)
        oc = occ * PI_1P5
        An0 = tl.load(A_ptr + atom * 5 + 0) * oc / tl.sqrt(tl.maximum(dt0, 1e-10))
        An1 = tl.load(A_ptr + atom * 5 + 1) * oc / tl.sqrt(tl.maximum(dt1, 1e-10))
        An2 = tl.load(A_ptr + atom * 5 + 2) * oc / tl.sqrt(tl.maximum(dt2, 1e-10))
        An3 = tl.load(A_ptr + atom * 5 + 3) * oc / tl.sqrt(tl.maximum(dt3, 1e-10))
        An4 = tl.load(A_ptr + atom * 5 + 4) * oc / tl.sqrt(tl.maximum(dt4, 1e-10))

        frac_x = ax * if0 + ay * if1 + az * if2
        frac_y = ax * if3 + ay * if4 + az * if5
        frac_z = ax * if6 + ay * if7 + az * if8
        frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
        frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
        frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)
        cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
        ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
        ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)
        # atom sub-voxel residual: w0 = frac @ (frac_wrapped - ci/n) (min-image displacement base)
        rx = frac_x - cix.to(DT) / nx
        ry = frac_y - ciy.to(DT) / ny
        rz = frac_z - ciz.to(DT) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = ((v.to(tl.float32) + 0.5) * inv_syz).to(tl.int32)  # floor via trunc
            rem = v - ix * syz
            iy = ((rem.to(tl.float32) + 0.5) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(DT); ofyf = off_y.to(DT); ofzf = off_z.to(DT)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # Write index, wrapped fully (the box may span the cell more than once) as
            # wrap does in the CPU kernel; % takes the dividend's sign, hence the add.
            vix = (cix + off_x) % nx; vix = vix + tl.where(vix < 0, nx, 0)
            viy = (ciy + off_y) % ny; viy = viy + tl.where(viy < 0, ny, 0)
            viz = (ciz + off_z) % nz; viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)
            xx = wx * wx; yy = wy * wy; zz = wz * wz
            xy = wx * wy; xz = wx * wz; yz = wy * wz
            q0 = p00_0 * xx + p11_0 * yy + p22_0 * zz + 2.0 * (p01_0 * xy + p02_0 * xz + p12_0 * yz)
            q1 = p00_1 * xx + p11_1 * yy + p22_1 * zz + 2.0 * (p01_1 * xy + p02_1 * xz + p12_1 * yz)
            q2 = p00_2 * xx + p11_2 * yy + p22_2 * zz + 2.0 * (p01_2 * xy + p02_2 * xz + p12_2 * yz)
            q3 = p00_3 * xx + p11_3 * yy + p22_3 * zz + 2.0 * (p01_3 * xy + p02_3 * xz + p12_3 * yz)
            q4 = p00_4 * xx + p11_4 * yy + p22_4 * zz + 2.0 * (p01_4 * xy + p02_4 * xz + p12_4 * yz)
            density = (An0 * tl.exp(-PI_SQ * q0) + An1 * tl.exp(-PI_SQ * q1)
                       + An2 * tl.exp(-PI_SQ * q2) + An3 * tl.exp(-PI_SQ * q3)
                       + An4 * tl.exp(-PI_SQ * q4))
            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            tl.atomic_add(dm_ptr + dm_flat, density, mask=wmask)
            v_start += BLOCK_V

    @triton.jit
    def _wq_grid_aniso_bwd_kernel(
        grad_density_map_ptr,
        member_ptr,
        xyz_ptr, u_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr,
        inv_frac_ptr, frac_ptr,
        grad_xyz_ptr, grad_u_ptr, grad_occ_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr, HAS_BATCH: tl.constexpr,
    ):
        """Anisotropic backward. v = Minv w; grad_xyz = sum_g 2*pi^2*dg*v_g;
        grad_U via S = -0.5*Minv + pi^2 v v^T (diag *2pi^2, offdiag *4pi^2);
        grad_occ = sum_g dg at occ = 1. Out-of-sphere voxels masked as in the forward."""
        if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
        if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
        if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)
        f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
        f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
        f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)
        # voxel step vectors (frac columns / n) + inv-frac row norms (loop-invariant)
        uax = f0 / nx; uay = f3 / nx; uaz = f6 / nx
        ubx = f1 / ny; uby = f4 / ny; ubz = f7 / ny
        ucx = f2 / nz; ucy = f5 / nz; ucz = f8 / nz
        inva = tl.sqrt(if0 * if0 + if1 * if1 + if2 * if2)
        invb = tl.sqrt(if3 * if3 + if4 * if4 + if5 * if5)
        invc = tl.sqrt(if6 * if6 + if7 * if7 + if8 * if8)
        v_lane = tl.arange(0, BLOCK_V)

        atom = tl.program_id(0)
        DT = xyz_ptr.dtype.element_ty
        dm_ptr = grad_density_map_ptr
        if HAS_BATCH:
            dm_ptr += tl.load(member_ptr + atom).to(tl.int64) * (nx * ny * nz)
        r2cut = tl.load(r2cut_ptr + atom)
        r = tl.sqrt(r2cut)
        # per-axis bounding box of the cartesian r-sphere in index space (triclinic-correct:
        # max|off_a| = n_a * r * ||inv_frac_row_a||). The sphere mask culls the corners.
        bhx = tl.extra.cuda.libdevice.ceil(r * nx * inva).to(tl.int32)
        bhy = tl.extra.cuda.libdevice.ceil(r * ny * invb).to(tl.int32)
        bhz = tl.extra.cuda.libdevice.ceil(r * nz * invc).to(tl.int32)
        sx_ = 2 * bhx + 1; sy_ = 2 * bhy + 1; sz_ = 2 * bhz + 1
        syz = sy_ * sz_
        n = sx_ * syz
        # Float reciprocals for the decode, avoiding the integer divide. Decoding at the
        # half index is exact for every box side up to 123 voxels with a reciprocal up to
        # 2 ulp off (175 if correctly rounded), checked exhaustively in float32 emulation.
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)

        occ = tl.load(occ_ptr + atom)
        ax = tl.load(xyz_ptr + atom * 3 + 0)
        ay = tl.load(xyz_ptr + atom * 3 + 1)
        az = tl.load(xyz_ptr + atom * 3 + 2)
        u11 = tl.load(u_ptr + atom * 6 + 0); u22 = tl.load(u_ptr + atom * 6 + 1)
        u33 = tl.load(u_ptr + atom * 6 + 2); u12 = tl.load(u_ptr + atom * 6 + 3)
        u13 = tl.load(u_ptr + atom * 6 + 4); u23 = tl.load(u_ptr + atom * 6 + 5)
        dd = TWO_PI_SQ * u12; ee = TWO_PI_SQ * u13; ff = TWO_PI_SQ * u23
        du = TWO_PI_SQ * u11; dv = TWO_PI_SQ * u22; dw = TWO_PI_SQ * u33

        p00_0, p11_0, p22_0, p01_0, p02_0, p12_0, dt0 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 0) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 0) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 0) * 0.25 + dw, dd, ee, ff)
        p00_1, p11_1, p22_1, p01_1, p02_1, p12_1, dt1 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 1) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 1) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 1) * 0.25 + dw, dd, ee, ff)
        p00_2, p11_2, p22_2, p01_2, p02_2, p12_2, dt2 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 2) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 2) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 2) * 0.25 + dw, dd, ee, ff)
        p00_3, p11_3, p22_3, p01_3, p02_3, p12_3, dt3 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 3) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 3) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 3) * 0.25 + dw, dd, ee, ff)
        p00_4, p11_4, p22_4, p01_4, p02_4, p12_4, dt4 = _sym3_inv(
            tl.load(B_ptr + atom * 5 + 4) * 0.25 + du,
            tl.load(B_ptr + atom * 5 + 4) * 0.25 + dv,
            tl.load(B_ptr + atom * 5 + 4) * 0.25 + dw, dd, ee, ff)
        # Occupancy-free amplitudes, as in _wq_grid_bwd_kernel.
        An0 = tl.load(A_ptr + atom * 5 + 0) * PI_1P5 / tl.sqrt(tl.maximum(dt0, 1e-10))
        An1 = tl.load(A_ptr + atom * 5 + 1) * PI_1P5 / tl.sqrt(tl.maximum(dt1, 1e-10))
        An2 = tl.load(A_ptr + atom * 5 + 2) * PI_1P5 / tl.sqrt(tl.maximum(dt2, 1e-10))
        An3 = tl.load(A_ptr + atom * 5 + 3) * PI_1P5 / tl.sqrt(tl.maximum(dt3, 1e-10))
        An4 = tl.load(A_ptr + atom * 5 + 4) * PI_1P5 / tl.sqrt(tl.maximum(dt4, 1e-10))

        frac_x = ax * if0 + ay * if1 + az * if2
        frac_y = ax * if3 + ay * if4 + az * if5
        frac_z = ax * if6 + ay * if7 + az * if8
        frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
        frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
        frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)
        cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
        ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
        ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)
        # atom sub-voxel residual: w0 = frac @ (frac_wrapped - ci/n) (min-image displacement base)
        rx = frac_x - cix.to(DT) / nx
        ry = frac_y - ciy.to(DT) / ny
        rz = frac_z - ciz.to(DT) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        # Per-lane partial sums, as in _wq_grid_bwd_kernel.
        g_ax = tl.zeros([BLOCK_V], dtype=DT); g_ay = tl.zeros([BLOCK_V], dtype=DT)
        g_az = tl.zeros([BLOCK_V], dtype=DT); g_occ = tl.zeros([BLOCK_V], dtype=DT)
        g_u0 = tl.zeros([BLOCK_V], dtype=DT); g_u1 = tl.zeros([BLOCK_V], dtype=DT)
        g_u2 = tl.zeros([BLOCK_V], dtype=DT); g_u3 = tl.zeros([BLOCK_V], dtype=DT)
        g_u4 = tl.zeros([BLOCK_V], dtype=DT); g_u5 = tl.zeros([BLOCK_V], dtype=DT)
        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = ((v.to(tl.float32) + 0.5) * inv_syz).to(tl.int32)  # floor via trunc
            rem = v - ix * syz
            iy = ((rem.to(tl.float32) + 0.5) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(DT); ofyf = off_y.to(DT); ofzf = off_z.to(DT)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # Write index, wrapped fully (the box may span the cell more than once) as
            # wrap does in the CPU kernel; % takes the dividend's sign, hence the add.
            vix = (cix + off_x) % nx; vix = vix + tl.where(vix < 0, nx, 0)
            viy = (ciy + off_y) % ny; viy = viy + tl.where(viy < 0, ny, 0)
            viz = (ciz + off_z) % nz; viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)

            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            grad_out = tl.load(dm_ptr + dm_flat, mask=vmask, other=0.0)

            vx0 = p00_0 * wx + p01_0 * wy + p02_0 * wz
            vy0 = p01_0 * wx + p11_0 * wy + p12_0 * wz
            vz0 = p02_0 * wx + p12_0 * wy + p22_0 * wz
            dg0 = An0 * tl.exp(-PI_SQ * (wx * vx0 + wy * vy0 + wz * vz0))
            vx1 = p00_1 * wx + p01_1 * wy + p02_1 * wz
            vy1 = p01_1 * wx + p11_1 * wy + p12_1 * wz
            vz1 = p02_1 * wx + p12_1 * wy + p22_1 * wz
            dg1 = An1 * tl.exp(-PI_SQ * (wx * vx1 + wy * vy1 + wz * vz1))
            vx2 = p00_2 * wx + p01_2 * wy + p02_2 * wz
            vy2 = p01_2 * wx + p11_2 * wy + p12_2 * wz
            vz2 = p02_2 * wx + p12_2 * wy + p22_2 * wz
            dg2 = An2 * tl.exp(-PI_SQ * (wx * vx2 + wy * vy2 + wz * vz2))
            vx3 = p00_3 * wx + p01_3 * wy + p02_3 * wz
            vy3 = p01_3 * wx + p11_3 * wy + p12_3 * wz
            vz3 = p02_3 * wx + p12_3 * wy + p22_3 * wz
            dg3 = An3 * tl.exp(-PI_SQ * (wx * vx3 + wy * vy3 + wz * vz3))
            vx4 = p00_4 * wx + p01_4 * wy + p02_4 * wz
            vy4 = p01_4 * wx + p11_4 * wy + p12_4 * wz
            vz4 = p02_4 * wx + p12_4 * wy + p22_4 * wz
            dg4 = An4 * tl.exp(-PI_SQ * (wx * vx4 + wy * vy4 + wz * vz4))
            dens = dg0 + dg1 + dg2 + dg3 + dg4
            sx_acc = dg0 * vx0 + dg1 * vx1 + dg2 * vx2 + dg3 * vx3 + dg4 * vx4
            sy_acc = dg0 * vy0 + dg1 * vy1 + dg2 * vy2 + dg3 * vy3 + dg4 * vy4
            sz_acc = dg0 * vz0 + dg1 * vz1 + dg2 * vz2 + dg3 * vz3 + dg4 * vz4
            gu0_l = (dg0 * (-0.5 * p00_0 + PI_SQ * vx0 * vx0)
                     + dg1 * (-0.5 * p00_1 + PI_SQ * vx1 * vx1)
                     + dg2 * (-0.5 * p00_2 + PI_SQ * vx2 * vx2)
                     + dg3 * (-0.5 * p00_3 + PI_SQ * vx3 * vx3)
                     + dg4 * (-0.5 * p00_4 + PI_SQ * vx4 * vx4))
            gu1_l = (dg0 * (-0.5 * p11_0 + PI_SQ * vy0 * vy0)
                     + dg1 * (-0.5 * p11_1 + PI_SQ * vy1 * vy1)
                     + dg2 * (-0.5 * p11_2 + PI_SQ * vy2 * vy2)
                     + dg3 * (-0.5 * p11_3 + PI_SQ * vy3 * vy3)
                     + dg4 * (-0.5 * p11_4 + PI_SQ * vy4 * vy4))
            gu2_l = (dg0 * (-0.5 * p22_0 + PI_SQ * vz0 * vz0)
                     + dg1 * (-0.5 * p22_1 + PI_SQ * vz1 * vz1)
                     + dg2 * (-0.5 * p22_2 + PI_SQ * vz2 * vz2)
                     + dg3 * (-0.5 * p22_3 + PI_SQ * vz3 * vz3)
                     + dg4 * (-0.5 * p22_4 + PI_SQ * vz4 * vz4))
            gu3_l = (dg0 * (-0.5 * p01_0 + PI_SQ * vx0 * vy0)
                     + dg1 * (-0.5 * p01_1 + PI_SQ * vx1 * vy1)
                     + dg2 * (-0.5 * p01_2 + PI_SQ * vx2 * vy2)
                     + dg3 * (-0.5 * p01_3 + PI_SQ * vx3 * vy3)
                     + dg4 * (-0.5 * p01_4 + PI_SQ * vx4 * vy4))
            gu4_l = (dg0 * (-0.5 * p02_0 + PI_SQ * vx0 * vz0)
                     + dg1 * (-0.5 * p02_1 + PI_SQ * vx1 * vz1)
                     + dg2 * (-0.5 * p02_2 + PI_SQ * vx2 * vz2)
                     + dg3 * (-0.5 * p02_3 + PI_SQ * vx3 * vz3)
                     + dg4 * (-0.5 * p02_4 + PI_SQ * vx4 * vz4))
            gu5_l = (dg0 * (-0.5 * p12_0 + PI_SQ * vy0 * vz0)
                     + dg1 * (-0.5 * p12_1 + PI_SQ * vy1 * vz1)
                     + dg2 * (-0.5 * p12_2 + PI_SQ * vy2 * vz2)
                     + dg3 * (-0.5 * p12_3 + PI_SQ * vy3 * vz3)
                     + dg4 * (-0.5 * p12_4 + PI_SQ * vy4 * vz4))
            sxyz = grad_out * 2.0 * PI_SQ
            g_ax += tl.where(wmask, sxyz * sx_acc, 0.0)
            g_ay += tl.where(wmask, sxyz * sy_acc, 0.0)
            g_az += tl.where(wmask, sxyz * sz_acc, 0.0)
            g_u0 += tl.where(wmask, grad_out * 2.0 * PI_SQ * gu0_l, 0.0)
            g_u1 += tl.where(wmask, grad_out * 2.0 * PI_SQ * gu1_l, 0.0)
            g_u2 += tl.where(wmask, grad_out * 2.0 * PI_SQ * gu2_l, 0.0)
            g_u3 += tl.where(wmask, grad_out * 4.0 * PI_SQ * gu3_l, 0.0)
            g_u4 += tl.where(wmask, grad_out * 4.0 * PI_SQ * gu4_l, 0.0)
            g_u5 += tl.where(wmask, grad_out * 4.0 * PI_SQ * gu5_l, 0.0)
            g_occ += tl.where(wmask, grad_out * dens, 0.0)
            v_start += BLOCK_V

        tl.atomic_add(grad_xyz_ptr + atom * 3 + 0, occ * tl.sum(g_ax, axis=0))
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 1, occ * tl.sum(g_ay, axis=0))
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 2, occ * tl.sum(g_az, axis=0))
        tl.atomic_add(grad_u_ptr + atom * 6 + 0, occ * tl.sum(g_u0, axis=0))
        tl.atomic_add(grad_u_ptr + atom * 6 + 1, occ * tl.sum(g_u1, axis=0))
        tl.atomic_add(grad_u_ptr + atom * 6 + 2, occ * tl.sum(g_u2, axis=0))
        tl.atomic_add(grad_u_ptr + atom * 6 + 3, occ * tl.sum(g_u3, axis=0))
        tl.atomic_add(grad_u_ptr + atom * 6 + 4, occ * tl.sum(g_u4, axis=0))
        tl.atomic_add(grad_u_ptr + atom * 6 + 5, occ * tl.sum(g_u5, axis=0))
        tl.atomic_add(grad_occ_ptr + atom, tl.sum(g_occ, axis=0))

    # ====================================================================
    # SECOND-ORDER kernels: the backward of the backward. The backward maps
    # the incoming map gradient G to the atom gradients g_p = sum_v G dρ/dp;
    # given a cotangent t on those, these kernels return its two partials:
    #   tangent map  T(v) = sum_p t_p dρ(v)/dp          (scattered, atomics)
    #   per atom     h_p  = sum_v G(v) d/dp [T(v)]       (gathered, H·t)
    # in one pass over the same truncated box. The per-Gaussian constants are
    # recomputed inside the voxel loop under tl.static_range: they are scalar
    # work per block, negligible against the BLOCK_V-wide voxel math, and it
    # keeps the 5-Gaussian sums as one loop rather than five unrolled copies.
    # ====================================================================
    @triton.jit
    def _wq_grid_hvp_kernel(
        grad_density_map_ptr,
        member_ptr,
        xyz_ptr, b_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr,
        inv_frac_ptr, frac_ptr,
        tangent_map_ptr, t_xyz_ptr, t_b_ptr, t_occ_ptr,
        h_xyz_ptr, h_b_ptr, h_occ_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr, HAS_BATCH: tl.constexpr, WRITE_TANGENT: tl.constexpr,
    ):
        """Isotropic second order. Per Gaussian, with s = 1/Bt, kap = dBt/db and
        L = d ln(ρ_g)/dBt, the occupancy-free sums D, X1 = Σ s ρ_g, X2 = Σ s² ρ_g,
        Bb1 = dρ/db, XB and BB (the xyz-b and b-b curvatures) give T and H·t in
        closed form. Out-of-sphere voxels masked as in the forward."""
        if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
        if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
        if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)
        f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
        f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
        f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)
        uax = f0 / nx; uay = f3 / nx; uaz = f6 / nx
        ubx = f1 / ny; uby = f4 / ny; ubz = f7 / ny
        ucx = f2 / nz; ucy = f5 / nz; ucz = f8 / nz
        inva = tl.sqrt(if0 * if0 + if1 * if1 + if2 * if2)
        invb = tl.sqrt(if3 * if3 + if4 * if4 + if5 * if5)
        invc = tl.sqrt(if6 * if6 + if7 * if7 + if8 * if8)
        v_lane = tl.arange(0, BLOCK_V)

        atom = tl.program_id(0)
        DT = xyz_ptr.dtype.element_ty
        dm_ptr = grad_density_map_ptr
        tm_ptr = tangent_map_ptr
        if HAS_BATCH:
            slab = tl.load(member_ptr + atom).to(tl.int64) * (nx * ny * nz)
            dm_ptr += slab
            tm_ptr += slab
        r2cut = tl.load(r2cut_ptr + atom)
        r = tl.sqrt(r2cut)
        bhx = tl.extra.cuda.libdevice.ceil(r * nx * inva).to(tl.int32)
        bhy = tl.extra.cuda.libdevice.ceil(r * ny * invb).to(tl.int32)
        bhz = tl.extra.cuda.libdevice.ceil(r * nz * invc).to(tl.int32)
        sx_ = 2 * bhx + 1; sy_ = 2 * bhy + 1; sz_ = 2 * bhz + 1
        syz = sy_ * sz_
        n = sx_ * syz
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)

        b_iso = tl.load(b_ptr + atom)
        occ = tl.load(occ_ptr + atom)
        ax = tl.load(xyz_ptr + atom * 3 + 0)
        ay = tl.load(xyz_ptr + atom * 3 + 1)
        az = tl.load(xyz_ptr + atom * 3 + 2)
        tx = tl.load(t_xyz_ptr + atom * 3 + 0)
        ty = tl.load(t_xyz_ptr + atom * 3 + 1)
        tz = tl.load(t_xyz_ptr + atom * 3 + 2)
        tb = tl.load(t_b_ptr + atom)
        to = tl.load(t_occ_ptr + atom)

        frac_x = ax * if0 + ay * if1 + az * if2
        frac_y = ax * if3 + ay * if4 + az * if5
        frac_z = ax * if6 + ay * if7 + az * if8
        frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
        frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
        frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)
        cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
        ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
        ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)
        rx = frac_x - cix.to(DT) / nx
        ry = frac_y - ciy.to(DT) / ny
        rz = frac_z - ciz.to(DT) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        h_wx = tl.zeros([BLOCK_V], dtype=DT); h_wy = tl.zeros([BLOCK_V], dtype=DT)
        h_wz = tl.zeros([BLOCK_V], dtype=DT); h_x1 = tl.zeros([BLOCK_V], dtype=DT)
        h_b = tl.zeros([BLOCK_V], dtype=DT); h_o = tl.zeros([BLOCK_V], dtype=DT)
        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = ((v.to(tl.float32) + 0.5) * inv_syz).to(tl.int32)  # floor via trunc
            rem = v - ix * syz
            iy = ((rem.to(tl.float32) + 0.5) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(DT); ofyf = off_y.to(DT); ofzf = off_z.to(DT)
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            vix = (cix + off_x) % nx; vix = vix + tl.where(vix < 0, nx, 0)
            viy = (ciy + off_y) % ny; viy = viy + tl.where(viy < 0, ny, 0)
            viz = (ciz + off_z) % nz; viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)
            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            grad_out = tl.load(dm_ptr + dm_flat, mask=vmask, other=0.0)

            D = tl.zeros([BLOCK_V], dtype=DT); X1 = tl.zeros([BLOCK_V], dtype=DT)
            X2 = tl.zeros([BLOCK_V], dtype=DT); Bb1 = tl.zeros([BLOCK_V], dtype=DT)
            XB = tl.zeros([BLOCK_V], dtype=DT); BB = tl.zeros([BLOCK_V], dtype=DT)
            for g in tl.static_range(5):
                raw = (tl.load(B_ptr + atom * 5 + g) + b_iso) * 0.25
                Bt = tl.maximum(raw, 0.1)
                kap = 0.25 * (raw > 0.1).to(DT)  # dBt/db, zero where the clamp holds
                s = 1.0 / Bt
                d = tl.load(A_ptr + atom * 5 + g) * PI_1P5 / (Bt * tl.sqrt(Bt)) * tl.exp(-PI_SQ * r2 * s)
                L = s * (PI_SQ * r2 * s - 1.5)
                sd = s * d
                kd = kap * d
                D += d; X1 += sd; X2 += s * sd; Bb1 += kd * L
                XB += kd * s * (L - s)
                BB += kap * kd * (L * L + s * s * (1.5 - TWO_PI_SQ * r2 * s))
            wt = wx * tx + wy * ty + wz * tz
            J = TWO_PI_SQ * X1 * wt + Bb1 * tb  # dρ/d(xyz, b) . t at occ = 1
            if WRITE_TANGENT:
                tl.atomic_add(tm_ptr + dm_flat, occ * J + to * D, mask=wmask)
            alpha = occ * (TWO_PI_SQ * TWO_PI_SQ * X2 * wt + TWO_PI_SQ * XB * tb) + to * TWO_PI_SQ * X1
            h_wx += tl.where(wmask, grad_out * alpha * wx, 0.0)
            h_wy += tl.where(wmask, grad_out * alpha * wy, 0.0)
            h_wz += tl.where(wmask, grad_out * alpha * wz, 0.0)
            h_x1 += tl.where(wmask, grad_out * X1, 0.0)
            h_b += tl.where(wmask, grad_out * (occ * (TWO_PI_SQ * XB * wt + BB * tb) + to * Bb1), 0.0)
            h_o += tl.where(wmask, grad_out * J, 0.0)
            v_start += BLOCK_V

        # d²ρ/dxyz² carries -2π² X1 I besides the w w^T term; its t-contraction is an
        # atom constant times t_xyz, so it is applied once here.
        diag = occ * TWO_PI_SQ * tl.sum(h_x1, axis=0)
        tl.store(h_xyz_ptr + atom * 3 + 0, tl.sum(h_wx, axis=0) - diag * tx)
        tl.store(h_xyz_ptr + atom * 3 + 1, tl.sum(h_wy, axis=0) - diag * ty)
        tl.store(h_xyz_ptr + atom * 3 + 2, tl.sum(h_wz, axis=0) - diag * tz)
        tl.store(h_b_ptr + atom, tl.sum(h_b, axis=0))
        tl.store(h_occ_ptr + atom, tl.sum(h_o, axis=0))

    @triton.jit
    def _wq_grid_aniso_hvp_kernel(
        grad_density_map_ptr,
        member_ptr,
        xyz_ptr, u_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr,
        inv_frac_ptr, frac_ptr,
        tangent_map_ptr, t_xyz_ptr, t_u_ptr, t_occ_ptr,
        h_xyz_ptr, h_u_ptr, h_occ_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr, HAS_BATCH: tl.constexpr, WRITE_TANGENT: tl.constexpr,
    ):
        """Anisotropic second order. The cotangent of U enters as tM = 2*pi^2*t_U,
        the matching step of M. Per Gaussian, with v = Minv w and m = Minv (t_xyz +
        tM v), lam = d ln(ρ_g) . t gives T = Σ ρ_g (occ*lam + t_occ) and
            dT/dxyz = 2π² Σ ρ_g ((occ*lam + t_occ) v - occ m)
            dT/dM   = Σ ρ_g ((occ*lam + t_occ) S - 2π² occ sym(m v^T)
                             + occ/2 Minv tM Minv),  S = -Minv/2 + π² v v^T,
        mapped to U as the backward maps S (diag *2π², offdiag *4π²)."""
        if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
        if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
        if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)
        f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
        f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
        f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)
        uax = f0 / nx; uay = f3 / nx; uaz = f6 / nx
        ubx = f1 / ny; uby = f4 / ny; ubz = f7 / ny
        ucx = f2 / nz; ucy = f5 / nz; ucz = f8 / nz
        inva = tl.sqrt(if0 * if0 + if1 * if1 + if2 * if2)
        invb = tl.sqrt(if3 * if3 + if4 * if4 + if5 * if5)
        invc = tl.sqrt(if6 * if6 + if7 * if7 + if8 * if8)
        v_lane = tl.arange(0, BLOCK_V)

        atom = tl.program_id(0)
        DT = xyz_ptr.dtype.element_ty
        dm_ptr = grad_density_map_ptr
        tm_ptr = tangent_map_ptr
        if HAS_BATCH:
            slab = tl.load(member_ptr + atom).to(tl.int64) * (nx * ny * nz)
            dm_ptr += slab
            tm_ptr += slab
        r2cut = tl.load(r2cut_ptr + atom)
        r = tl.sqrt(r2cut)
        bhx = tl.extra.cuda.libdevice.ceil(r * nx * inva).to(tl.int32)
        bhy = tl.extra.cuda.libdevice.ceil(r * ny * invb).to(tl.int32)
        bhz = tl.extra.cuda.libdevice.ceil(r * nz * invc).to(tl.int32)
        sx_ = 2 * bhx + 1; sy_ = 2 * bhy + 1; sz_ = 2 * bhz + 1
        syz = sy_ * sz_
        n = sx_ * syz
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)

        occ = tl.load(occ_ptr + atom)
        ax = tl.load(xyz_ptr + atom * 3 + 0)
        ay = tl.load(xyz_ptr + atom * 3 + 1)
        az = tl.load(xyz_ptr + atom * 3 + 2)
        dd = TWO_PI_SQ * tl.load(u_ptr + atom * 6 + 3)
        ee = TWO_PI_SQ * tl.load(u_ptr + atom * 6 + 4)
        ff = TWO_PI_SQ * tl.load(u_ptr + atom * 6 + 5)
        du = TWO_PI_SQ * tl.load(u_ptr + atom * 6 + 0)
        dv = TWO_PI_SQ * tl.load(u_ptr + atom * 6 + 1)
        dw = TWO_PI_SQ * tl.load(u_ptr + atom * 6 + 2)
        tx = tl.load(t_xyz_ptr + atom * 3 + 0)
        ty = tl.load(t_xyz_ptr + atom * 3 + 1)
        tz = tl.load(t_xyz_ptr + atom * 3 + 2)
        to = tl.load(t_occ_ptr + atom)
        m00 = TWO_PI_SQ * tl.load(t_u_ptr + atom * 6 + 0)
        m11 = TWO_PI_SQ * tl.load(t_u_ptr + atom * 6 + 1)
        m22 = TWO_PI_SQ * tl.load(t_u_ptr + atom * 6 + 2)
        m01 = TWO_PI_SQ * tl.load(t_u_ptr + atom * 6 + 3)
        m02 = TWO_PI_SQ * tl.load(t_u_ptr + atom * 6 + 4)
        m12 = TWO_PI_SQ * tl.load(t_u_ptr + atom * 6 + 5)

        frac_x = ax * if0 + ay * if1 + az * if2
        frac_y = ax * if3 + ay * if4 + az * if5
        frac_z = ax * if6 + ay * if7 + az * if8
        frac_x = frac_x - tl.extra.cuda.libdevice.floor(frac_x)
        frac_y = frac_y - tl.extra.cuda.libdevice.floor(frac_y)
        frac_z = frac_z - tl.extra.cuda.libdevice.floor(frac_z)
        cix = tl.extra.cuda.libdevice.round(frac_x * nx).to(tl.int32)
        ciy = tl.extra.cuda.libdevice.round(frac_y * ny).to(tl.int32)
        ciz = tl.extra.cuda.libdevice.round(frac_z * nz).to(tl.int32)
        rx = frac_x - cix.to(DT) / nx
        ry = frac_y - ciy.to(DT) / ny
        rz = frac_z - ciz.to(DT) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        h_x = tl.zeros([BLOCK_V], dtype=DT); h_y = tl.zeros([BLOCK_V], dtype=DT)
        h_z = tl.zeros([BLOCK_V], dtype=DT); h_o = tl.zeros([BLOCK_V], dtype=DT)
        h00 = tl.zeros([BLOCK_V], dtype=DT); h11 = tl.zeros([BLOCK_V], dtype=DT)
        h22 = tl.zeros([BLOCK_V], dtype=DT); h01 = tl.zeros([BLOCK_V], dtype=DT)
        h02 = tl.zeros([BLOCK_V], dtype=DT); h12 = tl.zeros([BLOCK_V], dtype=DT)
        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = ((v.to(tl.float32) + 0.5) * inv_syz).to(tl.int32)  # floor via trunc
            rem = v - ix * syz
            iy = ((rem.to(tl.float32) + 0.5) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(DT); ofyf = off_y.to(DT); ofzf = off_z.to(DT)
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            vix = (cix + off_x) % nx; vix = vix + tl.where(vix < 0, nx, 0)
            viy = (ciy + off_y) % ny; viy = viy + tl.where(viy < 0, ny, 0)
            viz = (ciz + off_z) % nz; viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)
            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            grad_out = tl.load(dm_ptr + dm_flat, mask=vmask, other=0.0)

            tang = tl.zeros([BLOCK_V], dtype=DT); so = tl.zeros([BLOCK_V], dtype=DT)
            sx = tl.zeros([BLOCK_V], dtype=DT); sy = tl.zeros([BLOCK_V], dtype=DT)
            sz = tl.zeros([BLOCK_V], dtype=DT)
            s00 = tl.zeros([BLOCK_V], dtype=DT); s11 = tl.zeros([BLOCK_V], dtype=DT)
            s22 = tl.zeros([BLOCK_V], dtype=DT); s01 = tl.zeros([BLOCK_V], dtype=DT)
            s02 = tl.zeros([BLOCK_V], dtype=DT); s12 = tl.zeros([BLOCK_V], dtype=DT)
            for g in tl.static_range(5):
                Bq = tl.load(B_ptr + atom * 5 + g) * 0.25
                p00, p11, p22, p01, p02, p12, det = _sym3_inv(
                    Bq + du, Bq + dv, Bq + dw, dd, ee, ff)
                An = tl.load(A_ptr + atom * 5 + g) * PI_1P5 / tl.sqrt(tl.maximum(det, 1e-10))
                tau, q00, q11, q22, q01, q02, q12 = _sym3_sandwich(
                    p00, p11, p22, p01, p02, p12, m00, m11, m22, m01, m02, m12)
                vx = p00 * wx + p01 * wy + p02 * wz
                vy = p01 * wx + p11 * wy + p12 * wz
                vz = p02 * wx + p12 * wy + p22 * wz
                d = An * tl.exp(-PI_SQ * (wx * vx + wy * vy + wz * vz))
                mvx = m00 * vx + m01 * vy + m02 * vz
                mvy = m01 * vx + m11 * vy + m12 * vz
                mvz = m02 * vx + m12 * vy + m22 * vz
                lam = (TWO_PI_SQ * (vx * tx + vy * ty + vz * tz) - 0.5 * tau
                       + PI_SQ * (vx * mvx + vy * mvy + vz * mvz))
                qx = tx + mvx; qy = ty + mvy; qz = tz + mvz
                mx = p00 * qx + p01 * qy + p02 * qz
                my = p01 * qx + p11 * qy + p12 * qz
                mz = p02 * qx + p12 * qy + p22 * qz
                dc = d * (occ * lam + to)
                d_occ = d * occ
                tang += dc
                so += d * lam
                sx += dc * vx - d_occ * mx
                sy += dc * vy - d_occ * my
                sz += dc * vz - d_occ * mz
                s00 += dc * (PI_SQ * vx * vx - 0.5 * p00) + d_occ * (0.5 * q00 - TWO_PI_SQ * mx * vx)
                s11 += dc * (PI_SQ * vy * vy - 0.5 * p11) + d_occ * (0.5 * q11 - TWO_PI_SQ * my * vy)
                s22 += dc * (PI_SQ * vz * vz - 0.5 * p22) + d_occ * (0.5 * q22 - TWO_PI_SQ * mz * vz)
                s01 += dc * (PI_SQ * vx * vy - 0.5 * p01) + d_occ * (0.5 * q01 - PI_SQ * (mx * vy + my * vx))
                s02 += dc * (PI_SQ * vx * vz - 0.5 * p02) + d_occ * (0.5 * q02 - PI_SQ * (mx * vz + mz * vx))
                s12 += dc * (PI_SQ * vy * vz - 0.5 * p12) + d_occ * (0.5 * q12 - PI_SQ * (my * vz + mz * vy))
            if WRITE_TANGENT:
                tl.atomic_add(tm_ptr + dm_flat, tang, mask=wmask)
            h_x += tl.where(wmask, grad_out * sx, 0.0)
            h_y += tl.where(wmask, grad_out * sy, 0.0)
            h_z += tl.where(wmask, grad_out * sz, 0.0)
            h_o += tl.where(wmask, grad_out * so, 0.0)
            h00 += tl.where(wmask, grad_out * s00, 0.0)
            h11 += tl.where(wmask, grad_out * s11, 0.0)
            h22 += tl.where(wmask, grad_out * s22, 0.0)
            h01 += tl.where(wmask, grad_out * s01, 0.0)
            h02 += tl.where(wmask, grad_out * s02, 0.0)
            h12 += tl.where(wmask, grad_out * s12, 0.0)
            v_start += BLOCK_V

        tl.store(h_xyz_ptr + atom * 3 + 0, TWO_PI_SQ * tl.sum(h_x, axis=0))
        tl.store(h_xyz_ptr + atom * 3 + 1, TWO_PI_SQ * tl.sum(h_y, axis=0))
        tl.store(h_xyz_ptr + atom * 3 + 2, TWO_PI_SQ * tl.sum(h_z, axis=0))
        tl.store(h_u_ptr + atom * 6 + 0, TWO_PI_SQ * tl.sum(h00, axis=0))
        tl.store(h_u_ptr + atom * 6 + 1, TWO_PI_SQ * tl.sum(h11, axis=0))
        tl.store(h_u_ptr + atom * 6 + 2, TWO_PI_SQ * tl.sum(h22, axis=0))
        tl.store(h_u_ptr + atom * 6 + 3, 2.0 * TWO_PI_SQ * tl.sum(h01, axis=0))
        tl.store(h_u_ptr + atom * 6 + 4, 2.0 * TWO_PI_SQ * tl.sum(h02, axis=0))
        tl.store(h_u_ptr + atom * 6 + 5, 2.0 * TWO_PI_SQ * tl.sum(h12, axis=0))
        tl.store(h_occ_ptr + atom, tl.sum(h_o, axis=0))


def _launch(kernel, grid_ptr, member, r2cut, scene_buffers, extra, dims, block_v, warps,
            **flags):
    """Launch one of the six kernels over ``grid=(n_atoms,)``.

    ``member`` is the per-atom member id, or None for an unbatched call, in which case a
    placeholder pointer is passed that the compiled-out ``HAS_BATCH`` branch never reads.
    ``flags`` are further compile-time switches (``WRITE_TANGENT``).
    """
    nx, ny, nz = dims
    kernel[(r2cut.shape[0],)](
        grid_ptr,
        r2cut if member is None else member,
        *scene_buffers,
        r2cut,
        *extra,
        nx=nx, ny=ny, nz=nz, BLOCK_V=block_v, HAS_BATCH=member is not None,
        num_warps=warps, **flags,
    )


def _scene(xyz, third, A, B, occ):
    return tuple(t.contiguous() for t in (xyz, third, A, B, occ))


def _cell(inv_frac, frac):
    return inv_frac.contiguous().view(-1), frac.contiguous().view(-1)


def _forward(ctx, fwd_kernel, density_map, xyz, third, occ, A, B, r2cut, inv_frac, frac,
             member):
    """Forward shared by both splats: ``third`` is the scalar ``b`` or the 6-component
    ``u``. Returns ``density_map + splat``."""
    # Accumulate the splat into a copy of the running density_map (out =
    # density_map + splat) so the dispatch needs no separate zeros buffer + add.
    # A clone (not in-place) keeps this autograd-trivial and safe for the row's
    # on_failure="degrade": density_map is untouched if the kernel raises.
    dims = tuple(int(n) for n in density_map.shape[-3:])
    out = density_map.contiguous().clone()
    _launch(fwd_kernel, out.view(-1), member, r2cut, _scene(xyz, third, A, B, occ),
            _cell(inv_frac, frac), dims, FWD_BLOCK_V, FWD_NUM_WARPS)
    ctx.dims = dims
    ctx.save_for_backward(xyz, third, occ, A, B, r2cut, inv_frac, frac, member)
    return out


def _zeros_contiguous(t):
    # The kernels write per-atom arrays flat and row-major; zeros_like alone would keep a
    # strided input's layout and the gradient of a transposed xyz would come back permuted.
    return torch.zeros_like(t, memory_format=torch.contiguous_format)


class _SplatGrad(torch.autograd.Function):
    """The splat's atom gradients as a function of the map gradient and the atoms.

    The backward of both splats. Autograd records it under ``create_graph=True``, and its
    own backward runs the second-order kernel of ``kernels``, so a Hessian-vector product
    through the splat stays on the GPU. That backward is first-order only: a third
    derivative raises.
    """

    @staticmethod
    def forward(ctx, grad_density_map, xyz, third, occ, A, B, r2cut, inv_frac, frac,
                member, kernels, dims):
        grads = tuple(_zeros_contiguous(t) for t in (xyz, third, occ))
        _launch(kernels[0], grad_density_map.contiguous().view(-1), member, r2cut,
                _scene(xyz, third, A, B, occ), (*_cell(inv_frac, frac), *grads),
                dims, BWD_BLOCK_V, BWD_NUM_WARPS)
        ctx.kernels, ctx.dims = kernels, dims
        ctx.save_for_backward(grad_density_map, xyz, third, occ, A, B, r2cut, inv_frac,
                              frac, member)
        return grads

    @staticmethod
    @first_order_only
    def backward(ctx, t_xyz, t_third, t_occ):
        grad_density_map, xyz, third, occ, A, B, r2cut, inv_frac, frac, member = (
            ctx.saved_tensors
        )
        atoms = (xyz, third, occ)
        # A cotangent autograd leaves out (None) is zero.
        cotangents = tuple(
            _zeros_contiguous(x) if t is None else t.contiguous()
            for t, x in zip((t_xyz, t_third, t_occ), atoms)
        )
        hvps = tuple(_zeros_contiguous(x) for x in atoms)
        grad_map = grad_density_map.contiguous().view(-1)
        # The tangent map is the gradient wrt grad_density_map, wanted only when the map
        # gradient itself depends on the atoms (any loss nonlinear in the density).
        write_tangent = ctx.needs_input_grad[0]
        tangent = torch.zeros_like(grad_map) if write_tangent else None
        _launch(ctx.kernels[1], grad_map, member, r2cut,
                _scene(xyz, third, A, B, occ),
                (*_cell(inv_frac, frac), grad_map if tangent is None else tangent,
                 *cotangents, *hvps),
                ctx.dims, BWD_BLOCK_V, BWD_NUM_WARPS, WRITE_TANGENT=write_tangent)
        if tangent is not None:
            tangent = tangent.view(grad_density_map.shape)
        # grads for: grad_density_map, xyz, third, occ, A, B, r2cut, inv_frac, frac,
        # member, kernels, dims
        return (tangent, *hvps) + (None,) * 8


def _backward(ctx, kernels, grad_density_map):
    """Backward shared by both splats; the gradient of ``third`` has its shape."""
    xyz, third, occ, A, B, r2cut, inv_frac, frac, member = ctx.saved_tensors
    grads = _SplatGrad.apply(grad_density_map, xyz, third, occ, A, B, r2cut, inv_frac,
                             frac, member, kernels, ctx.dims)
    # out = density_map + splat -> grad wrt density_map is identity.
    # grads for: density_map, xyz, third, occ, A, B, r2cut, inv_frac, frac, member
    return (grad_density_map, *grads, None, None, None, None, None, None)


class WorkQueueGridDensity(torch.autograd.Function):
    """Isotropic variable-radius splat: grid=(n_atoms,), one program per atom.

    Each program iterates its atom's per-axis voxel box (decoded in-kernel from
    ``r2cut`` + cell metric) and truncates to the per-atom sphere ``r2 <= r2cut``, so every
    atom is splatted at its own radius. Twice differentiable in xyz, b, occ.
    """

    @staticmethod
    def forward(ctx, density_map, xyz, b, occ, A, B, r2cut, inv_frac, frac, member):
        return _forward(ctx, _wq_grid_fwd_kernel, density_map, xyz, b, occ, A, B,
                        r2cut, inv_frac, frac, member)

    @staticmethod
    def backward(ctx, grad_density_map):
        return _backward(ctx, (_wq_grid_bwd_kernel, _wq_grid_hvp_kernel),
                         grad_density_map)


class WorkQueueGridDensityAniso(torch.autograd.Function):
    """Anisotropic variable-radius splat: identical skeleton to
    ``WorkQueueGridDensity`` but carries the 6-component ``u`` instead of the
    scalar ``b`` and evaluates the 3D quadratic-form density; backward returns
    ``grad_u`` (6 components) in place of ``grad_b``."""

    @staticmethod
    def forward(ctx, density_map, xyz, u, occ, A, B, r2cut, inv_frac, frac, member):
        return _forward(ctx, _wq_grid_aniso_fwd_kernel, density_map, xyz, u, occ, A, B,
                        r2cut, inv_frac, frac, member)

    @staticmethod
    def backward(ctx, grad_density_map):
        return _backward(ctx, (_wq_grid_aniso_bwd_kernel, _wq_grid_aniso_hvp_kernel),
                         grad_density_map)


# ---------------------------------------------------------------------------
# Canonical-signature wrappers
# ---------------------------------------------------------------------------
# Every production splat exposes the same signature:
#
#     (density_map, xyz, adp_or_u, occ, A, B, inv_frac_matrix, frac_matrix,
#      radius_per_atom, batch_offsets=None)
#
# so ``electron_density/main.py`` dispatches with one ``select`` plus ``run_or_degrade``
# over ``DENSITY_BACKENDS`` and a test can drive any backend through one code path. These
# wrappers square the radius, as ``add_*_mps_var`` do for Metal, rather than leaving
# that to the caller.
#
# No coordinate grid is threaded anywhere: every voxel's Cartesian position is
# derived arithmetically in-kernel from ``frac`` and the grid dims, so ``density_map``
# alone carries the ``(nx, ny, nz)`` shape the launch needs.


def why_unavailable():
    """``None`` if these kernels can run, else why they cannot.

    The reason-returning half of the availability protocol every backend implements (see
    :mod:`torchref.utils.backends`). Not a restatement of ``triton_available()``: the
    ``@triton.jit`` bodies live under ``if _HAVE_TRITON:`` but the wrappers are defined
    unconditionally, so without Triton the module imports cleanly and the failure would
    otherwise surface as a bare ``NameError`` from inside ``_launch``.
    """
    if not _HAVE_TRITON:
        return (
            "triton is not importable, so the work-queue kernel bodies were never "
            "compiled into this module"
        )
    return None


def _member(density_map, xyz, batch_offsets):
    """Per-atom member ids on the map's device, or None for an unbatched call.

    Also refuses float64 on a GPU without float64 atomics (before sm_60), which the
    availability probe cannot see because it takes no tensors. The row degrades on the
    raise, so such a call runs on the portable splat with a warning.
    """
    if density_map.dtype is torch.float64:  # dtype-ok: capability check, not an allocation
        major, _ = torch.cuda.get_device_capability(density_map.device)
        if major < 6:
            raise RuntimeError(
                "float64 atomic_add needs compute capability 6.0 or newer; got "
                f"{major}.x on {density_map.device}"
            )
    _, offsets = check_batch(density_map, xyz.shape[0], batch_offsets)
    if offsets is None:
        return None
    return member_index(offsets.to(density_map.device), xyz.shape[0])


def add_isotropic_cuda_var(
    density_map, xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom,
    batch_offsets=None,
):
    """Isotropic variable-radius Triton splat; returns ``density_map + splat``.

    Canonical splat signature, identical to ``add_isotropic_plain_var``,
    ``add_isotropic_cpu_sphere_var`` and ``add_isotropic_mps_var``. CUDA float32 and
    float64 -- the gate is the ``cuda_triton`` row of ``DENSITY_BACKENDS``; this wrapper
    does not re-check. ``density_map`` is ``(nx, ny, nz)``, or ``(B, nx, ny, nz)`` with
    ``batch_offsets`` ``(B + 1,)``. Accumulation uses atomics, so results are not
    bit-reproducible run to run. Twice differentiable in ``xyz``, ``adp`` and ``occ``.
    """
    return WorkQueueGridDensity.apply(
        density_map,
        xyz,
        adp,
        occ,
        A,
        B,
        radius_per_atom * radius_per_atom,
        inv_frac_matrix,
        frac_matrix,
        _member(density_map, xyz, batch_offsets),
    )


def add_anisotropic_cuda_var(
    density_map, xyz, u, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom,
    batch_offsets=None,
):
    """Anisotropic variable-radius Triton splat; returns ``density_map + splat``.

    Canonical splat signature. ``u`` carries ``[U11, U22, U33, U12, U13, U23]``; the
    cutoff stays the Euclidean sphere at ``radius_per_atom`` while the density is the
    full Mahalanobis form, matching the Metal and fused-CPU kernels. Batching and dtypes
    as :func:`add_isotropic_cuda_var`.
    """
    return WorkQueueGridDensityAniso.apply(
        density_map,
        xyz,
        u,
        occ,
        A,
        B,
        radius_per_atom * radius_per_atom,
        inv_frac_matrix,
        frac_matrix,
        _member(density_map, xyz, batch_offsets),
    )
