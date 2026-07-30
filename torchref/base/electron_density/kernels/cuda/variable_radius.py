"""Variable-radius GPU density-splatting kernels (Triton).

One program per atom (``grid=(n_atoms,)``); each program iterates the cubic voxel
per-axis bounding box (sized in-kernel from ``r2cut`` + the inverse-cell metric)
around its atom, **decoding each voxel offset
arithmetically** from the lane index, and truncates to the per-atom sphere
(``r2 <= r2cut``). So every atom is splatted at its own ``N_sigma * sigma_eff``
radius with no host-built work plan / offset buffer. This is the production CUDA
float32 path, replacing the old single-radius ``fused_find_and_place_atoms``.

The per-voxel Gaussian math, PBC wrapping, and gradient formulae match the
reference fused kernel bit-for-bit (modulo atomic ordering); the only change vs the
earlier CSR-plan version is that offsets are computed in-kernel and the sphere
truncation is applied per voxel (``wmask``). The isotropic kernels carry the scalar
ADP ``b``; the anisotropic kernels carry the 6-component ``U`` and evaluate the 3D
quadratic form ``q = w^T Minv w`` with ``M = (B_g*I + 8*pi^2*U)/4`` inverted in-kernel.

Backward accumulates per-atom grads with ``atomic_add``; out-of-sphere voxels are
masked identically to the forward so they contribute neither density nor gradient.
"""

from __future__ import annotations

import torch

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
    def _wq_grid_fwd_kernel(
        n_items,
        grid_ptr, density_map_ptr,
        xyz_ptr, b_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr, mask_ptr,
        inv_frac_ptr, frac_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr,
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
        # float reciprocals for the decode (avoid the integer-divide on the int pipe;
        # exact for v < 2^24, i.e. any physical box: bh<=~14 -> n=side^3 << 2^24)
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)
        m0 = tl.load(mask_ptr + atom * 5 + 0); m1 = tl.load(mask_ptr + atom * 5 + 1)
        m2 = tl.load(mask_ptr + atom * 5 + 2); m3 = tl.load(mask_ptr + atom * 5 + 3)
        m4 = tl.load(mask_ptr + atom * 5 + 4)

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
        An0 = m0 * A0 * occ * PI_1P5 / (Bt0 * tl.sqrt(Bt0))
        An1 = m1 * A1 * occ * PI_1P5 / (Bt1 * tl.sqrt(Bt1))
        An2 = m2 * A2 * occ * PI_1P5 / (Bt2 * tl.sqrt(Bt2))
        An3 = m3 * A3 * occ * PI_1P5 / (Bt3 * tl.sqrt(Bt3))
        An4 = m4 * A4 * occ * PI_1P5 / (Bt4 * tl.sqrt(Bt4))

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
        rx = frac_x - cix.to(tl.float32) / nx
        ry = frac_y - ciy.to(tl.float32) / ny
        rz = frac_z - ciz.to(tl.float32) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = (v.to(tl.float32) * inv_syz).to(tl.int32)  # floor via trunc (v >= 0)
            rem = v - ix * syz
            iy = (rem.to(tl.float32) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(tl.float32); ofyf = off_y.to(tl.float32); ofzf = off_z.to(tl.float32)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # write index (PBC wrap); coords use the unwrapped offset above
            vix = cix + off_x; vix = vix - tl.where(vix >= nx, nx, 0); vix = vix + tl.where(vix < 0, nx, 0)
            viy = ciy + off_y; viy = viy - tl.where(viy >= ny, ny, 0); viy = viy + tl.where(viy < 0, ny, 0)
            viz = ciz + off_z; viz = viz - tl.where(viz >= nz, nz, 0); viz = viz + tl.where(viz < 0, nz, 0)
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
            tl.atomic_add(density_map_ptr + dm_flat, density, mask=wmask)
            v_start += BLOCK_V

    @triton.jit
    def _wq_grid_bwd_kernel(
        n_items,
        grid_ptr, grad_density_map_ptr,
        xyz_ptr, b_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr, mask_ptr,
        inv_frac_ptr, frac_ptr,
        grad_xyz_ptr, grad_b_ptr, grad_occ_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr,
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
        # float reciprocals for the decode (avoid the integer-divide on the int pipe;
        # exact for v < 2^24, i.e. any physical box: bh<=~14 -> n=side^3 << 2^24)
        inv_syz = 1.0 / syz.to(tl.float32)
        inv_sz = 1.0 / sz_.to(tl.float32)
        m0 = tl.load(mask_ptr + atom * 5 + 0); m1 = tl.load(mask_ptr + atom * 5 + 1)
        m2 = tl.load(mask_ptr + atom * 5 + 2); m3 = tl.load(mask_ptr + atom * 5 + 3)
        m4 = tl.load(mask_ptr + atom * 5 + 4)

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
        clamp0 = ((B0 + b_iso) * 0.25 > 0.1).to(tl.float32)
        clamp1 = ((B1 + b_iso) * 0.25 > 0.1).to(tl.float32)
        clamp2 = ((B2 + b_iso) * 0.25 > 0.1).to(tl.float32)
        clamp3 = ((B3 + b_iso) * 0.25 > 0.1).to(tl.float32)
        clamp4 = ((B4 + b_iso) * 0.25 > 0.1).to(tl.float32)
        An0 = m0 * A0 * occ * PI_1P5 / (Bt0 * tl.sqrt(Bt0))
        An1 = m1 * A1 * occ * PI_1P5 / (Bt1 * tl.sqrt(Bt1))
        An2 = m2 * A2 * occ * PI_1P5 / (Bt2 * tl.sqrt(Bt2))
        An3 = m3 * A3 * occ * PI_1P5 / (Bt3 * tl.sqrt(Bt3))
        An4 = m4 * A4 * occ * PI_1P5 / (Bt4 * tl.sqrt(Bt4))

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
        rx = frac_x - cix.to(tl.float32) / nx
        ry = frac_y - ciy.to(tl.float32) / ny
        rz = frac_z - ciz.to(tl.float32) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        g_ax = 0.0; g_ay = 0.0; g_az = 0.0; g_b = 0.0; g_occ = 0.0
        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = (v.to(tl.float32) * inv_syz).to(tl.int32)  # floor via trunc (v >= 0)
            rem = v - ix * syz
            iy = (rem.to(tl.float32) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(tl.float32); ofyf = off_y.to(tl.float32); ofzf = off_z.to(tl.float32)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # write index (PBC wrap); coords use the unwrapped offset above
            vix = cix + off_x; vix = vix - tl.where(vix >= nx, nx, 0); vix = vix + tl.where(vix < 0, nx, 0)
            viy = ciy + off_y; viy = viy - tl.where(viy >= ny, ny, 0); viy = viy + tl.where(viy < 0, ny, 0)
            viz = ciz + off_z; viz = viz - tl.where(viz >= nz, nz, 0); viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)

            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            grad_out = tl.load(grad_density_map_ptr + dm_flat, mask=vmask, other=0.0)
            e0 = tl.exp(-PI_SQ * r2 / Bt0); e1 = tl.exp(-PI_SQ * r2 / Bt1)
            e2 = tl.exp(-PI_SQ * r2 / Bt2); e3 = tl.exp(-PI_SQ * r2 / Bt3)
            e4 = tl.exp(-PI_SQ * r2 / Bt4)
            coeff_xyz = (
                An0 * e0 / Bt0 + An1 * e1 / Bt1 + An2 * e2 / Bt2
                + An3 * e3 / Bt3 + An4 * e4 / Bt4
            )
            scale_xyz = grad_out * 2.0 * PI_SQ * coeff_xyz
            g_ax += tl.sum(tl.where(wmask, scale_xyz * wx, 0.0), axis=0)
            g_ay += tl.sum(tl.where(wmask, scale_xyz * wy, 0.0), axis=0)
            g_az += tl.sum(tl.where(wmask, scale_xyz * wz, 0.0), axis=0)
            db0 = An0 * e0 * (-1.5 / Bt0 + PI_SQ * r2 / (Bt0 * Bt0)) * clamp0
            db1 = An1 * e1 * (-1.5 / Bt1 + PI_SQ * r2 / (Bt1 * Bt1)) * clamp1
            db2 = An2 * e2 * (-1.5 / Bt2 + PI_SQ * r2 / (Bt2 * Bt2)) * clamp2
            db3 = An3 * e3 * (-1.5 / Bt3 + PI_SQ * r2 / (Bt3 * Bt3)) * clamp3
            db4 = An4 * e4 * (-1.5 / Bt4 + PI_SQ * r2 / (Bt4 * Bt4)) * clamp4
            g_b += tl.sum(tl.where(wmask, grad_out * 0.25 * (db0 + db1 + db2 + db3 + db4), 0.0), axis=0)
            dens = An0 * e0 + An1 * e1 + An2 * e2 + An3 * e3 + An4 * e4
            g_occ += tl.sum(
                tl.where(wmask, grad_out * tl.where(occ != 0.0, dens / occ, 0.0), 0.0), axis=0)
            v_start += BLOCK_V

        tl.atomic_add(grad_xyz_ptr + atom * 3 + 0, g_ax)
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 1, g_ay)
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 2, g_az)
        tl.atomic_add(grad_b_ptr + atom, g_b)
        tl.atomic_add(grad_occ_ptr + atom, g_occ)

    # ====================================================================
    # ANISOTROPIC variable-radius kernels: same skeleton, scalar b replaced
    # by the 6-component U, density = An_g*exp(-pi^2 * w^T Minv w) with
    # M_g = (B_g*I + 8*pi^2*U)/4. (Math from the reference aniso kernel.)
    # ====================================================================
    @triton.jit
    def _wq_grid_aniso_fwd_kernel(
        n_items,
        grid_ptr, density_map_ptr,
        xyz_ptr, u_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr, mask_ptr,
        inv_frac_ptr, frac_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr,
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
        # float reciprocals for the decode (avoid the integer-divide on the int pipe;
        # exact for v < 2^24, i.e. any physical box: bh<=~14 -> n=side^3 << 2^24)
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
        An0 = tl.load(mask_ptr + atom * 5 + 0) * tl.load(A_ptr + atom * 5 + 0) * oc / tl.sqrt(tl.maximum(dt0, 1e-10))
        An1 = tl.load(mask_ptr + atom * 5 + 1) * tl.load(A_ptr + atom * 5 + 1) * oc / tl.sqrt(tl.maximum(dt1, 1e-10))
        An2 = tl.load(mask_ptr + atom * 5 + 2) * tl.load(A_ptr + atom * 5 + 2) * oc / tl.sqrt(tl.maximum(dt2, 1e-10))
        An3 = tl.load(mask_ptr + atom * 5 + 3) * tl.load(A_ptr + atom * 5 + 3) * oc / tl.sqrt(tl.maximum(dt3, 1e-10))
        An4 = tl.load(mask_ptr + atom * 5 + 4) * tl.load(A_ptr + atom * 5 + 4) * oc / tl.sqrt(tl.maximum(dt4, 1e-10))

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
        rx = frac_x - cix.to(tl.float32) / nx
        ry = frac_y - ciy.to(tl.float32) / ny
        rz = frac_z - ciz.to(tl.float32) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = (v.to(tl.float32) * inv_syz).to(tl.int32)  # floor via trunc (v >= 0)
            rem = v - ix * syz
            iy = (rem.to(tl.float32) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(tl.float32); ofyf = off_y.to(tl.float32); ofzf = off_z.to(tl.float32)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # write index (PBC wrap); coords use the unwrapped offset above
            vix = cix + off_x; vix = vix - tl.where(vix >= nx, nx, 0); vix = vix + tl.where(vix < 0, nx, 0)
            viy = ciy + off_y; viy = viy - tl.where(viy >= ny, ny, 0); viy = viy + tl.where(viy < 0, ny, 0)
            viz = ciz + off_z; viz = viz - tl.where(viz >= nz, nz, 0); viz = viz + tl.where(viz < 0, nz, 0)
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
            tl.atomic_add(density_map_ptr + dm_flat, density, mask=wmask)
            v_start += BLOCK_V

    @triton.jit
    def _wq_grid_aniso_bwd_kernel(
        n_items,
        grid_ptr, grad_density_map_ptr,
        xyz_ptr, u_ptr, A_ptr, B_ptr, occ_ptr,
        r2cut_ptr, mask_ptr,
        inv_frac_ptr, frac_ptr,
        grad_xyz_ptr, grad_u_ptr, grad_occ_ptr,
        nx: tl.constexpr, ny: tl.constexpr, nz: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Anisotropic backward. v = Minv w; grad_xyz = sum_g 2*pi^2*dg*v_g;
        grad_U via S = -0.5*Minv + pi^2 v v^T (diag *2pi^2, offdiag *4pi^2);
        grad_occ = sum_g dg/occ. Out-of-sphere voxels masked as in the forward."""
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
        # float reciprocals for the decode (avoid the integer-divide on the int pipe;
        # exact for v < 2^24, i.e. any physical box: bh<=~14 -> n=side^3 << 2^24)
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
        An0 = tl.load(mask_ptr + atom * 5 + 0) * tl.load(A_ptr + atom * 5 + 0) * oc / tl.sqrt(tl.maximum(dt0, 1e-10))
        An1 = tl.load(mask_ptr + atom * 5 + 1) * tl.load(A_ptr + atom * 5 + 1) * oc / tl.sqrt(tl.maximum(dt1, 1e-10))
        An2 = tl.load(mask_ptr + atom * 5 + 2) * tl.load(A_ptr + atom * 5 + 2) * oc / tl.sqrt(tl.maximum(dt2, 1e-10))
        An3 = tl.load(mask_ptr + atom * 5 + 3) * tl.load(A_ptr + atom * 5 + 3) * oc / tl.sqrt(tl.maximum(dt3, 1e-10))
        An4 = tl.load(mask_ptr + atom * 5 + 4) * tl.load(A_ptr + atom * 5 + 4) * oc / tl.sqrt(tl.maximum(dt4, 1e-10))

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
        rx = frac_x - cix.to(tl.float32) / nx
        ry = frac_y - ciy.to(tl.float32) / ny
        rz = frac_z - ciz.to(tl.float32) / nz
        w0x = f0 * rx + f1 * ry + f2 * rz
        w0y = f3 * rx + f4 * ry + f5 * rz
        w0z = f6 * rx + f7 * ry + f8 * rz

        g_ax = 0.0; g_ay = 0.0; g_az = 0.0; g_occ = 0.0
        g_u0 = 0.0; g_u1 = 0.0; g_u2 = 0.0; g_u3 = 0.0; g_u4 = 0.0; g_u5 = 0.0
        v_start = 0
        while v_start < n:
            v = v_start + v_lane
            vmask = v < n
            ix = (v.to(tl.float32) * inv_syz).to(tl.int32)  # floor via trunc (v >= 0)
            rem = v - ix * syz
            iy = (rem.to(tl.float32) * inv_sz).to(tl.int32)
            off_x = ix - bhx
            off_y = iy - bhy
            off_z = (rem - iy * sz_) - bhz
            ofxf = off_x.to(tl.float32); ofyf = off_y.to(tl.float32); ofzf = off_z.to(tl.float32)
            # direct atom->voxel displacement: w = di*u_a + dj*u_b + dk*u_c - w0
            wx = ofxf * uax + ofyf * ubx + ofzf * ucx - w0x
            wy = ofxf * uay + ofyf * uby + ofzf * ucy - w0y
            wz = ofxf * uaz + ofyf * ubz + ofzf * ucz - w0z
            # write index (PBC wrap); coords use the unwrapped offset above
            vix = cix + off_x; vix = vix - tl.where(vix >= nx, nx, 0); vix = vix + tl.where(vix < 0, nx, 0)
            viy = ciy + off_y; viy = viy - tl.where(viy >= ny, ny, 0); viy = viy + tl.where(viy < 0, ny, 0)
            viz = ciz + off_z; viz = viz - tl.where(viz >= nz, nz, 0); viz = viz + tl.where(viz < 0, nz, 0)
            r2 = wx * wx + wy * wy + wz * wz
            wmask = vmask & (r2 <= r2cut)

            dm_flat = ((vix * ny + viy) * nz + viz).to(tl.int64)
            grad_out = tl.load(grad_density_map_ptr + dm_flat, mask=vmask, other=0.0)

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
            g_ax += tl.sum(tl.where(wmask, sxyz * sx_acc, 0.0), axis=0)
            g_ay += tl.sum(tl.where(wmask, sxyz * sy_acc, 0.0), axis=0)
            g_az += tl.sum(tl.where(wmask, sxyz * sz_acc, 0.0), axis=0)
            g_u0 += tl.sum(tl.where(wmask, grad_out * 2.0 * PI_SQ * gu0_l, 0.0), axis=0)
            g_u1 += tl.sum(tl.where(wmask, grad_out * 2.0 * PI_SQ * gu1_l, 0.0), axis=0)
            g_u2 += tl.sum(tl.where(wmask, grad_out * 2.0 * PI_SQ * gu2_l, 0.0), axis=0)
            g_u3 += tl.sum(tl.where(wmask, grad_out * 4.0 * PI_SQ * gu3_l, 0.0), axis=0)
            g_u4 += tl.sum(tl.where(wmask, grad_out * 4.0 * PI_SQ * gu4_l, 0.0), axis=0)
            g_u5 += tl.sum(tl.where(wmask, grad_out * 4.0 * PI_SQ * gu5_l, 0.0), axis=0)
            g_occ += tl.sum(
                tl.where(wmask, grad_out * tl.where(occ != 0.0, dens / occ, 0.0), 0.0), axis=0)
            v_start += BLOCK_V

        tl.atomic_add(grad_xyz_ptr + atom * 3 + 0, g_ax)
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 1, g_ay)
        tl.atomic_add(grad_xyz_ptr + atom * 3 + 2, g_az)
        tl.atomic_add(grad_u_ptr + atom * 6 + 0, g_u0)
        tl.atomic_add(grad_u_ptr + atom * 6 + 1, g_u1)
        tl.atomic_add(grad_u_ptr + atom * 6 + 2, g_u2)
        tl.atomic_add(grad_u_ptr + atom * 6 + 3, g_u3)
        tl.atomic_add(grad_u_ptr + atom * 6 + 4, g_u4)
        tl.atomic_add(grad_u_ptr + atom * 6 + 5, g_u5)
        tl.atomic_add(grad_occ_ptr + atom, g_occ)


def _launch_grid_fwd(out_flat, r2cut, mask, scene_buffers, dims):
    """Isotropic grid=(n_atoms,) forward (fixed FWD_BLOCK_V/FWD_NUM_WARPS)."""
    (grid_flat, xyz, b, A, B, occ, inv_frac, frac) = scene_buffers
    nx, ny, nz = dims
    n_atoms = r2cut.shape[0]
    _wq_grid_fwd_kernel[(n_atoms,)](
        n_atoms,
        grid_flat, out_flat,
        xyz, b, A, B, occ,
        r2cut, mask,
        inv_frac, frac,
        nx=nx, ny=ny, nz=nz, BLOCK_V=FWD_BLOCK_V,
        num_warps=FWD_NUM_WARPS,
    )


def _launch_grid_aniso_fwd(out_flat, r2cut, mask, scene_buffers, dims):
    """Anisotropic grid=(n_atoms,) forward (fixed FWD_BLOCK_V/FWD_NUM_WARPS)."""
    (grid_flat, xyz, u, A, B, occ, inv_frac, frac) = scene_buffers
    nx, ny, nz = dims
    n_atoms = r2cut.shape[0]
    _wq_grid_aniso_fwd_kernel[(n_atoms,)](
        n_atoms,
        grid_flat, out_flat,
        xyz, u, A, B, occ,
        r2cut, mask,
        inv_frac, frac,
        nx=nx, ny=ny, nz=nz, BLOCK_V=FWD_BLOCK_V,
        num_warps=FWD_NUM_WARPS,
    )


class WorkQueueGridDensity(torch.autograd.Function):
    """Isotropic variable-radius splat: grid=(n_atoms,), one program per atom.

    Each program iterates its atom's per-axis voxel box (decoded in-kernel from
    ``r2cut`` + cell metric) and truncates to the per-atom sphere ``r2 <= r2cut``, so every
    atom is splatted at its own radius. Differentiable in xyz, b, occ.
    """

    @staticmethod
    def forward(ctx, density_map, real_space_grid, xyz, b, occ, A, B,
                r2cut, mask, inv_frac, frac):
        # Accumulate the splat into a copy of the running density_map (out =
        # density_map + splat) so the dispatch needs no separate zeros buffer + add.
        # A clone (not in-place) keeps this autograd-trivial AND safe for the AUTO
        # fallthrough: density_map is untouched if the kernel raises.
        nx, ny, nz = real_space_grid.shape[:3]
        grid_flat = real_space_grid.contiguous().view(-1)
        xyz = xyz.contiguous(); b = b.contiguous(); occ = occ.contiguous()
        A = A.contiguous(); B = B.contiguous()
        inv_frac_flat = inv_frac.contiguous().view(-1)
        frac_flat = frac.contiguous().view(-1)
        out = density_map.contiguous().clone().view(-1)
        _launch_grid_fwd(
            out, r2cut, mask,
            (grid_flat, xyz, b, A, B, occ, inv_frac_flat, frac_flat),
            (nx, ny, nz),
        )
        ctx.save_for_backward(real_space_grid, xyz, b, occ, A, B,
                              r2cut, mask, inv_frac, frac)
        return out.view(nx, ny, nz)

    @staticmethod
    def backward(ctx, grad_density_map):
        (real_space_grid, xyz, b, occ, A, B,
         r2cut, mask, inv_frac, frac) = ctx.saved_tensors
        nx, ny, nz = real_space_grid.shape[:3]
        grid_flat = real_space_grid.contiguous().view(-1)
        grad_dm = grad_density_map.contiguous().view(-1)
        inv_frac_flat = inv_frac.contiguous().view(-1)
        frac_flat = frac.contiguous().view(-1)
        grad_xyz = torch.zeros_like(xyz)
        grad_b = torch.zeros_like(b)
        grad_occ = torch.zeros_like(occ)
        _wq_grid_bwd_kernel[(r2cut.shape[0],)](
            r2cut.shape[0],
            grid_flat, grad_dm,
            xyz.contiguous(), b.contiguous(), A.contiguous(), B.contiguous(), occ.contiguous(),
            r2cut, mask,
            inv_frac_flat, frac_flat,
            grad_xyz, grad_b, grad_occ,
            nx=nx, ny=ny, nz=nz, BLOCK_V=BWD_BLOCK_V,
            num_warps=BWD_NUM_WARPS,
        )
        # out = density_map + splat -> grad wrt density_map is identity.
        # grads for: density_map, real_space_grid, xyz, b, occ, A, B, r2cut, mask, inv_frac, frac
        return (grad_density_map, None, grad_xyz, grad_b, grad_occ, None, None,
                None, None, None, None)


class WorkQueueGridDensityAniso(torch.autograd.Function):
    """Anisotropic variable-radius splat: identical skeleton to
    ``WorkQueueGridDensity`` but carries the 6-component ``u`` instead of the
    scalar ``b`` and evaluates the 3D quadratic-form density; backward returns
    ``grad_u`` (6 components) in place of ``grad_b``."""

    @staticmethod
    def forward(ctx, density_map, real_space_grid, xyz, u, occ, A, B,
                r2cut, mask, inv_frac, frac):
        # Accumulate into a copy of the running density_map (see the iso forward).
        nx, ny, nz = real_space_grid.shape[:3]
        grid_flat = real_space_grid.contiguous().view(-1)
        xyz = xyz.contiguous(); u = u.contiguous(); occ = occ.contiguous()
        A = A.contiguous(); B = B.contiguous()
        inv_frac_flat = inv_frac.contiguous().view(-1)
        frac_flat = frac.contiguous().view(-1)
        out = density_map.contiguous().clone().view(-1)
        _launch_grid_aniso_fwd(
            out, r2cut, mask,
            (grid_flat, xyz, u, A, B, occ, inv_frac_flat, frac_flat),
            (nx, ny, nz),
        )
        ctx.save_for_backward(real_space_grid, xyz, u, occ, A, B,
                              r2cut, mask, inv_frac, frac)
        return out.view(nx, ny, nz)

    @staticmethod
    def backward(ctx, grad_density_map):
        (real_space_grid, xyz, u, occ, A, B,
         r2cut, mask, inv_frac, frac) = ctx.saved_tensors
        nx, ny, nz = real_space_grid.shape[:3]
        grid_flat = real_space_grid.contiguous().view(-1)
        grad_dm = grad_density_map.contiguous().view(-1)
        inv_frac_flat = inv_frac.contiguous().view(-1)
        frac_flat = frac.contiguous().view(-1)
        grad_xyz = torch.zeros_like(xyz)
        grad_u = torch.zeros_like(u)
        grad_occ = torch.zeros_like(occ)
        _wq_grid_aniso_bwd_kernel[(r2cut.shape[0],)](
            r2cut.shape[0],
            grid_flat, grad_dm,
            xyz.contiguous(), u.contiguous(), A.contiguous(), B.contiguous(), occ.contiguous(),
            r2cut, mask,
            inv_frac_flat, frac_flat,
            grad_xyz, grad_u, grad_occ,
            nx=nx, ny=ny, nz=nz, BLOCK_V=BWD_BLOCK_V,
            num_warps=BWD_NUM_WARPS,
        )
        # out = density_map + splat -> grad wrt density_map is identity.
        return (grad_density_map, None, grad_xyz, grad_u, grad_occ, None, None,
                None, None, None, None)


# ---------------------------------------------------------------------------
# Canonical-signature wrappers
# ---------------------------------------------------------------------------
# Every production splat exposes the same signature:
#
#     (density_map, xyz, adp_or_u, occ, A, B, inv_frac_matrix, frac_matrix,
#      radius_per_atom)
#
# so the dispatch in ``electron_density/main.py`` is four structurally identical
# calls per ladder and a test can drive any backend through one code path. These
# wrappers do for CUDA what ``add_*_mps_var`` already did for Metal: square the
# radius and build the coefficient mask, rather than leaving that to the caller.
#
# ``density_map`` is passed where the ``autograd.Function`` wants
# ``real_space_grid``. That is exact, not a convenience: ``forward``/``backward``
# use that argument only for ``.shape[:3]`` and for a ``grid_flat`` pointer handed
# to the kernel as ``grid_ptr`` -- which none of the four Triton kernels ever
# loads, because voxel coordinates are derived arithmetically from ``frac`` and the
# grid dims. ``density_map`` has the same ``(nx, ny, nz)`` shape, so both uses are
# satisfied. If a kernel is ever changed to actually read ``grid_ptr``, this breaks
# and the right fix is to delete that dead parameter, not to thread a real grid
# back through here.


def why_unavailable():
    """``None`` if these kernels can run, else why they cannot.

    The reason-returning half of the availability protocol shared by every backend (see
    :mod:`torchref.utils.backends`).

    This probe closes a real gap rather than restating ``triton_available()``. The
    ``@triton.jit`` kernel bodies live inside ``if _HAVE_TRITON:`` above, but
    ``WorkQueueGridDensity`` and these wrappers are defined *unconditionally*, so on a host
    without Triton the module imports cleanly and the failure surfaces as a bare
    ``NameError`` from deep inside ``_launch_grid_fwd``. Every other backend answers
    "can I run" before being called; this one had nothing to ask.
    """
    if not _HAVE_TRITON:
        return (
            "triton is not importable, so the work-queue kernel bodies were never "
            "compiled into this module"
        )
    return None


def _coeff_mask(xyz):
    """All-ones per-atom coefficient mask, shape ``(n, 5)``.

    Every call site in the codebase passes all ones -- the mask exists so a caller
    *could* disable individual ITC92 Gaussians, and nothing does. Kept because it is
    part of the kernel's argument list.
    """
    return torch.ones(xyz.shape[0], 5, dtype=xyz.dtype, device=xyz.device)


def add_isotropic_cuda_var(
    density_map, xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Isotropic variable-radius Triton splat; returns ``density_map + splat``.

    Canonical splat signature, identical to ``add_isotropic_plain_var``,
    ``add_isotropic_cpu_sphere_var`` and ``add_isotropic_mps_var``. CUDA float32
    only -- the gate is ``should_use_triton``; this wrapper does not re-check.
    """
    return WorkQueueGridDensity.apply(
        density_map,
        density_map,  # stands in for real_space_grid; see the note above
        xyz,
        adp,
        occ,
        A,
        B,
        radius_per_atom * radius_per_atom,
        _coeff_mask(xyz),
        inv_frac_matrix,
        frac_matrix,
    )


def add_anisotropic_cuda_var(
    density_map, xyz, u, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Anisotropic variable-radius Triton splat; returns ``density_map + splat``.

    Canonical splat signature. ``u`` carries ``[U11, U22, U33, U12, U13, U23]``; the
    cutoff stays the Euclidean sphere at ``radius_per_atom`` while the density is the
    full Mahalanobis form, matching the Metal and fused-CPU kernels.
    """
    return WorkQueueGridDensityAniso.apply(
        density_map,
        density_map,  # stands in for real_space_grid; see the note above
        xyz,
        u,
        occ,
        A,
        B,
        radius_per_atom * radius_per_atom,
        _coeff_mask(xyz),
        inv_frac_matrix,
        frac_matrix,
    )
