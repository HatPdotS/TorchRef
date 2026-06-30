"""
Triton GPU kernel for fused electron density computation.

Fuses PBC wrapping, r² calculation, 5-Gaussian evaluation, and scatter-add
into a single GPU kernel, eliminating ~14 separate kernel launches and
~500MB of intermediate memory allocations.

Provides full autograd support for refinement of xyz, b, and occ.
"""

import torch
import triton
import triton.language as tl

# =============================================================================
# Constants
# =============================================================================

PI: float = 3.141592653589793
PI_SQ: float = PI * PI
PI_1P5: float = PI * 1.7724538509055159  # pi * sqrt(pi) = pi^1.5


# =============================================================================
# Forward kernel
# =============================================================================

@triton.jit
def _density_fwd_kernel(
    # Pointers
    surr_coords_ptr,   # (N_atoms, N_voxels, 3) float32
    voxel_idx_ptr,     # (N_atoms, N_voxels, 3) int32
    density_map_ptr,   # (nx*ny*nz,) float32
    xyz_ptr,           # (N_atoms, 3) float32
    b_ptr,             # (N_atoms,) float32
    inv_frac_ptr,      # (9,) float32 row-major
    frac_ptr,          # (9,) float32 row-major
    A_ptr,             # (N_atoms, 5) float32
    B_ptr,             # (N_atoms, 5) float32
    occ_ptr,           # (N_atoms,) float32
    # Dimensions
    N_voxels: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    # Block size
    BLOCK_V: tl.constexpr,
):
    """One program per atom. Threads process BLOCK_V voxels."""
    atom = tl.program_id(0)

    # Load per-atom scalars
    b_iso = tl.load(b_ptr + atom)
    occ = tl.load(occ_ptr + atom)
    ax = tl.load(xyz_ptr + atom * 3 + 0)
    ay = tl.load(xyz_ptr + atom * 3 + 1)
    az = tl.load(xyz_ptr + atom * 3 + 2)

    # Load ITC92 params (5 Gaussians)
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

    # Precompute B_total and A_norm for each Gaussian
    Bt0 = tl.maximum((B0 + b_iso) * 0.25, 0.1)
    Bt1 = tl.maximum((B1 + b_iso) * 0.25, 0.1)
    Bt2 = tl.maximum((B2 + b_iso) * 0.25, 0.1)
    Bt3 = tl.maximum((B3 + b_iso) * 0.25, 0.1)
    Bt4 = tl.maximum((B4 + b_iso) * 0.25, 0.1)

    pi_1p5: tl.constexpr = 5.568327996831708  # pi^1.5
    An0 = A0 * occ * pi_1p5 / (Bt0 * tl.sqrt(Bt0))
    An1 = A1 * occ * pi_1p5 / (Bt1 * tl.sqrt(Bt1))
    An2 = A2 * occ * pi_1p5 / (Bt2 * tl.sqrt(Bt2))
    An3 = A3 * occ * pi_1p5 / (Bt3 * tl.sqrt(Bt3))
    An4 = A4 * occ * pi_1p5 / (Bt4 * tl.sqrt(Bt4))

    # Load 3x3 matrices (row-major: M[i,j] = ptr[i*3+j])
    # inv_frac_matrix.T means we need columns of inv_frac = rows of inv_frac.T
    # diff @ inv_frac.T  =>  for each component: dot(diff, inv_frac.T[col]) = dot(diff, inv_frac[:,col])
    # In row-major, inv_frac[i,j] = inv_frac_ptr[i*3+j]
    # Column j of inv_frac = inv_frac_ptr[0*3+j], inv_frac_ptr[1*3+j], inv_frac_ptr[2*3+j]
    if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
    if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
    if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)

    f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
    f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
    f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)

    pi_sq: tl.constexpr = 9.869604401089358

    # Process voxels in blocks
    v_offsets = tl.arange(0, BLOCK_V)
    base = atom * N_voxels

    for v_start in range(0, N_voxels, BLOCK_V):
        v = v_start + v_offsets
        mask = v < N_voxels

        # Load surrounding_coords[atom, v, 0:3]
        idx3 = (base + v) * 3
        sx = tl.load(surr_coords_ptr + idx3 + 0, mask=mask)
        sy = tl.load(surr_coords_ptr + idx3 + 1, mask=mask)
        sz = tl.load(surr_coords_ptr + idx3 + 2, mask=mask)

        # diff = surrounding - xyz
        dx = sx - ax
        dy = sy - ay
        dz = sz - az

        # PBC: diff_frac = diff @ inv_frac_matrix.T
        # (diff @ M.T)[i] = sum_j diff[j] * M[i,j]  (dot with row i of M)
        # Row-major: M[i,j] = ptr[i*3 + j]
        fx = dx * if0 + dy * if1 + dz * if2
        fy = dx * if3 + dy * if4 + dz * if5
        fz = dx * if6 + dy * if7 + dz * if8

        # Round and correct
        tx = tl.extra.cuda.libdevice.round(fx)
        ty = tl.extra.cuda.libdevice.round(fy)
        tz = tl.extra.cuda.libdevice.round(fz)

        # correction = translation @ frac_matrix.T
        cx = tx * f0 + ty * f1 + tz * f2
        cy = tx * f3 + ty * f4 + tz * f5
        cz = tx * f6 + ty * f7 + tz * f8

        # Wrapped diff
        wx = dx - cx
        wy = dy - cy
        wz = dz - cz

        r2 = wx * wx + wy * wy + wz * wz

        # 5-Gaussian density
        density = (
            An0 * tl.exp(-pi_sq * r2 / Bt0)
            + An1 * tl.exp(-pi_sq * r2 / Bt1)
            + An2 * tl.exp(-pi_sq * r2 / Bt2)
            + An3 * tl.exp(-pi_sq * r2 / Bt3)
            + An4 * tl.exp(-pi_sq * r2 / Bt4)
        )

        # Load voxel indices and compute flat index
        vi3 = (base + v) * 3
        ix = tl.load(voxel_idx_ptr + vi3 + 0, mask=mask).to(tl.int64)
        iy = tl.load(voxel_idx_ptr + vi3 + 1, mask=mask).to(tl.int64)
        iz = tl.load(voxel_idx_ptr + vi3 + 2, mask=mask).to(tl.int64)
        flat_idx = ix * (ny * nz) + iy * nz + iz

        # Atomic add to density map
        tl.atomic_add(density_map_ptr + flat_idx, density, mask=mask)


# =============================================================================
# Backward kernel
# =============================================================================

@triton.jit
def _density_bwd_kernel(
    # Forward inputs (read-only)
    surr_coords_ptr,
    voxel_idx_ptr,
    grad_density_map_ptr,  # (nx*ny*nz,) gradient from upstream
    xyz_ptr,
    b_ptr,
    inv_frac_ptr,
    frac_ptr,
    A_ptr,
    B_ptr,
    occ_ptr,
    # Gradient outputs (accumulated via atomic add)
    grad_xyz_ptr,          # (N_atoms, 3)
    grad_b_ptr,            # (N_atoms,)
    grad_occ_ptr,          # (N_atoms,)
    # Dimensions
    N_voxels: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program per atom. Accumulates gradients across voxels."""
    atom = tl.program_id(0)

    # Load per-atom data (same as forward)
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

    # Clamp masks for b gradient (zero grad when clamped)
    clamp0 = ((B0 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp1 = ((B1 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp2 = ((B2 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp3 = ((B3 + b_iso) * 0.25 > 0.1).to(tl.float32)
    clamp4 = ((B4 + b_iso) * 0.25 > 0.1).to(tl.float32)

    pi_1p5: tl.constexpr = 5.568327996831708
    An0 = A0 * occ * pi_1p5 / (Bt0 * tl.sqrt(Bt0))
    An1 = A1 * occ * pi_1p5 / (Bt1 * tl.sqrt(Bt1))
    An2 = A2 * occ * pi_1p5 / (Bt2 * tl.sqrt(Bt2))
    An3 = A3 * occ * pi_1p5 / (Bt3 * tl.sqrt(Bt3))
    An4 = A4 * occ * pi_1p5 / (Bt4 * tl.sqrt(Bt4))

    # Matrices
    if0 = tl.load(inv_frac_ptr + 0); if1 = tl.load(inv_frac_ptr + 1); if2 = tl.load(inv_frac_ptr + 2)
    if3 = tl.load(inv_frac_ptr + 3); if4 = tl.load(inv_frac_ptr + 4); if5 = tl.load(inv_frac_ptr + 5)
    if6 = tl.load(inv_frac_ptr + 6); if7 = tl.load(inv_frac_ptr + 7); if8 = tl.load(inv_frac_ptr + 8)

    f0 = tl.load(frac_ptr + 0); f1 = tl.load(frac_ptr + 1); f2 = tl.load(frac_ptr + 2)
    f3 = tl.load(frac_ptr + 3); f4 = tl.load(frac_ptr + 4); f5 = tl.load(frac_ptr + 5)
    f6 = tl.load(frac_ptr + 6); f7 = tl.load(frac_ptr + 7); f8 = tl.load(frac_ptr + 8)

    pi_sq: tl.constexpr = 9.869604401089358

    # Accumulators for this atom's gradients
    g_ax = 0.0; g_ay = 0.0; g_az = 0.0
    g_b = 0.0
    g_occ = 0.0

    v_offsets = tl.arange(0, BLOCK_V)
    base = atom * N_voxels

    for v_start in range(0, N_voxels, BLOCK_V):
        v = v_start + v_offsets
        mask = v < N_voxels

        # Recompute forward quantities
        idx3 = (base + v) * 3
        sx = tl.load(surr_coords_ptr + idx3 + 0, mask=mask, other=0.0)
        sy = tl.load(surr_coords_ptr + idx3 + 1, mask=mask, other=0.0)
        sz = tl.load(surr_coords_ptr + idx3 + 2, mask=mask, other=0.0)

        dx = sx - ax; dy = sy - ay; dz = sz - az

        fx = dx * if0 + dy * if1 + dz * if2
        fy = dx * if3 + dy * if4 + dz * if5
        fz = dx * if6 + dy * if7 + dz * if8

        tx = tl.extra.cuda.libdevice.round(fx)
        ty = tl.extra.cuda.libdevice.round(fy)
        tz = tl.extra.cuda.libdevice.round(fz)

        cx = tx * f0 + ty * f1 + tz * f2
        cy = tx * f3 + ty * f4 + tz * f5
        cz = tx * f6 + ty * f7 + tz * f8

        wx = dx - cx; wy = dy - cy; wz = dz - cz
        r2 = wx * wx + wy * wy + wz * wz

        # Gather upstream gradient at voxel locations
        vi3 = (base + v) * 3
        ix = tl.load(voxel_idx_ptr + vi3 + 0, mask=mask, other=0).to(tl.int64)
        iy = tl.load(voxel_idx_ptr + vi3 + 1, mask=mask, other=0).to(tl.int64)
        iz = tl.load(voxel_idx_ptr + vi3 + 2, mask=mask, other=0).to(tl.int64)
        flat_idx = ix * (ny * nz) + iy * nz + iz
        grad_out = tl.load(grad_density_map_ptr + flat_idx, mask=mask, other=0.0)

        # Exponentials for each Gaussian
        e0 = tl.exp(-pi_sq * r2 / Bt0)
        e1 = tl.exp(-pi_sq * r2 / Bt1)
        e2 = tl.exp(-pi_sq * r2 / Bt2)
        e3 = tl.exp(-pi_sq * r2 / Bt3)
        e4 = tl.exp(-pi_sq * r2 / Bt4)

        # --- Gradient w.r.t. xyz ---
        # d(density)/d(xyz_i) = 2*pi^2 * diff_wrapped_i * sum_g(A_norm_g * exp_g / B_total_g)
        # (sign: diff = surr - xyz, so d(diff)/d(xyz) = -1, and d(r2)/d(diff_i) = 2*diff_i)
        coeff_xyz = (
            An0 * e0 / Bt0 + An1 * e1 / Bt1 + An2 * e2 / Bt2
            + An3 * e3 / Bt3 + An4 * e4 / Bt4
        )
        scale_xyz = grad_out * 2.0 * pi_sq * coeff_xyz

        # Positive because: d_loss/d_xyz = grad_out * d_density/d_xyz
        # d_density/d_xyz_i = sum_g An_g * exp_g * (-pi_sq / Bt_g) * 2 * w_i * (-1)
        #                   = 2 * pi_sq * w_i * sum_g(An_g * exp_g / Bt_g)
        g_ax += tl.sum(tl.where(mask, scale_xyz * wx, 0.0), axis=0)
        g_ay += tl.sum(tl.where(mask, scale_xyz * wy, 0.0), axis=0)
        g_az += tl.sum(tl.where(mask, scale_xyz * wz, 0.0), axis=0)

        # --- Gradient w.r.t. b ---
        # d(density)/d(B_total_g) = An_g * exp_g * (-1.5/Bt_g + pi_sq*r2/Bt_g^2)
        # d(B_total_g)/d(b) = 0.25 (if not clamped)
        db0 = An0 * e0 * (-1.5 / Bt0 + pi_sq * r2 / (Bt0 * Bt0)) * clamp0
        db1 = An1 * e1 * (-1.5 / Bt1 + pi_sq * r2 / (Bt1 * Bt1)) * clamp1
        db2 = An2 * e2 * (-1.5 / Bt2 + pi_sq * r2 / (Bt2 * Bt2)) * clamp2
        db3 = An3 * e3 * (-1.5 / Bt3 + pi_sq * r2 / (Bt3 * Bt3)) * clamp3
        db4 = An4 * e4 * (-1.5 / Bt4 + pi_sq * r2 / (Bt4 * Bt4)) * clamp4
        g_b += tl.sum(tl.where(mask, grad_out * 0.25 * (db0 + db1 + db2 + db3 + db4), 0.0), axis=0)

        # --- Gradient w.r.t. occ ---
        # d(density)/d(occ) = density / occ  (since An_g is linear in occ)
        density = An0 * e0 + An1 * e1 + An2 * e2 + An3 * e3 + An4 * e4
        # Avoid division by zero; if occ==0 the gradient is the density formula without occ
        g_occ += tl.sum(tl.where(mask, grad_out * tl.where(occ != 0.0, density / occ, 0.0), 0.0), axis=0)

    # Write accumulated gradients
    tl.store(grad_xyz_ptr + atom * 3 + 0, g_ax)
    tl.store(grad_xyz_ptr + atom * 3 + 1, g_ay)
    tl.store(grad_xyz_ptr + atom * 3 + 2, g_az)
    tl.store(grad_b_ptr + atom, g_b)
    tl.store(grad_occ_ptr + atom, g_occ)


# =============================================================================
# Autograd wrapper
# =============================================================================

class _FusedDensityFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        surrounding_coords,  # (N_atoms, N_voxels, 3)
        voxel_indices,       # (N_atoms, N_voxels, 3)
        density_map,         # (nx, ny, nz) — cloned, not modified in-place
        xyz,                 # (N_atoms, 3)
        b,                   # (N_atoms,)
        inv_frac_matrix,     # (3, 3)
        frac_matrix,         # (3, 3)
        A,                   # (N_atoms, 5)
        B,                   # (N_atoms, 5)
        occ,                 # (N_atoms,)
    ):
        N_atoms, N_voxels = surrounding_coords.shape[:2]
        ny, nz = density_map.shape[1], density_map.shape[2]

        # Ensure contiguous
        surrounding_coords = surrounding_coords.contiguous()
        voxel_indices = voxel_indices.contiguous()
        xyz = xyz.contiguous()
        b = b.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        occ = occ.contiguous()
        inv_frac_flat = inv_frac_matrix.contiguous().view(-1)
        frac_flat = frac_matrix.contiguous().view(-1)

        # Clone so the output is owned by this Function and safe for
        # subsequent in-place ops (e.g. anisotropic scatter_add).
        output = density_map.clone()

        # Choose block size for voxel dimension
        BLOCK_V = triton.next_power_of_2(min(N_voxels, 1024))

        grid = (N_atoms,)
        _density_fwd_kernel[grid](
            surrounding_coords, voxel_indices, output.view(-1),
            xyz, b, inv_frac_flat, frac_flat, A, B, occ,
            N_voxels=N_voxels, ny=ny, nz=nz, BLOCK_V=BLOCK_V,
        )

        # Save for backward
        ctx.save_for_backward(
            surrounding_coords, voxel_indices, xyz, b,
            inv_frac_matrix, frac_matrix, A, B, occ,
        )
        ctx.ny = ny
        ctx.nz = nz
        ctx.density_map_shape = density_map.shape

        return output

    @staticmethod
    def backward(ctx, grad_density_map):
        (surrounding_coords, voxel_indices, xyz, b,
         inv_frac_matrix, frac_matrix, A, B, occ) = ctx.saved_tensors
        ny, nz = ctx.ny, ctx.nz

        N_atoms, N_voxels = surrounding_coords.shape[:2]

        grad_density_map = grad_density_map.contiguous()
        inv_frac_flat = inv_frac_matrix.contiguous().view(-1)
        frac_flat = frac_matrix.contiguous().view(-1)

        grad_xyz = torch.zeros_like(xyz)
        grad_b = torch.zeros_like(b)
        grad_occ = torch.zeros_like(occ)

        BLOCK_V = triton.next_power_of_2(min(N_voxels, 1024))

        grid = (N_atoms,)
        _density_bwd_kernel[grid](
            surrounding_coords, voxel_indices, grad_density_map.view(-1),
            xyz, b, inv_frac_flat, frac_flat, A, B, occ,
            grad_xyz, grad_b, grad_occ,
            N_voxels=N_voxels, ny=ny, nz=nz, BLOCK_V=BLOCK_V,
        )

        # Return gradients in same order as forward args:
        # surrounding_coords, voxel_indices, density_map, xyz, b,
        # inv_frac_matrix, frac_matrix, A, B, occ
        return None, None, None, grad_xyz, grad_b, None, None, None, None, grad_occ


# =============================================================================
# Public API
# =============================================================================

def fused_add_to_map_gpu(
    surrounding_coords: torch.Tensor,
    voxel_indices: torch.Tensor,
    density_map: torch.Tensor,
    xyz: torch.Tensor,
    b: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    occ: torch.Tensor,
) -> torch.Tensor:
    """
    Fused GPU density computation using Triton.

    Drop-in replacement for the JIT GPU kernel with full autograd support.
    Fuses PBC wrapping, r² computation, 5-Gaussian evaluation, and scatter-add
    into a single GPU kernel launch.

    Parameters
    ----------
    surrounding_coords : torch.Tensor
        Cartesian coordinates of voxels, shape (N_atoms, N_voxels, 3).
    voxel_indices : torch.Tensor
        Grid indices of voxels, shape (N_atoms, N_voxels, 3).
    density_map : torch.Tensor
        Base electron density map, shape (nx, ny, nz). Not modified in
        place; a clone is taken and the atomic density is accumulated onto it.
    xyz : torch.Tensor
        Atom positions in Cartesian space, shape (N_atoms, 3).
    b : torch.Tensor
        Isotropic B-factors, shape (N_atoms,).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix, shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix, shape (3, 3).
    A : torch.Tensor
        ITC92 amplitude coefficients, shape (N_atoms, 5).
    B : torch.Tensor
        ITC92 width coefficients, shape (N_atoms, 5).
    occ : torch.Tensor
        Atomic occupancies, shape (N_atoms,).

    Returns
    -------
    torch.Tensor
        A new density map equal to ``density_map`` plus the accumulated
        atomic density. The input ``density_map`` is left unchanged.
    """
    return _FusedDensityFunction.apply(
        surrounding_coords, voxel_indices, density_map,
        xyz, b, inv_frac_matrix, frac_matrix, A, B, occ,
    )
