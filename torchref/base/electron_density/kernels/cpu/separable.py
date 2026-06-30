"""Separable Gaussian box-splat for isotropic atoms (CPU / MPS shared core).

Factorizes exp(-alpha * r^T G r) into 1D Gaussians per axis with 2D cross-term
corrections for non-orthogonal cells, keeping peak memory low. Used by the CPU
``Engine.AUTO`` path and (via ``_get_compiled_separable_density``) the MPS
single-pass path.
"""

import math

import torch

from torchref.base.electron_density.kernels.offsets import _get_box_radius
from torchref.base.electron_density.kernels.cpu.scatter_dispatch import _do_structured_scatter
from torchref.config import dtypes


def _separable_density(
    d_frac: torch.Tensor,
    alpha: torch.Tensor,
    A_norm: torch.Tensor,
    G: torch.Tensor,
    has_ab: bool,
    has_ac: bool,
    has_bc: bool,
) -> torch.Tensor:
    """Separable Gaussian density evaluation.

    Factorizes exp(-alpha * r^T G r) into 1D Gaussians per axis with 2D
    cross-term corrections for non-orthogonal cells. Batches all corrections
    across the 5 ITC92 components and uses einsum where possible.

    For non-orthogonal cells, cross-term exponents are combined with the
    relevant diagonal exponents before taking exp() to avoid float32 overflow
    (exp(-big) * exp(+big) = 0 * inf = NaN).  Each combined 2D block
    exponent corresponds to a principal sub-matrix of G (positive definite),
    guaranteeing the exponent is always <= 0 and exp() is in (0, 1].

    Dispatch by crystal system for optimal performance:
    - Orthogonal (no cross terms): separable 1D products + einsum
    - Hexagonal  (ab only):  combined ab exponent + einsum with e_c
    - Monoclinic (ac only):  combined ac exponent + einsum with e_b
    - General    (bc, or multiple cross terms): full 3D exponent per component

    Parameters
    ----------
    d_frac : (C, 3, n_axis) — fractional distances per axis, PBC-wrapped.
    alpha : (C, N_comp) — pi^2 / B_total.
    A_norm : (C, N_comp) — weighted amplitudes.
    G : (3, 3) — metric tensor frac_matrix.T @ frac_matrix.
    has_ab : bool — whether G[0,1] cross-term is significant.
    has_ac : bool — whether G[0,2] cross-term is significant.
    has_bc : bool — whether G[1,2] cross-term is significant.

    Returns
    -------
    (C, n_axis, n_axis, n_axis) density cube.
    """
    # --- Convert fractional → Cartesian per-axis ---
    cell_lengths = torch.sqrt(torch.diagonal(G))
    d_cart = d_frac * cell_lengths[None, :, None]  # (C, 3, n)

    # --- 1D exponents (always <= 0) ---
    da2 = d_cart[:, 0, :] ** 2
    db2 = d_cart[:, 1, :] ** 2
    dc2 = d_cart[:, 2, :] ** 2

    log_a = -alpha.unsqueeze(2) * da2.unsqueeze(1)  # (C, Nc, n)
    log_b = -alpha.unsqueeze(2) * db2.unsqueeze(1)
    log_c = -alpha.unsqueeze(2) * dc2.unsqueeze(1)

    if not (has_ab or has_ac or has_bc):
        # ---- Orthogonal cells: pure separable, all exp() args <= 0 ----
        e_a = torch.exp(log_a)
        e_b = torch.exp(log_b)
        e_c = torch.exp(log_c)
        e_ab = e_a.unsqueeze(3) * e_b.unsqueeze(2)
        return torch.einsum("cg,cgij,cgk->cijk", A_norm, e_ab, e_c)

    # --- Cross-term coefficients ---
    cos_gamma = G[0, 1] / (cell_lengths[0] * cell_lengths[1])
    cos_beta = G[0, 2] / (cell_lengths[0] * cell_lengths[2])
    cos_alpha = G[1, 2] / (cell_lengths[1] * cell_lengths[2])

    da = d_cart[:, 0, :]
    db = d_cart[:, 1, :]
    dc = d_cart[:, 2, :]
    alpha_4d = alpha[:, :, None, None]  # (C, Nc, 1, 1)

    if has_ab and not has_ac and not has_bc:
        # ---- Hexagonal / trigonal: only ab cross-term ----
        # Combined 2D exponent: -alpha*(da2 + db2 + 2*cos_gamma*da*db)
        # = -alpha * d_ab^T G_ab d_ab <= 0 (G_ab positive definite)
        prod_ab = da.unsqueeze(2) * db.unsqueeze(1)
        log_ab = (
            log_a[:, :, :, None]
            + log_b[:, :, None, :]
            + (-2.0 * alpha_4d * cos_gamma * prod_ab[:, None, :, :])
        )
        slice_ab = torch.exp(log_ab)  # (C, Nc, n, n), all in (0, 1]
        e_c = torch.exp(log_c)
        return torch.einsum("cg,cgij,cgk->cijk", A_norm, slice_ab, e_c)

    if has_ac and not has_ab and not has_bc:
        # ---- Monoclinic (beta != 90): only ac cross-term ----
        # Combined 2D exponent: -alpha*(da2 + dc2 + 2*cos_beta*da*dc)
        # = -alpha * d_ac^T G_ac d_ac <= 0 (G_ac positive definite)
        prod_ac = da.unsqueeze(2) * dc.unsqueeze(1)
        log_ac = (
            log_a[:, :, :, None]
            + log_c[:, :, None, :]
            + (-2.0 * alpha_4d * cos_beta * prod_ac[:, None, :, :])
        )
        e_ac = torch.exp(log_ac)  # (C, Nc, n_a, n_c), all in (0, 1]
        e_b = torch.exp(log_b)
        return torch.einsum("cg,cgj,cgik->cijk", A_norm, e_b, e_ac)

    # ---- General path (triclinic, or multiple cross-terms) ----
    # Combine ALL exponents into a single 3D value per voxel per component
    # to guarantee no overflow.  Component loop keeps memory at O(C*n^3).
    prod_ab = da.unsqueeze(2) * db.unsqueeze(1) if has_ab else None
    prod_ac = da.unsqueeze(2) * dc.unsqueeze(1) if has_ac else None
    prod_bc = db.unsqueeze(2) * dc.unsqueeze(1) if has_bc else None

    C = d_frac.shape[0]
    n = d_frac.shape[2]
    density_cube = d_frac.new_zeros(C, n, n, n)
    for g in range(alpha.shape[1]):
        # Full 3D exponent: -alpha * r^T G r  (always <= 0)
        exp_3d = (
            log_a[:, g, :, None, None]
            + log_b[:, g, None, :, None]
            + log_c[:, g, None, None, :]
        )
        if has_ab:
            exp_3d = exp_3d + (
                -2.0 * alpha[:, g, None, None] * cos_gamma * prod_ab
            ).unsqueeze(
                3
            )  # broadcast (C, n_a, n_b, 1)
        if has_ac:
            exp_3d = exp_3d + (
                -2.0 * alpha[:, g, None, None] * cos_beta * prod_ac
            ).unsqueeze(
                2
            )  # broadcast (C, n_a, 1, n_c)
        if has_bc:
            exp_3d = exp_3d + (
                -2.0 * alpha[:, g, None, None] * cos_alpha * prod_bc
            ).unsqueeze(
                1
            )  # broadcast (C, 1, n_b, n_c)
        density_cube += A_norm[:, g, None, None, None] * torch.exp(exp_3d)

    return density_cube


_compiled_separable_density = None


def _get_compiled_separable_density():
    """Return a torch.compile'd version of _separable_density (lazy, cached)."""
    global _compiled_separable_density
    if _compiled_separable_density is None:
        _compiled_separable_density = torch.compile(_separable_density)
    return _compiled_separable_density


_CHUNK_SIZES = (4096, 2048, 1024, 512)


def _add_isotropic_cpu_separable(
    density_map,
    xyz,
    adp,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    grid_shape_tuple,
    voxel_size,
    radius_angstrom,
):
    """Separable Gaussian splatting for isotropic atoms.

    Factorizes the 3D Gaussian into 1D components along each fractional axis,
    with 2D cross-term corrections for non-zero off-diagonal elements of the
    metric tensor. Uses a component loop for non-orthogonal cells to keep peak
    memory low (~1.7 MB vs ~22 MB for the full 5D intermediate).

    Reduces exp() calls from O(r^3) to O(r) per atom for orthogonal cells,
    and O(r^2) for monoclinic/hexagonal cells. Handles all cell geometries.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    # --- Box radius (cached) ---
    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)
    # int32 indices: the cpu_scatter C++ kernel takes int32, and for any
    # realistic crystallographic grid (nx*ny*nz < 2**31) all scatter indices
    # fit in int32. Halving index bandwidth speeds up the inner loop.
    axis_offsets = axis_offsets.to(dtypes.int)

    # --- Constants ---
    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix  # metric tensor
    inv_grid = 1.0 / grid_shape_float

    nx_val = int(grid_shape[0])
    ny_val = int(grid_shape[1])
    nz_val = int(grid_shape[2])
    ny_nz = ny_val * nz_val

    # --- Atom fractional coords & center indices ---
    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0  # only used for index computation
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(
        dtypes.int
    )  # (N, 3) int32

    # --- B_total, normalized amplitudes, and exponent coefficients ---
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)  # (N, 5)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))  # (N, 5)
    alpha = pi_sq / B_total  # (N, 5)

    # --- Cross-term flags (computed once) ---
    tol = 1e-3 * torch.norm(torch.diagonal(G))
    has_ab = bool(torch.abs(G[0, 1]) > tol)
    has_ac = bool(torch.abs(G[0, 2]) > tol)
    has_bc = bool(torch.abs(G[1, 2]) > tol)

    # --- Precompute fractional axis offsets (shared across chunks) ---
    # axis_offsets_frac[dim, i] = axis_offsets[i] / grid_shape[dim]
    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(
        1
    )  # (3, n_axis)

    # --- Sort atoms by 1D voxel center for cache-friendly scatter ---
    center_1d = center_idx[:, 0] * ny_nz + center_idx[:, 1] * nz_val + center_idx[:, 2]
    atom_order = torch.argsort(center_1d)
    xyz_frac = xyz_frac[atom_order]
    center_idx = center_idx[atom_order]
    alpha = alpha[atom_order]
    A_norm = A_norm[atom_order]

    # --- Precompute 1D scatter indices for ALL atoms (int32) ---
    # (N, n_axis) each, ~0.2 MB per axis for 3k atoms — avoids recomputing per chunk
    all_wa = (center_idx[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz
    all_wb = (center_idx[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    all_wc = (center_idx[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    # 2D yz-plane index: (N, n_axis, n_axis) — cuts outer sum from 2 adds to 1
    all_wbwc = all_wb.unsqueeze(2) + all_wc.unsqueeze(1)

    # --- Process in chunks ---
    N = xyz.shape[0]
    CHUNK = 1024
    map_size = density_map.numel()
    density_flat = density_map.view(-1)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)

        # Sub-grid offset: fractional displacement from atom to nearest grid point
        center_frac = center_idx[start:end].float() * inv_grid  # (C, 3)
        sub_grid_offset = xyz_frac[start:end] - center_frac  # (C, 3)

        # 1D fractional distances along each axis: (C, 3, n_axis)
        d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
        d_frac = d_frac - torch.round(d_frac)  # PBC

        # Density computation → (C, n_axis, n_axis, n_axis)
        density_cube = _separable_density(
            d_frac,
            alpha[start:end],
            A_norm[start:end],
            G,
            has_ab,
            has_ac,
            has_bc,
        )

        density_flat = _do_structured_scatter(
            density_cube,
            all_wa[start:end],
            all_wbwc[start:end],
            density_flat,
            map_size,
        )

    return density_flat.view(density_map.shape)


def _add_isotropic_cpu_separable_compiled(
    density_map,
    xyz,
    adp,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    grid_shape_tuple,
    voxel_size,
    radius_angstrom,
):
    """Compiled variant of separable Gaussian splatting.

    Same algorithm as _add_isotropic_cpu_separable but uses torch.compile on
    _separable_density with decreasing fixed chunk sizes (4096, 2048, 1024, 512)
    to keep compiled shapes stable across different proteins. The small remainder
    runs eagerly to avoid extra recompilation.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    # --- Box radius (cached) ---
    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)
    # int32 indices to match _do_structured_scatter's MPS / CPU C++ kernels;
    # the PyTorch scatter_add_ fallback casts to int64 inside the helper.
    axis_offsets = axis_offsets.to(dtypes.int)

    # --- Constants ---
    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix  # metric tensor
    inv_grid = 1.0 / grid_shape_float

    nx_val = int(grid_shape[0])
    ny_val = int(grid_shape[1])
    nz_val = int(grid_shape[2])
    ny_nz = ny_val * nz_val
    map_size = density_map.numel()

    # --- Atom fractional coords & center indices ---
    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0  # only used for index computation
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(
        dtypes.int
    )  # (N, 3) int32

    # --- B_total, normalized amplitudes, and exponent coefficients ---
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)  # (N, 5)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))  # (N, 5)
    alpha = pi_sq / B_total  # (N, 5)

    # --- Cross-term flags (computed once, passed as compile-time constants) ---
    tol = 1e-3 * torch.norm(torch.diagonal(G))
    has_ab = bool(torch.abs(G[0, 1]) > tol)
    has_ac = bool(torch.abs(G[0, 2]) > tol)
    has_bc = bool(torch.abs(G[1, 2]) > tol)

    # --- Precompute fractional axis offsets (shared across chunks) ---
    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(
        1
    )  # (3, n_axis)

    # --- Process with decreasing fixed chunk sizes for stable compiled shapes ---
    N = xyz.shape[0]
    compiled_fn = _get_compiled_separable_density()
    density_flat = density_map.view(-1)
    offset = 0
    remaining = N

    for chunk_size in _CHUNK_SIZES:
        while remaining >= chunk_size:
            end = offset + chunk_size
            density_flat = _splat_chunk(
                offset,
                end,
                center_idx,
                xyz_frac,
                axis_offsets_frac,
                inv_grid,
                alpha,
                A_norm,
                G,
                has_ab,
                has_ac,
                has_bc,
                axis_offsets,
                nx_val,
                ny_val,
                nz_val,
                ny_nz,
                map_size,
                density_flat,
                compiled_fn,
            )
            offset = end
            remaining -= chunk_size

    # --- Eager remainder (no recompilation for the tail) ---
    if remaining > 0:
        density_flat = _splat_chunk(
            offset,
            offset + remaining,
            center_idx,
            xyz_frac,
            axis_offsets_frac,
            inv_grid,
            alpha,
            A_norm,
            G,
            has_ab,
            has_ac,
            has_bc,
            axis_offsets,
            nx_val,
            ny_val,
            nz_val,
            ny_nz,
            map_size,
            density_flat,
            _separable_density,
        )

    return density_flat.view(density_map.shape)


def _splat_chunk(
    start,
    end,
    center_idx,
    xyz_frac,
    axis_offsets_frac,
    inv_grid,
    alpha,
    A_norm,
    G,
    has_ab,
    has_ac,
    has_bc,
    axis_offsets,
    nx_val,
    ny_val,
    nz_val,
    ny_nz,
    map_size,
    density_flat,
    density_fn,
):
    """Compute separable density for one chunk and scatter into the map.

    Returns the (possibly new) flat density tensor — the C++ and MPS
    structured-scatter backends produce out-of-place results that we have
    to rebind through the chunk loop.
    """
    # Sub-grid offset: fractional displacement from atom to nearest grid point
    center_frac = center_idx[start:end].float() * inv_grid  # (C, 3)
    sub_grid_offset = xyz_frac[start:end] - center_frac  # (C, 3)

    # 1D fractional distances along each axis: (C, 3, n_axis)
    d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
    d_frac = d_frac - torch.round(d_frac)  # PBC

    # Density computation → (C, n_axis, n_axis, n_axis)
    density_cube = density_fn(
        d_frac,
        alpha[start:end],
        A_norm[start:end],
        G,
        has_ab,
        has_ac,
        has_bc,
    )

    # Structured (wa, wbwc) indices — int32; cast to int64 happens inside
    # the helper's PyTorch fallback.
    chunk_center = center_idx[start:end]
    wa = (chunk_center[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz  # (C, n)
    wb = (chunk_center[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    wc = (chunk_center[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    wbwc = wb.unsqueeze(2) + wc.unsqueeze(1)  # (C, n, n)

    return _do_structured_scatter(
        density_cube,
        wa,
        wbwc,
        density_flat,
        map_size,
    )
