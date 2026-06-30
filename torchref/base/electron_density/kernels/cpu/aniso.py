"""Anisotropic Gaussian box-splat for CPU / MPS (``Engine.AUTO``).

Mirrors the isotropic separable box-splat (center index -> local voxel cube ->
per-axis ``d_frac`` -> structured scatter) but evaluates the full 3D anisotropic
Gaussian over the cube (the cross-terms U12/U13/U23 prevent 1D factorization).
"""

import math

import torch

from torchref.base.electron_density.kernels.offsets import _get_box_radius
from torchref.base.electron_density.kernels.cpu.scatter_dispatch import _do_structured_scatter
from torchref.config import dtypes


def _aniso_density_cube(d_frac, frac_matrix, Minv, A_norm):
    """Anisotropic Gaussian density over the local voxel cube.

    The aniso analogue of ``_separable_density`` — but the 3D Gaussian
    does NOT factorize across axes (cross-terms U12/U13/U23), so it builds the
    full Cartesian offset cube and evaluates the quadratic form directly. The
    component loop keeps peak memory at O(C*n^3).

    Parameters
    ----------
    d_frac : (C, 3, n) — PBC-wrapped fractional offsets per axis.
    frac_matrix : (3, 3) — fractional -> Cartesian.
    Minv : (C, 5, 3, 3) — inverse of M_g = (B_g*I + 8*pi^2*U)/4.
    A_norm : (C, 5) — A * occ * pi^1.5 / sqrt(det M_g).

    Returns
    -------
    (C, n, n, n) density cube.
    """
    pi_sq = math.pi * math.pi
    da = d_frac[:, 0, :][:, :, None, None]  # (C, n, 1, 1)
    db = d_frac[:, 1, :][:, None, :, None]  # (C, 1, n, 1)
    dc = d_frac[:, 2, :][:, None, None, :]  # (C, 1, 1, n)
    fm = frac_matrix
    # Cartesian offset cube r = frac_matrix @ d_frac  (each (C, n, n, n))
    cx = fm[0, 0] * da + fm[0, 1] * db + fm[0, 2] * dc
    cy = fm[1, 0] * da + fm[1, 1] * db + fm[1, 2] * dc
    cz = fm[2, 0] * da + fm[2, 1] * db + fm[2, 2] * dc

    C = d_frac.shape[0]
    n = d_frac.shape[2]
    density_cube = d_frac.new_zeros(C, n, n, n)
    for g in range(Minv.shape[1]):
        m00 = Minv[:, g, 0, 0][:, None, None, None]
        m11 = Minv[:, g, 1, 1][:, None, None, None]
        m22 = Minv[:, g, 2, 2][:, None, None, None]
        m01 = Minv[:, g, 0, 1][:, None, None, None]
        m02 = Minv[:, g, 0, 2][:, None, None, None]
        m12 = Minv[:, g, 1, 2][:, None, None, None]
        q = (
            m00 * cx * cx
            + m11 * cy * cy
            + m22 * cz * cz
            + 2.0 * (m01 * cx * cy + m02 * cx * cz + m12 * cy * cz)
        )
        density_cube = density_cube + A_norm[:, g, None, None, None] * torch.exp(
            -pi_sq * q
        )
    return density_cube


def _add_anisotropic_cpu(
    real_space_grid,
    density_map,
    xyz,
    u,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    radius_angstrom,
    voxel_size,
):
    """Optimized box-splat for anisotropic atoms (CPU/MPS).

    Mirrors ``_add_isotropic_cpu_separable`` (center index -> local voxel
    cube -> per-axis ``d_frac`` -> structured scatter) but evaluates the full
    3D anisotropic Gaussian over the cube (no separable 1D factorization).
    """
    device = density_map.device
    grid_shape_tuple = real_space_grid.shape[:3]
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)
    axis_offsets = axis_offsets.to(dtypes.int)

    pi = math.pi
    pi_1p5 = pi * math.sqrt(pi)
    eight_pi_sq = 8.0 * pi * pi
    inv_grid = 1.0 / grid_shape_float

    nx_val = int(grid_shape[0])
    ny_val = int(grid_shape[1])
    nz_val = int(grid_shape[2])
    ny_nz = ny_val * nz_val

    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(dtypes.int)

    # Per-atom, per-Gaussian 3x3 M = (B_g*I + 8*pi^2*U)/4  -> Minv, det, A_norm
    N = xyz.shape[0]
    eye = torch.eye(3, dtype=xyz.dtype, device=device)
    U3 = xyz.new_zeros(N, 3, 3)
    U3[:, 0, 0] = u[:, 0]
    U3[:, 1, 1] = u[:, 1]
    U3[:, 2, 2] = u[:, 2]
    U3[:, 0, 1] = U3[:, 1, 0] = u[:, 3]
    U3[:, 0, 2] = U3[:, 2, 0] = u[:, 4]
    U3[:, 1, 2] = U3[:, 2, 1] = u[:, 5]
    M = (B[:, :, None, None] * eye + eight_pi_sq * U3[:, None, :, :]) / 4.0  # (N,5,3,3)
    Minv = torch.linalg.inv(M)
    det = torch.linalg.det(M).clamp(min=1e-10)
    A_norm = A * occ[:, None] * pi_1p5 / torch.sqrt(det)  # (N, 5)

    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(1)

    # Sort atoms by 1D voxel center for cache-friendly scatter
    center_1d = center_idx[:, 0] * ny_nz + center_idx[:, 1] * nz_val + center_idx[:, 2]
    atom_order = torch.argsort(center_1d)
    xyz_frac = xyz_frac[atom_order]
    center_idx = center_idx[atom_order]
    Minv = Minv[atom_order]
    A_norm = A_norm[atom_order]

    all_wa = (center_idx[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz
    all_wb = (center_idx[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    all_wc = (center_idx[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    all_wbwc = all_wb.unsqueeze(2) + all_wc.unsqueeze(1)

    CHUNK = 1024
    map_size = density_map.numel()
    density_flat = density_map.view(-1)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        center_frac = center_idx[start:end].float() * inv_grid
        sub_grid_offset = xyz_frac[start:end] - center_frac
        d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
        d_frac = d_frac - torch.round(d_frac)  # PBC

        density_cube = _aniso_density_cube(
            d_frac, frac_matrix, Minv[start:end], A_norm[start:end]
        )
        density_flat = _do_structured_scatter(
            density_cube,
            all_wa[start:end],
            all_wbwc[start:end],
            density_flat,
            map_size,
        )

    return density_flat.view(density_map.shape)
