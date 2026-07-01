"""CPU per-atom variable-radius density splatting (grouped-separable / fused / aniso).

Each atom is truncated at its own ``N_sigma * sigma_eff`` radius instead of a
single structure-wide radius. The separable CPU path factorizes
``exp(-alpha * r^T G r)`` into 1D per-axis Gaussians (O(r) exp calls, not O(r^3))
which needs a *uniform* box per batch, so a per-atom radius forces grouping atoms
by box. Two choices keep the grouping cheap:

* **Bucket by integer box size** ``box_radius = ceil(r_i / min_voxel)`` (NOT the
  nominal 0.25-A radius): the kernels truncate by the box, so atoms rounding to
  the same box are identical and share one launch.
* **One global sort by (box, center_1d)** -> each bucket is a contiguous,
  cache-sorted slice; then **chunk atoms within a bucket** for L3 locality.

Work drops from ``N * max_box^3`` to ``sum_bucket n * box^3`` while the
factorization is preserved. The per-voxel cores (``_separable_density``,
``_aniso_density_cube``) and the structured scatter are reused verbatim from the
single-radius kernels, so a single-radius plan reproduces them bit-for-bit. No
``torch.compile``.

These functions ADD into the supplied ``density_map`` (so the isotropic and
anisotropic passes accumulate into the same map) and are autograd-connected in
xyz / adp / u / occ.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from torchref.config import dtypes
from torchref.base.electron_density.kernels.offsets import _get_radius_offsets
from torchref.base.electron_density.kernels.cpu.separable import _separable_density
from torchref.base.electron_density.kernels.cpu.aniso import _aniso_density_cube
from torchref.base.electron_density.kernels.cpu.scatter_dispatch import _do_structured_scatter
from torchref.base.electron_density.radius_policy import _u6_to_u3

_PI = math.pi
_PI_SQ = _PI * _PI
_PI_1P5 = _PI * math.sqrt(_PI)
_EIGHT_PI_SQ = 8.0 * _PI * _PI
_CHUNK = 1024


# =========================================================================
# Shared grouping helpers
# =========================================================================
def _cross_term_flags(G: torch.Tensor) -> Tuple[bool, bool, bool]:
    tol = 1e-3 * torch.norm(torch.diagonal(G))
    return (
        bool(torch.abs(G[0, 1]) > tol),
        bool(torch.abs(G[0, 2]) > tol),
        bool(torch.abs(G[1, 2]) > tol),
    )


def _box_radius_per_atom(radius: torch.Tensor, voxel_size: torch.Tensor) -> torch.Tensor:
    """Integer box radius per atom = ceil(r_i / min_voxel) -- the true bucket key."""
    min_voxel = float(voxel_size.min())
    return torch.ceil(radius / min_voxel).to(torch.int64)


def _bucket_by_box(box_radius: torch.Tensor, center_1d: torch.Tensor):
    """Sort atoms by (box_radius, center_1d); return (order, [(box_radius, start, end)]).

    ONE global sort -> each distinct-box bucket is a contiguous, cache-sorted slice.
    """
    uniq = torch.unique(box_radius)
    order_parts, spans, cursor = [], [], 0
    for b in uniq.tolist():
        idx = (box_radius == b).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(center_1d[idx])]
        order_parts.append(idx)
        spans.append((int(b), cursor, cursor + idx.numel()))
        cursor += idx.numel()
    order = (torch.cat(order_parts) if order_parts
             else box_radius.new_zeros(0, dtype=torch.long))
    return order, spans


def _axis_offsets(box_radius: int, device, int_dtype):
    return torch.arange(-box_radius, box_radius + 1, device=device).to(int_dtype)


# =========================================================================
# Isotropic separable (Engine.AUTO CPU path)
# =========================================================================
def _splat_chunked(density_flat, map_size, G, flags, inv_grid, grid_dims, device, dtype,
                   xyz_frac, center_idx, alpha, A_norm, spans, chunk=_CHUNK):
    """Bucket loop (per box size) x chunk loop: factorized cube + structured scatter."""
    nx, ny, nz = grid_dims
    ny_nz = ny * nz
    for box_radius, b0, b1 in spans:
        axis = _axis_offsets(box_radius, device, dtypes.int)
        axis_frac = axis.to(dtype).unsqueeze(0) * inv_grid.unsqueeze(1)
        for s in range(b0, b1, chunk):
            e = min(s + chunk, b1)
            ci = center_idx[s:e]
            center_frac = ci.to(dtype) * inv_grid
            sub = xyz_frac[s:e] - center_frac
            d_frac = axis_frac.unsqueeze(0) - sub.unsqueeze(2)
            d_frac = d_frac - torch.round(d_frac)
            cube = _separable_density(d_frac, alpha[s:e], A_norm[s:e], G, *flags)
            wa = (ci[:, 0:1] + axis.unsqueeze(0)) % nx * ny_nz
            wb = (ci[:, 1:2] + axis.unsqueeze(0)) % ny * nz
            wc = (ci[:, 2:3] + axis.unsqueeze(0)) % nz
            wbwc = wb.unsqueeze(2) + wc.unsqueeze(1)
            density_flat = _do_structured_scatter(cube, wa, wbwc, density_flat, map_size)
    return density_flat


def _iso_setup(xyz, adp, occ, A, B, inv_frac, frac, grid_shape, voxel_size,
               radius_per_atom):
    """Per-atom + shared setup; returns sorted tensors + bucket spans."""
    device, dtype = xyz.device, xyz.dtype
    nx, ny, nz = grid_shape
    ny_nz = ny * nz
    grid_f = torch.tensor(grid_shape, device=device, dtype=dtype)
    inv_grid = 1.0 / grid_f

    G = frac.T @ frac
    flags = _cross_term_flags(G)

    xyz_frac = xyz @ inv_frac.T
    center_idx = torch.round((xyz_frac % 1.0) * grid_f).to(dtypes.int)
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)
    A_norm = A * occ[:, None] * _PI_1P5 / (B_total * torch.sqrt(B_total))
    alpha = _PI_SQ / B_total

    box_radius = _box_radius_per_atom(radius_per_atom, voxel_size)
    center_1d = (center_idx[:, 0].long() * ny_nz
                 + center_idx[:, 1].long() * nz + center_idx[:, 2].long())
    order, spans = _bucket_by_box(box_radius, center_1d)
    return dict(
        G=G, flags=flags, inv_grid=inv_grid, grid_dims=(nx, ny, nz),
        xyz_frac=xyz_frac[order], center_idx=center_idx[order],
        alpha=alpha[order], A_norm=A_norm[order], B_total=B_total[order],
        spans=spans,
    )


def add_isotropic_cpu_separable_var(density_map, xyz, adp, occ, A, B,
                                    inv_frac_matrix, frac_matrix, grid_shape_tuple,
                                    voxel_size, radius_per_atom):
    """Variable-radius grouped-separable isotropic splat; adds into ``density_map``."""
    nx, ny, nz = grid_shape_tuple
    map_size = nx * ny * nz
    st = _iso_setup(xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix,
                    grid_shape_tuple, voxel_size, radius_per_atom)
    density_flat = density_map.reshape(-1)
    density_flat = _splat_chunked(
        density_flat, map_size, st["G"], st["flags"], st["inv_grid"],
        st["grid_dims"], xyz.device, xyz.dtype,
        st["xyz_frac"], st["center_idx"], st["alpha"], st["A_norm"], st["spans"],
    )
    return density_flat.view(nx, ny, nz)


# =========================================================================
# Isotropic fused sphere (Engine.EAGER CPU path)
# =========================================================================
def _splat_chunked_sphere(density_flat, G, inv_grid, grid_dims, voxel_size, device, dtype,
                          xyz_frac, center_idx, B_total, A_norm, spans, chunk=_CHUNK):
    """Per box-bucket x chunk: sphere offsets + r^2 + 5-Gaussian + plain scatter_add_."""
    grid_shape = torch.tensor(grid_dims, device=device)
    min_voxel = float(voxel_size.min())
    nx, ny, nz = grid_dims
    ny_nz = ny * nz
    strides = torch.tensor([ny_nz, nz, 1], device=device, dtype=torch.long)
    for box_radius, b0, b1 in spans:
        offsets = _get_radius_offsets(voxel_size, box_radius * min_voxel, device)  # (R,3)
        for s in range(b0, b1, chunk):
            e = min(s + chunk, b1)
            vi = (center_idx[s:e].unsqueeze(1) + offsets.unsqueeze(0)) % grid_shape
            voxel_frac = vi.to(dtype) * inv_grid
            diff = voxel_frac - xyz_frac[s:e].unsqueeze(1)
            diff = diff - torch.round(diff)
            r_sq = torch.einsum("avi,ij,avj->av", diff, G, diff)
            expo = -_PI_SQ * r_sq.unsqueeze(2) / B_total[s:e].unsqueeze(1)
            dens = torch.einsum("ag,avg->av", A_norm[s:e], torch.exp(expo))
            idx_flat = (vi.to(torch.long) * strides).sum(-1).view(-1)
            density_flat = density_flat.scatter_add(0, idx_flat, dens.reshape(-1))
    return density_flat


def add_isotropic_plain_var(density_map, xyz, adp, occ, A, B,
                            inv_frac_matrix, frac_matrix, grid_shape_tuple,
                            voxel_size, radius_per_atom):
    """Variable-radius grouped fused-sphere isotropic splat; adds into ``density_map``.

    Uses only plain ``scatter_add`` (no custom C++ scatter), so it runs on every
    device (CPU/CUDA/MPS), supports float64, and is double-differentiable -- the
    portable / eager per-atom path."""
    nx, ny, nz = grid_shape_tuple
    st = _iso_setup(xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix,
                    grid_shape_tuple, voxel_size, radius_per_atom)
    density_flat = density_map.reshape(-1)
    density_flat = _splat_chunked_sphere(
        density_flat, st["G"], st["inv_grid"], st["grid_dims"], voxel_size,
        xyz.device, xyz.dtype, st["xyz_frac"], st["center_idx"],
        st["B_total"], st["A_norm"], st["spans"],
    )
    return density_flat.view(nx, ny, nz)


# =========================================================================
# Anisotropic box-splat (full 3D Gaussian, no factorization)
# =========================================================================
def _splat_chunked_aniso(density_flat, map_size, frac, inv_grid, grid_dims, device, dtype,
                         xyz_frac, center_idx, Minv, A_norm, spans, chunk=_CHUNK):
    """Per box-bucket x chunk: full 3D aniso cube (`_aniso_density_cube`) + scatter."""
    nx, ny, nz = grid_dims
    ny_nz = ny * nz
    for box_radius, b0, b1 in spans:
        axis = _axis_offsets(box_radius, device, dtypes.int)
        axis_frac = axis.to(dtype).unsqueeze(0) * inv_grid.unsqueeze(1)
        for s in range(b0, b1, chunk):
            e = min(s + chunk, b1)
            ci = center_idx[s:e]
            sub = xyz_frac[s:e] - ci.to(dtype) * inv_grid
            d_frac = axis_frac.unsqueeze(0) - sub.unsqueeze(2)
            d_frac = d_frac - torch.round(d_frac)
            cube = _aniso_density_cube(d_frac, frac, Minv[s:e], A_norm[s:e])
            wa = (ci[:, 0:1] + axis.unsqueeze(0)) % nx * ny_nz
            wb = (ci[:, 1:2] + axis.unsqueeze(0)) % ny * nz
            wc = (ci[:, 2:3] + axis.unsqueeze(0)) % nz
            wbwc = wb.unsqueeze(2) + wc.unsqueeze(1)
            density_flat = _do_structured_scatter(cube, wa, wbwc, density_flat, map_size)
    return density_flat


def _aniso_setup(xyz, u, occ, A, B, inv_frac, frac, grid_shape, voxel_size,
                 radius_per_atom):
    """Per-atom aniso M/Minv/A_norm + box-size buckets."""
    device, dtype = xyz.device, xyz.dtype
    nx, ny, nz = grid_shape
    ny_nz = ny * nz
    grid_f = torch.tensor(grid_shape, device=device, dtype=dtype)
    inv_grid = 1.0 / grid_f
    N = xyz.shape[0]

    xyz_frac = xyz @ inv_frac.T
    center_idx = torch.round((xyz_frac % 1.0) * grid_f).to(dtypes.int)
    eye = torch.eye(3, dtype=dtype, device=device)
    U3 = _u6_to_u3(u)
    M = (B[:, :, None, None] * eye + _EIGHT_PI_SQ * U3[:, None, :, :]) / 4.0  # (N,5,3,3)
    Minv = torch.linalg.inv(M)
    det = torch.linalg.det(M).clamp(min=1e-10)
    A_norm = A * occ[:, None] * _PI_1P5 / torch.sqrt(det)  # (N,5)

    box_radius = _box_radius_per_atom(radius_per_atom, voxel_size)
    center_1d = (center_idx[:, 0].long() * ny_nz
                 + center_idx[:, 1].long() * nz + center_idx[:, 2].long())
    order, spans = _bucket_by_box(box_radius, center_1d)
    return dict(
        frac=frac, inv_grid=inv_grid, grid_dims=(nx, ny, nz),
        xyz_frac=xyz_frac[order], center_idx=center_idx[order],
        Minv=Minv[order], A_norm=A_norm[order], spans=spans,
    )


def add_anisotropic_cpu_var(real_space_grid, density_map, xyz, u, occ, A, B,
                            inv_frac_matrix, frac_matrix, radius_per_atom, voxel_size):
    """Variable-radius grouped anisotropic box-splat; adds into ``density_map``."""
    nx, ny, nz = real_space_grid.shape[:3]
    map_size = nx * ny * nz
    st = _aniso_setup(xyz, u, occ, A, B, inv_frac_matrix, frac_matrix,
                      (nx, ny, nz), voxel_size, radius_per_atom)
    density_flat = density_map.reshape(-1)
    density_flat = _splat_chunked_aniso(
        density_flat, map_size, st["frac"], st["inv_grid"], st["grid_dims"],
        xyz.device, xyz.dtype, st["xyz_frac"], st["center_idx"],
        st["Minv"], st["A_norm"], st["spans"],
    )
    return density_flat.view(nx, ny, nz)


def _splat_chunked_aniso_plain(density_flat, frac, inv_grid, grid_dims, device, dtype,
                               xyz_frac, center_idx, Minv, A_norm, spans, chunk=_CHUNK):
    """Per box-bucket x chunk: full 3D aniso cube + plain ``scatter_add`` (portable)."""
    nx, ny, nz = grid_dims
    ny_nz = ny * nz
    for box_radius, b0, b1 in spans:
        axis = _axis_offsets(box_radius, device, dtypes.int)
        axis_frac = axis.to(dtype).unsqueeze(0) * inv_grid.unsqueeze(1)
        for s in range(b0, b1, chunk):
            e = min(s + chunk, b1)
            ci = center_idx[s:e]
            sub = xyz_frac[s:e] - ci.to(dtype) * inv_grid
            d_frac = axis_frac.unsqueeze(0) - sub.unsqueeze(2)
            d_frac = d_frac - torch.round(d_frac)
            cube = _aniso_density_cube(d_frac, frac, Minv[s:e], A_norm[s:e])  # (C,L,L,L)
            ix = (ci[:, 0:1] + axis.unsqueeze(0)) % nx          # (C,L)
            iy = (ci[:, 1:2] + axis.unsqueeze(0)) % ny
            iz = (ci[:, 2:3] + axis.unsqueeze(0)) % nz
            flat = (ix[:, :, None, None].to(torch.long) * ny_nz
                    + iy[:, None, :, None].to(torch.long) * nz
                    + iz[:, None, None, :].to(torch.long))      # (C,L,L,L)
            density_flat = density_flat.scatter_add(0, flat.reshape(-1), cube.reshape(-1))
    return density_flat


def add_anisotropic_plain_var(real_space_grid, density_map, xyz, u, occ, A, B,
                              inv_frac_matrix, frac_matrix, radius_per_atom, voxel_size):
    """Variable-radius anisotropic box-splat using plain ``scatter_add``.

    The portable / eager anisotropic per-atom path: no custom C++ scatter, so it
    runs on every device, supports float64, and is double-differentiable."""
    nx, ny, nz = real_space_grid.shape[:3]
    st = _aniso_setup(xyz, u, occ, A, B, inv_frac_matrix, frac_matrix,
                      (nx, ny, nz), voxel_size, radius_per_atom)
    density_flat = density_map.reshape(-1)
    density_flat = _splat_chunked_aniso_plain(
        density_flat, st["frac"], st["inv_grid"], st["grid_dims"],
        xyz.device, xyz.dtype, st["xyz_frac"], st["center_idx"],
        st["Minv"], st["A_norm"], st["spans"],
    )
    return density_flat.view(nx, ny, nz)
