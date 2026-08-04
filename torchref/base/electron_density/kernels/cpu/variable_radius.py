"""Portable per-atom variable-radius density splatting.

Reached by ``force_portable`` on any device, by CUDA/MPS float64, and whenever the fused
C++ kernel could not be built. Plain ``scatter_add`` only, so it runs on every device,
supports float64, and is double-differentiable -- which makes it the reference the
accelerator kernels are checked against.

One truncation contract, shared with the Triton, Metal and fused-CPU kernels, so AUTO and
EAGER agree to float noise on every device:

    voxel v gets atom i's density iff ``||w||^2 <= r_i^2``, where ``w`` is the Cartesian
    atom->voxel vector (sphere centred on the ATOM, not on its anchor node) and ``r_i``
    is the raw radius_policy radius,

enumerated over the triclinic-correct per-axis box
``ceil(r * n_axis * ||inv_frac row_axis||)``. See ``sphere_splat.py`` for the canonical
statement.

Out-of-sphere voxels are zeroed rather than dropped, keeping the box dense so one
``scatter_add`` covers the chunk -- some wasted writes, the right trade for a portable
reference. These functions ADD into the supplied ``density_map`` (so the isotropic and
anisotropic passes accumulate into one map) and are autograd-connected in
xyz / adp / u / occ.
"""

from __future__ import annotations

import math

import torch

from torchref.base.electron_density.radius_policy import _u6_to_u3

_PI = math.pi
_PI_SQ = _PI * _PI
_PI_1P5 = _PI * math.sqrt(_PI)
_EIGHT_PI_SQ = 8.0 * _PI * _PI
_CHUNK = 1024


def _bucket_by_radius(radius: torch.Tensor, center_1d: torch.Tensor):
    """Sort atoms by (radius, center_1d); return ``(order, [(radius, start, end)])``.

    Buckets on the radius itself, of which the policy's quantization leaves at most 21
    distinct values; sorting by 1D centre within a bucket keeps the scatter cache-friendly.
    """
    order_parts, spans, cursor = [], [], 0
    for r in torch.unique(radius).tolist():
        idx = (radius == r).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(center_1d[idx])]
        order_parts.append(idx)
        spans.append((float(r), cursor, cursor + idx.numel()))
        cursor += idx.numel()
    order = (torch.cat(order_parts) if order_parts
             else torch.zeros(0, dtype=torch.long, device=radius.device))
    return order, spans


def _axis_half_widths(r: float, inv_frac: torch.Tensor, grid_dims):
    """``ceil(r * n_axis * ||inv_frac row_axis||)`` -- the kernels' enumeration box.

    The norm is taken in float64 **on the CPU**, never on the input's device: this path also
    serves ``force_portable`` on MPS, which has no float64 and raises on ``.double()``.
    float64 matters because the result feeds a ``ceil`` -- a value landing a hair under an
    integer in float32 shrinks the box by one voxel and silently clips the sphere.
    """
    row_norms = torch.linalg.norm(inv_frac.detach().cpu().double(), dim=1)
    return tuple(
        int(math.ceil(r * float(n) * float(row_norms[i])))
        for i, n in enumerate(grid_dims)
    )


def _box_offsets(bh, frac, grid_dims, device, dtype):
    """``(offsets (R,3) int64, off_cart (R,3))`` with ``off_cart = frac @ (offset / n)``
    -- the same ``ox*u_a + oy*u_b + oz*u_c`` the kernels accumulate.
    """
    axes = [torch.arange(-b, b + 1, device=device) for b in bh]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    offsets = torch.stack((gx, gy, gz), dim=-1).reshape(-1, 3)
    inv_grid = 1.0 / torch.tensor(grid_dims, device=device, dtype=dtype)
    return offsets, (offsets.to(dtype) * inv_grid) @ frac.T


def _canonical_setup(xyz, inv_frac, frac, grid_dims, radius_per_atom, dtype):
    """Anchor index, Cartesian sub-voxel residual ``w0``, and radius buckets."""
    device = xyz.device
    nx, ny, nz = grid_dims
    grid_f = torch.tensor(grid_dims, device=device, dtype=dtype)
    xyz_frac = (xyz @ inv_frac.T) % 1.0
    center_idx = torch.round(xyz_frac * grid_f).to(torch.long)
    # w0: atom position relative to its anchor node, in Cartesian. This is what
    # centres the sphere on the atom rather than on the node.
    w0 = (xyz_frac - center_idx.to(dtype) / grid_f) @ frac.T
    center_1d = ((center_idx[:, 0] % nx) * (ny * nz)
                 + (center_idx[:, 1] % ny) * nz + (center_idx[:, 2] % nz))
    order, spans = _bucket_by_radius(radius_per_atom, center_1d)
    return order, spans, center_idx[order], w0[order]


def add_isotropic_plain_var(density_map, xyz, adp, occ, A, B,
                            inv_frac_matrix, frac_matrix, radius_per_atom):
    """Portable canonical-sphere isotropic splat; adds into ``density_map``.

    Signature mirrors
    :func:`~torchref.base.electron_density.kernels.cpu.sphere_splat.add_isotropic_cpu_sphere_var`
    exactly, so the two are interchangeable at the dispatch site. No ``voxel_size`` or
    ``grid_shape``: the grid shape comes from ``density_map``, the box from
    ``inv_frac_matrix``.
    """
    device, dtype = xyz.device, density_map.dtype
    nx, ny, nz = (int(s) for s in density_map.shape)
    grid_dims = (nx, ny, nz)
    strides = torch.tensor([ny * nz, nz, 1], device=device, dtype=torch.long)
    grid_shape = torch.tensor(grid_dims, device=device, dtype=torch.long)

    order, spans, center_idx, w0 = _canonical_setup(
        xyz, inv_frac_matrix, frac_matrix, grid_dims, radius_per_atom, dtype)
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)[order]
    A_norm = (A * occ[:, None])[order] * _PI_1P5 / (B_total * torch.sqrt(B_total))
    r2cut = (radius_per_atom * radius_per_atom)[order]

    density_flat = density_map.reshape(-1)
    for r, b0, b1 in spans:
        bh = _axis_half_widths(r, inv_frac_matrix, grid_dims)
        offsets, off_cart = _box_offsets(bh, frac_matrix, grid_dims, device, dtype)
        for s in range(b0, b1, _CHUNK):
            e = min(s + _CHUNK, b1)
            w = off_cart.unsqueeze(0) - w0[s:e].unsqueeze(1)          # (C,R,3)
            r_sq = (w * w).sum(-1)                                    # (C,R)
            expo = -_PI_SQ * r_sq.unsqueeze(2) / B_total[s:e].unsqueeze(1)
            dens = torch.einsum("ag,avg->av", A_norm[s:e], torch.exp(expo))
            dens = torch.where(r_sq <= r2cut[s:e, None], dens, dens.new_zeros(()))
            vi = (center_idx[s:e].unsqueeze(1) + offsets.unsqueeze(0)) % grid_shape
            idx_flat = (vi * strides).sum(-1).reshape(-1)
            density_flat = density_flat.scatter_add(0, idx_flat, dens.reshape(-1))
    return density_flat.view(nx, ny, nz)


def add_anisotropic_plain_var(density_map, xyz, u, occ, A, B,
                              inv_frac_matrix, frac_matrix, radius_per_atom):
    """Portable canonical-sphere anisotropic splat; adds into ``density_map``.

    Signature mirrors
    :func:`~torchref.base.electron_density.kernels.cpu.sphere_splat.add_anisotropic_cpu_sphere_var`.
    Density is the full 3D Gaussian ``exp(-pi^2 w^T Minv w)`` with
    ``M_g = (B_g*I + 8*pi^2*U)/4``; the *cutoff* is the Euclidean sphere at
    ``radius_per_atom`` (the ellipsoid's isotropic bounding radius), matching the CUDA and
    Metal kernels.
    """
    device, dtype = xyz.device, density_map.dtype
    nx, ny, nz = (int(s) for s in density_map.shape)
    grid_dims = (nx, ny, nz)
    strides = torch.tensor([ny * nz, nz, 1], device=device, dtype=torch.long)
    grid_shape = torch.tensor(grid_dims, device=device, dtype=torch.long)

    order, spans, center_idx, w0 = _canonical_setup(
        xyz, inv_frac_matrix, frac_matrix, grid_dims, radius_per_atom, dtype)
    eye = torch.eye(3, dtype=dtype, device=device)
    M = (B[:, :, None, None] * eye + _EIGHT_PI_SQ * _u6_to_u3(u)[:, None, :, :]) / 4.0
    Minv = torch.linalg.inv(M)[order]                                   # (N,5,3,3)
    det = torch.linalg.det(M).clamp(min=1e-10)[order]
    A_norm = (A * occ[:, None])[order] * _PI_1P5 / torch.sqrt(det)       # (N,5)
    r2cut = (radius_per_atom * radius_per_atom)[order]

    density_flat = density_map.reshape(-1)
    for r, b0, b1 in spans:
        bh = _axis_half_widths(r, inv_frac_matrix, grid_dims)
        offsets, off_cart = _box_offsets(bh, frac_matrix, grid_dims, device, dtype)
        for s in range(b0, b1, _CHUNK):
            e = min(s + _CHUNK, b1)
            w = off_cart.unsqueeze(0) - w0[s:e].unsqueeze(1)            # (C,R,3)
            r_sq = (w * w).sum(-1)                                      # (C,R)
            # q[a,v,g] = w^T Minv_g w
            q = torch.einsum("avi,agij,avj->avg", w, Minv[s:e], w)
            dens = torch.einsum("ag,avg->av", A_norm[s:e], torch.exp(-_PI_SQ * q))
            dens = torch.where(r_sq <= r2cut[s:e, None], dens, dens.new_zeros(()))
            vi = (center_idx[s:e].unsqueeze(1) + offsets.unsqueeze(0)) % grid_shape
            idx_flat = (vi * strides).sum(-1).reshape(-1)
            density_flat = density_flat.scatter_add(0, idx_flat, dens.reshape(-1))
    return density_flat.view(nx, ny, nz)
