"""Fused CPU box-splat for isotropic atoms (``Engine.EAGER`` on CPU).

Computes voxel fractional coordinates directly from integer indices (no
Cartesian round-trip) and accumulates with a plain ``scatter_add_`` — the
double-differentiable reference for CPU eager.
"""

import math

import torch

from torchref.base.electron_density.kernels.offsets import _get_radius_offsets


def _add_isotropic_cpu_fused(
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
    """Fused CPU path for isotropic atoms.

    Avoids the expensive Cartesian↔Fractional round-trip by computing voxel
    fractional coordinates directly from integer indices. Processes atoms in
    chunks to keep intermediates in L3 cache.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    # --- Radius mask (cached) ---
    local_offsets = _get_radius_offsets(voxel_size, radius_angstrom, device)

    # --- Constants ---
    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix  # metric tensor
    inv_grid = 1.0 / grid_shape_float

    ny_nz = int(grid_shape[1]) * int(grid_shape[2])
    nz_val = int(grid_shape[2])
    strides = torch.tensor([ny_nz, nz_val, 1], device=device, dtype=torch.long)

    # --- Atom fractional coords & center indices ---
    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0  # only used for index computation
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).long()  # (N, 3)

    # --- B_total and normalized amplitudes (small, atom-level tensors) ---
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)  # (N, 5)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))  # (N, 5)

    # --- Process in chunks for cache efficiency ---
    N = xyz.shape[0]
    CHUNK = 1024

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)

        # Voxel indices (wrapped for scatter and frac coord computation)
        vi = (
            center_idx[start:end].unsqueeze(1) + local_offsets.unsqueeze(0)
        ) % grid_shape
        # shape: (C, R, 3)

        # Fractional voxel positions — direct from integer indices
        voxel_frac = vi.float() * inv_grid  # (C, R, 3)

        # Fractional diff with PBC — use unwrapped xyz_frac to preserve gradients
        diff_frac = voxel_frac - xyz_frac[start:end].unsqueeze(1)
        diff_frac = diff_frac - torch.round(diff_frac)

        # r² via metric tensor: exact for any cell geometry
        r_sq = torch.einsum("avi,ij,avj->av", diff_frac, G, diff_frac)

        # Gaussian density
        chunk_B = B_total[start:end]
        exponents = -pi_sq * r_sq.unsqueeze(2) / chunk_B.unsqueeze(1)
        density = torch.einsum("ag,avg->av", A_norm[start:end], torch.exp(exponents))

        # Scatter add to map
        idx_flat = (vi.to(torch.long) * strides).sum(-1).view(-1)
        density_map.view(-1).scatter_add_(0, idx_flat, density.reshape(-1))

    return density_map
