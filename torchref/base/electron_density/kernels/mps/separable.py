"""Single-pass separable Gaussian box-splat for isotropic atoms on MPS.

Evaluates the separable density for all atoms at once (via the torch.compile'd
``_separable_density``) and scatters with a single structured scatter, avoiding
the per-chunk op overhead of the chunked CPU paths.
"""

import math

import torch

from torchref.base.electron_density.kernels.cpu.separable import _get_compiled_separable_density
from torchref.base.electron_density.kernels.offsets import _get_box_radius
from torchref.base.electron_density.kernels.cpu.scatter_dispatch import _do_structured_scatter
from torchref.config import dtypes


def _add_isotropic_mps_single(
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
    """Single-pass separable Gaussian splat for isotropic atoms on MPS.

    Evaluates the separable density for all atoms at once (via the
    torch.compile'd ``_separable_density``) and scatters the result into the
    map with a single structured scatter, avoiding the per-chunk op overhead
    of the chunked CPU paths.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)
    axis_offsets = axis_offsets.to(dtypes.int)

    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix
    inv_grid = 1.0 / grid_shape_float

    nx_val = grid_shape_tuple[0]
    ny_val = grid_shape_tuple[1]
    nz_val = grid_shape_tuple[2]
    ny_nz = ny_val * nz_val
    map_size = density_map.numel()

    xyz_frac = xyz @ inv_frac_matrix.T
    xyz_frac_wrapped = xyz_frac % 1.0
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(dtypes.int)

    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))
    alpha = pi_sq / B_total

    tol = 1e-3 * torch.norm(torch.diagonal(G))
    has_ab = bool(torch.abs(G[0, 1]) > tol)
    has_ac = bool(torch.abs(G[0, 2]) > tol)
    has_bc = bool(torch.abs(G[1, 2]) > tol)

    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(1)

    # Math for ALL atoms at once
    center_frac = center_idx.float() * inv_grid
    sub_grid_offset = xyz_frac - center_frac
    d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
    d_frac = d_frac - torch.round(d_frac)

    # Compiled separable density — one shape per structure, one compile
    density_fn = _get_compiled_separable_density()
    density_cube = density_fn(d_frac, alpha, A_norm, G, has_ab, has_ac, has_bc)

    # Structured indices for all atoms
    all_wa = (center_idx[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz
    all_wb = (center_idx[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    all_wc = (center_idx[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    all_wbwc = all_wb.unsqueeze(2) + all_wc.unsqueeze(1)

    # Single scatter call into a flat view of density_map (zero-initialised
    # by the caller in build_electron_density).
    density_flat = density_map.view(-1)
    density_flat = _do_structured_scatter(
        density_cube,
        all_wa,
        all_wbwc,
        density_flat,
        map_size,
    )

    return density_flat.view(density_map.shape)
