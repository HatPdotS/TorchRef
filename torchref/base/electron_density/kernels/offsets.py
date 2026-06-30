"""Cached voxel-offset helpers for the box-splat density kernels.

Both helpers depend only on ``(voxel_size, radius_angstrom, device)`` and are
cached.  The caches are **dicts** (not single-slot): the dynamic per-atom radius
path launches the splat once per distinct voxel cutoff within a single density
build, so a single-slot cache keyed on the last radius would thrash, recomputing
the meshgrid on every group and every refinement step.
"""

import math

import torch

# =========================================================================
# Separable box radius (axis offsets) — used by the separable CPU / MPS /
# aniso box-splat paths.
# =========================================================================
_box_radius_cache = {}


def _get_box_radius(voxel_size, radius_angstrom, device):
    """Get axis offsets and box size for separable Gaussian splatting (cached).

    Returns
    -------
    axis_offsets : torch.Tensor
        Integer offsets [-box_radius, ..., box_radius], shape (n_axis,).
    n_axis : int
        Cube side length (2 * box_radius + 1).
    """
    key = (tuple(voxel_size.tolist()), float(radius_angstrom), device)
    cached = _box_radius_cache.get(key)
    if cached is not None:
        return cached

    min_voxelsize = voxel_size.min()
    box_radius = int(math.ceil(radius_angstrom / min_voxelsize.item()))
    n_axis = 2 * box_radius + 1
    r = torch.arange(-box_radius, box_radius + 1, device=device)

    result = (r, n_axis)
    _box_radius_cache[key] = result
    return result


# =========================================================================
# Spherical radius mask (local offsets) — used by the fused CPU path.
# =========================================================================
_radius_offsets_cache = {}


def _get_radius_offsets(voxel_size, radius_angstrom, device):
    """Get or compute the integer offsets within a spherical radius.

    Cached across calls since the result depends only on voxel_size and radius.
    """
    key = (tuple(voxel_size.tolist()), float(radius_angstrom), device)
    cached = _radius_offsets_cache.get(key)
    if cached is not None:
        return cached

    min_voxelsize = voxel_size.min()
    box_radius = int(math.ceil(radius_angstrom / min_voxelsize.item()))
    r = torch.arange(-box_radius, box_radius + 1, device=device)
    gx, gy, gz = torch.meshgrid(r, r, r, indexing="ij")
    offsets_all = torch.stack((gx, gy, gz), dim=-1)
    dist = torch.sqrt(((offsets_all.float() * voxel_size) ** 2).sum(-1))
    local_offsets = offsets_all[dist <= radius_angstrom]  # (R, 3) int

    _radius_offsets_cache[key] = local_offsets
    return local_offsets
