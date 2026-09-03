"""
Voxel utility functions for electron density calculations.

Functions for finding relevant voxels around atoms for map building.
"""

import torch

from torchref.config import dtypes


def find_relevant_voxels(real_space_grid, xyz, radius_angstrom=4, inv_frac_matrix=None):
    """Voxels within ``radius_angstrom`` of each atom, vectorized over atoms.

    Parameters
    ----------
    real_space_grid : torch.Tensor
        Cartesian coordinate at each grid point, shape ``(nx, ny, nz, 3)``.
    xyz : torch.Tensor
        Cartesian atom coordinates, shape ``(N, 3)`` or ``(3,)``.
    radius_angstrom : float, optional
        A single fixed radius, and a **legacy fallback**: the production splat
        (``main.build_electron_density``) derives a per-atom radius from each atom's B/U and
        ``torchref.sigma_cutoff_ed``. This helper is reached only from the legacy
        reference/JIT splat path.
    inv_frac_matrix : torch.Tensor, optional
        Cartesian-to-fractional, shape ``(3, 3)``. Required for non-orthogonal cells.

    Returns
    -------
    tuple
        ``(surrounding_coords, voxel_indices_wrapped)``, each ``(N, R, 3)`` for the ``R``
        voxels inside the radius. Only the voxel indices are wrapped -- atom coordinates are
        NOT, because ``smallest_diff()`` does the minimum-image work.
    """
    # Ensure xyz is 2D (N, 3)
    if xyz.ndim == 1:
        xyz = xyz.unsqueeze(0)

    grid_shape = torch.tensor(real_space_grid.shape[:3], device=xyz.device)

    # Get grid origin (first voxel corner)
    grid_origin = real_space_grid[0, 0, 0]

    # Convert atom positions to grid indices
    # For non-orthogonal cells, we must use fractional coordinates
    if inv_frac_matrix is not None:
        # Proper way: Cartesian -> Fractional -> Wrap to [0,1] -> Grid indices
        # This ensures atoms outside the unit cell are correctly wrapped
        xyz_frac = torch.matmul(inv_frac_matrix, xyz.T).T  # (N, 3)
        xyz_frac = xyz_frac % 1.0  # Wrap to [0, 1]
        center_idx = torch.round(xyz_frac * grid_shape.unsqueeze(0)).to(torch.int64)  # dtype-ok: rounded voxel center indices; torch indexing requires int64
    else:
        # Fallback for orthogonal cells (less accurate for non-orthogonal)
        voxelsize = real_space_grid[3, 3, 3] - real_space_grid[2, 2, 2]
        center_idx = torch.round(
            (xyz - grid_origin.unsqueeze(0)) / voxelsize.unsqueeze(0)
        ).to(torch.int64)  # dtype-ok: voxel index cast; torch indexing requires int64

    voxel_indices_wrapped = excise_angstrom_radius_around_coord(
        real_space_grid, center_idx, radius_angstrom
    )

    # Extract coordinates from real_space_grid
    # For each atom, get all surrounding voxel coordinates
    surrounding_coords = real_space_grid[
        voxel_indices_wrapped[..., 0],
        voxel_indices_wrapped[..., 1],
        voxel_indices_wrapped[..., 2],
    ]

    return surrounding_coords, voxel_indices_wrapped


def excise_angstrom_radius_around_coord(
    real_space_grid, start_indices, radius_angstrom=4.0
):
    """Wrapped voxel indices ``(N, R, 3)`` within ``radius_angstrom`` of each of
    ``start_indices`` ``(N, 3)`` on ``real_space_grid`` ``(nx, ny, nz, 3)``.

    Indices are wrapped for periodic boundaries so they stay valid array indices.
    """
    # Ensure xyz is 2D (N, 3)
    if start_indices.ndim == 1:
        start_indices = start_indices.unsqueeze(0)
    grid_shape = torch.tensor(real_space_grid.shape[:3], device=start_indices.device)
    # Get grid origin (first voxel corner)
    voxelsize = real_space_grid[3, 3, 3] - real_space_grid[2, 2, 2]

    min_box_radius = torch.ceil(radius_angstrom / torch.min(voxelsize)).to(dtypes.int)

    gridx = torch.arange(
        -min_box_radius, min_box_radius + 1, device=start_indices.device
    )
    gridy = torch.arange(
        -min_box_radius, min_box_radius + 1, device=start_indices.device
    )
    gridz = torch.arange(
        -min_box_radius, min_box_radius + 1, device=start_indices.device
    )
    x, y, z = torch.meshgrid(gridx, gridy, gridz, indexing="ij")
    coords = torch.stack((x, y, z), dim=-1)

    distance_map = torch.sqrt(
        torch.sum((coords * voxelsize.unsqueeze(0)) ** 2, axis=-1)
    )
    within_radius_mask = distance_map <= radius_angstrom
    local_offsets = coords[within_radius_mask]  # Shape: (N_voxels_within_radius, 3)

    voxel_indices = local_offsets.unsqueeze(0) + start_indices.unsqueeze(1)
    voxel_indices_wrapped = voxel_indices % grid_shape.unsqueeze(0).unsqueeze(0)
    return voxel_indices_wrapped
