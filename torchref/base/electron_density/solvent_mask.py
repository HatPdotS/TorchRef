"""
Solvent mask generation functions.

Functions for creating solvent masks in crystallographic unit cells.
"""

import torch

from torchref.config import dtypes
from torchref.base.coordinates.periodic_boundary import smallest_diff
from .map_building import scatter_add_nd


def add_to_solvent_mask(
    surrounding_coords, voxel_indices, mask, xyz, radius, inv_frac_matrix, frac_matrix
):
    """Set the solvent mask True inside ``radius`` of every atom.

    Parameters
    ----------
    surrounding_coords, voxel_indices : torch.Tensor
        Coordinates and map indices of the voxels around each atom,
        ``(N_atoms, N_voxels, 3)``.
    mask : torch.Tensor
        Solvent mask to update, ``(nx, ny, nz)``. Cast to ``dtypes.int`` first, so a
        mask *already* at that dtype is scatter-added into **in place** while any
        other dtype is copied -- use the return value either way.
    xyz : torch.Tensor
        Atom positions, ``(N_atoms, 3)``.
    radius : float
        Sphere radius per atom, in Angstrom.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Fractionalization matrix and its inverse, ``(3, 3)``.

    Returns
    -------
    torch.Tensor
        The updated mask, always ``torch.bool`` whatever dtype came in.
    """
    mask = mask.to(dtype=dtypes.int)
    # Calculate squared distances with periodic boundary conditions
    diff_coords_squared = smallest_diff(
        surrounding_coords - xyz.unsqueeze(1), inv_frac_matrix, frac_matrix
    )

    # Create boolean mask where distance squared is less than radius squared
    within_sphere = diff_coords_squared <= radius**2  # (N_atoms, N_voxels)

    # Convert boolean to float for addition
    values_to_add = within_sphere.to(dtype=mask.dtype).flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3).to(dtypes.int)

    # Add to mask
    mask = scatter_add_nd(values_to_add, voxel_indices_flat, mask)

    # Ensure mask is binary (0 or 1)
    mask = torch.clamp(mask, max=1.0)

    return mask.to(torch.bool)
