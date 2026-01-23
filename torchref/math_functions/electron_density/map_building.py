"""
Electron density map building functions.

Functions for adding atomic contributions to electron density maps
using ITC92 Gaussian parameterization.
"""

import numpy as np
import torch

from torchref.math_functions.coordinates.periodic_boundary import (
    smallest_diff,
    smallest_diff_aniso,
)


def scatter_add_nd_super_slow(source, index, map):
    """
    Non-vectorized n-dimensional scatter add operation (slow reference implementation).

    Parameters
    ----------
    source : torch.Tensor
        Values to add to the map of shape (N,).
    index : torch.Tensor
        Indices where values should be added of shape (N, ndim).
    map : torch.Tensor
        N-dimensional tensor to add values into.

    Returns
    -------
    torch.Tensor
        Modified map with values added.
    """
    for i in range(source.shape[0]):
        idx = tuple(index[i].tolist())
        map[idx] += source[i]
    return map


def scatter_add_nd(source, index, map):
    """
    Vectorized n-dimensional scatter add operation.

    Parameters
    ----------
    source : torch.Tensor
        Values to add to the map of shape (N,).
    index : torch.Tensor
        Indices where values should be added of shape (N, ndim).
    map : torch.Tensor
        N-dimensional tensor of shape (d1, d2, ..., dn) to add values into.

    Returns
    -------
    torch.Tensor
        Modified map with values added.
    """
    map_shape = torch.tensor(map.shape, device=index.device, dtype=torch.int64)

    # Convert n-dimensional indices to flat indices
    # For shape (d1, d2, d3, ..., dn), flat_index = i0 * (d1*d2*...*dn) + i1 * (d2*d3*...*dn) + ... + in
    strides = torch.ones(len(map_shape), device=index.device, dtype=torch.int64)
    for i in range(len(map_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * map_shape[i + 1]

    index_flat = torch.sum(index * strides.unsqueeze(0), dim=-1)

    map_flat = map.view(-1)
    try:
        map_flat.scatter_add_(0, index_flat, source)
    except RuntimeError as e:
        print("Error during scatter_add_: ", e)
        print(
            "Source shape: ",
            source.shape,
            "device: ",
            source.device,
            "dtype: ",
            source.dtype,
        )
        print(
            "Index shape: ",
            index.shape,
            "device: ",
            index.device,
            "dtype: ",
            index.dtype,
        )
        print("Map shape: ", map.shape, "device: ", map.device, "dtype: ", map.dtype)
        raise e
    return map


def vectorized_add_to_map(
    surrounding_coords,
    voxel_indices,
    map,
    xyz,
    b,
    inv_frac_matrix,
    frac_matrix,
    A,
    B,
    occ,
):
    """
    Add atoms to density map using ITC92 Gaussian parameterization.

    Parameters
    ----------
    surrounding_coords : torch.Tensor
        Coordinates of voxels around each atom of shape (N_atoms, N_voxels, 3).
    voxel_indices : torch.Tensor
        Indices of voxels in the map of shape (N_atoms, N_voxels, 3).
    map : torch.Tensor
        Electron density map of shape (nx, ny, nz).
    xyz : torch.Tensor
        Atom positions of shape (N_atoms, 3).
    b : torch.Tensor
        B-factors (thermal parameters) in Angstroms squared of shape (N_atoms,).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix of shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix of shape (3, 3).
    A : torch.Tensor
        ITC92 amplitude coefficients for each atom of shape (N_atoms, 5).
    B : torch.Tensor
        ITC92 width coefficients (b parameters) in Angstroms squared
        for each atom of shape (N_atoms, 5).
    occ : torch.Tensor
        Occupancies for each atom of shape (N_atoms,).

    Returns
    -------
    torch.Tensor
        Updated electron density map.
    """
    # Calculate squared distances with periodic boundary conditions
    # diff_coords shape: (N_atoms, N_voxels)

    diff_coords_squared = smallest_diff(
        surrounding_coords - xyz.unsqueeze(1), inv_frac_matrix, frac_matrix
    )

    B_total = ((B + b.unsqueeze(1)) / 4).clamp(min=1e-1)

    # Normalization constant: (π/B_total)^(3/2)
    normalization = (np.pi / B_total) ** 1.5

    # Scale amplitudes by occupancy and normalization
    A_normalized = A * occ.unsqueeze(1) * normalization

    # Calculate Gaussian with exponent: exp(-π²r²/B_total)
    # Note: diff_coords_squared already contains r²
    gaussian_terms = torch.exp(
        -(np.pi**2) * diff_coords_squared.unsqueeze(2) / B_total.unsqueeze(1)
    )

    # Sum over the 4 Gaussian components
    density = torch.sum(A_normalized.unsqueeze(1) * gaussian_terms, dim=2)

    # Flatten to (N_atoms * N_voxels,)
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)

    # Add to map
    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map


def vectorized_add_to_map_aniso(
    surrounding_coords,
    voxel_indices,
    map,
    xyz,
    U,
    inv_frac_matrix,
    frac_matrix,
    A,
    B,
    occ,
):
    """
    Add anisotropic atoms to density map using ITC92 Gaussian parameterization.

    For anisotropic atoms, the Gaussian is:
    rho(r) = sum_i A_i * exp(-2*pi^2 * dr^T * (U + U_i) * dr)

    where:
    - U is the atomic displacement parameter tensor (6 components)
    - U_i is the ITC92 Gaussian width tensor derived from B_i parameter
    - dr is the distance vector from atom center

    Parameters
    ----------
    surrounding_coords : torch.Tensor
        Coordinates of voxels around each atom of shape (N_atoms, N_voxels, 3).
    voxel_indices : torch.Tensor
        Indices of voxels in the map of shape (N_atoms, N_voxels, 3).
    map : torch.Tensor
        Electron density map of shape (nx, ny, nz).
    xyz : torch.Tensor
        Atom positions in Cartesian coordinates of shape (N_atoms, 3).
    U : torch.Tensor
        Anisotropic displacement parameters in Angstroms squared
        (u11, u22, u33, u12, u13, u23) of shape (N_atoms, 6).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix of shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix of shape (3, 3).
    A : torch.Tensor
        ITC92 amplitude coefficients for each atom of shape (N_atoms, 4).
    B : torch.Tensor
        ITC92 width coefficients (b parameters) in Angstroms squared
        for each atom of shape (N_atoms, 4).
    occ : torch.Tensor
        Occupancies for each atom of shape (N_atoms,).

    Returns
    -------
    torch.Tensor
        Updated electron density map.
    """
    # Calculate distance vectors with periodic boundary conditions
    # diff_coords shape: (N_atoms, N_voxels, 3)
    diff_coords = surrounding_coords - xyz.unsqueeze(1)

    diff_coords = smallest_diff_aniso(diff_coords, inv_frac_matrix, frac_matrix)

    four_pi_sq = 4 * np.pi**2

    # Diagonal U terms
    U_diag = U[:, :3]  # (N_atoms, 3) - u11, u22, u33

    # Compute u_total for each combination of ITC92 component and U diagonal
    B_expanded = B.unsqueeze(2)  # (N_atoms, 4, 1)
    U_diag_expanded = U_diag.unsqueeze(1)  # (N_atoms, 1, 3)

    U_total_diag = 1.0 / (B_expanded + four_pi_sq * U_diag_expanded)  # (N_atoms, 4, 3)

    # Build U_total tensor with proper format [u11, u22, u33, u12, u13, u23]
    U_total = torch.zeros(B.shape[0], B.shape[1], 6, device=B.device, dtype=B.dtype)
    U_total[:, :, :3] = U_total_diag  # u11, u22, u33

    # For off-diagonal terms, use similar approach
    U_off_diag = U[:, 3:]  # (N_atoms, 3) - u12, u13, u23
    U_total[:, :, 3:] = (U_off_diag.unsqueeze(1) / four_pi_sq).expand(
        -1, B.shape[1], -1
    )  # Approximate

    U_matrix = torch.zeros(
        U_total.shape[0],
        U_total.shape[1],
        3,
        3,
        device=U_total.device,
        dtype=U_total.dtype,
    )
    U_matrix[:, :, 0, 0] = U_total[:, :, 0]  # u11
    U_matrix[:, :, 1, 1] = U_total[:, :, 1]  # u22
    U_matrix[:, :, 2, 2] = U_total[:, :, 2]  # u33
    U_matrix[:, :, 0, 1] = U_total[:, :, 3]  # u12
    U_matrix[:, :, 1, 0] = U_total[:, :, 3]  # u12 (symmetric)
    U_matrix[:, :, 0, 2] = U_total[:, :, 4]  # u13
    U_matrix[:, :, 2, 0] = U_total[:, :, 4]  # u13 (symmetric)
    U_matrix[:, :, 1, 2] = U_total[:, :, 5]  # u23
    U_matrix[:, :, 2, 1] = U_total[:, :, 5]  # u23 (symmetric)

    # Compute quadratic form: Δr^T * U * Δr for each Gaussian component
    diff_coords_expanded = diff_coords.unsqueeze(2)  # (N_atoms, N_voxels, 1, 3)
    U_matrix_expanded = U_matrix.unsqueeze(1)  # (N_atoms, 1, 4, 3, 3)

    # First: U * Δr -> (N_atoms, N_voxels, 4, 3)
    U_times_diff = torch.einsum(
        "naijk,namk->naij", U_matrix_expanded, diff_coords_expanded
    )

    # Second: Δr^T * (U * Δr) -> (N_atoms, N_voxels, 4)
    quad_form = torch.einsum("namk,namk->nam", diff_coords_expanded, U_times_diff)

    # Apply occupancy scaling to amplitudes
    A_scaled = A * occ.unsqueeze(1)  # Shape: (N_atoms, 4)

    # Calculate Gaussian density for each component
    gaussian_terms = torch.exp(-2 * np.pi**2 * quad_form)  # (N_atoms, N_voxels, 4)

    # Sum over 4 Gaussian components
    density = torch.sum(
        A_scaled.unsqueeze(1) * gaussian_terms, dim=2
    )  # (N_atoms, N_voxels)

    # Flatten to (N_atoms * N_voxels,)
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)

    # Add to map
    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map
