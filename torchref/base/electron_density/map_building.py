"""
Electron density map building functions.

Functions for adding atomic contributions to electron density maps
using ITC92 Gaussian parameterization.
"""

import numpy as np
import torch

from torchref.base.coordinates.periodic_boundary import (
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

    # Sum over the 5 ITC92 Gaussian components (A/B have shape (N_atoms, 5))
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

    Uses the same convention as the isotropic case for consistency:
    - B_total = (B_itc92 + B_atomic) / 4
    - rho = A × (π/B_total)^(3/2) × exp(-π² r² / B_total)

    For anisotropic atoms, this generalizes to:
    - B_atomic_ij = 8π² × U_atomic_ij (standard crystallographic conversion)
    - B_total_ij = (B_itc92 × δ_ij + 8π² × U_atomic_ij) / 4
    - Normalization: (π³ / det(B_total))^(1/2)
    - Exponent: exp(-π² × r^T × B_total^(-1) × r)

    The isotropic ITC92 ``B`` is scalar and therefore adds only to the diagonal
    of ``B_total`` (via ``δ_ij``); the off-diagonal entries come solely from the
    anisotropic ``8π² × U_atomic_ij`` term. All 5 ITC92 Gaussian components are
    summed (``A``/``B`` have shape ``(N_atoms, 5)``).

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
    # Calculate distance vectors with periodic boundary conditions
    diff_coords = surrounding_coords - xyz.unsqueeze(1)
    diff_coords = smallest_diff_aniso(diff_coords, inv_frac_matrix, frac_matrix)

    n_atoms = B.shape[0]
    n_gauss = B.shape[1]

    # Convert atomic U to B using standard crystallographic convention: B = 8π² U
    eight_pi_sq = 8 * np.pi**2

    U_diag = U[:, :3]  # u11, u22, u33
    U_off_diag = U[:, 3:]  # u12, u13, u23

    # Convert U to B
    B_atomic_diag = eight_pi_sq * U_diag  # (N_atoms, 3)
    B_atomic_off = eight_pi_sq * U_off_diag  # (N_atoms, 3)

    # Compute B_total = (B_itc92 + B_atomic) / 4 for each ITC92 component
    # B_itc92 is isotropic, so it only adds to diagonal
    B_expanded = B.unsqueeze(2)  # (N_atoms, n_gauss, 1)
    B_atomic_diag_expanded = B_atomic_diag.unsqueeze(1)  # (N_atoms, 1, 3)

    B_total_diag = (B_expanded + B_atomic_diag_expanded) / 4  # (N_atoms, n_gauss, 3)
    B_total_diag = B_total_diag.clamp(min=0.1)  # Clamp for numerical stability

    # Off-diagonal B_total (ITC92 doesn't contribute to off-diagonals)
    B_total_off = B_atomic_off.unsqueeze(1).expand(-1, n_gauss, -1) / 4

    # Build the B_total matrix for each atom and Gaussian component
    B_matrix = torch.zeros(n_atoms, n_gauss, 3, 3, device=B.device, dtype=B.dtype)
    B_matrix[:, :, 0, 0] = B_total_diag[:, :, 0]
    B_matrix[:, :, 1, 1] = B_total_diag[:, :, 1]
    B_matrix[:, :, 2, 2] = B_total_diag[:, :, 2]
    B_matrix[:, :, 0, 1] = B_total_off[:, :, 0]
    B_matrix[:, :, 1, 0] = B_total_off[:, :, 0]
    B_matrix[:, :, 0, 2] = B_total_off[:, :, 1]
    B_matrix[:, :, 2, 0] = B_total_off[:, :, 1]
    B_matrix[:, :, 1, 2] = B_total_off[:, :, 2]
    B_matrix[:, :, 2, 1] = B_total_off[:, :, 2]

    # Compute inverse of B_total matrix for the exponent
    B_inv = torch.linalg.inv(B_matrix)  # (N_atoms, n_gauss, 3, 3)

    # Compute determinant for normalization
    det_B = torch.linalg.det(B_matrix)  # (N_atoms, n_gauss)
    det_B = det_B.clamp(min=1e-10)

    # Normalization: (π³ / det(B))^(1/2) - generalizes (π/B)^(3/2) to 3D
    normalization = (np.pi**3 / det_B) ** 0.5  # (N_atoms, n_gauss)

    # Scale amplitudes by occupancy and normalization
    A_normalized = A * occ.unsqueeze(1) * normalization  # (N_atoms, n_gauss)

    # Compute quadratic form: r^T × B^(-1) × r for each Gaussian component
    diff_coords_expanded = diff_coords.unsqueeze(2)  # (N_atoms, N_voxels, 1, 3)
    B_inv_expanded = B_inv.unsqueeze(1)  # (N_atoms, 1, n_gauss, 3, 3)

    # First: B^(-1) × r -> (N_atoms, N_voxels, n_gauss, 3)
    Binv_times_r = torch.einsum("naijk,namk->naij", B_inv_expanded, diff_coords_expanded)

    # Second: r^T × (B^(-1) × r) -> (N_atoms, N_voxels, n_gauss)
    quad_form = torch.einsum("namk,namk->nam", diff_coords_expanded, Binv_times_r)

    # Calculate Gaussian density: exp(-π² × r^T × B^(-1) × r)
    gaussian_terms = torch.exp(-np.pi**2 * quad_form)  # (N_atoms, N_voxels, n_gauss)

    # Sum over Gaussian components
    density = torch.sum(
        A_normalized.unsqueeze(1) * gaussian_terms, dim=2
    )  # (N_atoms, N_voxels)

    # Flatten and add to map
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)

    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map
