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
    """Slow reference n-dimensional scatter-add: ``source`` ``(N,)`` into ``map`` at
    ``index`` ``(N, ndim)``, returning the modified map.
    """
    for i in range(source.shape[0]):
        idx = tuple(index[i].tolist())
        map[idx] += source[i]
    return map


def scatter_add_nd(source, index, map):
    """Vectorized n-dimensional scatter-add: ``source`` ``(N,)`` into ``map``
    ``(d1..dn)`` at ``index`` ``(N, ndim)``, returning the modified map.
    """
    map_shape = torch.tensor(map.shape, device=index.device, dtype=torch.int64)  # dtype-ok: map_shape for stride/flat-index arithmetic feeding scatter_add; requires int64

    # Convert n-dimensional indices to flat indices
    # For shape (d1, d2, d3, ..., dn), flat_index = i0 * (d1*d2*...*dn) + i1 * (d2*d3*...*dn) + ... + in
    strides = torch.ones(len(map_shape), device=index.device, dtype=torch.int64)  # dtype-ok: strides for flat scatter_add index; requires int64
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
    """Add isotropic atoms to a density map using the ITC92 5-Gaussian parameterization.

    Parameters
    ----------
    surrounding_coords, voxel_indices : torch.Tensor
        Coordinates and map indices of the voxels around each atom,
        ``(N_atoms, N_voxels, 3)``.
    map : torch.Tensor
        Electron density map, ``(nx, ny, nz)``.
    xyz, b, occ : torch.Tensor
        Atom positions ``(N_atoms, 3)``, B-factors in A^2 and occupancies ``(N_atoms,)``.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Fractionalization matrix and its inverse, ``(3, 3)``.
    A, B : torch.Tensor
        ITC92 amplitudes and widths (A^2), ``(N_atoms, 5)`` each.

    Returns
    -------
    torch.Tensor
        The updated map.
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
    """Add anisotropic atoms to a density map, ITC92 5-Gaussian parameterization.

    Same convention as the isotropic case (``B_total = (B_itc92 + B_atomic) / 4``,
    ``rho = A (pi/B_total)^(3/2) exp(-pi^2 r^2 / B_total)``), generalised to::

        B_total_ij = (B_itc92 * delta_ij + 8pi^2 * U_atomic_ij) / 4
        norm       = (pi^3 / det(B_total))^(1/2)
        exponent   = exp(-pi^2 * r^T B_total^-1 r)

    The isotropic ITC92 ``B`` is scalar and so adds only to the **diagonal** of ``B_total``;
    the off-diagonals come solely from the ``8pi^2 U_atomic_ij`` term. All 5 Gaussian
    components are summed.

    Parameters
    ----------
    surrounding_coords, voxel_indices : torch.Tensor
        Coordinates and map indices of the voxels around each atom,
        ``(N_atoms, N_voxels, 3)``.
    map : torch.Tensor
        Electron density map, ``(nx, ny, nz)``.
    xyz, U, occ : torch.Tensor
        Cartesian positions ``(N_atoms, 3)``, ADPs ``(u11, u22, u33, u12, u13, u23)`` in A^2
        ``(N_atoms, 6)``, and occupancies ``(N_atoms,)``.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Fractionalization matrix and its inverse, ``(3, 3)``.
    A, B : torch.Tensor
        ITC92 amplitudes and widths (A^2), ``(N_atoms, 5)`` each.

    Returns
    -------
    torch.Tensor
        The updated map.
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
