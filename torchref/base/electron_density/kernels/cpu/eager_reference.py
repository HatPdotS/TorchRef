"""Eager two-step reference splats (find_relevant_voxels + scatter).

The device-agnostic, double-differentiable reference path used for
``Engine.EAGER`` everywhere and for the CUDA float64 fallback.
"""


def _add_isotropic_original(
    real_space_grid,
    density_map,
    xyz,
    adp,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    radius_angstrom,
):
    """Eager two-step isotropic splat: find_relevant_voxels + vectorized_add_to_map."""
    from torchref.base.electron_density.voxel_utils import find_relevant_voxels
    from torchref.base.electron_density.kernels import vectorized_add_to_map

    surrounding_coords, voxel_indices = find_relevant_voxels(
        real_space_grid,
        xyz,
        radius_angstrom=radius_angstrom,
        inv_frac_matrix=inv_frac_matrix,
    )
    return vectorized_add_to_map(
        surrounding_coords,
        voxel_indices,
        density_map,
        xyz,
        adp,
        inv_frac_matrix,
        frac_matrix,
        A,
        B,
        occ,
    )


def _add_anisotropic_original(
    real_space_grid,
    density_map,
    xyz,
    u,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    radius_angstrom,
):
    """Eager two-step anisotropic splat (find_relevant_voxels + scatter).

    The reference implementation; also the CUDA-eager fallback (float64 /
    ``Engine.EAGER``).
    """
    from torchref.base.electron_density.map_building import vectorized_add_to_map_aniso
    from torchref.base.electron_density.voxel_utils import find_relevant_voxels

    surrounding_coords, voxel_indices = find_relevant_voxels(
        real_space_grid,
        xyz,
        radius_angstrom=radius_angstrom,
        inv_frac_matrix=inv_frac_matrix,
    )
    return vectorized_add_to_map_aniso(
        surrounding_coords,
        voxel_indices,
        density_map,
        xyz,
        u,
        inv_frac_matrix,
        frac_matrix,
        A,
        B,
        occ,
    )
