"""
Mathematical functions for crystallographic computations.

This module provides PyTorch and NumPy implementations of:
- Coordinate transformations (Cartesian <-> fractional)
- Structure factor calculations
- R-factor computations
- French-Wilson intensity conversion
- Atomic scattering factors
- Grid and reciprocal space utilities

Submodules (New Organization)
-----------------------------
coordinates
    Coordinate transformation functions (Cartesian <-> fractional).
reciprocal
    Reciprocal space calculations (basis, HKL, d-spacing, grid operations).
structure_factors
    Structure factor calculations (isotropic, anisotropic, corrections).
electron_density
    Electron density map building functions.
fourier
    FFT operations and grid utilities.
scattering
    Atomic scattering factors (ITC92 parameterization).
alignment
    Coordinate alignment and superposition functions.
metrics
    R-factor and loss function calculations.
kernels
    Optimized GPU/CPU kernels for performance-critical operations.

Legacy Submodules (For Backward Compatibility)
----------------------------------------------
math_torch
    PyTorch implementations (deprecated, use domain-specific submodules).
math_numpy
    NumPy implementations (deprecated, use domain-specific submodules).
french_wilson
    French-Wilson treatment for negative intensities.
get_scattering_factor_torch
    Atomic scattering factor calculations (use scattering.itc92 instead).

Example
-------
New-style imports (recommended)::

    from torchref.base.coordinates import cartesian_to_fractional_torch
    from torchref.base.metrics import get_rfactor_torch
    from torchref.base.reciprocal import reciprocal_basis_matrix

Legacy imports (still supported)::

    from torchref.base import cartesian_to_fractional_torch
    from torchref.base import math_torch, math_numpy
"""

# =============================================================================
# Domain-specific submodules
# =============================================================================
from . import (
    coordinates,
    reciprocal,
    direct_summation,
    electron_density,
    fourier,
    scattering,
    alignment,
    metrics,
    kernels,
)

# =============================================================================
# Legacy submodules (for backward compatibility)
# =============================================================================
from . import (
    get_scattering_factor_torch,
    math_numpy,
    math_torch,
)

# =============================================================================
# Main classes
# =============================================================================
from .french_wilson import FrenchWilson

# =============================================================================
# Coordinate transformations (from coordinates submodule)
# =============================================================================
from .coordinates import (
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    get_fractional_matrix,
    get_inv_fractional_matrix_torch,
    cartesian_to_fractional,
    fractional_to_cartesian,
    get_inv_fractional_matrix,
    convert_coords_to_fractional,
    smallest_diff,
    smallest_diff_aniso,
)

# =============================================================================
# Reciprocal space (from reciprocal submodule)
# =============================================================================
from .reciprocal import (
    # Basis
    reciprocal_basis_matrix,
    reciprocal_basis_matrix_numpy,
    get_scattering_vectors,
    get_scattering_vectors_numpy,
    get_s,
    # HKL
    get_d_spacing,
    compute_d_spacing_batch,
    generate_possible_hkl,
    # Grid operations
    place_on_grid,
    extract_structure_factor_from_grid,
    apply_translation_phase,
    # Interpolation
    interpolate_structure_factor_from_grid,
    interpolate_complex_from_grid,
    trilinear_interpolate_patterson,
    interpolate_for_rotation,
    smooth_reciprocal_grid,
    # Symmetry
    compute_symmetry_equivalent_hkls,
    compute_translation_phases,
    extract_structure_factors_with_symmetry,
    ReciprocalSymmetryExtractor,
)

# =============================================================================
# Structure factors (from direct_summation submodule)
# =============================================================================
from .direct_summation import (
    iso_structure_factor_torched,
    iso_structure_factor_torched_no_complex,
    aniso_structure_factor_torched,
    aniso_structure_factor_torched_no_complex,
    anharmonic_correction,
    anharmonic_correction_no_complex,
    core_deformation,
    multiplication_quasi_complex_tensor,
)

# =============================================================================
# Electron density (from electron_density submodule)
# =============================================================================
from .electron_density import (
    vectorized_add_to_map,
    vectorized_add_to_map_aniso,
    scatter_add_nd,
    scatter_add_nd_super_slow,
    find_relevant_voxels,
    excise_angstrom_radius_around_coord,
    add_to_solvent_mask,
    add_to_phenix_mask,
    find_solvent_voids,
)

# =============================================================================
# Fourier (from fourier submodule)
# =============================================================================
from .fourier import (
    fft,
    ifft,
    get_real_grid,
    find_grid_size,
    get_real_grid_numpy,
    get_grids,
    put_hkl_on_grid,
)

# =============================================================================
# Scattering factors (from scattering submodule)
# =============================================================================
from .scattering import (
    get_scattering_factors,
    get_scattering_factors_unique,
    get_parametrization_for_elements,
    calc_scattering_factors_paramtetrization,
)


# Legacy scattering factor imports (from get_scattering_factor_torch)
from .get_scattering_factor_torch import (
    calc_scattering_factors_paramtetrization as calc_scattering_factors_paramtetrization_legacy,
)

# =============================================================================
# alignment (from alignment submodule)
# =============================================================================
from .alignment import (
    rotate_coords_torch,
    rotate_coords_numpy,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    quaternion_to_rotation_matrix,
    random_rotation_uniform,
    superpose_vectors_robust_torch,
    superpose_vectors_robust,
    align_torch,
    align_pdbs,
    get_alignment_matrix,
    apply_transformation,
    apply_transformation_numpy,
    invert_transformation_matrix,
    binned_correlation,
    rotation_matrix_euler_zyz,
    compute_radial_shells,
    assign_to_shells,
    compute_anisotropy_correction,
    compute_shell_cv,
    fit_anisotropy_correction,
    apply_anisotropy_correction,
    F_squared_to_E_values
)

# =============================================================================
# Metrics (from metrics submodule)
# =============================================================================
from .metrics import (
    get_rfactor_torch,
    get_rfactor,
    rfactor,
    get_rfactors,
    bin_wise_rfactors,
    calc_outliers,
    calc_outliers_numpy,
    nll_xray,
    nll_xray_sum,
    nll_xray_lognormal,
    log_loss,
    estimate_sigma_I,
    estimate_sigma_F,
    gaussian_to_lognormal_sigma,
    gaussian_to_lognormal_mu,
)

# =============================================================================
# Kernels (from kernels submodule)
# =============================================================================
from .kernels import (
    compute_metric_tensor,
    precompute_fractional_coords,
    warmup,
    get_cache_dir,
    clear_cache,
    fused_gaussian_density,
    fused_aniso_gaussian_density,
    warmup_cuda_operations,
    compute_smallest_diff_squared,
    CachedRadiusMask,
    get_cached_radius_offsets,
)



# =============================================================================
# __all__ - Public API
# =============================================================================
__all__ = [
    # -------------------------------------------------------------------------
    # New submodules
    # -------------------------------------------------------------------------
    "coordinates",
    "reciprocal",
    "structure_factors",
    "electron_density",
    "fourier",
    "scattering",
    "alignment",
    "metrics",
    "kernels",
    # -------------------------------------------------------------------------
    # Legacy submodules (backward compatibility)
    # -------------------------------------------------------------------------
    "math_torch",
    "math_numpy",
    "get_scattering_factor_torch",
    # -------------------------------------------------------------------------
    # Classes
    # -------------------------------------------------------------------------
    "FrenchWilson",
    "CachedRadiusMask",
    "ReciprocalSymmetryExtractor",
    # -------------------------------------------------------------------------
    # Coordinate transforms
    # -------------------------------------------------------------------------
    "cartesian_to_fractional_torch",
    "fractional_to_cartesian_torch",
    "get_fractional_matrix",
    "get_inv_fractional_matrix_torch",
    "cartesian_to_fractional",
    "fractional_to_cartesian",
    "get_inv_fractional_matrix",
    "convert_coords_to_fractional",
    "smallest_diff",
    "smallest_diff_aniso",
    # -------------------------------------------------------------------------
    # Reciprocal space
    # -------------------------------------------------------------------------
    "reciprocal_basis_matrix",
    "reciprocal_basis_matrix_numpy",
    "get_scattering_vectors",
    "get_scattering_vectors_numpy",
    "get_s",
    "get_d_spacing",
    "compute_d_spacing_batch",
    "generate_possible_hkl",
    "place_on_grid",
    "extract_structure_factor_from_grid",
    "apply_translation_phase",
    "interpolate_structure_factor_from_grid",
    "interpolate_complex_from_grid",
    "trilinear_interpolate_patterson",
    "compute_symmetry_equivalent_hkls",
    "compute_translation_phases",
    "extract_structure_factors_with_symmetry",
    "interpolate_for_rotation",
    "smooth_reciprocal_grid",
    # Structure factors
    # -------------------------------------------------------------------------
    "iso_structure_factor_torched",
    "iso_structure_factor_torched_no_complex",
    "aniso_structure_factor_torched",
    "aniso_structure_factor_torched_no_complex",
    "anharmonic_correction",
    "anharmonic_correction_no_complex",
    "core_deformation",
    "multiplication_quasi_complex_tensor",
    # -------------------------------------------------------------------------
    # Electron density
    # -------------------------------------------------------------------------
    "vectorized_add_to_map",
    "vectorized_add_to_map_aniso",
    "scatter_add_nd",
    "scatter_add_nd_super_slow",
    "find_relevant_voxels",
    "excise_angstrom_radius_around_coord",
    "add_to_solvent_mask",
    "add_to_phenix_mask",
    "find_solvent_voids",
    # -------------------------------------------------------------------------
    # Fourier
    # -------------------------------------------------------------------------
    "fft",
    "ifft",
    "get_real_grid",
    "find_grid_size",
    "get_real_grid_numpy",
    "get_grids",
    "put_hkl_on_grid",
    # -------------------------------------------------------------------------
    # Scattering factors
    # -------------------------------------------------------------------------
    "get_scattering_factors",
    "get_scattering_factors_unique",
    "get_parametrization_for_elements",
    "calc_scattering_factors_paramtetrization",
    # -------------------------------------------------------------------------
    # alignment
    # -------------------------------------------------------------------------
    "compute_radial_shells",
    "assign_to_shells",
    "compute_anisotropy_correction",
    "compute_shell_cv",
    "fit_anisotropy_correction",
    "apply_anisotropy_correction",
    "F_squared_to_E_values",
    "rotate_coords_torch",
    "rotate_coords_numpy",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_to_axis_angle",
    "quaternion_to_rotation_matrix",
    "random_rotation_uniform",
    "superpose_vectors_robust_torch",
    "superpose_vectors_robust",
    "align_torch",
    "align_pdbs",
    "get_alignment_matrix",
    "apply_transformation",
    "apply_transformation_numpy",
    "invert_transformation_matrix",
    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    "get_rfactor_torch",
    "get_rfactor",
    "rfactor",
    "get_rfactors",
    "bin_wise_rfactors",
    "calc_outliers",
    "calc_outliers_numpy",
    "nll_xray",
    "nll_xray_sum",
    "nll_xray_lognormal",
    "log_loss",
    "estimate_sigma_I",
    "estimate_sigma_F",
    "gaussian_to_lognormal_sigma",
    "gaussian_to_lognormal_mu",
    # -------------------------------------------------------------------------
    # Kernels
    # -------------------------------------------------------------------------
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
    "fused_gaussian_density",
    "fused_aniso_gaussian_density",
    "warmup_cuda_operations",
    "compute_smallest_diff_squared",
    "get_cached_radius_offsets",
]
