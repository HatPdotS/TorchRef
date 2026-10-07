"""
Mathematical functions for crystallographic computations.

This module provides PyTorch implementations of:
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
direct_summation
    Structure factor calculations (isotropic, anisotropic, corrections).
electron_density
    Electron density map building functions.
fourier
    FFT operations and grid utilities.
scattering
    Atomic scattering factors (ITC92 parameterization).
alignment
    Euler-angle rotation matrices.
metrics
    R-factor and loss function calculations.
kernels
    Optimized GPU/CPU kernels for performance-critical operations.

Legacy Submodule (For Backward Compatibility)
---------------------------------------------
math_torch
    PyTorch implementations (deprecated, use domain-specific submodules).
french_wilson
    French-Wilson treatment for negative intensities.

Example
-------
New-style imports (recommended)::

    from torchref.base.coordinates import cartesian_to_fractional_torch
    from torchref.base.metrics import get_rfactors
    from torchref.base.reciprocal import reciprocal_basis_matrix

Legacy imports (still supported)::

    from torchref.base import cartesian_to_fractional_torch
    from torchref.base import math_torch
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
# Coordinate transformations (from coordinates submodule)
# =============================================================================
from .coordinates import (
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    get_fractional_matrix,
    get_inv_fractional_matrix_torch,
    smallest_diff,
    smallest_diff_aniso,
)

# =============================================================================
# Reciprocal space (from reciprocal submodule)
# =============================================================================
from .reciprocal import (
    # Basis
    reciprocal_basis_matrix,
    get_scattering_vectors,
    # HKL
    get_d_spacing,
    generate_possible_hkl,
    # Grid operations
    place_on_grid,
    extract_structure_factor_from_grid,
    # Symmetry
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
)

# =============================================================================
# Scattering factors (from scattering submodule)
# =============================================================================
# =============================================================================
# alignment (from alignment submodule)
# =============================================================================
from .alignment import (
    rotation_matrix_euler_zyz,
)

# =============================================================================
# Metrics (from metrics submodule)
# =============================================================================
from .metrics import (
    get_rfactors,
    binwise_scale,
    nll_xray,
    nll_xray_mean,
    nll_xray_lognormal,
    estimate_sigma_F,
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
    "direct_summation",
    "electron_density",
    "fourier",
    "scattering",
    "alignment",
    "metrics",
    "kernels",
    # -------------------------------------------------------------------------
    # Classes
    # -------------------------------------------------------------------------
    "ReciprocalSymmetryExtractor",
    # -------------------------------------------------------------------------
    # Coordinate transforms
    # -------------------------------------------------------------------------
    "cartesian_to_fractional_torch",
    "fractional_to_cartesian_torch",
    "get_fractional_matrix",
    "get_inv_fractional_matrix_torch",
    "smallest_diff",
    "smallest_diff_aniso",
    # -------------------------------------------------------------------------
    # Reciprocal space
    # -------------------------------------------------------------------------
    "reciprocal_basis_matrix",
    "get_scattering_vectors",
    "get_d_spacing",
    "generate_possible_hkl",
    "place_on_grid",
    "extract_structure_factor_from_grid",
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
    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    "get_rfactors",
    "binwise_scale",
    "nll_xray",
    "nll_xray_mean",
    "nll_xray_lognormal",
    "estimate_sigma_F",
    # -------------------------------------------------------------------------
    # Kernels
    # -------------------------------------------------------------------------
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
]
