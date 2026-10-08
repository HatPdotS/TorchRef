"""Low-level crystallographic math on PyTorch tensors, the layer the model, data,
refinement and scaling packages build on.

Subpackages
-----------
coordinates
    Cartesian <-> fractional transformations.
reciprocal
    Reciprocal basis, HKL generation and d-spacings, symmetry, grid placement.
direct_summation
    Structure factors by direct summation (isotropic, anisotropic, corrections).
electron_density
    Real-space density building, voxel selection and solvent masks.
fourier
    FFTs and real-space grids.
scattering
    Atomic scattering factors (ITC92) and anomalous corrections.
alignment
    Euler-angle rotation matrices.
metrics
    R-factors, amplitude-space likelihoods and per-bin scaling.
targets
    Tensor-only kernels behind the refinement targets, eager and Triton.
kernels
    Compatibility shim re-exporting :mod:`torchref.base.electron_density.kernels`.

Modules
-------
french_wilson
    French-Wilson conversion of merged intensities to amplitudes.
wilson_outliers
    Model-free outlier detection from Wilson statistics.
math_torch
    Deprecated flat namespace over a subset of the subpackages; importing it warns
    ``DeprecationWarning``.

Re-exported here (``__all__``): every subpackage above except ``targets``, and the
commonly used functions of ``coordinates``, ``reciprocal``, ``direct_summation``,
``electron_density``, ``fourier``, ``metrics`` and ``kernels``. Not re-exported: the
names of ``scattering``, ``alignment`` and ``targets`` and the three modules, which
are imported from their own paths; ``math_torch`` is not imported at all.
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
    # Submodules
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
