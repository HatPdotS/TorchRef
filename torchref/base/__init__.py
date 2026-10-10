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
``electron_density`` (with its kernel-cache helpers), ``fourier`` and ``metrics``.
Not re-exported: the names of ``scattering``, ``alignment`` and ``targets`` and the
three modules, which are imported from their own paths; ``math_torch`` is not
imported at all.
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
# Electron density (from electron_density submodule)
# =============================================================================
from .electron_density import (
    vectorized_add_to_map,
    vectorized_add_to_map_aniso,
    scatter_add_nd,
    find_relevant_voxels,
    excise_angstrom_radius_around_coord,
    add_to_solvent_mask,
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
# Kernels (from electron_density.kernels)
# =============================================================================
from .electron_density.kernels import (
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
    # -------------------------------------------------------------------------
    # Electron density
    # -------------------------------------------------------------------------
    "vectorized_add_to_map",
    "vectorized_add_to_map_aniso",
    "scatter_add_nd",
    "find_relevant_voxels",
    "excise_angstrom_radius_around_coord",
    "add_to_solvent_mask",
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
