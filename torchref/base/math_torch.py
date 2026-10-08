"""Deprecated flat namespace over a subset of the ``torchref.base`` domain modules.

.. deprecated:: 0.6.0
    Importing this module warns ``DeprecationWarning``. Import each name from the
    module that defines it: :mod:`torchref.base.coordinates`,
    :mod:`torchref.base.reciprocal`, :mod:`torchref.base.direct_summation`,
    :mod:`torchref.base.electron_density`, :mod:`torchref.base.fourier`,
    :mod:`torchref.base.metrics`, or :mod:`torchref.base.targets.adp` for
    ``U_to_matrix``.

Only the names in ``__all__`` are re-exported, and nothing is defined here.
"""

import warnings

# =============================================================================
# Re-exports from coordinates submodule
# =============================================================================
from torchref.base.coordinates import (
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    get_fractional_matrix,
    get_inv_fractional_matrix_torch,
    smallest_diff,
    smallest_diff_aniso,
)

# =============================================================================
# Re-exports from reciprocal submodule
# =============================================================================
from torchref.base.reciprocal import (
    reciprocal_basis_matrix,
    get_scattering_vectors,
    get_d_spacing,
    place_on_grid,
    extract_structure_factor_from_grid,
)

# =============================================================================
# Re-exports from electron_density submodule
# =============================================================================
from torchref.base.electron_density import (
    vectorized_add_to_map,
    vectorized_add_to_map_aniso,
    scatter_add_nd,
    find_relevant_voxels,
    excise_angstrom_radius_around_coord,
    add_to_solvent_mask,
)

# =============================================================================
# Re-exports from fourier submodule
# =============================================================================
from torchref.base.fourier import (
    fft,
    ifft,
    get_real_grid,
)

# =============================================================================
# Re-exports from metrics submodule
# =============================================================================
from torchref.base.metrics import (
    get_rfactors,
    nll_xray,
    nll_xray_mean,
    nll_xray_lognormal,
    estimate_sigma_F,
)

# =============================================================================
# Re-exports from targets submodule
# =============================================================================
from torchref.base.targets.adp import U_to_matrix

warnings.warn(
    "torchref.base.math_torch is deprecated: import from the torchref.base domain "
    "modules (coordinates, reciprocal, direct_summation, electron_density, fourier, "
    "metrics, targets.adp) instead",
    DeprecationWarning,
    stacklevel=2,
)

# =============================================================================
# __all__ - Public API
# =============================================================================
__all__ = [
    # Coordinate transforms
    "cartesian_to_fractional_torch",
    "fractional_to_cartesian_torch",
    "get_fractional_matrix",
    "get_inv_fractional_matrix_torch",
    "smallest_diff",
    "smallest_diff_aniso",
    # Reciprocal space
    "reciprocal_basis_matrix",
    "get_scattering_vectors",
    "get_d_spacing",
    "place_on_grid",
    "extract_structure_factor_from_grid",
    # Electron density
    "vectorized_add_to_map",
    "vectorized_add_to_map_aniso",
    "scatter_add_nd",
    "find_relevant_voxels",
    "excise_angstrom_radius_around_coord",
    "add_to_solvent_mask",
    # Fourier
    "fft",
    "ifft",
    "get_real_grid",
    # Metrics
    "get_rfactors",
    "nll_xray",
    "nll_xray_mean",
    "nll_xray_lognormal",
    "estimate_sigma_F",
    # ADPs
    "U_to_matrix",
]
