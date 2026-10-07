"""
PyTorch implementations of mathematical functions for crystallography.

.. deprecated:: 0.6.0
    This module is deprecated. Please import from domain-specific submodules:

    - ``torchref.base.coordinates`` - Coordinate transformations
    - ``torchref.base.reciprocal`` - Reciprocal space calculations
    - ``torchref.base.direct_summation`` - Structure factor calculations
    - ``torchref.base.electron_density`` - Electron density map building
    - ``torchref.base.fourier`` - FFT operations
    - ``torchref.base.scattering`` - Atomic scattering factors
    - ``torchref.base.alignment`` - Coordinate alignment
    - ``torchref.base.metrics`` - R-factors and loss functions
    - ``torchref.base.kernels`` - Optimized kernels

This module is maintained for backward compatibility and re-exports all
functions from the new submodules.

Example (new style - recommended)::

    from torchref.base.coordinates import cartesian_to_fractional_torch
    from torchref.base.metrics import get_rfactors

Example (old style - still works)::

    from torchref.base.math_torch import cartesian_to_fractional_torch
"""

import torch

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
# Re-exports from direct_summation submodule
# =============================================================================
from torchref.base.direct_summation import (
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
# Re-exports from electron_density submodule
# =============================================================================
from torchref.base.electron_density import (
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
# Utility functions (kept here as they don't fit a specific domain)
# =============================================================================


def U_to_matrix(U: torch.Tensor) -> torch.Tensor:
    """
    Convert anisotropic displacement parameters from 6-component vector to 3x3 matrix.

    Parameters
    ----------
    U : torch.Tensor
        Anisotropic displacement parameters in the order
        [u11, u22, u33, u12, u13, u23] of shape (..., 6).

    Returns
    -------
    torch.Tensor
        Anisotropic displacement parameter matrices of shape (..., 3, 3).
    """
    u11 = U[..., 0]
    u22 = U[..., 1]
    u33 = U[..., 2]
    u12 = U[..., 3]
    u13 = U[..., 4]
    u23 = U[..., 5]

    # Build rows and stack to preserve gradient flow
    row0 = torch.stack([u11, u12, u13], dim=-1)
    row1 = torch.stack([u12, u22, u23], dim=-1)
    row2 = torch.stack([u13, u23, u33], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


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
    # Structure factors
    "iso_structure_factor_torched",
    "iso_structure_factor_torched_no_complex",
    "aniso_structure_factor_torched",
    "aniso_structure_factor_torched_no_complex",
    "anharmonic_correction",
    "anharmonic_correction_no_complex",
    "core_deformation",
    "multiplication_quasi_complex_tensor",
    # Electron density
    "vectorized_add_to_map",
    "vectorized_add_to_map_aniso",
    "scatter_add_nd",
    "scatter_add_nd_super_slow",
    "find_relevant_voxels",
    "excise_angstrom_radius_around_coord",
    "add_to_solvent_mask",
    "add_to_phenix_mask",
    "find_solvent_voids",
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
    # Utility functions
    "U_to_matrix",
]
