"""
Mathematical functions for crystallographic computations.

This module provides PyTorch and NumPy implementations of:
- Coordinate transformations (Cartesian <-> fractional)
- Structure factor calculations
- R-factor computations
- French-Wilson intensity conversion
- Atomic scattering factors
- Grid and reciprocal space utilities

Submodules
----------
math_torch
    PyTorch implementations for GPU-accelerated computations.
math_numpy
    NumPy implementations for CPU computations.
french_wilson
    French-Wilson treatment for negative intensities.
get_scattering_factor_torch
    Atomic scattering factor calculations.
reciprocal_space
    Reciprocal space grid generation and utilities.

Example
-------
::

    from torchref.math_functions import FrenchWilson
    from torchref.math_functions import math_torch, math_numpy

    # French-Wilson conversion
    fw = FrenchWilson(spacegroup='P21', cell=cell)
    F_french_wilson = fw(I_obs, sigma_I)

    # Coordinate transformations
    frac_coords = math_torch.cartesian_to_fractional_torch(cart_coords, cell)
"""

# Submodules for direct access
from . import get_scattering_factor_torch, math_numpy, math_torch, reciprocal_space

# Main classes
from .french_wilson import FrenchWilson

# Scattering factor functions
from .get_scattering_factor_torch import (
    calc_scattering_factors_paramtetrization,
    get_parametrization_for_elements,
    get_scattering_factors,
    get_scattering_factors_unique,
)

# Commonly used functions from math_torch
from .math_torch import (
    cartesian_to_fractional_torch,
    find_grid_size,
    fractional_to_cartesian_torch,
    get_d_spacing,
    get_real_grid,
    get_rfactor_torch,
    get_scattering_vectors,
    reciprocal_basis_matrix,
)
from .optimized_kernels import CachedRadiusMask

# Reciprocal space utilities
from .reciprocal_space import (
    compute_d_spacing_batch,
    generate_possible_hkl,
)

__all__ = [
    # Classes
    "FrenchWilson",
    "CachedRadiusMask",
    # Submodules
    "math_torch",
    "math_numpy",
    "get_scattering_factor_torch",
    "reciprocal_space",
    # Coordinate transforms
    "cartesian_to_fractional_torch",
    "fractional_to_cartesian_torch",
    # R-factor
    "get_rfactor_torch",
    # Reciprocal space
    "reciprocal_basis_matrix",
    "get_scattering_vectors",
    "get_d_spacing",
    "generate_possible_hkl",
    "compute_d_spacing_batch",
    # Grid utilities
    "get_real_grid",
    "find_grid_size",
    # Scattering factors
    "get_scattering_factors",
    "get_scattering_factors_unique",
    "get_parametrization_for_elements",
    "calc_scattering_factors_paramtetrization",
]
