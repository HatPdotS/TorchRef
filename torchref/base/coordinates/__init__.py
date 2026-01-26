"""
Coordinate transformation functions for crystallography.

This submodule provides functions for transforming between different
coordinate systems used in crystallography:
- Cartesian <-> fractional coordinate conversions
- Periodic boundary condition handling
- Transformation matrix computations

Both PyTorch (GPU-accelerated) and NumPy (CPU) implementations are provided.
"""

from .transforms_torch import (
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    get_fractional_matrix,
    get_inv_fractional_matrix_torch,
)

from .transforms_numpy import (
    cartesian_to_fractional,
    fractional_to_cartesian,
    get_fractional_matrix as get_fractional_matrix_numpy,
    get_inv_fractional_matrix,
    convert_coords_to_fractional,
)

from .periodic_boundary import (
    smallest_diff,
    smallest_diff_aniso,
)

__all__ = [
    # PyTorch implementations
    "cartesian_to_fractional_torch",
    "fractional_to_cartesian_torch",
    "get_fractional_matrix",
    "get_inv_fractional_matrix_torch",
    # NumPy implementations
    "cartesian_to_fractional",
    "fractional_to_cartesian",
    "get_fractional_matrix_numpy",
    "get_inv_fractional_matrix",
    "convert_coords_to_fractional",
    # Periodic boundary
    "smallest_diff",
    "smallest_diff_aniso",
]
