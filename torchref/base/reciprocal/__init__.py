"""Reciprocal space functions: basis, Miller indices, symmetry, grids.

Covers the reciprocal basis matrix, HKL generation and d-spacings, reciprocal-space
("late") symmetry, and structure-factor placement/extraction on a grid.
"""

from .basis import (
    reciprocal_basis_matrix,
    get_scattering_vectors,
)

from .hkl import (
    generate_possible_hkl,
    get_d_spacing,
)

from .grid_operations import (
    place_on_grid,
    extract_structure_factor_from_grid,
)

from .symmetry import ReciprocalSymmetryExtractor

__all__ = [
    # Basis functions
    "reciprocal_basis_matrix",
    "get_scattering_vectors",
    # HKL functions
    "generate_possible_hkl",
    "get_d_spacing",
    # Grid operations
    "place_on_grid",
    "extract_structure_factor_from_grid",
    # Symmetry
    "ReciprocalSymmetryExtractor",
]
