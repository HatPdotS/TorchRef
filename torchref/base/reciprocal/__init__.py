"""Reciprocal space functions: basis, Miller indices, symmetry, grids, interpolation.

Covers the reciprocal basis matrix, HKL generation and d-spacings, reciprocal-space
("late") symmetry, structure-factor placement/extraction on a grid, and the
interpolators used by rotation and translation searches.
"""

from .basis import (
    reciprocal_basis_matrix,
    reciprocal_basis_matrix_numpy,
    get_scattering_vectors,
    get_scattering_vectors_numpy,
    get_s,
)

from .hkl import (
    generate_possible_hkl,
    get_d_spacing,
    compute_d_spacing_batch,
)

from .grid_operations import (
    place_on_grid,
    extract_structure_factor_from_grid,
    apply_translation_phase,
)

from .interpolation import (
    interpolate_structure_factor_from_grid,
    interpolate_complex_from_grid,
    trilinear_interpolate_patterson,
    interpolate_for_rotation,
    smooth_reciprocal_grid
)

from .symmetry import ReciprocalSymmetryExtractor

__all__ = [
    # Basis functions
    "reciprocal_basis_matrix",
    "reciprocal_basis_matrix_numpy",
    "get_scattering_vectors",
    "get_scattering_vectors_numpy",
    "get_s",
    # HKL functions
    "generate_possible_hkl",
    "get_d_spacing",
    "compute_d_spacing_batch",
    # Grid operations
    "place_on_grid",
    "extract_structure_factor_from_grid",
    "apply_translation_phase",
    # Interpolation
    "interpolate_structure_factor_from_grid",
    "interpolate_complex_from_grid",
    "trilinear_interpolate_patterson",
    "interpolate_for_rotation",
    "smooth_reciprocal_grid",
    # Symmetry
    "ReciprocalSymmetryExtractor",
]
