"""
Reciprocal space functions for crystallography.

This submodule provides functions for working with reciprocal space:
- Reciprocal basis matrix calculations
- Miller index operations (HKL generation, d-spacing)
- Symmetry operations in reciprocal space
- Grid operations for structure factor placement/extraction
- Interpolation functions for rotation/translation searches
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
)

from .symmetry import (
    compute_symmetry_equivalent_hkls,
    compute_translation_phases,
    extract_structure_factors_with_symmetry,
    ReciprocalSymmetryExtractor,
)

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
    # Symmetry
    "compute_symmetry_equivalent_hkls",
    "compute_translation_phases",
    "extract_structure_factors_with_symmetry",
    "ReciprocalSymmetryExtractor",
]
