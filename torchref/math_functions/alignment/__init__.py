"""
Structure alignment and superposition functions.

This submodule provides functions for:
- Superposition of coordinate sets (Kabsch algorithm)
- Rotation operations and conversions
- PDB alignment utilities
"""

from .superposition import (
    superpose_vectors_robust_torch,
    superpose_vectors_robust,
    align_torch,
    get_alignement_matrix,
    align_pdbs,
    get_alignment_matrix,
    apply_transformation,
    apply_transformation_numpy,
    invert_transformation_matrix,
)

from .rotation import (
    rotate_coords_torch,
    rotate_coords_numpy,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    quaternion_to_rotation_matrix,
    random_rotation_uniform,
)

__all__ = [
    # Superposition
    "superpose_vectors_robust_torch",
    "superpose_vectors_robust",
    "align_torch",
    "get_alignement_matrix",
    "align_pdbs",
    "get_alignment_matrix",
    "apply_transformation",
    "apply_transformation_numpy",
    "invert_transformation_matrix",
    # Rotation
    "rotate_coords_torch",
    "rotate_coords_numpy",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_to_axis_angle",
    "quaternion_to_rotation_matrix",
    "random_rotation_uniform",
]
