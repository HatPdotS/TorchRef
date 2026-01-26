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
from .scoring import binned_correlation


from .rotation import (
    rotate_coords_torch,
    rotate_coords_numpy,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    quaternion_to_rotation_matrix,
    random_rotation_uniform,
    rotation_matrix_euler_zyz,
)

from .normalization import (
    compute_radial_shells,
    assign_to_shells,
    compute_anisotropy_correction,
    compute_shell_cv,
    fit_anisotropy_correction,
    apply_anisotropy_correction,
    F_squared_to_E_values,
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
    "rotation_matrix_euler_zyz",
    # Normalization
    "compute_radial_shells",
    "assign_to_shells",
    "compute_anisotropy_correction",
    "compute_shell_cv",
    "fit_anisotropy_correction",
    "apply_anisotropy_correction",
    "F_squared_to_E_values",
]
