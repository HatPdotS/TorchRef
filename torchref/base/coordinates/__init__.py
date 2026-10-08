"""
Coordinate transformation functions for crystallography.

This submodule provides functions for transforming between different
coordinate systems used in crystallography:
- Cartesian <-> fractional coordinate conversions
- Periodic boundary condition handling
- Transformation matrix computations
- Symmetry images: a position under a space-group operation and lattice translation

All of them are PyTorch functions; the cell metric is written out once, in
:func:`~.transforms_torch.get_fractional_matrix`.
"""

from .transforms_torch import (
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    get_fractional_matrix,
    get_inv_fractional_matrix_torch,
)

from .periodic_boundary import (
    smallest_diff,
    smallest_diff_aniso,
)

from .symmetry_images import (
    is_symmetry_image,
    symmetry_image_positions,
)

from .local_frame import (
    frame_is_degenerate,
    local_frame_axes,
    local_frame_coordinates,
    place_local_frame,
)

__all__ = [
    # PyTorch implementations
    "cartesian_to_fractional_torch",
    "fractional_to_cartesian_torch",
    "get_fractional_matrix",
    "get_inv_fractional_matrix_torch",
    # Periodic boundary
    "smallest_diff",
    "smallest_diff_aniso",
    # Symmetry images (non-bonded pair building and scoring)
    "symmetry_image_positions",
    "is_symmetry_image",
    # Local frames (riding hydrogens)
    "local_frame_axes",
    "place_local_frame",
    "local_frame_coordinates",
    "frame_is_degenerate",
]
