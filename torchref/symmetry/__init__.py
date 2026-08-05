"""Crystallographic symmetry: space groups, unit cells, map and HKL symmetry.

:class:`SpaceGroup` (``nn.Module`` holding the operations as buffers) is the entry
point, with ``Symmetry`` a bare alias for it. :func:`MapSymmetry` handles real-space
density and :func:`ReciprocalSymmetry` structure-factor grids; all three accept a
space group as a string, an int 1-230, or a gemmi object. :class:`Cell` is separate
-- it wraps the six cell parameters, not a space group.

The grid utilities re-exported here come from ``grid_utils``, which delegates to
``spacegroup``. ``spacegroup`` also defines its own same-named copies, which are
the source of truth and are *not* re-exported.
"""

from .cell import Cell, CellTensor
from .grid_utils import (
    calculate_optimal_grid_size,
    check_grid_compatibility,
    find_fft_friendly_size,
    get_symmetry_grid_requirements,
    is_fft_friendly,
    recommend_grid_size,
)
from .map_symmetry import MapSymmetry, MapSymmetryDirect
from .reciprocal_symmetry import (
    ReciprocalSymmetry,
    ReciprocalSymmetryGrid,
    canonicalize_hkl,
    complete_hkl,
    expand_hkl,
    expand_reciprocal_grid,
    expand_reflections,
    reduce_hkl,
)
from .spacegroup import (
    SpaceGroup,
    SpaceGroupLike,
    get_crystal_system,
    get_operations_as_tensors,
    get_point_group,
    get_symmetry_operations,
    is_centrosymmetric,
    is_same_spacegroup,
    n_operations,
    spacegroup_to_str,
)
from .symmetry import Symmetry

__all__ = [
    # Unit cell
    "Cell",
    # Space group utilities
    "SpaceGroup",
    "SpaceGroupLike",
    "spacegroup_to_str",
    "get_symmetry_operations",
    "get_operations_as_tensors",
    "is_same_spacegroup",
    "get_point_group",
    "get_crystal_system",
    "is_centrosymmetric",
    "n_operations",
    # Base symmetry
    "Symmetry",
    # Real space map symmetry
    "MapSymmetry",
    "MapSymmetryDirect",
    # Reciprocal space symmetry
    "ReciprocalSymmetry",
    "ReciprocalSymmetryGrid",
    "expand_hkl",
    "complete_hkl",
    "reduce_hkl",
    "canonicalize_hkl",
    "expand_reflections",
    "expand_reciprocal_grid",
    # Grid utilities
    "get_symmetry_grid_requirements",
    "check_grid_compatibility",
    "recommend_grid_size",
    "find_fft_friendly_size",
    "is_fft_friendly",
    "calculate_optimal_grid_size",
]
