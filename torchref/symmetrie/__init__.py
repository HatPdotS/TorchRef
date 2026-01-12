"""
Symmetry operations module for TorchRef.

Provides crystallographic symmetry operations for:
- Space group handling (using gemmi.SpaceGroup as canonical representation)
- Fractional coordinates (base Symmetry class)
- Real space density maps (MapSymmetry)
- Reciprocal space structure factor grids (ReciprocalSymmetry)

Example
-------
>>> from torchref.symmetrie import SpaceGroup, Symmetry, MapSymmetry, ReciprocalSymmetry
>>>
>>> # Normalize space group from various inputs
>>> sg = SpaceGroup('P21')        # From string
>>> sg = SpaceGroup(4)            # From number
>>> sg = SpaceGroup(gemmi_sg)     # Pass-through gemmi object
>>>
>>> # Base symmetry for coordinates
>>> sym = Symmetry('P21')
>>> transformed_coords = sym(fractional_coords)
>>>
>>> # Real space map symmetry
>>> map_sym = MapSymmetry('P21', map_shape=(64, 64, 64), cell_params=cell)
>>> symmetric_map = map_sym(density_map)
>>>
>>> # Reciprocal space symmetry
>>> recip_sym = ReciprocalSymmetry('P21', grid_shape=(64, 64, 64))
>>> F_averaged = recip_sym(F_grid, mode='average')
"""

from .spacegroup import (
    SpaceGroup,
    SpaceGroupLike,
    spacegroup_to_str,
    get_symmetry_operations,
    get_operations_as_tensors,
    is_same_spacegroup,
    get_point_group,
    get_crystal_system,
    is_centrosymmetric,
    n_operations,
)
from .symmetrie import Symmetry
from .map_symmetry import MapSymmetry, MapSymmetryDirect
from .reciprocal_symmetry import (
    ReciprocalSymmetry,
    ReciprocalSymmetryGrid,
    expand_hkl,
    complete_hkl,
    reduce_hkl,
    expand_reflections,
    expand_reciprocal_grid,
)
from .grid_utils import (
    get_symmetry_grid_requirements,
    check_grid_compatibility,
    recommend_grid_size,
    find_fft_friendly_size,
    is_fft_friendly,
    calculate_optimal_grid_size
)

__all__ = [
    # Space group utilities
    'SpaceGroup',
    'SpaceGroupLike',
    'spacegroup_to_str',
    'get_symmetry_operations',
    'get_operations_as_tensors',
    'is_same_spacegroup',
    'get_point_group',
    'get_crystal_system',
    'is_centrosymmetric',
    'n_operations',
    # Base symmetry
    'Symmetry',
    # Real space map symmetry
    'MapSymmetry',
    'MapSymmetryDirect',
    # Reciprocal space symmetry
    'ReciprocalSymmetry',
    'ReciprocalSymmetryGrid',
    'expand_hkl',
    'complete_hkl',
    'reduce_hkl',
    'expand_reflections',
    'expand_reciprocal_grid',
    # Grid utilities
    'get_symmetry_grid_requirements',
    'check_grid_compatibility',
    'recommend_grid_size',
    'find_fft_friendly_size',
    'is_fft_friendly',
    'calculate_optimal_grid_size',
]
