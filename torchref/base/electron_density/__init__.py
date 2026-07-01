"""
Electron density map building functions.

This submodule provides functions for:
- Building electron density maps from atomic models
- Finding relevant voxels around atoms
- Solvent mask generation
- Scatter operations for map building

The headline entry point is :func:`build_electron_density` (from ``main``): the
central, ``Engine``-dispatched per-atom variable-radius density builder that
takes atomic parameters and returns the full density map.
"""

from .map_building import (
    vectorized_add_to_map_aniso,
    scatter_add_nd,
    scatter_add_nd_super_slow,
)

# Import optimized JIT kernel version
from torchref.base.electron_density.kernels import vectorized_add_to_map

from .voxel_utils import (
    find_relevant_voxels,
    excise_angstrom_radius_around_coord,
)

from .main import build_electron_density

from .solvent_mask import (
    add_to_solvent_mask,
    add_to_phenix_mask,
    find_solvent_voids,
)

__all__ = [
    # Map building
    "vectorized_add_to_map",
    "vectorized_add_to_map_aniso",
    "scatter_add_nd",
    "scatter_add_nd_super_slow",
    # Voxel utilities
    "find_relevant_voxels",
    "excise_angstrom_radius_around_coord",
    # Solvent mask
    "add_to_solvent_mask",
    "add_to_phenix_mask",
    "find_solvent_voids",
    # Central dispatch
    "build_electron_density",
]
