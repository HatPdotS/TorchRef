"""Electron density map building: maps from atomic models, voxel selection around
atoms, solvent masks and the scatter operations behind them.

The headline entry point is :func:`~.main.build_electron_density`, the central
table-dispatched per-atom variable-radius builder: atomic parameters in, full density map
out.
"""

from .map_building import (
    vectorized_add_to_map_aniso,
    scatter_add_nd,
)

# Import optimized JIT kernel version
from torchref.base.electron_density.kernels import vectorized_add_to_map

from .voxel_utils import (
    find_relevant_voxels,
    excise_angstrom_radius_around_coord,
)

from .main import build_electron_density

from .solvent_mask import add_to_solvent_mask

__all__ = [
    # Map building
    "vectorized_add_to_map",
    "vectorized_add_to_map_aniso",
    "scatter_add_nd",
    # Voxel utilities
    "find_relevant_voxels",
    "excise_angstrom_radius_around_coord",
    # Solvent mask
    "add_to_solvent_mask",
    # Central dispatch
    "build_electron_density",
]
