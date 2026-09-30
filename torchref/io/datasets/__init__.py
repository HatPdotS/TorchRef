"""
Crystallographic dataset containers.

- :class:`CrystalDataset` -- base dataclass: fields, device moves, save/load
- :class:`ReflectionData` -- one crystal's raw observed reflections
- :class:`ScaledDataset` -- live observations corrected by a shared DatasetScaler
- :class:`FcalcDataset` -- calculated structure factors on a generated HKL set
- :class:`DatasetCollection` -- several ReflectionData on one common HKL grid
- :func:`merge_to_spacegroup` -- merge a dataset into another space group, with
  Rmerge / Rmeas / CC_sym as :class:`MergeStats`
"""

from .base import CrystalDataset
from .collection import DatasetCollection
from .fcalc_data import FcalcDataset
from .merging import MergeShell, MergeStats, merge_to_spacegroup
from .reflection_data import ReflectionData
from .scaled_dataset import ScaledDataset

__all__ = [
    "CrystalDataset",
    "ReflectionData",
    "ScaledDataset",
    "FcalcDataset",
    "DatasetCollection",
    "merge_to_spacegroup",
    "MergeStats",
    "MergeShell",
]
