"""Crystallographic dataset containers.

- :class:`~.base.CrystalDataset` -- base dataclass: fields, device moves, save/load
- :class:`~.reflection_data.ReflectionData` -- one crystal's raw observed reflections
- :class:`~.scaled_dataset.ScaledDataset` -- live observations corrected by a
  shared DatasetScaler
- :class:`~.fcalc_data.FcalcDataset` -- ``F_calc`` on a generated HKL set
- :class:`~.collection.DatasetCollection` -- several ReflectionData on a common HKL grid
- :func:`~.merging.merge_to_spacegroup` -- merge a dataset into another space
  group, with Rmerge / Rmeas / CC_sym as :class:`~.merging.MergeStats`
- :func:`~.french_wilson.french_wilson_auto` -- French-Wilson amplitudes from a
  dataset's intensities, Miller indices and space group
"""

from .base import CrystalDataset
from .collection import DatasetCollection
from .fcalc_data import FcalcDataset
from .french_wilson import french_wilson_auto
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
    "french_wilson_auto",
]
