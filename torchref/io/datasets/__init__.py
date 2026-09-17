"""
Crystallographic dataset containers.

- :class:`CrystalDataset` -- base dataclass: fields, device moves, save/load
- :class:`ReflectionData` -- one crystal's raw observed reflections
- :class:`ScaledDataset` -- live observations corrected by a shared DatasetScaler
- :class:`FcalcDataset` -- calculated structure factors on a generated HKL set
- :class:`DatasetCollection` -- several ReflectionData on one common HKL grid
"""

from .base import CrystalDataset
from .collection import DatasetCollection
from .fcalc_data import FcalcDataset
from .reflection_data import ReflectionData
from .scaled_dataset import ScaledDataset

__all__ = [
    "CrystalDataset",
    "ReflectionData",
    "ScaledDataset",
    "FcalcDataset",
    "DatasetCollection",
]
