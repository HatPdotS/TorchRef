"""
Crystallographic dataset containers.

- :class:`CrystalDataset` -- base dataclass: fields, device moves, save/load
- :class:`ReflectionData` -- one crystal's observed reflections
- :class:`FcalcDataset` -- calculated structure factors on a generated HKL set
- :class:`DatasetCollection` -- several ReflectionData on one common HKL grid
"""

from .base import CrystalDataset
from .collection import DatasetCollection
from .fcalc_data import FcalcDataset
from .reflection_data import ReflectionData

__all__ = [
    "CrystalDataset",
    "ReflectionData",
    "FcalcDataset",
    "DatasetCollection",
]
