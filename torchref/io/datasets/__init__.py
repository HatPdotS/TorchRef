"""
Crystallographic dataset classes.

This module provides PyTorch-based dataset classes for handling
crystallographic data:

- CrystalDataset: Abstract base class
- ReflectionData: Single crystal reflection dataset
- DatasetCollection: Container for multiple related datasets

Examples
--------
>>> from torchref.io.datasets import ReflectionData
>>> data = ReflectionData(device='cuda')
>>> data.load_mtz('observed.mtz')
>>> print(f"Loaded {len(data)} reflections")

>>> from torchref.io.datasets import DatasetCollection
>>> collection = DatasetCollection()
>>> collection.add_dataset('native', native_data)
>>> collection.add_dataset('derivative', derivative_data)
"""

from .base import CrystalDataset
from .reflection_data import ReflectionData
from .collection import DatasetCollection

__all__ = [
    'CrystalDataset',
    'ReflectionData',
    'DatasetCollection',
]
