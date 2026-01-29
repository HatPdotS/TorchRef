"""
Crystallographic dataset classes.

This module provides PyTorch-based dataset classes for handling
crystallographic data:

- CrystalDataset: Abstract base class
- ReflectionData: Single crystal reflection dataset
- FcalcDataset: Dataset for calculated structure factors
- DatasetCollection: Container for multiple related datasets

Examples
--------
::

    from torchref.io.datasets import ReflectionData
    data = ReflectionData(device='cuda')
    data.load_mtz('observed.mtz')
    print(f"Loaded {len(data)} reflections")

    from torchref.io.datasets import FcalcDataset
    fcalc = FcalcDataset.from_cell_and_resolution(
        cell=[50.0, 60.0, 70.0, 90.0, 90.0, 90.0],
        spacegroup='P212121',
        d_min=2.0,
    )

    from torchref.io.datasets import DatasetCollection
    collection = DatasetCollection()
    collection.add_dataset('native', native_data)
    collection.add_dataset('derivative', derivative_data)
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
