"""
I/O module for crystallographic data files.

This module provides:
- Dataset classes for handling reflection data
- Format-specific readers and writers (MTZ, PDB, CIF)
- Automatic format detection via DataRouter

High-level API
--------------
Load a single dataset::

    from torchref.io import ReflectionData
    data = ReflectionData(verbose=1)
    data.load_mtz('structure.mtz')

Multi-dataset handling::

    from torchref.io import DatasetCollection
    collection = DatasetCollection()
    collection.add_dataset('native', native_data)
    collection.add_dataset('derivative', derivative_data)

Direct format access::

    from torchref.io import mtz
    reader = mtz.read('data.mtz')
    data_dict, cell, spacegroup = reader()
"""

# Format modules
from . import cif, mtz, pdb
from .cif import (
    CIFReader,
    ModelCIFReader,
    ReflectionCIFReader,
    RestraintCIFReader,
)

# Data router
from .data_router import (
    DataRouter,
    DataRouterError,
)

# Dataset classes (primary API)
from .datasets import (
    CrystalDataset,
    DatasetCollection,
    ReflectionData,
    FcalcDataset,
)

# Reader classes (from format modules)
from .mtz import MTZReader
from .pdb import PDBReader

# Legacy aliases for backwards compatibility
MTZ = MTZReader
PDB = PDBReader

__all__ = [
    # Primary API - Datasets
    "CrystalDataset",
    "ReflectionData",
    "DatasetCollection",
    # Format modules
    "mtz",
    "pdb",
    "cif",
    # Reader classes
    "MTZReader",
    "PDBReader",
    "CIFReader",
    "ReflectionCIFReader",
    "ModelCIFReader",
    "RestraintCIFReader",
    # Router
    "DataRouter",
    "DataRouterError",
    # Legacy aliases
    "MTZ",
    "PDB",
]
