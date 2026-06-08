"""
I/O module for crystallographic data files.

This module provides:
- Dataset classes for handling reflection data
- Format-specific readers and writers (MTZ, PDB, CIF)
- Top-level object-creation readers: read_mtz, read_cif, read_pdb

High-level API
--------------
Load objects directly::

    from torchref.io import read_mtz, read_cif, read_pdb
    data = read_mtz('structure.mtz')   # -> ReflectionData
    model = read_pdb('structure.pdb')  # -> ModelFT

Or construct and load explicitly::

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

# Metadata
from .metadata import RefinementMetadata

# Top-level object-creation readers
from .readers import read_cif, read_mtz, read_pdb

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

# IHM ensemble support (mapping always available; reader/writer need python-ihm)
from .ihm_mapping import IHMEnsembleMapping, IHMModelGroupInfo, IHMStateInfo

__all__ = [
    # Primary API - Datasets
    "CrystalDataset",
    "ReflectionData",
    "DatasetCollection",
    "FcalcDataset",
    # Top-level readers
    "read_mtz",
    "read_cif",
    "read_pdb",
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
    # IHM ensemble support
    "IHMEnsembleMapping",
    "IHMStateInfo",
    "IHMModelGroupInfo",
    # Metadata
    "RefinementMetadata",
]
