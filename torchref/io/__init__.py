"""
I/O for crystallographic data files: dataset containers, MTZ/PDB/CIF format
modules, and the top-level object-creation readers.

The three layers, loosest to tightest::

    data = read_mtz('structure.mtz')          # -> ReflectionData
    data = ReflectionData(verbose=1); data.load_mtz('structure.mtz')
    data_dict, cell, spacegroup = mtz.read('data.mtz')()

:class:`DatasetCollection` handles several datasets jointly. The IHM reader and
writer need ``python-ihm``; :class:`IHMEnsembleMapping` does not.
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
