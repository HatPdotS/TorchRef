"""
TorchRef - A PyTorch-based crystallographic refinement library.

TorchRef provides GPU-accelerated crystallographic structure refinement
using PyTorch's automatic differentiation and nn.Module architecture.

Key Features
------------
- Native PyTorch integration with nn.Module architecture
- Automatic differentiation for custom target functions
- GPU acceleration for structure factor calculations
- Modular design for easy extension

Quick Start
-----------
::

    from torchref import Refinement, ReflectionData, Model

    # Load data and model
    data = ReflectionData().load_mtz('data.mtz')
    model = Model().load_pdb('structure.pdb')

    # Run refinement
    refinement = Refinement(data=data, model=model, device='cuda')
    refinement.run_refinement(macro_cycles=10)

Modules
-------
io
    File I/O for MTZ, PDB, CIF formats.
model
    Atomic structure models (coordinates, B-factors, occupancies).
refinement
    Core refinement framework with targets and weighting schemes.
restraints
    Geometry restraints (bonds, angles, torsions, planes). (initialized lazily as it requires downloading the monomer library)
scaling
    Structure factor scaling and bulk solvent models.
symmetry
    Crystallographic symmetry operations.
alignment
    Patterson-based structure alignment.
math_functions
    Mathematical utilities for crystallography.
utils
    General utilities and debugging tools.
"""

__version__ = "0.3.2"


import os
import warnings
from pathlib import Path

from torchref._bootstrap import configure_threading, detect_available_cpus

# Configure threading before importing torch
if "TORCHREF_NUM_THREADS" in os.environ:
    N_CPUS = int(os.environ["TORCHREF_NUM_THREADS"])
    warnings.warn(
        f"TorchRef using user-specified {N_CPUS} threads from TORCHREF_NUM_THREADS.",
        stacklevel=2,
    )
else:
    N_CPUS = detect_available_cpus()
    os.environ["TORCHREF_NUM_THREADS"] = str(N_CPUS)
    warnings.warn(
        f"TorchRef auto-configured {N_CPUS} threads. Set TORCHREF_NUM_THREADS to override.",
        stacklevel=2,
    )

configure_threading(N_CPUS)

import torch

torch.set_num_threads(N_CPUS)

# Dtype configuration (must be imported after torch)
from torchref.config import dtypes


# Project root path for referencing package files
ROOT_TORCHREF = Path(__file__).parent.parent.resolve()

# Package path for referencing internal files
PATH_TORCHREF = Path(__file__).parent.resolve()

PATH_TORCHREF_DATA = PATH_TORCHREF / "data"

# =============================================================================
# Convenience imports for common classes
# =============================================================================


# Data I/O
from torchref.io import DatasetCollection, ReflectionData

# Model
from torchref.model import Model, ModelFT

# Refinement
from torchref.refinement import LBFGSRefinement, Refinement
from torchref.symmetry import Cell, SpaceGroup

# Restraints
# from torchref.restraints import Restraints # Initialized lazily due to monomer library download requirement

# Scaling
from torchref.scaling import Scaler, SolventModel

# Maps
from torchref.maps import DifferenceMap, Map

__all__ = [
    # Version and paths
    "__version__",
    "ROOT_TORCHREF",
    "PATH_TORCHREF",
    "N_CPUS",
    # Dtype configuration
    "dtypes",
    # Data I/O
    "ReflectionData",
    "DatasetCollection",
    # Model
    "Model",
    "ModelFT",
    # Refinement
    "Refinement",
    "LBFGSRefinement",
    # Scaling
    "Scaler",
    "SolventModel",
    # Symmetry
    "Cell",
    "SpaceGroup",
    # Maps
    "Map",
    "DifferenceMap",
]
