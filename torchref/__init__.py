"""TorchRef - GPU-accelerated crystallographic refinement built on PyTorch.

Refinement as ``nn.Module``s and autograd, so a custom target function
differentiates itself. Start from :class:`LBFGSRefinement`, which takes the MTZ and
PDB paths directly::

    from torchref import LBFGSRefinement

    refinement = LBFGSRefinement(
        data_file='data.mtz', pdb='model.pdb', device='cuda'
    )
    refinement.refine(macro_cycles=10)

Modules
-------
io
    File I/O for MTZ, PDB, CIF formats.
model
    Atomic structure models (coordinates, B-factors, occupancies).
refinement
    Core refinement framework with targets and weighting schemes.
restraints
    Geometry restraints (bonds, angles, torsions, planes); initialized lazily, since it
    needs the monomer library downloaded.
scaling
    Structure factor scaling and bulk solvent models.
symmetry
    Crystallographic symmetry operations.
alignment
    Patterson-based structure alignment.
maps
    Map calculation, including difference maps (``Map``, ``DifferenceMap``).
base
    Low-level building blocks, including the math/crystallography utilities.
cli
    Command-line entry points (``torchref.refine``, ``torchref.difference-refine``, ...).
experimental
    Experimental features (e.g. kinetic targets, monolithic refinement).
utils
    General utilities and debugging tools.
"""

__version__ = "0.7.0"


import os

# Must be set before torch is imported below, or it has no effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import warnings
from pathlib import Path

from torchref._bootstrap import configure_threading, detect_available_cpus

# Threading must be configured before torch is imported.
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

# Must come after torch: reads its dtype/device state at import.
from torchref.config import device, dtypes, sigma_cutoff_ed

# Project root path for referencing package files
ROOT_TORCHREF = Path(__file__).parent.parent.resolve()

# Package path for referencing internal files
PATH_TORCHREF = Path(__file__).parent.resolve()

PATH_TORCHREF_DATA = PATH_TORCHREF / "data"

# =============================================================================
# Convenience imports for common classes
# =============================================================================


# Data I/O
from torchref.io import (
    DatasetCollection,
    ReflectionData,
    FcalcDataset,
    read_mtz,
    read_cif,
    read_pdb,
)

# Model
from torchref.model import Model, ModelFT
from torchref.model.rigid_xyz import RigidXYZTensor

# Refinement
from torchref.refinement import LBFGSRefinement, Refinement
from torchref.refinement.rigid_body_refinement import RigidBodyRefinementStep
from torchref.symmetry import Cell, SpaceGroup, Symmetry

# Restraints
# torchref.topology.restraints.Restraints is not imported here: constructing it can
# trigger a monomer-library download, so it stays lazy.

# Scaling
from torchref.scaling import Scaler, SolventModel, ScalerBase

# Maps
from torchref.maps import DifferenceMap, Map

# Device movement mixin (public API for extension code)
from torchref.utils.device_mixin import DeviceMixin

__all__ = [
    # Version and paths
    "__version__",
    "ROOT_TORCHREF",
    "PATH_TORCHREF",
    "N_CPUS",
    # Dtype and device configuration
    "dtypes",
    "device",
    "sigma_cutoff_ed",
    # Data I/O
    "ReflectionData",
    "DatasetCollection",
    "read_mtz",
    "read_cif",
    "read_pdb",
    # Model
    "Model",
    "ModelFT",
    "RigidXYZTensor",
    # Refinement
    "Refinement",
    "LBFGSRefinement",
    "RigidBodyRefinementStep",
    # Scaling
    "Scaler",
    "ScalerBase",
    "SolventModel",
    # Symmetry
    "Cell",
    "SpaceGroup",
    "Symmetry",
    # Maps
    "Map",
    "DifferenceMap",
    # Device mixin
    "DeviceMixin",
]
