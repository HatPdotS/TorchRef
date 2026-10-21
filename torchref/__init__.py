"""TorchRef - GPU-accelerated crystallographic refinement built on PyTorch.

Refinement as ``nn.Module`` objects and autograd, so a custom target function
differentiates itself. :class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement`
is the entry point and takes the MTZ and PDB paths directly::

    from torchref import LBFGSRefinement

    refinement = LBFGSRefinement(
        data_file='data.mtz', pdb='model.pdb', device='cuda'
    )
    refinement.refine(macro_cycles=10)

``import torchref`` sets the thread count and ``PYTORCH_ENABLE_MPS_FALLBACK`` but does
not import torch: the re-exported names below (``Model``, ``ReflectionData``,
``dtypes``, ...) and the subpackages are imported on first access. The environment
variables :mod:`torchref.config` reads are therefore read on that first access, not at
``import torchref``.

Modules
-------
io
    File I/O for MTZ, PDB, CIF formats.
model
    Atomic structure models (coordinates, B-factors, occupancies).
refinement
    Core refinement framework with targets and weighting schemes.
topology
    Connectivity graph, geometry restraints and hydrogens; the monomer library is
    resolved lazily.
scaling
    Structure factor scaling and bulk solvent models.
symmetry
    Crystallographic symmetry operations.
maps
    Map calculation, including difference maps (``Map``, ``DifferenceMap``).
base
    Low-level building blocks, including the math/crystallography utilities.
cli
    Command-line entry points (``torchref.refine``, ``torchref.difference-refine``, ...).
experimental
    Experimental features (e.g. kinetic targets, monolithic refinement, Patterson
    alignment).
utils
    General utilities and debugging tools.
"""

__version__ = "0.7.0"


import importlib
import os
import sys
import warnings
from pathlib import Path

# Must be set before torch is first imported, or it has no effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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

# A torch already imported has read its thread count; one imported later reads it from
# the OMP_NUM_THREADS just set.
if "torch" in sys.modules:
    sys.modules["torch"].set_num_threads(N_CPUS)

# Project root path for referencing package files
ROOT_TORCHREF = Path(__file__).parent.parent.resolve()

# Package path for referencing internal files
PATH_TORCHREF = Path(__file__).parent.resolve()

PATH_TORCHREF_DATA = PATH_TORCHREF / "data"

# Public name -> defining module. Resolved on first access by ``__getattr__`` so that
# ``import torchref`` (and with it every ``torchref.cli`` module, whose ``--help`` must
# not wait for torch) does not import torch. ``Restraints`` is deliberately absent:
# constructing it can trigger a monomer-library download.
_LAZY_EXPORTS = {
    "dtypes": "torchref.config",
    "device": "torchref.config",
    "sigma_cutoff_ed": "torchref.config",
    "DatasetCollection": "torchref.io",
    "ReflectionData": "torchref.io",
    "ScaledDataset": "torchref.io",
    "read_cif": "torchref.io",
    "read_mtz": "torchref.io",
    "read_pdb": "torchref.io",
    "DifferenceMap": "torchref.maps",
    "Map": "torchref.maps",
    "Model": "torchref.model",
    "ModelFT": "torchref.model",
    "RigidXYZTensor": "torchref.model.rigid_xyz",
    "LBFGSRefinement": "torchref.refinement",
    "Refinement": "torchref.refinement",
    "RigidBodyRefinementStep": "torchref.refinement.rigid_body_refinement",
    "Scaler": "torchref.scaling",
    "ScalerBase": "torchref.scaling",
    "SolventModel": "torchref.scaling",
    "Cell": "torchref.symmetry",
    "SpaceGroup": "torchref.symmetry",
    "Symmetry": "torchref.symmetry",
    "DeviceMixin": "torchref.utils.device_mixin",
}


def __getattr__(name: str):
    """Import a re-exported name, or a subpackage, on first access."""
    if name in _LAZY_EXPORTS:
        value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    elif not name.startswith("_") and (
        (PATH_TORCHREF / name / "__init__.py").exists()
        or (PATH_TORCHREF / f"{name}.py").exists()
    ):
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


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
    "ScaledDataset",
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
