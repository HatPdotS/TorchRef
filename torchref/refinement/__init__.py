"""
Refinement module for crystallographic structure refinement.

This module provides the core refinement framework including:
- Refinement classes for running optimization
- Target (loss) functions for X-ray, geometry, and ADP restraints
- Loss aggregation and state tracking

Example
-------
Basic refinement::

    from torchref.refinement import LBFGSRefinement

    refinement = LBFGSRefinement(
        data_file='reflections.mtz',
        pdb='structure.pdb',
    )
    refinement.refine_everything(macro_cycles=5)

Access targets::

    from torchref.refinement.targets import XrayTarget, BondTarget
"""

# Submodules
from . import targets
from .base_refinement import Refinement
from .lbfgs_refinement import LBFGSRefinement
from .rigid_body_refinement import RigidBodyRefinementStep
from .logger import Logger
from .loss_state import LossState
from .targets import Target, DataTarget, ModelTarget

__all__ = [
    # Main refinement classes
    "Refinement",
    "LBFGSRefinement",
    "RigidBodyRefinementStep",
    # Loss handling
    "LossState",
    # Logging
    "Logger",
    # Submodules
    "targets",
    # Base target classes
    "Target",
    "DataTarget",
    "ModelTarget",
]
