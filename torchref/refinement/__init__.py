"""
Refinement module for crystallographic structure refinement.

This module provides the core refinement framework including:
- Refinement classes for running optimization
- Target (loss) functions for X-ray, geometry, and ADP restraints
- Weighting schemes for balancing loss components
- Loss aggregation and state tracking

Example
-------
>>> from torchref.refinement import Refinement, LBFGSRefinement
>>> from torchref.refinement import targets, weighting
>>>
>>> # Basic refinement
>>> refinement = Refinement(
...     data_file='reflections.mtz',
...     pdb='structure.pdb',
...     device='cuda'
... )
>>> refinement.run_refinement(macro_cycles=10)
>>>
>>> # Access targets and weighting schemes
>>> from torchref.refinement.targets import XrayTarget, BondTarget
>>> from torchref.refinement.weighting import ComponentWeighting
"""

# Submodules
from . import targets, weighting
from .base_refinement import Refinement
from .lbfgs_refinement import LBFGSRefinement
from .loss_state import LossState
from .optimizers import AdamWithAdaptiveNoise

__all__ = [
    # Main refinement classes
    "Refinement",
    "LBFGSRefinement",
    # Loss handling
    "LossState",
    # Optimizers
    "AdamWithAdaptiveNoise",
    # Submodules
    "targets",
    "weighting",
]
