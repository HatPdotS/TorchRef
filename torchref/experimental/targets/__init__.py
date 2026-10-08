"""Experimental refinement targets.

This namespace collects loss/target modules whose APIs are still under
active development and may change without notice:

* :class:`~torchref.experimental.targets.amber_target.AmberTarget` --
  differentiable AMBER14/GAFF2 force-field target via an OpenMM autograd
  bridge.
* :class:`~torchref.experimental.targets.forcefield_target.ForceFieldTarget`
  -- generic force-field target scaffold.
* :class:`~torchref.experimental.targets.realspace.RealSpaceTarget` and
  friends -- real-space correlation / difference targets.
* :class:`~torchref.experimental.targets.occupancy_floor_diagnostic.OccupancyFloorDiagnostic`
  -- activation-fraction floor diagnostic.

These targets are not used by the headline AlphaFold-start benchmark or
the difference-refinement showcase in the main text; they are exposed
here for users prototyping new refinement workflows.
"""

# AMBER target (optional dependency: openmm) -- imported lazily so a
# missing OpenMM install does not break the wider experimental namespace.
try:
    from .amber_target import AMBER14_STANDARD, AmberTarget
except ImportError:
    AmberTarget = None
    AMBER14_STANDARD = None

from .forcefield_target import ForceFieldTarget
from .occupancy_floor_diagnostic import OccupancyFloorDiagnostic
from .realspace import (
    RealSpaceCorrelationTarget,
    RealSpaceDifferenceTarget,
    RealSpaceTarget,
)

__all__ = [
    "AmberTarget",
    "AMBER14_STANDARD",
    "ForceFieldTarget",
    "OccupancyFloorDiagnostic",
    "RealSpaceTarget",
    "RealSpaceCorrelationTarget",
    "RealSpaceDifferenceTarget",
]
