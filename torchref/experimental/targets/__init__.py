"""Experimental refinement targets.

This namespace collects loss/target modules whose APIs are still under
active development and may change without notice:

* :class:`~torchref.experimental.targets.amber_target.AmberTarget` --
  differentiable AMBER ff14SB/GAFF2 force-field target over
  :class:`~torchref.experimental.mm.OpenMMAdapter` (OpenMM is needed only to
  construct it).
* :class:`~torchref.experimental.targets.forcefield_target.ForceFieldTarget`
  -- generic force-field target scaffold.
* :class:`~torchref.experimental.targets.realspace.RealSpaceTarget` and
  friends -- real-space correlation / difference / extrapolation targets.
* :class:`~torchref.experimental.targets.sampled_ml_phase_target.SampledMLPhaseTarget`
  -- phase-sampled maximum-likelihood targets.
* :class:`~torchref.experimental.targets.occupancy_floor_diagnostic.OccupancyFloorDiagnostic`
  and related regularisers (:class:`NegativeDensityPenalty`,
  :class:`DisplacementRegularizer`, :class:`DifferenceAmplitudeRegularizer`)
  -- diagnostic / regularisation utilities.

These targets are not used by the headline AlphaFold-start benchmark or
the difference-refinement showcase in the main text; they are exposed
here for users prototyping new refinement workflows.
"""

from .amber_target import AmberTarget
from .forcefield_target import ForceFieldTarget
from .occupancy_floor_diagnostic import (
    DifferenceAmplitudeRegularizer,
    DisplacementRegularizer,
    NegativeDensityPenalty,
    OccupancyFloorDiagnostic,
)
from .realspace import (
    RealSpaceCorrelationTarget,
    RealSpaceDifferenceTarget,
    RealSpaceExtrapolatedTarget,
    RealSpaceTarget,
)
from .sampled_ml_phase_target import (
    SampledMLDifferenceTarget,
    SampledMLPhaseTarget,
    create_sampled_ml_difference_target,
    create_sampled_ml_target,
)

__all__ = [
    "AmberTarget",
    "ForceFieldTarget",
    "OccupancyFloorDiagnostic",
    "NegativeDensityPenalty",
    "DisplacementRegularizer",
    "DifferenceAmplitudeRegularizer",
    "RealSpaceTarget",
    "RealSpaceCorrelationTarget",
    "RealSpaceDifferenceTarget",
    "RealSpaceExtrapolatedTarget",
    "SampledMLPhaseTarget",
    "SampledMLDifferenceTarget",
    "create_sampled_ml_target",
    "create_sampled_ml_difference_target",
]
