"""
Weighting schemes for loss component aggregation.

This module provides weighting schemes that adjust the relative
importance of different loss components during refinement.
"""

from .component_weighting import (
    ComponentWeighting,
    ManualWeighting,
    OverfittingWeighting,
    TargetOffsetWeighting,
    WeightingScheme,
    XrayScaleWeighting,
)
from .policy_weighting import (
    COMPONENT_TO_LOSS_STATE,
    COMPONENTS,
    LOSS_STATE_TO_COMPONENT,
    PolicyComponentWeighting,
    StepRecord,
    StepState,
    TrajectoryData,
    trajectory_to_dict,
)

__all__ = [
    # Base weighting
    "WeightingScheme",
    "TargetOffsetWeighting",
    "OverfittingWeighting",
    "ManualWeighting",
    "XrayScaleWeighting",
    "ComponentWeighting",
    # Policy weighting
    "PolicyComponentWeighting",
    "StepState",
    "StepRecord",
    "TrajectoryData",
    "trajectory_to_dict",
    "COMPONENTS",
    "COMPONENT_TO_LOSS_STATE",
    "LOSS_STATE_TO_COMPONENT",
]
