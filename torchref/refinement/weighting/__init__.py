"""
Weighting schemes for loss component aggregation.

This module provides weighting schemes that adjust the relative
importance of different loss components during refinement.

All weighting schemes inherit from BaseWeighting and receive data
through LossState rather than direct refinement references.
"""

from .base_weighting import BaseWeighting
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
from .random_weighting import (
    DEFAULT_LOG_WEIGHTS,
    DEFAULT_STEP_SIGMAS,
    DEFAULT_TRAJECTORY_SIGMAS,
    RandomComponentWeighting,
    RandomWeightingScheme,
)

__all__ = [
    # Base weighting class
    "BaseWeighting",
    "WeightingScheme",  # Alias for backward compatibility
    # Weighting schemes
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
    # Random weighting
    "RandomWeightingScheme",
    "RandomComponentWeighting",
    "DEFAULT_LOG_WEIGHTS",
    "DEFAULT_TRAJECTORY_SIGMAS",
    "DEFAULT_STEP_SIGMAS",
]
