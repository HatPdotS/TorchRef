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
    ResolutionWeighting,
    WeightingScheme,
)

__all__ = [
    # Base weighting class
    "BaseWeighting",
    "WeightingScheme",  # Alias for backward compatibility
    # Weighting schemes
    "ResolutionWeighting",
    "OverfittingWeighting",
    "ManualWeighting",
    "ComponentWeighting",
]
