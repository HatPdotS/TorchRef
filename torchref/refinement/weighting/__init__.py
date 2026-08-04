"""Weighting schemes for loss-component aggregation.

A scheme inherits :class:`BaseWeighting` and returns a ``{component: weight}`` dict from a
``LossState`` without mutating it. Only the **static** scheme is provided --
:class:`ManualWeighting`, the canonical home for the default base weights
(:data:`torchref.refinement.base_refinement.DEFAULT_GROUP_WEIGHTS`).
"""

from .base_weighting import BaseWeighting
from .static_weighting import ManualWeighting, WeightingScheme

__all__ = ["BaseWeighting", "WeightingScheme", "ManualWeighting"]
