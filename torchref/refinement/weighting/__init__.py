"""Weighting schemes for loss-component aggregation.

A weighting scheme inherits :class:`BaseWeighting` and returns a
``{component: weight}`` dict from a ``LossState`` (it does not mutate the state).
Only the **static** scheme is provided — :class:`ManualWeighting`, which returns
a fixed set of weights and is the canonical home for the default base weights
(:data:`torchref.refinement.base_refinement.DEFAULT_GROUP_WEIGHTS`). The adaptive
policy/random/ES schemes from the pre-Springclean package were removed and are
not reinstated.
"""

from .base_weighting import BaseWeighting
from .static_weighting import ManualWeighting, WeightingScheme

__all__ = ["BaseWeighting", "WeightingScheme", "ManualWeighting"]
