"""Weighting schemes for loss-component aggregation.

A scheme subclasses :class:`~torchref.refinement.weighting.base_weighting.BaseWeighting`
and maps a ``LossState`` to a ``{component: weight}`` dict without mutating it. Only the
**static** :class:`~torchref.refinement.weighting.static_weighting.ManualWeighting` is
provided; it carries :data:`torchref.refinement.base_refinement.DEFAULT_GROUP_WEIGHTS`.
"""

from .base_weighting import BaseWeighting
from .static_weighting import ManualWeighting

__all__ = ["BaseWeighting", "ManualWeighting"]
