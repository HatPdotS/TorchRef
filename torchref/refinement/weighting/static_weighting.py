"""Static (manual) weighting scheme.

A weighting scheme returns a ``{component: weight}`` dict from a
:class:`~torchref.refinement.loss_state.LossState` and the refinement applies it.
``ManualWeighting`` is the **static** one: state-independent, returning a fixed set of
weights, and the canonical home for the defaults
(:data:`torchref.refinement.base_refinement.DEFAULT_GROUP_WEIGHTS`).
"""

from typing import TYPE_CHECKING, Dict

import torch

from torchref.refinement.weighting.base_weighting import BaseWeighting

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState

# Alias retained for backward compatibility with the pre-Springclean API.
WeightingScheme = BaseWeighting


class ManualWeighting(BaseWeighting):
    """Apply fixed manual weights, ignoring the ``LossState`` entirely.

    Lets a static weighting scheme (e.g. the default group base weights) be a first-class
    object rather than a bare dict.

    Parameters
    ----------
    weights : dict
        Weight key -- group (``'geometry'``) or component (``'geometry/bond'``) -- to value.
    device : torch.device, optional
        Computation device.
    """

    name = "manual_weighting"

    def __init__(self, weights: Dict[str, float], device: torch.device = None):
        super().__init__(device)
        self._weights = {k: float(v) for k, v in weights.items()}

    def forward(self, state: "LossState" = None) -> Dict[str, float]:
        """Return the fixed weights (the ``state`` argument is unused)."""
        return dict(self._weights)


__all__ = ["WeightingScheme", "ManualWeighting"]
