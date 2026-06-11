"""Static (manual) weighting scheme.

A weighting scheme returns a ``{component: weight}`` dict from a
:class:`~torchref.refinement.loss_state.LossState`; the refinement applies it
to the state's weight dict. ``ManualWeighting`` is the **static** scheme: it is
state-independent and simply returns a fixed set of weights — the canonical home
for the default base weights (see
:data:`torchref.refinement.base_refinement.DEFAULT_GROUP_WEIGHTS`).

This restores the minimal static piece of the pre-Springclean weighting package
(the adaptive policy/random/ES schemes were removed and are not reinstated).
"""

from typing import TYPE_CHECKING, Dict

import torch

from torchref.refinement.weighting.base_weighting import BaseWeighting

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState

# Alias retained for backward compatibility with the pre-Springclean API.
WeightingScheme = BaseWeighting


class ManualWeighting(BaseWeighting):
    """Apply fixed manual weights.

    State-independent: :meth:`forward` ignores the ``LossState`` and returns the
    weights supplied at construction. Use this to express a static weighting
    scheme (e.g. the default group base weights) as a first-class object rather
    than a bare dict.

    Parameters
    ----------
    weights : dict
        Mapping of weight key (group like ``'geometry'`` or component like
        ``'geometry/bond'``) to weight value.
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
