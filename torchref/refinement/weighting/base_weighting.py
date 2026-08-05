"""
Base weighting class using LossState for all data access.

All weighting schemes inherit from BaseWeighting and receive their data
through LossState rather than direct refinement references.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

from torchref.config import get_default_device
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.stats import StatEntry

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState


class BaseWeighting(DeviceMixin, nn.Module, ABC):
    """Abstract base for weighting schemes driven by a LossState.

    A scheme is constructed with only a device and hyperparameters, receives the state only
    when computing weights, and **returns** a weights dict rather than mutating the state.
    Register tunable parameters as buffers so they are reachable through ``state_dict``.

    Parameters
    ----------
    device : torch.device, optional
        Computation device. Defaults to the configured default.

    Attributes
    ----------
    name : str
        Unique name for this scheme.
    """

    name: str = "base_weighting"

    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__()
        self.device = device or get_default_device()

    @abstractmethod
    def forward(self, state: "LossState") -> Dict[str, float]:
        """The ``{component: weight}`` dict for the current ``state``.

        Read data through ``state["key"]`` / ``state.get(...)``; model-level metrics live in
        ``state.meta`` and cached per-target losses in ``state._losses``.
        """
        raise NotImplementedError

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        """Reporting statistics as ``{name: StatEntry}``, optionally drawn from
        ``state``."""
        return {}


__all__ = ["BaseWeighting"]
