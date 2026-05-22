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
    """
    Abstract base class for weighting schemes using LossState.

    Weighting schemes:
    - Are set up without state (only device and hyperparameters)
    - Receive state only when computing weights via forward()
    - Return weights dict (do not modify state directly)

    All tunable parameters should be registered as buffers using register_buffer()
    so they can be accessed/modified via state_dict notation.

    Parameters
    ----------
    device : torch.device, optional
        Computation device. Defaults to the configured device.current.

    Attributes
    ----------
    name : str
        Unique name for this weighting scheme.
    device : torch.device
        Computation device.
    """

    name: str = "base_weighting"

    def __init__(self, device: torch.device = None, **kwargs):
        super().__init__()
        self.device = device or get_default_device()

    @abstractmethod
    def forward(self, state: "LossState") -> Dict[str, float]:
        """
        Compute weights from the current LossState.

        Access data via state["key"] or state.get("key", default).
        Meta data (rwork, rfree, etc.) is in state.meta.
        Cached losses are in state._losses.

        Parameters
        ----------
        state : LossState
            Current loss state with meta and _losses populated.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping component names to weights.
        """
        raise NotImplementedError

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        """
        Return statistics for reporting.

        Parameters
        ----------
        state : LossState, optional
            If provided, can pull data from LossState.

        Returns
        -------
        Dict[str, StatEntry]
            Statistics dictionary with StatEntry objects.
        """
        return {}


__all__ = ["BaseWeighting"]
