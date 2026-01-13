"""
LossAggregator - Combines losses with weights into total loss.

This module provides the LossAggregator class which takes a LossState
containing computed losses and weights, and produces the total weighted loss.

Design Pattern:
- LossAggregator receives a LossState with all losses and weights computed
- It applies the weights to the losses and sums them
- Returns the total weighted loss for backpropagation
"""

from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState


class LossAggregator(nn.Module):
    """
    Combines losses with weights into a total weighted loss.

    The aggregator takes a LossState containing:
    - losses: Dict[str, torch.Tensor] - individual loss values
    - weights: Dict[str, torch.Tensor] - corresponding weights

    And computes:
        total_loss = sum(losses[k] * weights[k] for k in losses)

    For losses without corresponding weights, a default weight of 1.0 is used.

    Parameters
    ----------
    default_weight : float, optional
        Default weight for losses without explicit weights. Default is 1.0.

    Examples
    --------
    >>> aggregator = LossAggregator()
    >>> state = LossState(device=device)
    >>> state.add_loss('xray_work', xray_loss)
    >>> state.add_loss('bond', bond_loss)
    >>> state.add_weight('xray_work', 1.0)
    >>> state.add_weight('bond', 0.5)
    >>> total = aggregator(state)
    """

    def __init__(self, default_weight: float = 1.0):
        """
        Initialize LossAggregator.

        Parameters
        ----------
        default_weight : float, optional
            Default weight for losses without explicit weights. Default is 1.0.
        """
        super().__init__()
        self.default_weight = default_weight

    def forward(self, state: "LossState") -> torch.Tensor:
        """
        Compute total weighted loss from a LossState.

        Parameters
        ----------
        state : LossState
            LossState containing losses and weights.

        Returns
        -------
        torch.Tensor
            Total weighted loss (scalar tensor).
        """
        total = torch.tensor(0.0, device=state.device)

        for name, loss in state.losses.items():
            weight = state.get_weight(name, default=self.default_weight)
            total = total + weight * loss

        return total

    def aggregate(self, state: "LossState") -> torch.Tensor:
        """
        Alias for forward() - compute total weighted loss.

        Parameters
        ----------
        state : LossState
            LossState containing losses and weights.

        Returns
        -------
        torch.Tensor
            Total weighted loss.
        """
        return self.forward(state)

    def aggregate_with_breakdown(self, state: "LossState") -> Dict[str, torch.Tensor]:
        """
        Compute total loss and return breakdown of weighted components.

        Parameters
        ----------
        state : LossState
            LossState containing losses and weights.

        Returns
        -------
        dict
            Dictionary containing:
            - 'total': Total weighted loss
            - 'components': Dict mapping loss names to weighted values
            - 'raw_losses': Dict mapping loss names to unweighted values
            - 'weights': Dict mapping loss names to weights used
        """
        total = torch.tensor(0.0, device=state.device)
        components = {}
        raw_losses = {}
        weights_used = {}

        for name, loss in state.losses.items():
            weight = state.get_weight(name, default=self.default_weight)
            weighted_loss = weight * loss

            components[name] = weighted_loss
            raw_losses[name] = loss
            weights_used[name] = weight
            total = total + weighted_loss

        return {
            "total": total,
            "components": components,
            "raw_losses": raw_losses,
            "weights": weights_used,
        }


__all__ = ["LossAggregator"]
