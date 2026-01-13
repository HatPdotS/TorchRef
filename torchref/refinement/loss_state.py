"""
LossState - Hierarchical loss computation with lazy evaluation.

Design:
- Targets stored as callables, evaluated only on aggregation
- Hierarchical naming via '/' separator (e.g., 'geometry/bond', 'adp/simu')
- Weights computed at initialization, not recalculated each pass
- Internal history logging for debugging/analysis
- Aggregation handled directly in this class

Example:
    state = LossState(device)
    state.register_target('xray/work', xray_work_target)
    state.register_target('geometry/bond', bond_target)
    state.set_weight('xray', 1.0)
    state.set_weight('geometry', 0.5)
    state.set_weight('geometry/bond', 1.0)

    total = state.aggregate()  # Evaluates targets, applies hierarchical weights
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class LossState:
    """
    Hierarchical loss state with lazy evaluation.

    Attributes
    ----------
    device : torch.device
        Computation device.
    targets : Dict[str, Callable]
        Target functions keyed by hierarchical name (e.g., 'geometry/bond').
    weights : Dict[str, float]
        Weights keyed by name. Can be group weights ('geometry') or
        component weights ('geometry/bond').
    history : List[Dict]
        Log of computed values per aggregation call.
    """

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Targets as callables - only evaluated on aggregate()
    targets: Dict[str, Callable] = field(default_factory=dict)

    # Weights - computed at init, hierarchical via naming
    weights: Dict[str, float] = field(default_factory=dict)

    # History log
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Cache for computed losses (cleared on each aggregate)
    _losses: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    # =========================================================================
    # Target Registration
    # =========================================================================

    def register_target(self, name: str, target: Callable) -> "LossState":
        """
        Register a target function.

        Parameters
        ----------
        name : str
            Hierarchical name (e.g., 'geometry/bond', 'adp/simu').
        target : Callable
            Function that returns a loss tensor when called.

        Returns
        -------
        LossState
            Self for chaining.
        """
        self.targets[name] = target
        return self

    def register_targets(self, targets: Dict[str, Callable]) -> "LossState":
        """Register multiple targets."""
        for name, target in targets.items():
            self.register_target(name, target)
        return self

    # =========================================================================
    # Weight Management
    # =========================================================================

    def set_weight(self, name: str, weight: float) -> "LossState":
        """
        Set a weight value.

        Parameters
        ----------
        name : str
            Weight name. Can be a group ('geometry') or component ('geometry/bond').
        weight : float
            Weight value.

        Returns
        -------
        LossState
            Self for chaining.
        """
        self.weights[name] = weight
        return self

    def set_weights(self, weights: Dict[str, float]) -> "LossState":
        """Set multiple weights."""
        for name, weight in weights.items():
            self.set_weight(name, weight)
        return self

    def get_weight(self, name: str, default: float = 1.0) -> float:
        """Get a weight value."""
        return self.weights.get(name, default)

    def get_effective_weight(self, name: str) -> float:
        """
        Get effective weight for a target, including group weights.

        For 'geometry/bond', returns: weights['geometry'] * weights['geometry/bond']
        Missing weights default to 1.0.

        Parameters
        ----------
        name : str
            Target name (e.g., 'geometry/bond').

        Returns
        -------
        float
            Product of all hierarchical weights.
        """
        parts = name.split("/")
        effective = 1.0

        # Apply weights at each level
        path = ""
        for part in parts:
            path = f"{path}/{part}" if path else part
            effective *= self.weights.get(path, 1.0)

        return effective

    # =========================================================================
    # History Logging
    # =========================================================================

    def log(self, name: str, value: Any) -> None:
        """
        Log a value to the current history entry.

        Creates a new history entry if needed.

        Parameters
        ----------
        name : str
            Key for the logged value.
        value : Any
            Value to log. Tensors are converted to Python floats.
        """
        # Ensure we have a current entry
        if not self.history:
            self.history.append({})

        # Convert tensor to float
        if isinstance(value, torch.Tensor):
            value = value.detach().item()

        self.history[-1][name] = value

    def new_entry(self) -> None:
        """Start a new history entry."""
        self.history.append({})

    def get_history(self, name: str) -> List[Any]:
        """Get all logged values for a key across history."""
        return [entry.get(name) for entry in self.history if name in entry]

    # =========================================================================
    # Aggregation
    # =========================================================================

    def aggregate(self, log_values: bool = False) -> torch.Tensor:
        """
        Evaluate all targets and compute weighted sum.

        Parameters
        ----------
        log_values : bool
            If True, log all losses, weights, and total to history.

        Returns
        -------
        torch.Tensor
            Total weighted loss.
        """
        if log_values:
            self.new_entry()

        total = torch.tensor(0.0, device=self.device)
        self._losses.clear()

        for name, target in self.targets.items():
            # Evaluate target
            loss = target()
            self._losses[name] = loss

            # Get effective weight
            weight = self.get_effective_weight(name)

            # Accumulate
            weighted_loss = weight * loss
            total = total + weighted_loss

            # Log
            if log_values:
                self.log(f"loss/{name}", loss)
                self.log(f"weight/{name}", weight)
                self.log(f"weighted/{name}", weighted_loss)

        if log_values:
            self.log("total", total)

        return total

    def get_loss(self, name: str) -> Optional[torch.Tensor]:
        """
        Get a cached loss value (after aggregate() was called).

        Parameters
        ----------
        name : str
            Target name.

        Returns
        -------
        torch.Tensor or None
            Cached loss, or None if not computed.
        """
        return self._losses.get(name)

    # =========================================================================
    # Breakdown / Analysis
    # =========================================================================

    def get_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get breakdown of losses by group.

        Returns
        -------
        Dict
            Nested dict: {group: {component: {'loss': ..., 'weight': ..., 'weighted': ...}}}
        """
        breakdown = defaultdict(dict)

        for name, loss in self._losses.items():
            parts = name.split("/")
            group = parts[0] if len(parts) > 1 else "root"
            component = "/".join(parts[1:]) if len(parts) > 1 else parts[0]

            weight = self.get_effective_weight(name)

            breakdown[group][component] = {
                "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                "weight": weight,
                "weighted": (
                    (weight * loss).item()
                    if isinstance(loss, torch.Tensor)
                    else weight * loss
                ),
            }

        return dict(breakdown)

    def get_group_totals(self) -> Dict[str, float]:
        """
        Get total weighted loss per group.

        Returns
        -------
        Dict[str, float]
            {group_name: total_weighted_loss}
        """
        totals = defaultdict(float)

        for name, loss in self._losses.items():
            parts = name.split("/")
            group = parts[0]
            weight = self.get_effective_weight(name)
            weighted = (
                (weight * loss).item()
                if isinstance(loss, torch.Tensor)
                else weight * loss
            )
            totals[group] += weighted

        return dict(totals)

    # =========================================================================
    # Utility
    # =========================================================================

    def clear(self) -> "LossState":
        """Clear cached losses (not targets or weights)."""
        self._losses.clear()
        return self

    def clear_history(self) -> "LossState":
        """Clear history log."""
        self.history.clear()
        return self

    def __repr__(self) -> str:
        n_targets = len(self.targets)
        n_weights = len(self.weights)
        n_history = len(self.history)
        return f"LossState(device={self.device}, targets={n_targets}, weights={n_weights}, history={n_history})"


def create_loss_state(
    device: torch.device,
    targets: Dict[str, Callable] = None,
    weights: Dict[str, float] = None,
) -> LossState:
    """
    Factory function to create a LossState.

    Parameters
    ----------
    device : torch.device
        Computation device.
    targets : Dict[str, Callable], optional
        Initial targets to register.
    weights : Dict[str, float], optional
        Initial weights to set.

    Returns
    -------
    LossState
        Configured LossState instance.
    """
    state = LossState(device=device)

    if targets:
        state.register_targets(targets)
    if weights:
        state.set_weights(weights)

    return state


__all__ = ["LossState", "create_loss_state"]
