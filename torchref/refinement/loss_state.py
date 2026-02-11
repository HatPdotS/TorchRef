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

from torchref.utils.device_mixin import DeviceMovementMixin


@dataclass
class LossState(DeviceMovementMixin):
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
    meta : Dict[str, Any]
        Model-level data (rwork, rfree, n_atoms, etc.) populated by refinement.
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

    # Model-level data for weighting schemes
    meta: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # Item Access (meta and _losses)
    # =========================================================================

    def __getitem__(self, key: str) -> Any:
        """
        Get value from meta or _losses by key.

        Parameters
        ----------
        key : str
            Key to look up. Checks meta first, then _losses.

        Returns
        -------
        Any
            Value from meta or _losses.

        Raises
        ------
        KeyError
            If key not found in either dict.
        """
        if key in self.meta:
            return self.meta[key]
        if key in self._losses:
            return self._losses[key]
        raise KeyError(f"Key '{key}' not found in meta or _losses")

    def __contains__(self, key: str) -> bool:
        """Check if key exists in meta or _losses."""
        return key in self.meta or key in self._losses

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with default fallback.

        Parameters
        ----------
        key : str
            Key to look up.
        default : Any
            Value to return if key not found.

        Returns
        -------
        Any
            Value from meta, _losses, or default.
        """
        if key in self.meta:
            return self.meta[key]
        if key in self._losses:
            return self._losses[key]
        return default

    def cache_losses(self, force: bool = False) -> "LossState":
        """
        Cache all target losses.

        Evaluates all registered targets and stores results in _losses.

        Parameters
        ----------
        force : bool
            If True, re-evaluate all targets even if already cached.

        Returns
        -------
        LossState
            Self for chaining.
        """
        if force:
            self._losses.clear()

        for name, target in self.targets.items():
            if name not in self._losses:
                self._losses[name] = target()

        return self

    def update_meta(self, data: Dict[str, Any]) -> "LossState":
        """
        Update meta dict with model-level data.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to add to meta.

        Returns
        -------
        LossState
            Self for chaining.
        """
        self.meta.update(data)
        return self

    # =========================================================================
    # Target Registration
    # =========================================================================

    def register_target(
        self, name: str, target: Callable, prefix: str = None
    ) -> "LossState":
        """
        Register a target function.

        Automatically detects combined targets (like TotalGeometryTarget,
        TotalADPTarget) and expands them into their component targets.

        Parameters
        ----------
        name : str
            Hierarchical name (e.g., 'geometry/bond', 'adp/simu').
        target : Callable
            Function that returns a loss tensor when called. Can also be a
            combined target with .items() method, which will be auto-expanded.
        prefix : str, optional
            Prefix to prepend to the name (e.g., 'model1' -> 'model1/geometry/bond').
            Useful for registering targets from multiple models in the same state.

        Returns
        -------
        LossState
            Self for chaining.
        """
        # Check if target is a combined/dictionary-like target with .items()
        # This handles CombinedTargets, TotalGeometryTarget, TotalADPTarget, etc.
        if hasattr(target, 'items') and callable(getattr(target, 'items', None)):
            # Use name as prefix to maintain hierarchy (e.g., "geometry" -> "geometry/bond")
            combined_prefix = f"{prefix}/{name}" if prefix else name
            return self.register_targets(target, prefix=combined_prefix)

        # Normal single target registration
        key = f"{prefix}/{name}" if prefix else name
        self.targets[key] = target
        return self

    def register_targets(self, targets, prefix: str = None) -> "LossState":
        """Register multiple targets from a component target or dict.

        For targets with a .name attribute, uses target.name as the key.
        For plain callables, uses the dict key.

        Parameters
        ----------
        targets : dict
            Dictionary of name -> target mappings.
        prefix : str, optional
            Prefix to prepend to all target names.
        """
        for name, target in targets.items():
            target_name = getattr(target, "name", name)
            self.register_target(target_name, target, prefix=prefix)
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
    
    def summary(self) -> str:
        print("LossState Summary:")
        for name, loss in self._losses.items():
            weight = self.get_effective_weight(name)
            weighted_loss = weight * loss
            print(f"  {name}: loss={loss.item():.4f}, weight={weight:.4f}, weighted={weighted_loss.item():.4f}")

    # =========================================================================
    # Device Management
    # =========================================================================

    def to(self, device) -> "LossState":
        """
        Move LossState to the specified device.

        Updates device attribute and moves any cached tensors in _losses and meta.

        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns
        -------
        LossState
            Self, for method chaining.
        """
        self.device = torch.device(device)

        # Move cached losses
        for name, loss in self._losses.items():
            if isinstance(loss, torch.Tensor):
                self._losses[name] = loss.to(self.device)

        # Move tensors in meta
        for key, value in self.meta.items():
            if isinstance(value, torch.Tensor):
                self.meta[key] = value.to(self.device)

        return self

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
        n_meta = len(self.meta)
        return f"LossState(device={self.device}, targets={n_targets}, weights={n_weights}, meta={n_meta}, history={n_history})"


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
