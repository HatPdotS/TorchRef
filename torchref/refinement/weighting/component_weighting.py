from typing import TYPE_CHECKING, Any, Dict, List

import torch
from torch import nn
from torch.nn.functional import softplus

from torchref.refinement.weighting.base_weighting import BaseWeighting
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_ESSENTIAL,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState

from torchref.utils import TensorDict

# WeightingScheme is now an alias for BaseWeighting for backward compatibility
WeightingScheme = BaseWeighting


class TargetOffsetWeighting(BaseWeighting):
    """
    Weights components based on distance from target value.
    weight = 1 + softplus(loss - target_value) / sigma

    Uses state._losses for current loss values and state.meta for target info.
    If info is missing for a component, it is skipped.
    """

    name = "target_offset_weighting"

    def __init__(self, device: torch.device = None):
        super().__init__(device)

    def forward(self, state: "LossState") -> Dict[str, float]:
        """
        Compute weights based on distance from target values.

        Target info should be stored in state.meta as "target_info/{name}".
        """
        # Ensure losses are cached
        state.cache_losses()
        target_info = state.get(f"target_info", {})


        weights = {}
        for name, loss in state._losses.items():
            if not name in target_info:
                continue
            
            target_value = target_info[name].get("target_value", 0.0)
            sigma = target_info[name].get("sigma", 0.5)

            if sigma > 0:
                if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                    loss_val = loss.sum()
                else:
                    loss_val = loss

                weight = 1 + softplus(loss_val - target_value) / (sigma + 1e-6)
                weights[name] = weight.detach().item()

        return weights

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        """
        Get target offset weighting statistics.

        Parameters
        ----------
        state : LossState, optional
            If provided, pull weights from LossState.
        """
        stats = {}

        if state is not None:
            # Pull weights from state
            for name, weight in state.weights.items():
                if isinstance(weight, torch.Tensor):
                    stats[f"{name}_weight"] = stat(weight.item(), VERBOSITY_STANDARD)
                else:
                    stats[f"{name}_weight"] = stat(weight, VERBOSITY_STANDARD)

        return stats


class OverfittingWeighting(BaseWeighting):
    """
    Scale-invariant overfitting correction weight.

    Uses state.meta["xray_loss_work"] and state.meta["xray_loss_test"].

    Tunable parameters (as buffers):
    - target_gap: float, target overfitting gap threshold
    - min_weight: float, minimum weight value
    - sharpness: float, exponential sharpness
    - smoothing: float, exponential smoothing factor (0-1)
    """

    name = "overfitting_weighting"

    def __init__(
        self,
        device: torch.device = None,
        target_gap: float = 0.2,
        min_weight: float = 0.2,
        sharpness: float = 5.0,
        smoothing: float = 0.8,
    ):
        super().__init__(device)
        # Register tunable parameters as buffers for state_dict access
        self.register_buffer("target_gap", torch.tensor(target_gap))
        self.register_buffer("min_weight", torch.tensor(min_weight))
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("smoothing", torch.tensor(smoothing))
        self.register_buffer("weight_reg", torch.tensor(1.0))  # Current weight value

    def forward(self, state: "LossState") -> Dict[str, float]:
        """
        Compute overfitting correction weights from state.meta.
        """
        train_nll = state.get("xray_loss_work", 0.0)
        test_nll = state.get("xray_loss_test", 0.0)

        # Convert to tensors if needed
        if not isinstance(train_nll, torch.Tensor):
            train_nll = torch.tensor(train_nll, device=self.device)
        if not isinstance(test_nll, torch.Tensor):
            test_nll = torch.tensor(test_nll, device=self.device)

        scale = torch.abs(test_nll) + 1e-6
        relative_gap = (test_nll - train_nll) / scale

        # Calculate target weight with hard penalty
        target_weight = self.min_weight + torch.exp(
            self.sharpness * (relative_gap - self.target_gap)
        )
        target_weight = target_weight.detach()

        # Apply smoothing
        self.weight_reg = (
            self.smoothing * self.weight_reg + (1 - self.smoothing) * target_weight
        )

        # Apply to all regularization terms (geom and adp)
        return {
            "geometry": self.weight_reg.detach().item(),
            "adp": self.weight_reg.detach().item(),
        }

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        """
        Get overfitting weighting statistics.

        Parameters
        ----------
        state : LossState, optional
            If provided, pull losses from state.meta.
        """
        if state is not None:
            train_nll = state.get("xray_loss_work", 0.0)
            test_nll = state.get("xray_loss_test", 0.0)
        else:
            train_nll = 0.0
            test_nll = 0.0

        return {
            "overfitting_weight": stat(self.weight_reg.item(), VERBOSITY_ESSENTIAL),
            "target_gap": stat(self.target_gap.item(), VERBOSITY_DEBUG),
            "min_weight": stat(self.min_weight.item(), VERBOSITY_DEBUG),
            "sharpness": stat(self.sharpness.item(), VERBOSITY_DEBUG),
            "train_nll": stat(train_nll, VERBOSITY_STANDARD),
            "test_nll": stat(test_nll, VERBOSITY_STANDARD),
        }

class ManualWeighting(BaseWeighting):
    """
    Apply fixed manual weights.

    This scheme doesn't need any state data - just returns the present weights.
    """

    name = "manual_weighting"

    def __init__(self, weights: Dict[str, float], device: torch.device = None):
        super().__init__(device)
        weights_as_tensor = {k: torch.tensor(v, device=self.device) for k, v in weights.items()}
        self.manual_weights = TensorDict(weights_as_tensor)

    def forward(self, state: "LossState") -> Dict[str, float]:
        """Return manual weights (state is not used)."""
        return {k: v.item() for k, v in self.manual_weights.items()}

class XrayScaleWeighting(BaseWeighting):
    """
    Scale X-ray weight so that the effective X-ray loss is at a fixed target value.

    The X-ray NLL can vary widely (5-200+) depending on data quality, resolution,
    and model state. This scheme computes a weight such that:

        weight * xray_loss ≈ target_scale

    So: weight = target_scale / xray_loss

    Uses state.meta["xray_loss_work"] for the current X-ray loss.

    Tunable parameters (as buffers):
    - target_scale: float, target effective X-ray loss value
    - min_weight: float, minimum allowed weight
    - max_weight: float, maximum allowed weight
    - smoothing: float, exponential smoothing factor (0-1)

    Parameters
    ----------
    device : torch.device, optional
        Computation device.
    target_scale : float, optional
        Target effective X-ray loss value. Default is 50.0.
    min_weight : float, optional
        Minimum allowed weight (prevents division by small loss). Default is 0.01.
    max_weight : float, optional
        Maximum allowed weight (prevents explosion when loss is tiny). Default is 10.0.
    smoothing : float, optional
        Exponential smoothing factor (0-1). Higher = more smoothing. Default is 0.5.
    initial_xray_loss : float, optional
        Initial X-ray loss for scaling. If None, uses first state value.
    """

    name = "xray_scale_weighting"

    def __init__(
        self,
        device: torch.device = None,
        target_scale: float = 50.0,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
        smoothing: float = 0.5,
        initial_xray_loss: float = None,
    ):
        super().__init__(device)
        # Register tunable parameters as buffers for state_dict access
        self.register_buffer("target_scale", torch.tensor(target_scale))
        self.register_buffer("min_weight", torch.tensor(min_weight))
        self.register_buffer("max_weight", torch.tensor(max_weight))
        self.register_buffer("smoothing", torch.tensor(smoothing))
        self.register_buffer("xray_weight", torch.tensor(1.0))  # Current weight value
        self.register_buffer(
            "xray_loss_initial", torch.tensor(initial_xray_loss or 0.0)
        )
        self._initialized = initial_xray_loss is not None

    def forward(self, state: "LossState") -> Dict[str, float]:
        """
        Compute X-ray scale weight from state.meta["xray_loss_work"].
        """
        xray_loss = state.get("xray_loss_work", 1.0)

        # Initialize on first call if not provided at construction
        if not self._initialized:
            self.xray_loss_initial.fill_(xray_loss)
            self._initialized = True

        # weight = target_scale / initial_loss
        target_weight = self.target_scale / (self.xray_loss_initial + 1e-6)

        # Clamp to reasonable range
        target_weight = torch.clamp(
            target_weight, self.min_weight.item(), self.max_weight.item()
        )
        target_weight = target_weight.detach()

        # Apply smoothing
        self.xray_weight = (
            self.smoothing * self.xray_weight + (1 - self.smoothing) * target_weight
        )

        return {"xray": self.xray_weight.detach().item()}

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        """
        Get X-ray scale weighting statistics.

        Parameters
        ----------
        state : LossState, optional
            If provided, pull xray loss from state.meta.
        """
        if state is not None:
            raw_xray = state.get("xray_loss_work", 0.0)
        else:
            raw_xray = 0.0

        return {
            "xray_weight": stat(self.xray_weight.item(), VERBOSITY_ESSENTIAL),
            "raw_xray_loss": stat(raw_xray, VERBOSITY_STANDARD),
            "effective_xray_loss": stat(
                raw_xray * self.xray_weight.item() if raw_xray else 0.0,
                VERBOSITY_STANDARD,
            ),
            "target_scale": stat(self.target_scale.item(), VERBOSITY_DEBUG),
            "min_weight": stat(self.min_weight.item(), VERBOSITY_DEBUG),
            "max_weight": stat(self.max_weight.item(), VERBOSITY_DEBUG),
        }


class ComponentWeighting(nn.Module):
    """
    Combines multiple weighting schemes using nn.ModuleDict.

    Holds weighting schemes but does NOT hold a refinement reference.
    Weighting is computed via compute_weights(state) which receives a LossState.

    Default schemes:
    - 'xray_scale': XrayScaleWeighting - normalizes X-ray loss to consistent scale
    - 'target_offset': TargetOffsetWeighting - weights based on distance from target
    - 'overfitting': OverfittingWeighting - prevents overfitting via Rfree gap

    Parameters
    ----------
    device : torch.device, optional
        Computation device.
    weights : dict, optional
        Manual weight overrides.
    component_weights : dict, optional
        Manual weight overrides for specific components.
    schemes : list of BaseWeighting, optional
        Additional custom weighting schemes.
    initial_xray_loss : float, optional
        Initial X-ray loss for XrayScaleWeighting.

    Attributes
    ----------
    schemes : nn.ModuleDict
        Dictionary of weighting schemes.

    Examples
    --------
    >>> cw = ComponentWeighting(device=device, initial_xray_loss=50.0)
    >>> weights = cw.compute_weights(state)
    >>> xray_scheme = cw['xray_scale']
    >>> xray_scheme.target_scale.fill_(100.0)
    """

    def __init__(
        self,
        device: torch.device = None,
        weights: Dict[str, float] = None,
        component_weights: Dict[str, float] = None,
        schemes: List[BaseWeighting] = None,
        initial_xray_loss: float = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")

        # Build schemes dict - no refinement reference
        schemes_dict = {
            "xray_scale": XrayScaleWeighting(
                device, initial_xray_loss=initial_xray_loss
            ),
            "target_offset": TargetOffsetWeighting(device),
            "overfitting": OverfittingWeighting(device),
        }

        # Add manual weights if provided
        manual_weights_dict = {}
        if weights:
            manual_weights_dict.update(weights)
        if component_weights:
            manual_weights_dict.update(component_weights)

        if manual_weights_dict:
            schemes_dict["manual"] = ManualWeighting(manual_weights_dict, device)

        # Add additional schemes
        if schemes:
            for i, scheme in enumerate(schemes):
                key = getattr(scheme, "name", f"custom_{i}")
                schemes_dict[key] = scheme

        self.schemes = nn.ModuleDict(schemes_dict)

    def __getitem__(self, key: str) -> BaseWeighting:
        """Get a scheme by name using dictionary-style access."""
        return self.schemes[key]

    def __contains__(self, key: str) -> bool:
        """Check if a scheme exists."""
        return key in self.schemes

    def keys(self):
        """Return scheme names."""
        return self.schemes.keys()

    def values(self):
        """Return scheme instances."""
        return self.schemes.values()

    def items(self):
        """Return (name, scheme) pairs."""
        return self.schemes.items()

    def add_scheme(self, name: str, scheme: BaseWeighting):
        """
        Add a new weighting scheme.

        Parameters
        ----------
        name : str
            Name/key for the scheme.
        scheme : BaseWeighting
            The scheme to add.
        """
        self.schemes[name] = scheme

    def forward(self, state: "LossState") -> Dict[str, float]:
        """
        Compute weights from all schemes.

        Returns combined weights (multiplicative for shared keys).
        Does NOT modify state - just returns the computed weights.

        Parameters
        ----------
        state : LossState
            State with meta and _losses populated.

        Returns
        -------
        Dict[str, float]
            Dictionary of computed weights for each component.
        """
        combined = {}

        for scheme in self.schemes.values():
            scheme_weights = scheme.forward(state)
            for k, v in scheme_weights.items():
                if k in combined:
                    combined[k] = combined[k] * v
                else:
                    combined[k] = v

        return combined

    def total_loss_from_state(self, state: "LossState") -> torch.Tensor:
        """
        Compute total weighted loss from a LossState.

        Calls aggregate() which evaluates targets and applies weights.

        Parameters
        ----------
        state : LossState
            LossState with targets and weights registered.

        Returns
        -------
        torch.Tensor
            Total weighted loss.
        """
        return state.aggregate()

    def stats(self, state: "LossState" = None) -> Dict[str, Any]:
        """
        Return statistics for reporting.

        Parameters
        ----------
        state : LossState, optional
            If provided, pull data from LossState.

        Returns
        -------
        dict
            Stats dictionary with StatEntry objects.
        """
        stats = {}

        # Collect stats from schemes
        for name, scheme in self.schemes.items():
            scheme_stats = scheme.stats(state)
            if scheme_stats:
                stats[name] = scheme_stats

        # Add current weights from state
        if state is not None:
            stats["weights"] = {
                k: stat(v if isinstance(v, (int, float)) else v, VERBOSITY_STANDARD)
                for k, v in state.weights.items()
            }

            # Add xray stats from state.meta
            work_nll = state.get("xray_loss_work", 0.0)
            test_nll = state.get("xray_loss_test", 0.0)
            stats["xray"] = {
                "work_nll": stat(work_nll, VERBOSITY_ESSENTIAL),
                "test_nll": stat(test_nll, VERBOSITY_ESSENTIAL),
            }

        return stats


__all__ = [
    # Base class
    "BaseWeighting",
    # Weighting classes
    "WeightingScheme",  # Alias for backward compatibility
    "TargetOffsetWeighting",
    "OverfittingWeighting",
    "ManualWeighting",
    "XrayScaleWeighting",
    "ComponentWeighting",
]
