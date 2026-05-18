from typing import TYPE_CHECKING, Any, Dict, List

import torch
from torch import nn

from torchref.refinement.weighting.base_weighting import BaseWeighting
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
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


class ResolutionWeighting(BaseWeighting):
    """
    Base prior strength with optional resolution-dependent correction.

    With proper NLL sums and perfectly calibrated sigmas, w=1 would be
    the pure Bayesian answer. However empirical sweep on 50 structures
    (1.15-3.0 A) showed the monomer library sigmas lead to too-loose
    geometry at w=1. The optimal base is:

        w_geometry ~= 10  (from median Rfree minimum)
        w_adp      ~= 1   (lower is better for most resolutions)

    Geometry and ADP have different base weights because the monomer
    library geometry sigmas are tighter than the ADP restraint sigmas
    are loose, so they need different compensation.

    Optional resolution dependence:

        w_geometry = base_w_geometry * (d_min / d_ref) ^ alpha
        w_adp      = base_w_adp * (d_min / d_ref) ^ alpha

    alpha=0 disables resolution correction. The sweep found the
    resolution dependence was weak (<20% effect), so alpha=0 is the
    default.

    Parameters
    ----------
    device : torch.device, optional
        Computation device.
    base_w_geometry : float, optional
        Base geometry weight. Default 10.0 (from empirical sweep).
    base_w_adp : float, optional
        Base ADP weight. Default 1.0 (from empirical sweep).
    d_ref : float, optional
        Reference resolution for the optional power-law correction.
        Default 2.0 A.
    alpha : float, optional
        Resolution sensitivity. Default 0.0 (disabled).
    """

    name = "resolution_weighting"

    def __init__(
        self,
        device: torch.device = None,
        base_w_geometry: float = 1.0,
        base_w_adp: float = 1.0,
        d_ref: float = 2.0,
        alpha: float = 0.0,
    ):
        super().__init__(device)
        self.register_buffer("base_w_geometry", torch.tensor(base_w_geometry))
        self.register_buffer("base_w_adp", torch.tensor(base_w_adp))
        self.register_buffer("d_ref", torch.tensor(d_ref))
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, state: "LossState") -> Dict[str, float]:
        """Compute base weights with optional resolution correction."""
        d_min = state.get("resolution_min", 2.0)
        if not isinstance(d_min, torch.Tensor):
            d_min = torch.tensor(d_min, device=self.device)

        res_factor = (d_min / self.d_ref) ** self.alpha
        w_geom = torch.clamp(
            self.base_w_geometry * res_factor, 0.1, 100.0
        ).detach().item()
        w_adp = torch.clamp(
            self.base_w_adp * res_factor, 0.1, 100.0
        ).detach().item()

        return {"geometry": w_geom, "adp": w_adp}

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        if state is not None:
            d_min = state.get("resolution_min", 0.0)
            w = self.forward(state)
        else:
            d_min = 0.0
            w = {"geometry": self.base_w_geometry.item(),
                 "adp": self.base_w_adp.item()}
        return {
            "resolution_w_geom": stat(w["geometry"], VERBOSITY_ESSENTIAL),
            "resolution_w_adp": stat(w["adp"], VERBOSITY_ESSENTIAL),
            "d_min": stat(d_min, VERBOSITY_STANDARD),
            "d_ref": stat(self.d_ref.item(), VERBOSITY_DEBUG),
            "alpha": stat(self.alpha.item(), VERBOSITY_DEBUG),
        }


class OverfittingWeighting(BaseWeighting):
    """
    Dynamic overfitting correction based on Rfree - Rwork gap.

    Uses R-factors (scale-invariant, per-reflection-normalized) rather
    than NLL values, which are incomparable between work/test sets after
    switching to summed NLLs.

    When the Rfree-Rwork gap exceeds target_gap, exponentially increases
    regularization. **The correction is applied primarily to ADP weights
    and only weakly to geometry weights**: in crystallographic refinement,
    overfitting is typically driven by B-factors (which have one parameter
    per atom and relatively weak restraints) rather than coordinates
    (which are held tightly by the geometry prior).

    Effective correction:
        factor = min_weight + exp(sharpness * (gap - target_gap))
        w_adp      *= factor
        w_geometry *= 1 + geom_share * (factor - 1)

    With geom_share = 0.2, only 20% of the overfitting correction is
    applied to geometry, keeping most of the effect on ADP.

    Tunable parameters (as buffers):
    - target_gap: R-factor gap threshold. Default 0.05 (5%).
    - min_weight: base correction factor. Default 1.0.
    - sharpness: exponential response steepness. Default 30.0.
    - geom_share: fraction of correction applied to geometry. Default 1.0.
    - smoothing: EMA smoothing factor (0-1). Default 0.8.
    """

    name = "overfitting_weighting"

    def __init__(
        self,
        device: torch.device = None,
        target_gap: float = 0.05,
        min_weight: float = 1.0,
        sharpness: float = 30.0,
        geom_share: float = 1.0,
        smoothing: float = 0.8,
    ):
        super().__init__(device)
        self.register_buffer("target_gap", torch.tensor(target_gap))
        self.register_buffer("min_weight", torch.tensor(min_weight))
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("geom_share", torch.tensor(geom_share))
        self.register_buffer("smoothing", torch.tensor(smoothing))
        self.register_buffer("weight_reg", torch.tensor(1.0))

    def forward(self, state: "LossState") -> Dict[str, float]:
        """Compute overfitting correction weights from R-factor gap."""
        rwork = state.get("rwork", 0.0)
        rfree = state.get("rfree", 0.0)

        if not isinstance(rwork, torch.Tensor):
            rwork = torch.tensor(rwork, device=self.device)
        if not isinstance(rfree, torch.Tensor):
            rfree = torch.tensor(rfree, device=self.device)

        gap = rfree - rwork

        target_weight = self.min_weight + torch.exp(
            self.sharpness * (gap - self.target_gap)
        )
        target_weight = target_weight.detach()

        self.weight_reg = (
            self.smoothing * self.weight_reg + (1 - self.smoothing) * target_weight
        )

        adp_factor = self.weight_reg.detach().item()
        # Apply only a share of the correction to geometry
        geom_factor = 1.0 + self.geom_share.item() * (adp_factor - 1.0)

        return {
            "geometry": geom_factor,
            "adp": adp_factor,
        }

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        if state is not None:
            rwork = state.get("rwork", 0.0)
            rfree = state.get("rfree", 0.0)
        else:
            rwork = 0.0
            rfree = 0.0

        return {
            "overfitting_weight": stat(self.weight_reg.item(), VERBOSITY_ESSENTIAL),
            "target_gap": stat(self.target_gap.item(), VERBOSITY_DEBUG),
            "min_weight": stat(self.min_weight.item(), VERBOSITY_DEBUG),
            "sharpness": stat(self.sharpness.item(), VERBOSITY_DEBUG),
            "rwork": stat(rwork, VERBOSITY_STANDARD),
            "rfree": stat(rfree, VERBOSITY_STANDARD),
        }


class ManualWeighting(BaseWeighting):
    """
    Apply fixed manual weights.

    This scheme doesn't need any state data - just returns the present weights.
    """

    name = "manual_weighting"

    def __init__(self, weights: Dict[str, float], device: torch.device = None):
        super().__init__(device)
        weights_as_tensor = {
            k: torch.tensor(v, device=self.device) for k, v in weights.items()
        }
        self.manual_weights = TensorDict(weights_as_tensor)

    def forward(self, state: "LossState") -> Dict[str, float]:
        """Return manual weights (state is not used)."""
        return {k: v.item() for k, v in self.manual_weights.items()}


class ComponentWeighting(DeviceMixin, nn.Module):
    """
    Combines multiple weighting schemes using nn.ModuleDict.

    Holds weighting schemes but does NOT hold a refinement reference.
    Weighting is computed via forward(state) which receives a LossState.

    Default schemes:
    - 'resolution': ResolutionWeighting - resolution-dependent prior strength
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

    Attributes
    ----------
    schemes : nn.ModuleDict
        Dictionary of weighting schemes.
    """

    def __init__(
        self,
        device: torch.device = None,
        weights: Dict[str, float] = None,
        component_weights: Dict[str, float] = None,
        schemes: List[BaseWeighting] = None,
        # Legacy parameter, ignored
        initial_xray_loss: float = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")

        schemes_dict = {
            # "resolution": ResolutionWeighting(device),
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
        """Add a new weighting scheme."""
        self.schemes[name] = scheme

    def forward(self, state: "LossState") -> Dict[str, float]:
        """
        Compute weights from all schemes.

        Returns combined weights (multiplicative for shared keys).
        Does NOT modify state - just returns the computed weights.
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
        """Compute total weighted loss from a LossState."""
        return state.aggregate()

    def stats(self, state: "LossState" = None) -> Dict[str, Any]:
        """Return statistics for reporting."""
        stats = {}

        for name, scheme in self.schemes.items():
            scheme_stats = scheme.stats(state)
            if scheme_stats:
                stats[name] = scheme_stats

        if state is not None:
            stats["weights"] = {
                k: stat(v if isinstance(v, (int, float)) else v, VERBOSITY_STANDARD)
                for k, v in state.weights.items()
            }

            rwork = state.get("rwork", 0.0)
            rfree = state.get("rfree", 0.0)
            stats["xray"] = {
                "rwork": stat(rwork, VERBOSITY_ESSENTIAL),
                "rfree": stat(rfree, VERBOSITY_ESSENTIAL),
            }

        return stats


__all__ = [
    # Base class
    "BaseWeighting",
    # Weighting classes
    "WeightingScheme",  # Alias for backward compatibility
    "ResolutionWeighting",
    "OverfittingWeighting",
    "ManualWeighting",
    "ComponentWeighting",
]
