"""
Random Weighting Scheme for Policy Training Data Collection.

This module implements a random weighting scheme that samples component weights
from log-normal distributions around default values. This is used to generate
diverse training trajectories for initializing a policy network via AWR
(Advantage-Weighted Regression).

The key insight is that by randomly sampling weights during refinement and
recording the resulting R-free trajectories, we can train a policy to predict
optimal weights based on refinement state.

Weight Sampling Strategy
------------------------
Two-level sampling for stable yet diverse trajectories:

1. **Trajectory-level base weights**: Sampled once at initialization with larger
   sigma (trajectory_sigma). These define the "character" of the trajectory.

2. **Step-level perturbations**: Small perturbations applied at each step with
   smaller sigma (step_sigma). These add local variation while keeping the
   trajectory coherent.

For each component at each step:
    base_log_weight ~ N(default, trajectory_sigma)  [sampled once]
    step_perturbation ~ N(0, step_sigma)            [sampled each step]
    log_weight = base_log_weight + step_perturbation
    weight = exp(log_weight)

References
----------
- "Advantage-Weighted Regression" (Peng et al., 2019)
- design_choices.md in meta_weighting_2/
"""

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
from torch import nn

from torchref.config import get_default_device
from torchref.refinement.weighting.base_weighting import BaseWeighting
from torchref.refinement.weighting.component_weighting import (
    ComponentWeighting,
    ResolutionWeighting,
)
from torchref.utils.stats import (
    VERBOSITY_DETAILED,
    VERBOSITY_ESSENTIAL,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState

# Default log-space weights (these are the log of the actual weights)
# weight = exp(log_weight)
DEFAULT_LOG_WEIGHTS = {
    "xray": 0.0,
    # Geometry components
    "bond": 0.0,
    "angle": 0.0,
    "torsion": 0.0,
    "planarity": 0.0,
    "chiral": 0.0,
    "nonbonded": 0.0,
    # ADP components
    "simu": 0.0,
    "locality": 0.0,
    "KL": 0.0,
}

# Default trajectory-level sigmas (larger - defines trajectory character)
# Increased for more diverse exploration
DEFAULT_TRAJECTORY_SIGMAS = {
    "xray": 1.5,
    "bond": 1.5,
    "angle": 1.5,
    "torsion": 1.5,
    "planarity": 1.5,
    "chiral": 1.5,
    "nonbonded": 1.5,
    "simu": 1.5,
    "locality": 1.5,
    "KL": 1.5,
}

# Default step-level sigmas (smaller - local perturbations)
# Increased for more variation within trajectories
DEFAULT_STEP_SIGMAS = {
    "xray": 0.3,
    "bond": 0.3,
    "angle": 0.3,
    "torsion": 0.3,
    "planarity": 0.3,
    "chiral": 0.3,
    "nonbonded": 0.3,
    "simu": 0.3,
    "locality": 0.3,
    "KL": 0.3,
}

# Bounds for log-weights to prevent extreme values
LOG_WEIGHT_MIN = -3.0
LOG_WEIGHT_MAX = 3.0


class RandomWeightingScheme(BaseWeighting):
    """
    Weighting scheme with two-level random sampling for stable trajectories.

    Sampling strategy:
    1. Base weights sampled once at initialization (trajectory_sigma)
    2. Small perturbations applied at each step (step_sigma)

    This creates diverse trajectories while keeping them coherent within
    a single refinement run.

    Parameters
    ----------
    device : torch.device, optional
        Computation device. Defaults to the configured device.current.
    default_log_weights : dict, optional
        Default log-space weights for each component (mean of distribution).
        Default is DEFAULT_LOG_WEIGHTS.
    trajectory_sigmas : dict, optional
        Standard deviations for trajectory-level base weight sampling.
        Default is DEFAULT_TRAJECTORY_SIGMAS (0.8 for all).
    step_sigmas : dict, optional
        Standard deviations for step-level perturbations.
        Default is DEFAULT_STEP_SIGMAS (0.1 for all).
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    base_log_weights : dict
        Trajectory-level base weights (sampled once at init).
    current_log_weights : dict
        Current weights including step perturbation.
    current_weights : dict
        Current weights (exp of log_weights).
    sample_count : int
        Number of times step perturbations have been applied.
    """

    name = "random_weighting"

    def __init__(
        self,
        device: torch.device = None,
        default_log_weights: Dict[str, float] = None,
        trajectory_sigmas: Dict[str, float] = None,
        step_sigmas: Dict[str, float] = None,
        seed: int = None,
    ):
        super().__init__(device)

        # Set defaults
        self.default_log_weights = default_log_weights or DEFAULT_LOG_WEIGHTS.copy()
        self.trajectory_sigmas = trajectory_sigmas or DEFAULT_TRAJECTORY_SIGMAS.copy()
        self.step_sigmas = step_sigmas or DEFAULT_STEP_SIGMAS.copy()

        # Initialize RNG
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        # Register buffers for base weights (trajectory-level, sampled once)
        for name, default in self.default_log_weights.items():
            self.register_buffer(f"base_log_weight_{name}", torch.tensor(default))
            self.register_buffer(f"log_weight_{name}", torch.tensor(default))

        # Track sampling
        self.register_buffer("_sample_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_trajectory_initialized", torch.tensor(False))

        # Initialize base weights and current weights
        self.base_log_weights = {}
        self.current_log_weights = {}
        self.current_weights = {}

        # Sample trajectory base weights
        self._sample_base_weights()
        self._update_weight_tensors()

    def _sample_base_weights(self):
        """Sample trajectory-level base weights (called once at init)."""
        for name, default_log in self.default_log_weights.items():
            sigma = self.trajectory_sigmas.get(name, 0.8)

            # Sample base weight in log-space
            base_log_weight = self.rng.normal(default_log, sigma)

            # Clip to bounds
            base_log_weight = np.clip(base_log_weight, LOG_WEIGHT_MIN, LOG_WEIGHT_MAX)

            # Update buffer
            getattr(self, f"base_log_weight_{name}").fill_(base_log_weight)
            getattr(self, f"log_weight_{name}").fill_(base_log_weight)

            self.base_log_weights[name] = float(base_log_weight)

        self._trajectory_initialized.fill_(True)

    def _update_weight_tensors(self):
        """Update weight tensors from current log_weight buffers."""
        self.current_log_weights = {}
        self.current_weights = {}

        for name in self.default_log_weights.keys():
            log_w = getattr(self, f"log_weight_{name}")
            self.current_log_weights[name] = float(log_w.item())
            self.current_weights[name] = float(torch.exp(log_w).item())

    def apply_step_perturbation(self) -> Dict[str, float]:
        """
        Apply small perturbation to base weights for current step.

        Returns
        -------
        dict
            Dictionary of perturbed weights.
        """
        for name in self.default_log_weights.keys():
            base_log = self.base_log_weights[name]
            step_sigma = self.step_sigmas.get(name, 0.1)

            # Sample small perturbation
            perturbation = self.rng.normal(0.0, step_sigma)

            # Apply perturbation to base weight
            log_weight = base_log + perturbation

            # Clip to bounds
            log_weight = np.clip(log_weight, LOG_WEIGHT_MIN, LOG_WEIGHT_MAX)

            # Update buffer
            getattr(self, f"log_weight_{name}").fill_(log_weight)

        self._sample_count.add_(1)
        self._update_weight_tensors()

        return self.current_weights.copy()

    def resample_trajectory(self) -> Dict[str, float]:
        """
        Resample trajectory-level base weights.

        Call this to start a completely new trajectory with different
        base weights.

        Returns
        -------
        dict
            Dictionary of new base weights.
        """
        self._sample_base_weights()
        self._update_weight_tensors()
        self._sample_count.fill_(0)
        return self.current_weights.copy()

    def resample_weights(self) -> Dict[str, float]:
        """
        Apply step perturbation (for backwards compatibility).

        This is equivalent to apply_step_perturbation().

        Returns
        -------
        dict
            Dictionary of perturbed weights.
        """
        return self.apply_step_perturbation()

    def forward(self, state: "LossState" = None) -> Dict[str, float]:
        """
        Return current weights.

        Parameters
        ----------
        state : LossState, optional
            Current loss state (not used by this scheme, but required by interface).

        Returns
        -------
        dict
            Dictionary mapping component names to weight values.
        """
        return self.current_weights.copy()

    @property
    def sample_count(self) -> int:
        """Get number of step perturbations applied."""
        return self._sample_count.item()

    def get_log_weights(self) -> Dict[str, float]:
        """
        Get current log-space weights (base + perturbation).

        Returns
        -------
        dict
            Dictionary of current log-space weights.
        """
        return self.current_log_weights.copy()

    def get_base_log_weights(self) -> Dict[str, float]:
        """
        Get trajectory-level base log-space weights.

        Returns
        -------
        dict
            Dictionary of base log-space weights.
        """
        return self.base_log_weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights (linear space).

        Returns
        -------
        dict
            Dictionary of current weights.
        """
        return self.current_weights.copy()

    def stats(self, state: "LossState" = None) -> Dict[str, StatEntry]:
        """
        Return statistics for reporting.

        Parameters
        ----------
        state : LossState, optional
            If provided, can pull data from LossState (not used by this scheme).

        Returns
        -------
        dict
            Statistics dictionary with StatEntry objects.
        """
        stats = {
            "sample_count": stat(self.sample_count, VERBOSITY_STANDARD),
        }

        # Add current weights and base weights
        for name, weight in self.current_weights.items():
            stats[f"weight_{name}"] = stat(weight, VERBOSITY_STANDARD)
            stats[f"log_weight_{name}"] = stat(
                self.current_log_weights[name], VERBOSITY_DETAILED
            )
            stats[f"base_log_weight_{name}"] = stat(
                self.base_log_weights.get(name, 0.0), VERBOSITY_DETAILED
            )

        return stats


class RandomComponentWeighting(ComponentWeighting):
    """
    Component weighting with two-level random sampling for data collection.

    This is a drop-in replacement for ComponentWeighting that samples
    weights using a two-level strategy:
    1. Trajectory-level base weights (sampled once at init)
    2. Step-level perturbations (applied at each update)

    The weighting combines:
    1. XrayScaleWeighting - normalizes X-ray loss to consistent scale
    2. RandomWeightingScheme - two-level random weight sampling

    Parameters
    ----------
    device : torch.device, optional
        Computation device. Defaults to the configured device.current.
    default_log_weights : dict, optional
        Default log-space weights for each component.
    trajectory_sigmas : dict, optional
        Standard deviations for trajectory-level sampling (default 0.8).
    step_sigmas : dict, optional
        Standard deviations for step-level perturbations (default 0.1).
    seed : int, optional
        Random seed for reproducibility.
    resample_each_step : bool, optional
        If True, apply step perturbation at each compute_weights() call.
        If False, only apply perturbation when resample_weights() is called.
        Default is True.
    initial_xray_loss : float, optional
        Initial X-ray loss for XrayScaleWeighting.

    Examples
    --------
    ::

        from torchref.refinement.weighting import RandomComponentWeighting
        weighting = RandomComponentWeighting(device=device, seed=42)
        # Base weights are sampled at init
        base = weighting.get_base_log_weights()
        # Each compute_weights applies small perturbation
        weights = weighting.compute_weights(state)
        current = weighting.get_sampled_log_weights()  # base + perturbation
    """

    def __init__(
        self,
        device: torch.device = None,
        default_log_weights: Dict[str, float] = None,
        trajectory_sigmas: Dict[str, float] = None,
        step_sigmas: Dict[str, float] = None,
        seed: int = None,
        resample_each_step: bool = True,
        initial_xray_loss: float = None,
    ):
        # Don't call parent __init__ - we'll set up our own schemes
        nn.Module.__init__(self)
        self.device = device or get_default_device()
        self.resample_each_step = resample_each_step

        # Build schemes dict with resolution weighting and random
        schemes_dict = {
            "resolution": ResolutionWeighting(device),
            "random": RandomWeightingScheme(
                device,
                default_log_weights=default_log_weights,
                trajectory_sigmas=trajectory_sigmas,
                step_sigmas=step_sigmas,
                seed=seed,
            ),
        }

        self.schemes = nn.ModuleDict(schemes_dict)

    @property
    def random_scheme(self) -> RandomWeightingScheme:
        """Get the random weighting scheme."""
        return self.schemes["random"]

    def resample_trajectory(self) -> Dict[str, float]:
        """
        Resample trajectory-level base weights for a new trajectory.

        Call this to start a completely new trajectory with different
        base weights.

        Returns
        -------
        dict
            Dictionary of new base weights from the random scheme.
        """
        return self.random_scheme.resample_trajectory()

    def resample_weights(self) -> Dict[str, float]:
        """
        Apply step perturbation.

        Returns
        -------
        dict
            Dictionary of newly perturbed weights.
        """
        return self.random_scheme.apply_step_perturbation()

    def compute_weights(self, state: "LossState") -> Dict[str, float]:
        """
        Compute weights from all schemes.

        If resample_each_step is True, also applies step perturbation.
        Returns combined weights (multiplicative for shared keys).
        Does NOT modify state - just returns the computed weights.

        Parameters
        ----------
        state : LossState
            State with meta and _losses populated.

        Returns
        -------
        dict
            Dictionary of computed weights for each component.
        """
        if self.resample_each_step:
            self.random_scheme.apply_step_perturbation()

        combined = {}

        # Combine weights from all schemes (multiply for shared keys)
        for scheme in self.schemes.values():
            scheme_weights = scheme.forward(state)
            for k, v in scheme_weights.items():
                if k in combined:
                    combined[k] = combined[k] * v
                else:
                    combined[k] = v

        return combined

    def get_sampled_log_weights(self) -> Dict[str, float]:
        """
        Get the current log-space weights (base + perturbation).

        These are the "actions" taken by the random policy.

        Returns
        -------
        dict
            Dictionary of log-space weights.
        """
        return self.random_scheme.get_log_weights()

    def get_base_log_weights(self) -> Dict[str, float]:
        """
        Get the trajectory-level base log-space weights.

        Returns
        -------
        dict
            Dictionary of base log-space weights.
        """
        return self.random_scheme.get_base_log_weights()

    def get_sampled_weights(self) -> Dict[str, float]:
        """
        Get the weights from the random scheme (before xray scaling).

        Returns
        -------
        dict
            Dictionary of sampled weights.
        """
        return self.random_scheme.get_weights()

    def stats(self, state: "LossState" = None) -> Dict[str, Any]:
        """
        Return statistics for reporting.

        Parameters
        ----------
        state : LossState, optional
            If provided, pull data from state.meta.

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

        # Add sampled log-weights (the "actions")
        stats["sampled_log_weights"] = {
            k: stat(v, VERBOSITY_STANDARD)
            for k, v in self.get_sampled_log_weights().items()
        }

        # Add xray stats from state.meta
        if state is not None:
            work_nll = state.get("xray_loss_work", 0.0)
            test_nll = state.get("xray_loss_test", 0.0)
            stats["xray"] = {
                "work_nll": stat(work_nll, VERBOSITY_ESSENTIAL),
                "test_nll": stat(test_nll, VERBOSITY_ESSENTIAL),
            }

            # Add current weights from state
            stats["weights"] = {
                k: stat(v if isinstance(v, (int, float)) else v, VERBOSITY_STANDARD)
                for k, v in state.weights.items()
            }

        return stats


__all__ = [
    "RandomWeightingScheme",
    "RandomComponentWeighting",
    "DEFAULT_LOG_WEIGHTS",
    "DEFAULT_TRAJECTORY_SIGMAS",
    "DEFAULT_STEP_SIGMAS",
]
