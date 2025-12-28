"""
Policy-based component weighting for crystallographic refinement.

This module implements a weighting scheme that uses a trained neural network
policy to predict component weights from the current refinement state.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from torchref.refinement.component_weighting import WeightingScheme
from torchref.utils.utils import ModuleReference
from torchref.utils.stats import stat, VERBOSITY_ESSENTIAL, VERBOSITY_STANDARD, VERBOSITY_DEBUG


# Component names (must match policy network)
COMPONENTS = [
    'xray', 'bond', 'angle', 'torsion', 'planarity',
    'chiral', 'nonbonded', 'simu', 'locality', 'KL'
]


class PolicyComponentWeighting(WeightingScheme):
    """
    Policy-based component weighting using a trained neural network.

    This weighting scheme uses a policy network to predict component weights
    from the current refinement state. The policy can optionally predict
    uncertainties for exploration.

    Parameters
    ----------
    refinement : Refinement
        Reference to Refinement object
    policy_path : str
        Path to trained policy checkpoint (.pt file)
    use_uncertainty : bool, optional
        Whether to use predicted uncertainties for sampling (default: False)
    temperature : float, optional
        Sampling temperature for uncertainty-based exploration (default: 0.0)
        - 0.0: deterministic (use mean predictions)
        - > 0.0: sample from distribution with scaled variance

    Attributes
    ----------
    policy : WeightPolicyNetwork
        Loaded policy network
    use_uncertainty : bool
        Whether to sample from predicted distribution
    temperature : float
        Sampling temperature for exploration
    predicted_weights : dict
        Most recently predicted weights
    predicted_uncertainties : dict
        Most recently predicted uncertainties

    Examples
    --------
    >>> # Setup in refinement
    >>> refinement.setup_policy_weighting('policy.pt')
    >>>
    >>> # Or manually add to ComponentWeighting
    >>> policy_scheme = PolicyComponentWeighting(refinement, 'policy.pt')
    >>> refinement.component_weighting.add_scheme('policy', policy_scheme)
    """

    name = 'policy_weighting'

    def __init__(
        self,
        refinement,
        policy_path: str,
        use_uncertainty: bool = False,
        temperature: float = 0.0
    ):
        super().__init__(refinement)

        # Load policy
        from meta_weigthing_2.policy_network import load_policy
        self.policy = load_policy(policy_path)
        self.policy.eval()  # Set to evaluation mode

        # Sampling parameters
        self.use_uncertainty = use_uncertainty
        self.temperature = temperature

        # Cache for most recent predictions
        self.predicted_weights = {}
        self.predicted_uncertainties = {}

        # Track previous step R-factors for delta calculation
        self.prev_rwork = None
        self.prev_rfree = None

        print(f"PolicyComponentWeighting initialized")
        print(f"  Policy: {policy_path}")
        print(f"  Parameters: {self.policy.get_param_count():,}")
        print(f"  Use uncertainty: {use_uncertainty}")
        print(f"  Temperature: {temperature}")

    def extract_state_features(self) -> torch.Tensor:
        """
        Extract state features from current refinement state.

        Returns
        -------
        torch.Tensor, shape (state_dim,)
            State feature vector for policy input

        Features extracted (21 total):
        - R-work, R-free, R-free gap (3)
        - Delta R-work, delta R-free (2, from previous step, 0 for first step)
        - X-ray loss work + test (2, RAW values)
        - Geometry loss, ADP loss (2, RAW values)
        - Geometry metrics: bond RMSD, angle RMSD, mean B / 100 (3)
        - Component losses: 9 per-component losses (6 geometry + 3 ADP, RAW values)

        NOTE: All losses are RAW (not log-transformed) to match training data
        """
        ref = self.refinement

        # Get current R-factors
        rwork, rfree = ref.get_rfactor()
        rfree_gap = rfree - rwork

        # Compute deltas from previous step (0 for first step)
        if self.prev_rwork is not None:
            delta_rwork = rwork - self.prev_rwork
            delta_rfree = rfree - self.prev_rfree
        else:
            delta_rwork = 0.0
            delta_rfree = 0.0

        # Get losses
        xray_loss_work = ref.xray_target_work().item()
        xray_loss_test = ref.xray_target_test().item()

        # Geometry losses
        if hasattr(ref, 'geometry_target'):
            geom_losses = ref.geometry_target.target_losses()
            geom_loss_total = sum(loss.item() for loss in geom_losses.values())
        else:
            geom_loss_total = 0.0
            geom_losses = {}

        # ADP losses
        if hasattr(ref, 'adp_target'):
            adp_losses = ref.adp_target.target_losses()
            adp_loss_total = sum(loss.item() for loss in adp_losses.values())
        else:
            adp_loss_total = 0.0
            adp_losses = {}

        # Geometry metrics
        if hasattr(ref, 'geometry_target'):
            geom_stats = ref.geometry_target.stats()
            bond_rmsd = geom_stats.get('bond_rmsd', 0.0)
            angle_rmsd = geom_stats.get('angle_rmsd', 0.0)
        else:
            bond_rmsd = 0.0
            angle_rmsd = 0.0

        # Mean B-factor
        if hasattr(ref, 'atoms') and hasattr(ref.atoms, 'b_iso'):
            mean_b = ref.atoms.b_iso.mean().item()
        else:
            mean_b = 0.0

        # Build feature vector
        # Note: For first step, deltas are 0
        # IMPORTANT: Use RAW (non-logged) values to match training data
        features = [
            # R-factors
            rwork,
            rfree,
            rfree_gap,
            delta_rwork,
            delta_rfree,

            # Losses (RAW values, NOT logged - must match training)
            xray_loss_work,
            xray_loss_test,
            geom_loss_total,
            adp_loss_total,

            # Geometry metrics
            bond_rmsd,
            angle_rmsd,
            mean_b / 100.0,  # Normalize by typical scale

            # Component losses (RAW values, NOT logged)
            # Note: 9 component losses (xray is already in base features above)
        ]

        # Add component losses in fixed order
        # Note: 'xray' is NOT included here - it's already in base features
        # Only add the 9 geometry/ADP component losses
        for comp in COMPONENTS:
            # Skip xray - it's already in base features
            if comp == 'xray':
                continue

            # Map component names to target names
            target_name = comp
            if comp in ['bond', 'angle', 'torsion', 'planarity', 'chiral', 'nonbonded']:
                target_name = comp  # Geometry components
            elif comp in ['simu', 'locality', 'KL']:
                target_name = comp  # ADP components

            # Get loss value
            if target_name in geom_losses:
                loss_val = geom_losses[target_name].item()
            elif target_name in adp_losses:
                loss_val = adp_losses[target_name].item()
            else:
                loss_val = 0.0

            # Use RAW value (NOT logged) to match training data
            features.append(loss_val)

        # Update previous R-factors for next step
        self.prev_rwork = rwork
        self.prev_rfree = rfree

        # Convert to tensor
        state_tensor = torch.tensor(features, dtype=torch.float32)

        return state_tensor

    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Predict component weights from current refinement state.

        Returns
        -------
        dict
            Dictionary of {component: weight} for all components
        """
        # Extract state features
        state_features = self.extract_state_features()

        # Get predictions from policy
        with torch.no_grad():
            if self.use_uncertainty and self.temperature > 0:
                # Sample from distribution
                weights_dict = self.policy.sample_weights(
                    state_features,
                    temperature=self.temperature,
                    use_uncertainty=True
                )
                # Policy returns linear-space weights
                # Also get uncertainties for stats
                self.predicted_weights, self.predicted_uncertainties = \
                    self.policy.predict_weights(state_features)
            else:
                # Deterministic prediction
                weights_dict, self.predicted_uncertainties = \
                    self.policy.predict_weights(state_features)
                self.predicted_weights = weights_dict

        # Convert to torch tensors on correct device
        device = self.refinement.device
        torch_weights = {
            comp: torch.tensor(weight, device=device, dtype=torch.float32)
            for comp, weight in weights_dict.items()
        }

        return torch_weights

    def stats(self) -> Dict[str, any]:
        """
        Return statistics for reporting.

        Returns
        -------
        dict
            Statistics dictionary with predicted weights and uncertainties
        """
        stats_dict = {}

        # Predicted weights (essential)
        if self.predicted_weights:
            stats_dict['predicted_weights'] = {
                comp: stat(weight, VERBOSITY_STANDARD)
                for comp, weight in self.predicted_weights.items()
            }

        # Predicted uncertainties (detailed)
        if self.predicted_uncertainties:
            stats_dict['predicted_uncertainties'] = {
                comp: stat(sigma, VERBOSITY_DEBUG)
                for comp, sigma in self.predicted_uncertainties.items()
            }

        # Sampling parameters
        stats_dict['use_uncertainty'] = stat(self.use_uncertainty, VERBOSITY_DEBUG)
        stats_dict['temperature'] = stat(self.temperature, VERBOSITY_DEBUG)

        return stats_dict


# Integration helper function for base_refinement.py
def setup_policy_weighting(
    refinement,
    policy_path: str,
    use_uncertainty: bool = False,
    temperature: float = 0.0,
    replace_existing: bool = True
):
    """
    Setup policy-based component weighting for refinement.

    This is a helper function to integrate PolicyComponentWeighting into
    an existing refinement object. It can either replace the existing
    component weighting or add as an additional scheme.

    Parameters
    ----------
    refinement : Refinement
        Refinement object to configure
    policy_path : str
        Path to trained policy checkpoint (.pt file)
    use_uncertainty : bool, optional
        Whether to use predicted uncertainties for sampling (default: False)
    temperature : float, optional
        Sampling temperature for exploration (default: 0.0)
    replace_existing : bool, optional
        If True, replace entire ComponentWeighting with policy only
        If False, add policy as additional scheme to existing weighting
        (default: True)

    Examples
    --------
    >>> # Replace all weighting schemes with policy
    >>> refinement.setup_policy_weighting('policy.pt', replace_existing=True)
    >>>
    >>> # Add policy to existing weighting schemes
    >>> refinement.setup_policy_weighting('policy.pt', replace_existing=False)
    >>>
    >>> # Use with uncertainty-based exploration
    >>> refinement.setup_policy_weighting(
    ...     'policy.pt',
    ...     use_uncertainty=True,
    ...     temperature=0.5
    ... )
    """
    from torchref.refinement.component_weighting import ComponentWeighting

    # Create policy scheme
    policy_scheme = PolicyComponentWeighting(
        refinement,
        policy_path=policy_path,
        use_uncertainty=use_uncertainty,
        temperature=temperature
    )

    if replace_existing:
        # Replace entire component weighting with policy only
        refinement.component_weighting = ComponentWeighting(
            refinement,
            schemes=[policy_scheme]
        )
        print("Replaced component weighting with policy")
    else:
        # Add to existing schemes
        if not hasattr(refinement, 'component_weighting'):
            # No existing weighting, create new
            refinement.component_weighting = ComponentWeighting(
                refinement,
                schemes=[policy_scheme]
            )
            print("Created component weighting with policy")
        else:
            # Add to existing
            refinement.component_weighting.add_scheme('policy', policy_scheme)
            print("Added policy to existing component weighting")

    return refinement.component_weighting


__all__ = [
    'PolicyComponentWeighting',
    'setup_policy_weighting',
    'COMPONENTS',
]
