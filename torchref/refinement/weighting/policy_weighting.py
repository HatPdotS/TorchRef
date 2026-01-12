"""
Policy-based component weighting for crystallographic refinement.

This module implements a weighting scheme that uses a trained neural network
policy to predict component weights from the current refinement state.

Supports both:
- Training mode: Sample from policy distribution for exploration
- Evaluation mode: Use mean predictions for deterministic refinement

Integrates with LossState for hierarchical weight management.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path

from torchref.refinement.weighting.component_weighting import WeightingScheme
from torchref.utils.stats import stat, VERBOSITY_ESSENTIAL, VERBOSITY_STANDARD, VERBOSITY_DEBUG

if TYPE_CHECKING:
    from torchref.refinement.loss_state import LossState


# Component names (must match policy network output order)
COMPONENTS = [
    'xray', 'bond', 'angle', 'torsion', 'planarity',
    'chiral', 'nonbonded', 'simu', 'locality', 'KL'
]

# Mapping from policy component names to LossState hierarchical names
COMPONENT_TO_LOSS_STATE = {
    'xray': 'xray',
    'bond': 'geometry/bond',
    'angle': 'geometry/angle',
    'torsion': 'geometry/torsion',
    'planarity': 'geometry/planarity',
    'chiral': 'geometry/chiral',
    'nonbonded': 'geometry/nonbonded',
    'simu': 'adp/simu',
    'locality': 'adp/locality',
    'KL': 'adp/KL',
}

# Reverse mapping
LOSS_STATE_TO_COMPONENT = {v: k for k, v in COMPONENT_TO_LOSS_STATE.items()}


@dataclass
class StepState:
    """State representation at a single refinement step for trajectory recording.

    Matches the 31-feature format used in meta_weighting_4.
    """
    step: int
    progress: float  # step / n_steps (normalized)
    improvement_rate: float  # (initial_rfree - rfree) / initial_rfree
    rwork: float
    rfree: float
    rfree_gap: float  # rfree - rwork (overfitting indicator)
    delta_rfree: float
    xray_loss: float
    xray_loss_test: float
    xray_work_test_ratio: float  # xray_work / xray_test
    geometry_loss: float
    adp_loss: float
    bond_rmsd: float
    angle_rmsd: float
    mean_b_normalized: float  # mean_b / wilson_b
    b_std_normalized: float  # b_std / wilson_b
    component_losses: Dict[str, float] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class StepRecord:
    """Record for a single refinement step (state-action-reward tuple)."""
    state: StepState
    log_weights: Dict[str, float]  # Actions in log-space
    weights: Dict[str, float]       # Effective weights used
    reward: float                   # Per-step reward (-delta_rfree)


@dataclass
class TrajectoryData:
    """Complete trajectory data for training."""
    pdb_id: str
    structure_path: str
    sf_path: str
    steps: List[StepRecord] = field(default_factory=list)
    initial_rfree: float = 0.0
    final_rfree: float = 0.0
    initial_rwork: float = 0.0
    final_rwork: float = 0.0
    total_time: float = 0.0
    random_seed: Optional[int] = None
    policy_version: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class PolicyComponentWeighting(WeightingScheme):
    """
    Policy-based component weighting using a trained neural network.

    This weighting scheme uses a policy network to predict component weights
    from the current refinement state. Supports both training (sampling) and
    evaluation (deterministic) modes.

    Integrates with LossState for hierarchical weight management.

    Parameters
    ----------
    refinement : Refinement
        Reference to Refinement object
    policy_path : str, optional
        Path to trained policy checkpoint (.pt file). If None, uses random init.
    sample : bool, optional
        Whether to sample from policy distribution (default: False for eval)
    temperature : float, optional
        Sampling temperature for exploration (default: 1.0)
        - 0.0: deterministic (use mean predictions)
        - > 0.0: sample from distribution with scaled variance

    Attributes
    ----------
    policy : nn.Module
        Policy network (loaded or randomly initialized)
    sample : bool
        Whether to sample from predicted distribution
    temperature : float
        Sampling temperature for exploration
    current_step : int
        Current refinement step index
    last_log_weights : dict
        Most recent log-space weights (actions for training)
    last_weights : dict
        Most recent linear-space weights
    trajectory : TrajectoryData
        Current trajectory being recorded (if recording enabled)

    Examples
    --------
    >>> # Evaluation mode (deterministic)
    >>> policy = PolicyComponentWeighting(refinement, 'policy.pt', sample=False)
    >>>
    >>> # Training mode (sampling for exploration)
    >>> policy = PolicyComponentWeighting(refinement, 'policy.pt', sample=True, temperature=1.0)
    >>>
    >>> # Use with LossState
    >>> state = refinement.create_loss_state()
    >>> state = policy.apply_to_state(state)
    >>> loss = state.aggregate()
    """

    name = 'policy_weighting'

    def __init__(
        self,
        refinement,
        policy_path: Optional[str] = None,
        sample: bool = False,
        temperature: float = 1.0,
        n_steps: int = 10,
    ):
        super().__init__(refinement)

        # Sampling parameters
        self.sample = sample
        self.temperature = temperature
        self.n_steps = n_steps

        # Load or create policy network
        self.policy = self._create_policy_network()
        if policy_path is not None:
            self._load_policy(policy_path)
        self.policy.to(refinement.device)
        self.policy.eval()

        # State tracking
        self.current_step = 0
        self.prev_rfree = None
        self.initial_rfree = None

        # Last predictions (for trajectory recording)
        self.last_log_weights: Dict[str, float] = {}
        self.last_weights: Dict[str, float] = {}
        self.last_log_sigmas: Dict[str, float] = {}

        # Trajectory recording
        self.trajectory: Optional[TrajectoryData] = None
        self._recording = False

        # For backward compatibility
        self.predicted_weights = self.last_weights
        self.predicted_uncertainties = {}

        # Compute static features (computed once at initialization)
        self._compute_static_features()

        if refinement.verbose > 0:
            param_count = sum(p.numel() for p in self.policy.parameters())
            print(f"PolicyComponentWeighting initialized")
            print(f"  Policy: {policy_path or 'random initialization'}")
            print(f"  Parameters: {param_count:,}")
            print(f"  Sample mode: {sample}")
            print(f"  Temperature: {temperature}")
            print(f"  N steps: {n_steps}")
            print(f"  Static features: resolution={self.static_features['resolution_min']:.2f}Å, "
                  f"n_atoms={int(np.exp(self.static_features['log_n_atoms']))}, "
                  f"wilson_b={self.static_features['wilson_b']:.1f}")

    def _create_policy_network(self) -> nn.Module:
        """Create the policy network architecture.

        Uses 31-dimensional state vector matching meta_weighting_4 format.
        """
        class PolicyNetwork(nn.Module):
            def __init__(self, state_dim=31, hidden_dim=256):
                super().__init__()
                self.input_norm = nn.LayerNorm(state_dim)
                self.fc1 = nn.Linear(state_dim, hidden_dim)
                self.ln1 = nn.LayerNorm(hidden_dim)
                self.relu = nn.ReLU()
                self.fc_out = nn.Linear(hidden_dim, 20)  # 10 weights + 10 sigmas
                self.tanh = nn.Tanh()
                self.log_weight_scale = 3.0
                self.log_sigma_scale = 2.0

            def forward(self, x):
                x = self.input_norm(x)
                x = self.fc1(x)
                x = self.ln1(x)
                x = self.relu(x)
                x = self.fc_out(x)
                x = self.tanh(x)
                log_weights = x[:, :10] * self.log_weight_scale
                log_sigmas = x[:, 10:] * self.log_sigma_scale
                return log_weights, log_sigmas

        return PolicyNetwork()

    def _compute_static_features(self):
        """Compute static features from structure/data (computed once at init).

        Static features (7 total):
        - resolution_min: Best resolution in Angstroms
        - inv_resolution: 1/resolution (higher = better data)
        - log_n_atoms: Log of number of atoms (structure size)
        - log_n_hkl: Log of number of reflections
        - data_to_param_ratio: n_hkl / n_atoms
        - log_wilson_b: Log of Wilson B-factor (data quality)
        - b_cv: Coefficient of variation of B-factors
        - wilson_b: Wilson B (kept for normalization)
        """
        ref = self.refinement

        with torch.no_grad():
            # Structure properties
            n_atoms = len(ref.model.pdb)
            n_hkl = ref.reflection_data.hkl.shape[0]
            resolution_min = float(ref.reflection_data.resolution.min())
            wilson_b = float(ref.reflection_data.wilson_b)

            # Initial B-factor statistics
            b_factors = ref.model.b()
            b_mean = float(b_factors.mean())
            b_std = float(b_factors.std())

        self.static_features = {
            'resolution_min': resolution_min,
            'inv_resolution': 1.0 / resolution_min,
            'log_n_atoms': np.log(n_atoms),
            'log_n_hkl': np.log(n_hkl),
            'data_to_param_ratio': n_hkl / n_atoms,
            'log_wilson_b': np.log(wilson_b),
            'b_cv': b_std / (b_mean + 1e-6),  # Coefficient of variation
            'wilson_b': wilson_b,  # Keep for B-factor normalization
        }

    def _load_policy(self, checkpoint_path: str):
        """Load policy weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'policy_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            self.policy.load_state_dict(checkpoint)
        self.policy.eval()

    def set_training_mode(self, sample: bool = True, temperature: float = 1.0):
        """Enable training mode (sampling from policy distribution)."""
        self.sample = sample
        self.temperature = temperature

    def set_eval_mode(self):
        """Enable evaluation mode (deterministic predictions)."""
        self.sample = False

    def reset_trajectory(self, pdb_id: str = "", structure_path: str = "", sf_path: str = ""):
        """Reset for a new trajectory and optionally start recording."""
        self.current_step = 0
        self.prev_rfree = None
        self.initial_rfree = None
        self.last_log_weights = {}
        self.last_weights = {}

        # Capture initial R-free for improvement_rate calculation
        with torch.no_grad():
            _, rfree = self.refinement.get_rfactor()
            self.initial_rfree = float(rfree)

    def start_recording(self, pdb_id: str, structure_path: str, sf_path: str,
                       seed: Optional[int] = None, policy_version: Optional[str] = None):
        """Start recording a new trajectory."""
        self.reset_trajectory()
        self.trajectory = TrajectoryData(
            pdb_id=pdb_id,
            structure_path=structure_path,
            sf_path=sf_path,
            random_seed=seed,
            policy_version=policy_version,
        )
        self._recording = True

        # Record initial R-factors
        with torch.no_grad():
            rwork, rfree = self.refinement.get_rfactor()
            self.trajectory.initial_rwork = float(rwork)
            self.trajectory.initial_rfree = float(rfree)

    def stop_recording(self) -> Optional[TrajectoryData]:
        """Stop recording and return the trajectory data."""
        if not self._recording or self.trajectory is None:
            return None

        # Record final R-factors
        with torch.no_grad():
            rwork, rfree = self.refinement.get_rfactor()
            self.trajectory.final_rwork = float(rwork)
            self.trajectory.final_rfree = float(rfree)

        self._recording = False
        trajectory = self.trajectory
        self.trajectory = None
        return trajectory

    def increment_step(self):
        """Increment the step counter after a refinement step."""
        self.current_step += 1

    def extract_state_features(self, state: 'LossState' = None) -> Tuple[torch.Tensor, StepState]:
        """
        Extract 31-dimensional state features from current refinement state.

        Parameters
        ----------
        state : LossState, optional
            If provided, extract losses from the LossState cache.

        Returns
        -------
        features : torch.Tensor, shape (31,)
            State feature vector for policy input
        step_state : StepState
            State object for trajectory recording

        Features extracted (31 total):
        - Progress metrics (2): progress, improvement_rate
        - R-factors (4): rwork, rfree, gap, delta_rfree
        - Static structure/data features (7): resolution, inv_resolution, log_n_atoms,
          log_n_hkl, data_to_param_ratio, log_wilson_b, b_cv
        - X-ray losses (3): work, test, work/test ratio
        - Geometry losses (7): total, bond, angle, torsion, planarity, chiral, nonbonded
        - Geometry RMSD (2): bond_rmsd, angle_rmsd
        - ADP losses (4): total, simu, locality, KL
        - B-factor stats (2): mean_b/wilson_b, b_std/wilson_b
        """
        ref = self.refinement

        # Get current R-factors
        with torch.no_grad():
            rwork, rfree = ref.get_rfactor()
        rwork = float(rwork)
        rfree = float(rfree)
        rfree_gap = rfree - rwork

        # Progress metrics
        progress = self.current_step / self.n_steps
        if self.initial_rfree is not None and self.initial_rfree > 1e-6:
            improvement_rate = (self.initial_rfree - rfree) / self.initial_rfree
        else:
            improvement_rate = 0.0

        # Compute delta from previous step (0 for first step)
        if self.prev_rfree is not None:
            delta_rfree = rfree - self.prev_rfree
        else:
            delta_rfree = 0.0

        # Get losses - prefer from LossState if available
        component_losses = {}
        if state is not None and state._losses:
            # Extract from LossState cache
            xray_loss_work = state.get_loss('xray/work')
            xray_loss_work = xray_loss_work.item() if xray_loss_work is not None else 0.0

            # Component losses
            for comp in COMPONENTS[1:]:  # Skip 'xray'
                loss_state_name = COMPONENT_TO_LOSS_STATE[comp]
                loss = state.get_loss(loss_state_name)
                component_losses[comp] = loss.item() if loss is not None else 0.0

            geom_loss_total = sum(component_losses.get(c, 0.0) for c in ['bond', 'angle', 'torsion', 'planarity', 'chiral', 'nonbonded'])
            adp_loss_total = sum(component_losses.get(c, 0.0) for c in ['simu', 'locality', 'KL'])
        else:
            # Compute fresh
            with torch.no_grad():
                xray_loss_work = ref.xray_target_work().item() if hasattr(ref, 'xray_target_work') else 0.0

                # Try to get component losses
                if hasattr(ref, 'geometry_target'):
                    geom_losses = ref.geometry_target.target_losses()
                    for name, loss in geom_losses.items():
                        component_losses[name] = loss.item()
                    geom_loss_total = sum(loss.item() for loss in geom_losses.values())
                else:
                    geom_loss_total = 0.0

                if hasattr(ref, 'adp_target'):
                    adp_losses = ref.adp_target.target_losses()
                    for name, loss in adp_losses.items():
                        component_losses[name] = loss.item()
                    adp_loss_total = sum(loss.item() for loss in adp_losses.values())
                else:
                    adp_loss_total = 0.0

        # X-ray test loss
        with torch.no_grad():
            xray_loss_test = ref.xray_target_test().item() if hasattr(ref, 'xray_target_test') else xray_loss_work

        # X-ray work/test ratio
        xray_work_test_ratio = xray_loss_work / (xray_loss_test + 1e-6)

        # Geometry metrics
        with torch.no_grad():
            if hasattr(ref, 'restraints') and hasattr(ref.restraints, 'bond_deviations'):
                bond_devs, _ = ref.restraints.bond_deviations()
                bond_rmsd = torch.sqrt((bond_devs ** 2).mean()).item()
            else:
                bond_rmsd = 0.0

            if hasattr(ref, 'restraints') and hasattr(ref.restraints, 'angle_deviations'):
                angle_devs, _ = ref.restraints.angle_deviations()
                angle_rmsd = torch.sqrt((angle_devs ** 2).mean()).item()
            else:
                angle_rmsd = 0.0

            # B-factor statistics (dynamic)
            b_factors = ref.model.b()
            mean_b = float(b_factors.mean())
            b_std = float(b_factors.std())

        # Normalize B-factor stats by Wilson B
        wilson_b = self.static_features['wilson_b']
        mean_b_normalized = mean_b / wilson_b
        b_std_normalized = b_std / wilson_b

        # Build 31-feature tensor matching meta_weighting_4/worker.py
        features = torch.tensor([
            # Progress (2)
            progress,                                           # 0: normalized progress
            improvement_rate,                                   # 1: relative improvement

            # R-factors (4)
            rwork,                                              # 2
            rfree,                                              # 3
            rfree_gap,                                          # 4: gap (overfitting indicator)
            delta_rfree,                                        # 5: recent change

            # Structure/Data properties (7) - STATIC
            self.static_features['resolution_min'],             # 6: best resolution (Å)
            self.static_features['inv_resolution'],             # 7: inverted (higher = better data)
            self.static_features['log_n_atoms'],                # 8: structure size (log-scaled)
            self.static_features['log_n_hkl'],                  # 9: number of reflections (log-scaled)
            self.static_features['data_to_param_ratio'],        # 10: data-to-parameter ratio
            self.static_features['log_wilson_b'],               # 11: data quality (log-scaled)
            self.static_features['b_cv'],                       # 12: disorder coefficient of variation

            # X-ray losses (3) - RAW
            xray_loss_work,                                     # 13
            xray_loss_test,                                     # 14
            xray_work_test_ratio,                               # 15: work/test ratio

            # Geometry losses (7) - RAW
            geom_loss_total,                                    # 16
            component_losses.get('bond', 0.0),                  # 17
            component_losses.get('angle', 0.0),                 # 18
            component_losses.get('torsion', 0.0),               # 19
            component_losses.get('planarity', 0.0),             # 20
            component_losses.get('chiral', 0.0),                # 21
            component_losses.get('nonbonded', 0.0),             # 22

            # Geometry RMSD (2)
            bond_rmsd,                                          # 23
            angle_rmsd,                                         # 24

            # ADP losses (4) - RAW
            adp_loss_total,                                     # 25
            component_losses.get('simu', 0.0),                  # 26
            component_losses.get('locality', 0.0),              # 27
            component_losses.get('KL', 0.0),                    # 28

            # B-factor stats (2) - normalized by Wilson B
            mean_b_normalized,                                  # 29: normalized mean B
            b_std_normalized,                                   # 30: normalized B spread
        ], dtype=torch.float32, device=ref.device)

        # Create StepState for recording
        step_state = StepState(
            step=self.current_step,
            progress=progress,
            improvement_rate=improvement_rate,
            rwork=rwork,
            rfree=rfree,
            rfree_gap=rfree_gap,
            delta_rfree=delta_rfree,
            xray_loss=xray_loss_work,
            xray_loss_test=xray_loss_test,
            xray_work_test_ratio=xray_work_test_ratio,
            geometry_loss=geom_loss_total,
            adp_loss=adp_loss_total,
            bond_rmsd=bond_rmsd,
            angle_rmsd=angle_rmsd,
            mean_b_normalized=mean_b_normalized,
            b_std_normalized=b_std_normalized,
            component_losses=component_losses.copy(),
            component_weights={},  # Will be filled by forward()
        )

        # Update previous R-free for delta computation
        self.prev_rfree = rfree

        return features, step_state

    def forward(self, state: 'LossState' = None) -> Dict[str, float]:
        """
        Predict component weights from current refinement state.

        Parameters
        ----------
        state : LossState, optional
            If provided, extract features from LossState cache.

        Returns
        -------
        dict
            Dictionary of {loss_state_name: weight} for all components
        """
        # Extract state features
        features, step_state = self.extract_state_features(state)
        features = features.unsqueeze(0)  # Add batch dimension

        # Get policy predictions
        with torch.no_grad():
            log_weights, log_sigmas = self.policy(features)

        log_weights = log_weights[0]  # Remove batch dimension
        log_sigmas = log_sigmas[0]

        # Store log-weights for trajectory recording
        self.last_log_weights = {
            comp: log_weights[i].item()
            for i, comp in enumerate(COMPONENTS)
        }
        self.last_log_sigmas = {
            comp: log_sigmas[i].item()
            for i, comp in enumerate(COMPONENTS)
        }

        # Sample or use mean
        if self.sample and self.temperature > 0:
            # Sample from N(log_weight, temperature * exp(log_sigma))
            sigmas = torch.exp(log_sigmas) * self.temperature
            sampled_log_weights = torch.normal(log_weights, sigmas)
            weights_tensor = torch.exp(sampled_log_weights)
        else:
            # Deterministic: use mean
            weights_tensor = torch.exp(log_weights)

        # Clip to reasonable range
        weights_tensor = torch.clamp(weights_tensor, 0.01, 100.0)

        # Build output dictionaries
        self.last_weights = {
            comp: weights_tensor[i].item()
            for i, comp in enumerate(COMPONENTS)
        }

        # Return with LossState naming convention
        weights = {}
        for comp in COMPONENTS:
            loss_state_name = COMPONENT_TO_LOSS_STATE[comp]
            weights[loss_state_name] = self.last_weights[comp]

        # Update step_state with weights for recording
        step_state.component_weights = self.last_weights.copy()

        # Record step if trajectory recording is enabled
        if self._recording and self.trajectory is not None:
            reward = -step_state.delta_rfree  # Negative delta = improvement = positive reward
            reward = max(-0.05, min(0.05, reward))  # Clip to [-0.05, 0.05]

            step_record = StepRecord(
                state=step_state,
                log_weights=self.last_log_weights.copy(),
                weights=self.last_weights.copy(),
                reward=reward,
            )
            self.trajectory.steps.append(step_record)

        # For backward compatibility
        self.predicted_weights = self.last_weights
        self.predicted_uncertainties = {
            comp: np.exp(self.last_log_sigmas[comp])
            for comp in COMPONENTS
        }

        return weights

    def apply_to_state(self, state: 'LossState') -> 'LossState':
        """
        Apply policy-predicted weights to a LossState.

        This is the main integration point with the LossState architecture.
        Call this after creating a LossState to apply policy weights.

        Parameters
        ----------
        state : LossState
            Loss state to update with policy weights.

        Returns
        -------
        LossState
            State with policy weights applied.

        Example
        -------
        >>> state = refinement.create_loss_state()
        >>> state = policy_weighting.apply_to_state(state)
        >>> loss = state.aggregate()
        """
        weights = self.forward(state)
        for name, weight in weights.items():
            state.set_weight(name, weight)
        return state

    def stats(self, state: 'LossState' = None) -> Dict[str, Any]:
        """
        Return statistics for reporting.

        Returns
        -------
        dict
            Statistics dictionary with predicted weights and uncertainties
        """
        stats_dict = {}

        # Predicted weights
        if self.last_weights:
            for comp, weight in self.last_weights.items():
                stats_dict[f'policy_weight_{comp}'] = stat(weight, VERBOSITY_STANDARD)

        # Predicted log-sigmas (uncertainties)
        if self.last_log_sigmas:
            for comp, log_sigma in self.last_log_sigmas.items():
                stats_dict[f'policy_sigma_{comp}'] = stat(np.exp(log_sigma), VERBOSITY_DEBUG)

        # Sampling parameters
        stats_dict['sample_mode'] = stat(self.sample, VERBOSITY_DEBUG)
        stats_dict['temperature'] = stat(self.temperature, VERBOSITY_DEBUG)
        stats_dict['current_step'] = stat(self.current_step, VERBOSITY_DEBUG)

        return stats_dict


def trajectory_to_dict(trajectory: TrajectoryData) -> Dict[str, Any]:
    """Convert TrajectoryData to a JSON-serializable dictionary."""
    return {
        'pdb_id': trajectory.pdb_id,
        'structure_path': trajectory.structure_path,
        'sf_path': trajectory.sf_path,
        'initial_rfree': trajectory.initial_rfree,
        'final_rfree': trajectory.final_rfree,
        'initial_rwork': trajectory.initial_rwork,
        'final_rwork': trajectory.final_rwork,
        'total_time': trajectory.total_time,
        'random_seed': trajectory.random_seed,
        'policy_version': trajectory.policy_version,
        'success': trajectory.success,
        'error_message': trajectory.error_message,
        'steps': [
            {
                'state': asdict(step.state),
                'log_weights': step.log_weights,
                'weights': step.weights,
                'reward': step.reward,
            }
            for step in trajectory.steps
        ]
    }


__all__ = [
    'PolicyComponentWeighting',
    'StepState',
    'StepRecord',
    'TrajectoryData',
    'trajectory_to_dict',
    'COMPONENTS',
    'COMPONENT_TO_LOSS_STATE',
    'LOSS_STATE_TO_COMPONENT',
]
