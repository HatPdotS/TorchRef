"""
Evolution Strategy (ES) based policy training for weight prediction.

This approach:
1. Starts from a reasonably initialized policy
2. Spawns perturbations of the policy
3. Evaluates perturbations on structures
4. Scores by population-normalized final Rfree
5. Updates policy by advantage-weighted parameter averaging

Advantages over AWR:
- Directly optimizes final Rfree (not per-step delta)
- Population normalization controls for structure difficulty
- Simpler, more robust to noise in the signal
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ESConfig:
    """Configuration for ES training."""

    population_size: int = 20  # Number of perturbations per structure
    sigma: float = 0.1  # Perturbation standard deviation
    learning_rate: float = 0.01  # Update step size
    n_structures_per_gen: int = 10  # Structures evaluated per generation
    n_steps: int = 10  # Refinement steps per trajectory
    normalize_returns: bool = True  # Normalize within structure
    antithetic: bool = True  # Use mirrored perturbations (more stable)


@dataclass
class ESResult:
    """Results from one ES evaluation."""

    pdb_id: str
    perturbation_idx: int
    epsilon: np.ndarray  # The noise vector used
    final_rfree: float
    initial_rfree: float
    delta_rfree: float
    success: bool
    error: Optional[str] = None


@dataclass
class GenerationResult:
    """Results from one generation of ES."""

    generation: int
    results: List[ESResult]
    mean_final_rfree: float
    best_final_rfree: float
    gradient_norm: float
    param_update_norm: float


class ESPolicyTrainer:
    """
    Evolution Strategy trainer for weight prediction policy.

    Uses Natural Evolution Strategies (NES) with population-normalized
    fitness to optimize a policy network for weight prediction.

    Parameters
    ----------
    policy : nn.Module
        Policy network to optimize (outputs log-weights)
    config : ESConfig
        Training configuration
    device : str
        Computation device

    Example
    -------
    ::

        policy = create_policy_network()
        trainer = ESPolicyTrainer(policy, ESConfig())

        # Run one generation
        results = trainer.evaluate_generation(structures, run_trajectory_fn)
        trainer.update(results)
    """

    def __init__(
        self,
        policy: nn.Module,
        config: ESConfig = None,
        device: str = "cpu",
    ):
        self.policy = policy
        self.config = config or ESConfig()
        self.device = torch.device(device)
        self.policy.to(self.device)

        # Get total number of parameters
        self.n_params = sum(p.numel() for p in policy.parameters())

        # Flatten parameters for easier manipulation
        self._param_shapes = [p.shape for p in policy.parameters()]

        # Training history
        self.generation = 0
        self.history: List[GenerationResult] = []

    def get_flat_params(self) -> np.ndarray:
        """Get flattened policy parameters."""
        return np.concatenate(
            [p.detach().cpu().numpy().flatten() for p in self.policy.parameters()]
        )

    def set_flat_params(self, flat_params: np.ndarray):
        """Set policy parameters from flattened array."""
        offset = 0
        for p, shape in zip(self.policy.parameters(), self._param_shapes):
            size = np.prod(shape)
            p.data.copy_(
                torch.tensor(
                    flat_params[offset : offset + size].reshape(shape),
                    dtype=p.dtype,
                    device=p.device,
                )
            )
            offset += size

    def sample_perturbations(self) -> List[np.ndarray]:
        """
        Sample perturbation vectors.

        If antithetic=True, returns mirrored pairs for variance reduction.
        """
        n = self.config.population_size

        if self.config.antithetic:
            # Sample half, mirror to get full population
            half_n = n // 2
            epsilons = [
                np.random.randn(self.n_params) * self.config.sigma
                for _ in range(half_n)
            ]
            # Add mirrored versions
            epsilons = epsilons + [-e for e in epsilons]
        else:
            epsilons = [
                np.random.randn(self.n_params) * self.config.sigma for _ in range(n)
            ]

        return epsilons

    def create_perturbed_policy(self, epsilon: np.ndarray) -> nn.Module:
        """Create a copy of policy with perturbed parameters."""
        import copy

        perturbed = copy.deepcopy(self.policy)

        base_params = self.get_flat_params()
        perturbed_params = base_params + epsilon

        # Set parameters on the copy
        offset = 0
        for p, shape in zip(perturbed.parameters(), self._param_shapes):
            size = np.prod(shape)
            p.data.copy_(
                torch.tensor(
                    perturbed_params[offset : offset + size].reshape(shape),
                    dtype=p.dtype,
                    device=self.device,
                )
            )
            offset += size

        return perturbed

    def compute_advantages(self, results: List[ESResult]) -> Dict[int, float]:
        """
        Compute advantages for each perturbation.

        Normalizes within each structure to remove structure difficulty effect.
        Returns {perturbation_idx: advantage}
        """
        # Group by structure
        by_structure: Dict[str, List[ESResult]] = {}
        for r in results:
            if r.success:
                by_structure.setdefault(r.pdb_id, []).append(r)

        advantages = {}

        for pdb_id, struct_results in by_structure.items():
            # Get final Rfree values
            rfrees = np.array([r.final_rfree for r in struct_results])

            if self.config.normalize_returns:
                # Normalize within structure: z-score
                mean_rfree = rfrees.mean()
                std_rfree = rfrees.std()
                if std_rfree > 1e-8:
                    normalized = (rfrees - mean_rfree) / std_rfree
                else:
                    normalized = np.zeros_like(rfrees)

                # Lower Rfree = higher advantage (negate)
                struct_advantages = -normalized
            else:
                # Just negate: lower Rfree = higher advantage
                struct_advantages = -rfrees

            # Assign to perturbation indices
            for r, adv in zip(struct_results, struct_advantages):
                advantages[r.perturbation_idx] = adv

        return advantages

    def compute_gradient(
        self, epsilons: List[np.ndarray], advantages: Dict[int, float]
    ) -> np.ndarray:
        """
        Compute ES gradient estimate.

        gradient = (1/n*sigma^2) * sum(advantage_i * epsilon_i)
        """
        n = len(epsilons)
        sigma = self.config.sigma

        gradient = np.zeros(self.n_params)

        for idx, eps in enumerate(epsilons):
            if idx in advantages:
                gradient += advantages[idx] * eps

        # Normalize by population size and sigma squared
        gradient /= n * sigma**2

        return gradient

    def update(self, generation_result: GenerationResult):
        """Apply gradient update to policy."""
        # Already computed in evaluate_generation
        self.generation += 1
        self.history.append(generation_result)

    def evaluate_generation(
        self,
        structures: List[
            Dict[str, str]
        ],  # [{'pdb': path, 'mtz': path, 'pdb_id': id}, ...]
        run_trajectory_fn,  # callable(policy, pdb, mtz, n_steps) -> (final_rfree, initial_rfree, success, error)
    ) -> GenerationResult:
        """
        Evaluate one generation of ES.

        Parameters
        ----------
        structures : list
            List of structure dicts with 'pdb', 'mtz', 'pdb_id' keys
        run_trajectory_fn : callable
            Function to run a trajectory with a given policy

        Returns
        -------
        GenerationResult
            Results including gradient update
        """
        epsilons = self.sample_perturbations()
        results = []

        for struct in structures:
            pdb_id = struct.get("pdb_id", Path(struct["pdb"]).stem)

            for idx, epsilon in enumerate(epsilons):
                # Create perturbed policy
                perturbed_policy = self.create_perturbed_policy(epsilon)

                try:
                    final_rfree, initial_rfree, success, error = run_trajectory_fn(
                        perturbed_policy,
                        struct["pdb"],
                        struct["mtz"],
                        self.config.n_steps,
                    )

                    results.append(
                        ESResult(
                            pdb_id=pdb_id,
                            perturbation_idx=idx,
                            epsilon=epsilon,
                            final_rfree=final_rfree if success else float("inf"),
                            initial_rfree=initial_rfree,
                            delta_rfree=final_rfree - initial_rfree if success else 0,
                            success=success,
                            error=error,
                        )
                    )
                except Exception as e:
                    results.append(
                        ESResult(
                            pdb_id=pdb_id,
                            perturbation_idx=idx,
                            epsilon=epsilon,
                            final_rfree=float("inf"),
                            initial_rfree=0,
                            delta_rfree=0,
                            success=False,
                            error=str(e),
                        )
                    )

        # Compute advantages and gradient
        advantages = self.compute_advantages(results)
        gradient = self.compute_gradient(epsilons, advantages)

        # Update parameters
        current_params = self.get_flat_params()
        new_params = current_params + self.config.learning_rate * gradient
        self.set_flat_params(new_params)

        # Compute statistics
        successful = [r for r in results if r.success]
        mean_rfree = (
            np.mean([r.final_rfree for r in successful]) if successful else float("inf")
        )
        best_rfree = (
            min([r.final_rfree for r in successful]) if successful else float("inf")
        )

        gen_result = GenerationResult(
            generation=self.generation,
            results=results,
            mean_final_rfree=mean_rfree,
            best_final_rfree=best_rfree,
            gradient_norm=np.linalg.norm(gradient),
            param_update_norm=np.linalg.norm(self.config.learning_rate * gradient),
        )

        self.update(gen_result)
        return gen_result

    def save_checkpoint(self, path: str, state_dim: int = 31, hidden_dim: int = 256):
        """Save training checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "config": self.config.__dict__,
            "generation": self.generation,
            "n_params": self.n_params,
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
            "history": [
                {
                    "generation": h.generation,
                    "mean_final_rfree": h.mean_final_rfree,
                    "best_final_rfree": h.best_final_rfree,
                    "gradient_norm": h.gradient_norm,
                }
                for h in self.history
            ],
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.generation = checkpoint.get("generation", 0)
        # Config and history are informational


def create_policy_network(state_dim: int = 31, hidden_dim: int = 256) -> nn.Module:
    """Create the standard policy network architecture.

    Uses 31-dimensional state vector matching meta_weighting_4 format:
    - Progress (2): progress, improvement_rate
    - R-factors (4): rwork, rfree, gap, delta_rfree
    - Static features (7): resolution_min, inv_resolution, log_n_atoms,
      log_n_hkl, data_to_param_ratio, log_wilson_b, b_cv
    - X-ray losses (3): work, test, work/test ratio
    - Geometry losses (7): total, bond, angle, torsion, planarity, chiral, nonbonded
    - Geometry RMSD (2): bond_rmsd, angle_rmsd
    - ADP losses (4): total, simu, locality, KL
    - B-factor stats (2): mean_b/wilson_b, b_std/wilson_b
    """

    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(state_dim)
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.relu = nn.ReLU()
            self.fc_out = nn.Linear(hidden_dim, 20)  # 10 log-weights + 10 log-sigmas
            self.tanh = nn.Tanh()
            self.log_weight_scale = 3.0
            self.log_sigma_scale = 2.0

        def forward(self, x):
            x = self.input_norm(x)
            x = self.relu(self.ln1(self.fc1(x)))
            x = self.tanh(self.fc_out(x))
            log_weights = x[:, :10] * self.log_weight_scale
            log_sigmas = x[:, 10:] * self.log_sigma_scale
            return log_weights, log_sigmas

    return PolicyNetwork()


def initialize_policy_sensibly(state_dim: int = 31, hidden_dim: int = 256) -> nn.Module:
    """
    Initialize policy with sensible default weights.

    Uses 31-dimensional state vector matching meta_weighting_4 format.

    Based on correlation analysis:
    - Higher xray weight is better
    - Lower torsion/KL weights are better
    """
    policy = create_policy_network(state_dim=state_dim, hidden_dim=hidden_dim)

    # The output layer maps tanh output ([-1, 1]) to log-weights ([-3, 3])
    # Initialize bias to produce reasonable default weights:
    # xray: high (~5-10) -> log = 1.6-2.3 -> tanh input = 0.5-0.8
    # torsion, KL: low (~0.3) -> log = -1.2 -> tanh input = -0.4
    # others: medium (~1-3) -> log = 0-1 -> tanh input = 0-0.3

    with torch.no_grad():
        # Set output bias for log-weights (first 10 outputs)
        default_log_weights = torch.tensor(
            [
                1.5,  # xray: high
                0.5,  # bond: medium
                0.0,  # angle: medium
                -0.5,  # torsion: low
                0.0,  # planarity: medium
                0.0,  # chiral: medium
                0.0,  # nonbonded: medium
                0.5,  # simu: medium
                0.0,  # locality: medium
                -0.5,  # KL: low
            ]
        )

        # Convert to tanh input space
        tanh_targets = default_log_weights / 3.0  # Divide by log_weight_scale

        # Set bias (assuming weights are near zero initially)
        policy.fc_out.bias.data[:10] = torch.atanh(tanh_targets.clamp(-0.95, 0.95))

        # Set sigma biases to produce moderate uncertainty
        policy.fc_out.bias.data[10:] = 0.0  # log_sigma = 0 -> sigma = 1

    return policy
