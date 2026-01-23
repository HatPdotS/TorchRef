"""

Combined targets for refinement (e.g., geometry + ADP).

Integrate multiple component targets into a single combined target
using nn.ModuleDict for clean organization and easy access.

Integrate into lossState via add_to_state"""

from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

from torchref.refinement.targets import targets
from torchref.utils.stats import (
    VERBOSITY_DETAILED,
    filter_stats,
)

if TYPE_CHECKING:
    from torchref.model.model import Model
    from torchref.refinement.base_refinement import Refinement


class CombinedTargets(targets.Target):
    """
    Base class for combined targets.

    Uses nn.ModuleDict to store component targets for clean organization
    and easy access via dictionary-style notation.

    Subclasses should override `_create_targets()` to define their component targets.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level. Default is 0.

    Attributes
    ----------
    _targets : nn.ModuleDict
        Dictionary of component targets.
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize CombinedTargets.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Default is 0.
        """
        super().__init__(verbose=verbose)
        self._targets = nn.ModuleDict(self._create_targets())

    def _create_targets(self) -> Dict[str, "targets.Target"]:
        """
        Create and return component targets as a dictionary.

        Subclasses must override this method to define their component targets.

        Returns
        -------
        Dict[str, Target]
            Dictionary mapping target names to Target instances.
        """
        raise NotImplementedError("Subclasses must implement _create_targets() method.")

    def targets(self) -> nn.ModuleDict:
        """Return registered sub-targets as ModuleDict."""
        return self._targets

    def __getitem__(self, key: str) -> "targets.Target":
        """Get a target by name using dictionary-style access."""
        return self._targets[key]

    def __contains__(self, key: str) -> bool:
        """Check if a target exists."""
        return key in self._targets

    def keys(self):
        """Return target names."""
        return self._targets.keys()

    def values(self):
        """Return target instances."""
        return self._targets.values()

    def items(self):
        """Return (name, target) pairs."""
        return self._targets.items()

    def target_losses(self) -> Dict[str, torch.Tensor]:
        """Get individual component losses (without weights)."""
        return {name: target() for name, target in self._targets.items()}

    def forward(self) -> torch.Tensor:
        """Compute total combined target loss."""
        losses = list(self.target_losses().values())
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).sum()

    def stats(self) -> Dict[str, any]:
        """Get statistics from all registered targets."""
        statistics = {}
        for name, target in self._targets.items():
            if hasattr(target, "stats"):
                target_stats = target.stats()
                if target_stats:
                    statistics[name] = target_stats
        return statistics

    def get(self) -> dict:
        """Get individual component losses."""
        return self.target_losses()

    def add_to_state(self, state):
        for name, target in self._targets.items():
            target.add_to_state(state)
        return state


class CombinedModelTargets(targets.ModelTarget):
    """
    Base class for combined targets that only need Model (geometry/ADP targets).

    Uses nn.ModuleDict to store component targets for clean organization
    and easy access via dictionary-style notation.

    Subclasses should override `_create_targets()` to define their component targets.

    Parameters
    ----------
    model : Model, optional
        Reference to Model object.
    verbose : int, optional
        Verbosity level. Default is 0.

    Attributes
    ----------
    _targets : nn.ModuleDict
        Dictionary of component targets.
    """

    def __init__(self, model: "Model" = None, verbose: int = 0):
        """
        Initialize CombinedModelTargets.

        Parameters
        ----------
        model : Model, optional
            Reference to Model object.
        verbose : int, optional
            Verbosity level. Default is 0.
        """
        super().__init__(model, verbose)
        self._targets = nn.ModuleDict(self._create_targets())

    def _create_targets(self) -> Dict[str, "targets.Target"]:
        """
        Create and return component targets as a dictionary.

        Subclasses must override this method to define their component targets.

        Returns
        -------
        Dict[str, Target]
            Dictionary mapping target names to Target instances.
        """
        raise NotImplementedError("Subclasses must implement _create_targets() method.")

    def targets(self) -> nn.ModuleDict:
        """Return registered sub-targets as ModuleDict."""
        return self._targets

    def __getitem__(self, key: str) -> "targets.Target":
        """Get a target by name using dictionary-style access."""
        return self._targets[key]

    def __contains__(self, key: str) -> bool:
        """Check if a target exists."""
        return key in self._targets

    def keys(self):
        """Return target names."""
        return self._targets.keys()

    def values(self):
        """Return target instances."""
        return self._targets.values()

    def items(self):
        """Return (name, target) pairs."""
        return self._targets.items()

    def target_losses(self) -> Dict[str, torch.Tensor]:
        """Get individual component losses (without weights)."""
        return {name: target() for name, target in self._targets.items()}

    def forward(self) -> torch.Tensor:
        """Compute total combined target loss."""
        losses = list(self.target_losses().values())
        if not losses:
            return torch.tensor(0.0, device=self.model.xyz().device if self.model else "cpu")
        return torch.stack(losses).sum()

    def stats(self) -> Dict[str, any]:
        """Get statistics from all registered targets."""
        statistics = {}
        for name, target in self._targets.items():
            if hasattr(target, "stats"):
                target_stats = target.stats()
                if target_stats:
                    statistics[name] = target_stats
        return statistics

    def get(self) -> dict:
        """Get individual component losses."""
        return self.target_losses()

    def add_to_state(self, state):
        for name, target in self._targets.items():
            target.add_to_state(state)
        return state


class TotalGeometryTarget(CombinedModelTargets):
    """
    Computes weighted sum of all geometry restraint NLLs.

    Uses nn.ModuleDict to store component targets:
    - 'bond': BondTarget
    - 'angle': AngleTarget
    - 'torsion': TorsionTarget
    - 'planarity': PlanarityTarget
    - 'chiral': ChiralTarget
    - 'nonbonded': NonBondedTarget

    The torsion weight is reduced because:

    1. Protein torsions naturally deviate from ideal (Ramachandran plot)
    2. Side chain rotamers have discrete populations, not single ideals
    3. High torsion weight can over-constrain the structure

    The nonbonded weight is very low because:

    1. PROLSQ repulsion is already steep (E ~ violation^4)
    2. Most contacts should be satisfied by covalent geometry
    3. High VDW weight can prevent proper packing

    Set weight to 0 to disable a component.

    Parameters
    ----------
    model : Model, optional
        Reference to Model object.
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    ::

        geom_target = TotalGeometryTarget(model)
        loss = geom_target()
        bond_loss = geom_target['bond']()
        for name, target in geom_target.items():
            print(f"{name}: {target()}")
    """

    def _create_targets(self) -> Dict[str, targets.Target]:
        """
        Create geometry component targets.

        Returns
        -------
        Dict[str, Target]
            Dictionary of geometry targets.
        """
        print("Initializing TotalGeometryTarget with component targets...")
        return {
            "bond": targets.BondTarget(self.model, self.verbose),
            "angle": targets.AngleTarget(self.model, self.verbose),
            "torsion": targets.TorsionTarget(self.model, self.verbose),
            "planarity": targets.PlanarityTarget(self.model, self.verbose),
            "chiral": targets.ChiralTarget(self.model, self.verbose),
            "nonbonded": targets.NonBondedTarget(self.model, verbose=self.verbose),
        }

    def get_metrics(self, verbosity: int = VERBOSITY_DETAILED) -> Dict[str, float]:
        """
        Get all geometry metrics as a flat dictionary for logging/reporting.

        Parameters
        ----------
        verbosity : int, optional
            Verbosity level for filtering. Default is VERBOSITY_DETAILED.

        Returns
        -------
        dict
            Dictionary with validation metrics from all component targets.
            All values are Python floats (not tensors).
        """
        metrics = {}

        # Total loss (always include)
        total_loss = self.forward()
        metrics["geom_total_loss"] = (
            total_loss.item() if torch.is_tensor(total_loss) else total_loss
        )

        # Get losses from target_losses()
        for name, loss in self.target_losses().items():
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            metrics[f"geom_{name}_loss"] = loss_val

        # Get statistics from parent stats() method and filter here
        filtered_stats = filter_stats(self.stats(), verbosity)
        for name, target_stats in filtered_stats.items():
            for stat_name, stat_val in target_stats.items():
                metrics[f"geom_{name}_{stat_name}"] = stat_val

        return metrics

    def print_statistics(self):
        """Print REFMAC-style geometry statistics with losses."""
        # Temporarily disable verbose to prevent duplicate output during loss calculation
        saved_verbose = self.verbose
        self.verbose = 0

        print("\n" + "=" * 90)
        print("Geometry Restraint Statistics (REFMAC-style)")
        print("=" * 90)

        # Show component targets
        print(f"Components: {', '.join(self._targets.keys())}")
        print("-" * 90)
        print(
            f"{'Restraint Type':<25} {'N':>8} {'RMS Delta':>12} {'RMS Z':>10} {'Av(Sigma)':>12} {'Loss':>12}"
        )
        print("-" * 90)

        # Get losses and stats using parent methods
        losses = self.target_losses()
        all_stats = self.stats()

        for name, loss in losses.items():
            try:
                loss_val = loss.item() if torch.is_tensor(loss) else loss
                stats = all_stats.get(name, {})

                # Format based on available stats
                if "n" in stats:
                    n = stats["n"]
                    rms_delta = stats.get("rms_delta", 0.0)
                    rms_z = stats.get("rms_z", 0.0)
                    mean_sigma = stats.get("mean_sigma", 0.0)
                    print(
                        f"{name:<25} {n:>8} {rms_delta:>12.4f} {rms_z:>10.2f} {mean_sigma:>12.4f} {loss_val:>12.4f}"
                    )
                elif "n_violations" in stats:  # NonBonded format
                    n = stats.get("n", 0)
                    n_viol = stats.get("n_violations", 0)
                    pct_viol = 100.0 * n_viol / n if n > 0 else 0.0
                    print(
                        f"{name:<25} {n:>8} pairs, {n_viol:>6} violations ({pct_viol:>5.1f}%)"
                    )
                    print(
                        f"{'  RMS violation (Å)':<25} {stats.get('rms_violation', 0.0):>12.4f}   Max: {stats.get('max_violation', 0.0):.4f} Å"
                    )
                    print(f"{'  Loss':<25} {loss_val:>12.4f}")
                else:
                    print(
                        f"{name:<25} {'':>8} {'':>12} {'':>10} {'':>12} {loss_val:>12.4f}"
                    )
            except Exception:
                pass

        # Total loss
        print("-" * 90)
        total_loss = self.forward().item()

        # Restore verbose now
        self.verbose = saved_verbose

        print(
            f"{'TOTAL GEOMETRY LOSS':<25} {'':>8} {'':>12} {'':>10} {'':>12} {total_loss:>12.4f}"
        )

        print("=" * 90)
        print("Target: RMS Z should be ~1.0 for well-refined structure")
        print("        Phenix typical: Bond RMS ~0.007Å, Angle RMS ~1.2°")
        print("=" * 90 + "\n")


class TotalADPTarget(CombinedModelTargets):
    """
    Total ADP restraint target combining global, similarity, and local components.

    Uses nn.ModuleDict to store component targets:
    - 'simu': ADPSimilarityTarget (SIMU-like bond similarity)
    - 'locality': ADPLocalityTarget (spatial smoothness)
    - 'KL': ADPEntropyTarget (KL divergence regularization)

    B-factors follow a LOG-NORMAL distribution (B > 0, right-skewed).
    If B ~ LogNormal(μ, σ), then log(B) ~ Normal(μ, σ).

    This target combines:

    1. **Similarity restraint (SIMU-like)**: Bond-based B-factor similarity
       - Enforces bonded atoms have similar B-factors
       - Based on covalent bond topology (strongest local constraint)

    2. **Locality restraint**: Spatial smoothness - nearby atoms should have similar B
       - Uses K-NN with distance-based sigma (d² scaling)
       - Medium-range spatial correlation

    3. **KL divergence**: Controls the spread of B-factor distribution
       - Prevents overfitting by controlling distribution width

    Log-normal distribution properties:

    - If log(B) ~ N(μ, σ), then:
    - Mean of B: exp(μ + σ²/2)
    - Mode of B: exp(μ - σ²)
    - For typical proteins: σ_logB ≈ 0.3-0.5 (in log space)

    Parameters
    ----------
    model : Model
        Reference to Model object.
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    ::

        adp_target = TotalADPTarget(model)
        loss = adp_target()
        simu_loss = adp_target['simu']()
        for name, target in adp_target.items():
            print(f"{name}: {target()}")
    """

    def _create_targets(self) -> Dict[str, targets.Target]:
        """
        Create ADP component targets.

        Returns
        -------
        Dict[str, Target]
            Dictionary of ADP targets.
        """
        print("Initializing TotalADPTarget with component targets...")
        return {
            "simu": targets.ADPSimilarityTarget(self.model, verbose=self.verbose),
            "locality": targets.ADPLocalityTarget(
                self.model, verbose=self.verbose
            ),
            "KL": targets.ADPEntropyTarget(self.model, verbose=self.verbose),
        }

    def print_statistics(self) -> None:
        """
        Print comprehensive ADP restraint statistics.

        Displays statistics from all registered ADP targets.
        """
        print("\n" + "=" * 90)
        print("ADP RESTRAINT STATISTICS")
        print("=" * 90)

        # Component losses and statistics
        print(f"\n{'COMPONENT LOSSES':^90}")
        print("-" * 90)
        print(f"{'Component':<25} {'Loss':>15}")
        print("-" * 90)

        # Get losses and stats using parent methods
        losses = self.target_losses()
        all_stats = self.stats()

        for name, loss in losses.items():
            try:
                loss_val = loss.item() if torch.is_tensor(loss) else loss
                print(f"{name:<25} {loss_val:>15.4f}")

                # Print detailed stats from parent stats() method
                stats = all_stats.get(name, {})
                for stat_name, stat_val in stats.items():
                    if isinstance(stat_val, float):
                        print(f"  {stat_name:<23} {stat_val:>15.4f}")
                    else:
                        print(f"  {stat_name:<23} {stat_val:>15}")
            except Exception as e:
                print(f"{name:<25} Error: {e}")

        print("-" * 90)
        total_loss = self.forward().item()
        print(f"{'TOTAL ADP LOSS':<25} {total_loss:>15.4f}")

        print("=" * 90 + "\n")

    def get_metrics(self, verbosity: int = VERBOSITY_DETAILED) -> Dict[str, float]:
        """
        Get all ADP metrics as a flat dictionary for logging/reporting.

        Parameters
        ----------
        verbosity : int, optional
            Verbosity level for filtering. Default is VERBOSITY_DETAILED.

        Returns
        -------
        dict
            Dictionary with validation metrics from all component targets.
            All values are Python floats (not tensors).
        """
        metrics = {}

        # Total loss (always include)
        total_loss = self.forward()
        metrics["adp_total_loss"] = (
            total_loss.item() if torch.is_tensor(total_loss) else total_loss
        )

        # Get losses from target_losses()
        for name, loss in self.target_losses().items():
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            metrics[f"adp_{name}_loss"] = loss_val

        # Get statistics from parent stats() method (filter here)
        filtered_stats = filter_stats(self.stats(), verbosity)
        for name, target_stats in filtered_stats.items():
            for stat_name, stat_val in target_stats.items():
                metrics[f"adp_{name}_{stat_name}"] = stat_val

        return metrics
