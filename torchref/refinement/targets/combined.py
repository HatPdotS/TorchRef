"""
Combined targets for crystallographic refinement (e.g., geometry + ADP).

This module provides combined target classes that group several component
targets into a single target using ``nn.ModuleDict`` for clean organization
and dictionary-style access. Combined targets register their components with a
``LossState`` via :meth:`add_to_state`.
"""

from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

from torchref.config import get_default_device
from torchref.refinement.targets.base import Target, ModelTarget
from torchref.refinement.targets.geometry import (
    BondTarget, AngleTarget, TorsionTarget, PlanarityTarget,
    ChiralTarget, NonBondedHTarget, RamachandranTarget,
)
from torchref.refinement.targets.adp import (
    ADPSimilarityTarget, ADPLocalityTarget, ADPEntropyTarget,
)
from torchref.utils.stats import (
    VERBOSITY_DETAILED,
    filter_stats,
)

if TYPE_CHECKING:
    from torchref.model.model import Model
    from torchref.refinement.base_refinement import Refinement


class CombinedTargets(Target):
    """Base class for combined targets, summing components held in a ModuleDict.

    Subclasses override :meth:`_create_targets`. Components are reachable by
    name (``self['bond']``) and through ``keys``/``values``/``items``.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level. Default is 0.
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

    def _create_targets(self) -> Dict[str, "Target"]:
        """Build the ``{name: Target}`` components. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _create_targets() method.")

    def targets(self) -> nn.ModuleDict:
        """Return registered sub-targets as ModuleDict."""
        return self._targets

    def __getitem__(self, key: str) -> "Target":
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
        """Add each component's loss to ``state`` under its own name, not one total."""
        for name, target in self._targets.items():
            target.add_to_state(state)
        return state


class CombinedModelTargets(ModelTarget):
    """Combined-target base for components that need only a Model.

    The :class:`CombinedTargets` counterpart for geometry and ADP restraints;
    subclasses override :meth:`_create_targets`. Not listed in
    ``targets/__init__.__all__``, unlike ``CombinedTargets``.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
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

    def _create_targets(self) -> Dict[str, "Target"]:
        """Build the ``{name: Target}`` components. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _create_targets() method.")

    def targets(self) -> nn.ModuleDict:
        """Return registered sub-targets as ModuleDict."""
        return self._targets

    def __getitem__(self, key: str) -> "Target":
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
            return torch.tensor(
                0.0,
                device=self.model.xyz().device if self.model else get_default_device(),
            )
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
        """Add each component's loss to ``state`` under its own name, not one total."""
        for name, target in self._targets.items():
            target.add_to_state(state)
        return state


class TotalGeometryTarget(CombinedModelTargets):
    """Sum of every geometry restraint NLL.

    Components, keyed for individual access (``target['bond']()``): 'bond',
    'angle', 'torsion', 'planarity', 'chiral', 'nonbonded' (a
    ``NonBondedHTarget``, so riding-hydrogen VDW is included) and
    'ramachandran'. Set a component's weight to 0 to disable it.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    def _create_targets(self) -> Dict[str, Target]:
        """Build the seven geometry component targets."""
        print("Initializing TotalGeometryTarget with component targets...")
        return {
            "bond": BondTarget(self.model, self.verbose),
            "angle": AngleTarget(self.model, self.verbose),
            "torsion": TorsionTarget(self.model, self.verbose),
            "planarity": PlanarityTarget(self.model, self.verbose),
            "chiral": ChiralTarget(self.model, self.verbose),
            "nonbonded": NonBondedHTarget(self.model, verbose=self.verbose),
            "ramachandran": RamachandranTarget(self.model, self.verbose),
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

        total_loss = self.forward()
        metrics["geom_total_loss"] = (
            total_loss.item() if torch.is_tensor(total_loss) else total_loss
        )

        for name, loss in self.target_losses().items():
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            metrics[f"geom_{name}_loss"] = loss_val

        filtered_stats = filter_stats(self.stats(), verbosity)
        for name, target_stats in filtered_stats.items():
            for stat_name, stat_val in target_stats.items():
                metrics[f"geom_{name}_{stat_name}"] = stat_val

        return metrics

    def print_statistics(self):
        """Print REFMAC-style geometry statistics with losses."""
        # Silence sub-target verbosity: the loss calls below would double-print.
        saved_verbose = self.verbose
        self.verbose = 0

        print("\n" + "=" * 90)
        print("Geometry Restraint Statistics (REFMAC-style)")
        print("=" * 90)

        print(f"Components: {', '.join(self._targets.keys())}")
        print("-" * 90)
        print(
            f"{'Restraint Type':<25} {'N':>8} {'RMS Delta':>12} {'RMS Z':>10} {'Av(Sigma)':>12} {'Loss':>12}"
        )
        print("-" * 90)

        losses = self.target_losses()
        all_stats = self.stats()

        for name, loss in losses.items():
            try:
                loss_val = loss.item() if torch.is_tensor(loss) else loss
                stats = all_stats.get(name, {})

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

        print("-" * 90)
        total_loss = self.forward().item()

        self.verbose = saved_verbose

        print(
            f"{'TOTAL GEOMETRY LOSS':<25} {'':>8} {'':>12} {'':>10} {'':>12} {total_loss:>12.4f}"
        )

        print("=" * 90)
        print("Target: RMS Z should be ~1.0 for well-refined structure")
        print("        Phenix typical: Bond RMS ~0.007Å, Angle RMS ~1.2°")
        print("=" * 90 + "\n")


class TotalADPTarget(CombinedModelTargets):
    """Sum of the ADP restraints, from covalent to spatial to distribution-wide.

    Components, keyed for individual access (``target['simu']()``):

    - 'simu': :class:`ADPSimilarityTarget`, bonded atoms should share a B --
      covalent topology, the strongest local constraint.
    - 'locality': :class:`ADPLocalityTarget`, K-NN spatial smoothness,
      inverse-distance weighted, for medium-range correlation.
    - 'KL': :class:`ADPEntropyTarget`, controls the width of the B
      distribution, which is where overfitting shows up.

    'locality' and 'KL' work in log space, since B > 0 and right-skewed
    (B ~ LogNormal(μ, σ) means log B ~ Normal(μ, σ)); 'simu' restrains the raw
    ΔB of bonded atoms.

    Parameters
    ----------
    model : Model
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    def _create_targets(self) -> Dict[str, Target]:
        """Build the three ADP component targets."""
        print("Initializing TotalADPTarget with component targets...")
        return {
            "simu": ADPSimilarityTarget(self.model, verbose=self.verbose),
            "locality": ADPLocalityTarget(
                self.model, verbose=self.verbose
            ),
            "KL": ADPEntropyTarget(self.model, verbose=self.verbose),
        }

    def print_statistics(self) -> None:
        """
        Print comprehensive ADP restraint statistics.

        Displays statistics from all registered ADP targets.
        """
        print("\n" + "=" * 90)
        print("ADP RESTRAINT STATISTICS")
        print("=" * 90)

        print(f"\n{'COMPONENT LOSSES':^90}")
        print("-" * 90)
        print(f"{'Component':<25} {'Loss':>15}")
        print("-" * 90)

        losses = self.target_losses()
        all_stats = self.stats()

        for name, loss in losses.items():
            try:
                loss_val = loss.item() if torch.is_tensor(loss) else loss
                print(f"{name:<25} {loss_val:>15.4f}")

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

        total_loss = self.forward()
        metrics["adp_total_loss"] = (
            total_loss.item() if torch.is_tensor(total_loss) else total_loss
        )

        for name, loss in self.target_losses().items():
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            metrics[f"adp_{name}_loss"] = loss_val

        filtered_stats = filter_stats(self.stats(), verbosity)
        for name, target_stats in filtered_stats.items():
            for stat_name, stat_val in target_stats.items():
                metrics[f"adp_{name}_{stat_name}"] = stat_val

        return metrics
