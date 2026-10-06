"""
Combined targets for crystallographic refinement (e.g., geometry + ADP).

This module provides combined target classes that group several component
targets into a single target using ``nn.ModuleDict`` for clean organization
and dictionary-style access. Registering one with
:meth:`~torchref.refinement.loss_state.LossState.register_target` expands it
through ``items()``, so each component is registered under its own name.
"""

from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

from torchref.refinement.targets.base import Target, ModelTarget
from torchref.refinement.targets.geometry import (
    BondTarget, AngleTarget, TorsionTarget, PlanarityTarget,
    ChiralTarget, NonBondedHTarget, RamachandranTarget,
)
from torchref.refinement.targets.adp import (
    ADPSimilarityTarget, ADPLocalityTarget, ADPSigdTarget,
    NodeLoadTarget,
    NodeSmoothnessTarget,
)

if TYPE_CHECKING:
    from torchref.model.model import Model
    from torchref.refinement.base_refinement import Refinement


class CombinedModelTargets(ModelTarget):
    """Combined-target base for components that need only a Model.

    Sums geometry or ADP restraint components held in a ModuleDict; subclasses
    override ``_create_targets``. Components are reachable by name
    (``self['bond']``) and through ``keys``/``values``/``items``. Not listed in
    ``targets/__init__.__all__``.

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
            return torch.zeros((), device=self.device, dtype=self.dtype_float)
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
        if self.verbose > 0:
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


class TotalADPTarget(CombinedModelTargets):
    """Sum of the ADP restraints, from covalent to spatial to distribution-wide.

    Components, keyed for individual access (``target['sigd']()``), follow the ADP
    representation: per-atom ADPs register 'simu', 'locality' and 'sigd', and a
    node-field model (``model.adp_is_field``) 'sigd', 'node_load' and 'node_smoothness'.

    - 'simu': :class:`~torchref.refinement.targets.adp.similarity.ADPSimilarityTarget`,
      bonded atoms should share a B -- covalent topology, the strongest local
      constraint.
    - 'locality': :class:`~torchref.refinement.targets.adp.locality.ADPLocalityTarget`,
      K-NN spatial smoothness, inverse-distance weighted, for medium-range correlation.
    - 'sigd': :class:`~torchref.refinement.targets.adp.sigd.ADPSigdTarget`, a shifted
      inverse-gamma prior on the whole B distribution, which is where overfitting shows
      up.
    - 'node_load': :class:`~torchref.refinement.targets.adp.NodeLoadTarget`, keeps
      every node carrying a fair share of atoms.
    - 'node_smoothness': :class:`~torchref.refinement.targets.adp.NodeSmoothnessTarget`,
      penalises a node whose B departs from the nodes around it.

    'locality' works in log space, since B > 0 and right-skewed; 'sigd' uses the
    shifted inverse-gamma distribution that Masmaliyeva & Murshudov (2019) showed
    macromolecular B values actually follow -- it shares the log-normal's useful
    property that shape and scale separate (``std(log B) = sqrt(trigamma(alpha))``
    is independent of the scale), but fits deposited structures measurably
    better. 'simu' restrains the raw ΔB of bonded atoms.

    Parameters
    ----------
    model : Model
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    def _create_targets(self) -> Dict[str, Target]:
        """Build the ADP component targets that apply to the model's representation.

        Only the applicable ones are registered, rather than registering all of them and
        zero-weighting the inapplicable half. ``simu`` and ``locality`` restrain by
        penalty exactly the spatial smoothness a node field enforces by construction, so
        in field mode they are not a weak prior but a duplicate of the parametrisation;
        and ``node_load`` / ``node_smoothness`` have nothing to act on off it.

        Registering-then-zeroing would cost nothing at run time --- ``LossState.aggregate``
        skips a zero-weight target --- but it leaves the correctness of the setup resting
        on a weight, so anyone who touches the ``adp`` group weight for their own reasons
        silently re-enables a double-counted restraint. Whether a term applies is a
        property of the representation, not a number to be tuned.

        ``sigd`` applies either way: it is a prior on the marginal B distribution, which
        a field constrains no more than a per-atom parametrisation does.
        """
        if self.model.adp_is_field:
            targets = {
                "sigd": ADPSigdTarget(self.model, verbose=self.verbose),
                "node_load": NodeLoadTarget(self.model, verbose=self.verbose),
                "node_smoothness": NodeSmoothnessTarget(
                    self.model, verbose=self.verbose
                ),
            }
        else:
            targets = {
                "simu": ADPSimilarityTarget(self.model, verbose=self.verbose),
                "locality": ADPLocalityTarget(self.model, verbose=self.verbose),
                "sigd": ADPSigdTarget(self.model, verbose=self.verbose),
            }
        if self.verbose > 0:
            print(
                "Initializing TotalADPTarget with component targets: "
                + ", ".join(targets)
            )
        return targets
