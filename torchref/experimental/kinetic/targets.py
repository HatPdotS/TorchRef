"""
Kinetic refinement targets.

The generic collection-aware targets now live in
:mod:`torchref.refinement.targets.collection` and are **re-exported here** for
backward compatibility:

CollectionDifferenceTarget
    Multi-timepoint difference target (primary optimization driver).
CollectionRiceTarget
    Multi-timepoint Rice maximum-likelihood amplitude target.
CollectionMLTarget
    Multi-timepoint maximum-likelihood target with a shared Luzzati σ_A term.
MultiModelGeometryTarget
    Geometry restraints applied to the shared base models.
MultiModelADPTarget
    ADP restraints applied to the shared base models.

This module now only *defines* the kinetic-specific target:

KineticPriorTarget
    Regularizes per-timepoint fractions towards a kinetic model.
"""

from typing import TYPE_CHECKING, Dict

import torch

from torchref.refinement.targets.base import Target

# Back-compat re-exports of the relocated generic collection targets.
from torchref.refinement.targets.collection import (  # noqa: F401
    CollectionDifferenceTarget,
    CollectionMLTarget,
    CollectionRiceTarget,
    MultiModelADPTarget,
    MultiModelGeometryTarget,
)
from torchref.refinement.targets.collection._util import (  # noqa: F401
    _LOG_2PI,
    _scale_fcalc,
    _unpack_masked_data,
)

if TYPE_CHECKING:
    from torchref.model.model_collection import ModelCollection


# =========================================================================
# KineticPriorTarget (kinetic-specific)
# =========================================================================


class KineticPriorTarget(Target):
    """
    Regularize per-timepoint fractions towards a kinetic model.

    The kinetic model provides a smooth prior over how population fractions
    should evolve over time.  The fractions in ModelCollection are free
    parameters; this target penalizes deviation from the kinetic prediction.

    Periodically call ``refit_prior()`` to update the kinetic model to
    match the current free fractions (EM-style alternation).

    Parameters
    ----------
    model_collection : ModelCollection
    kinetic_model : occupancies_kinetics
        The kinetic occupancy model whose ``forward()`` returns
        shape ``[n_states, n_timepoints]``.
    timepoints_map : Dict[str, int]
        Maps timepoint names to indices into the kinetic model's time axis.
        E.g. ``{"1ps": 0, "5ps": 1, "10ps": 2}``.
    verbose : int
        Verbosity level.
    """

    name: str = "kinetic_prior"

    def __init__(
        self,
        model_collection: "ModelCollection",
        kinetic_model,
        timepoints_map: Dict[str, int],
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._model_collection = model_collection
        self.add_module("_kinetic_model", kinetic_model)
        self.timepoints_map = timepoints_map

    def forward(self) -> torch.Tensor:
        """
        Squared difference between current fractions and kinetic predictions.
        """
        mc = self._model_collection

        # Kinetic model predictions: [n_states, n_timepoints]
        kinetic_occ = self._kinetic_model()

        total_loss = torch.tensor(0.0, device=mc.device)

        for tp_name in mc.timepoint_names:
            if tp_name not in self.timepoints_map:
                continue

            t_idx = self.timepoints_map[tp_name]

            # Kinetic prediction for this timepoint (detached — prior is fixed)
            predicted = kinetic_occ[:, t_idx].detach()

            # Current free fractions
            current = mc[tp_name].fractions

            # Match dimensions: kinetic states may differ from base models
            # if state_mapping collapses states. Use min of both lengths.
            n_match = min(len(predicted), len(current))
            total_loss = total_loss + torch.sum(
                (current[:n_match] - predicted[:n_match]) ** 2
            )

        return total_loss

    def refit_prior(self, niter: int = 50, lr: float = 1e-2):
        """
        Refit kinetic model to match current free fractions (M-step).

        Freezes model fractions, optimizes kinetic model parameters to
        minimize prediction error against current fractions.

        Parameters
        ----------
        niter : int
            Number of optimizer steps.
        lr : float
            Learning rate for Adam optimizer.
        """
        mc = self._model_collection

        # Collect current fractions as optimization targets
        target_fractions = []
        target_indices = []

        sorted_names = sorted(self.timepoints_map, key=lambda n: self.timepoints_map[n])
        for name in sorted_names:
            if name in mc:
                target_fractions.append(mc[name].fractions.detach())
                target_indices.append(self.timepoints_map[name])

        if not target_fractions:
            return

        target_matrix = torch.stack(target_fractions)  # [n_tp, n_models]
        target_indices = torch.tensor(target_indices, device=mc.device)

        # Optimize kinetic model
        optimizer = torch.optim.Adam(self._kinetic_model.parameters(), lr=lr)

        for i in range(niter):
            optimizer.zero_grad()
            predicted = self._kinetic_model()  # [n_states, n_timepoints]

            # Select matching timepoint columns
            predicted_at_tp = predicted[:, target_indices].T  # [n_tp, n_states]

            # Match dimensions
            n_match = min(predicted_at_tp.shape[1], target_matrix.shape[1])
            loss = torch.sum(
                (predicted_at_tp[:, :n_match] - target_matrix[:, :n_match]) ** 2
            )
            loss.backward()
            optimizer.step()

        if self.verbose > 0:
            print(
                f"  Kinetic prior refit: {niter} steps, "
                f"final loss = {loss.item():.6f}"
            )
