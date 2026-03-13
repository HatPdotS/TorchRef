"""
Collection-aware targets for kinetic refinement.

All targets extend ``torchref.refinement.targets.base.Target`` and operate
on paired DatasetCollection + ModelCollection instances.  Keys are matched
automatically so that each timepoint dataset is paired with its
corresponding mixed model.

Targets
-------
CollectionDifferenceTarget
    Multi-timepoint difference target (primary optimization driver).
CollectionMLTarget
    Multi-timepoint maximum-likelihood amplitude target.
MultiModelGeometryTarget
    Geometry restraints applied to the shared base models.
MultiModelADPTarget
    ADP restraints applied to the shared base models.
KineticPriorTarget
    Regularizes per-timepoint fractions towards a kinetic model.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from torchref.refinement.targets.base import Target

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.scaler_base import ScalerBase


# =========================================================================
# Utility functions
# =========================================================================


def nll_difference_amplitude(
    delta_F_obs: torch.Tensor,
    delta_F_calc: torch.Tensor,
    sigma_diff: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL for difference amplitudes.

    NLL = 0.5 * ((ΔF_obs - ΔF_calc) / σ_diff)^2 + log(σ_diff)

    Parameters
    ----------
    delta_F_obs : torch.Tensor
        Observed difference amplitudes.
    delta_F_calc : torch.Tensor
        Calculated difference amplitudes.
    sigma_diff : torch.Tensor
        Propagated uncertainty on the differences.

    Returns
    -------
    torch.Tensor
        Mean NLL over reflections (scalar).
    """
    residual = (delta_F_obs - delta_F_calc) / sigma_diff
    nll = 0.5 * residual**2 + torch.log(sigma_diff)
    return nll.mean()


def nll_xray_amplitude(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian NLL for X-ray amplitudes.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factor amplitudes.
    sigma : torch.Tensor
        Experimental uncertainties.

    Returns
    -------
    torch.Tensor
        Mean NLL (scalar).
    """
    residual = (F_obs - F_calc) / sigma
    nll = 0.5 * residual**2 + torch.log(sigma)
    return nll.mean()


def propagate_sigma_difference(
    sigma_t: torch.Tensor, sigma_ref: torch.Tensor
) -> torch.Tensor:
    """
    Propagate uncertainties for difference amplitudes.

    σ_diff = sqrt(σ_t² + σ_ref²)
    """
    return torch.sqrt(sigma_t**2 + sigma_ref**2)


# =========================================================================
# CollectionDifferenceTarget
# =========================================================================


class CollectionDifferenceTarget(Target):
    """
    Multi-timepoint difference target using DatasetCollection + ModelCollection.

    For each timepoint name present in both collections::

        ΔF_obs  = F_obs(dark) - F_obs(timepoint)
        ΔF_calc = |F_calc_scaled(dark_model)| - |F_calc_scaled(mixed_model)|
        NLL    += nll_difference(ΔF_obs, ΔF_calc, σ_diff)

    Pairs are matched automatically by key name.

    A single scaler is used for both dark and mixed F_calc.  This avoids
    artificial scale differences that corrupt the difference signal and
    confuse kinetic refinement.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection of reflection datasets (keyed by timepoint name).
    model_collection : ModelCollection
        Collection of mixed models (keyed by timepoint name).
    scaler : ScalerBase
        Single scaler applied to both dark and mixed F_calc.
    normalize : bool
        If True, divide total NLL by number of matched timepoints.
    use_work_set : bool
        If True, compute loss only on the work set (non-free reflections).
    verbose : int
        Verbosity level.
    """

    name: str = "difference_xray"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        scaler: "ScalerBase" = None,
        normalize: bool = True,
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._dataset_collection = dataset_collection
        self._model_collection = model_collection
        self.add_module("_scaler", scaler)
        self.normalize = normalize
        self.use_work_set = use_work_set

    def forward(self) -> torch.Tensor:
        dc = self._dataset_collection
        mc = self._model_collection

        dark_data = dc[mc.dark_key]
        dark_model = mc.dark_model

        hkl = dark_data.hkl
        F_obs_dark = dark_data.F
        sigma_dark = dark_data.F_sigma

        # Dark F_calc (scaled with fraction-weighted solvent if available)
        f_calc_dark = dark_model(hkl)
        if self._scaler is not None:
            if hasattr(self._scaler, "forward_mixed"):
                f_calc_dark = self._scaler.forward_mixed(f_calc_dark, dark_model.fractions)
            else:
                f_calc_dark = self._scaler(f_calc_dark)
        F_calc_dark = torch.abs(f_calc_dark)

        # Work-set mask for dark
        if self.use_work_set and hasattr(dark_data, "rfree_flags"):
            mask_dark = ~dark_data.rfree_flags  # work set = non-free
        else:
            mask_dark = torch.ones(len(hkl), dtype=torch.bool, device=hkl.device)

        total_nll = torch.tensor(0.0, device=hkl.device)
        n_timepoints = 0

        for tp_name in mc.timepoint_names:
            if tp_name not in dc:
                continue

            data = dc[tp_name]
            model = mc[tp_name]

            # Timepoint F_calc (same scaler, fraction-weighted solvent)
            f_calc = model(hkl)
            if self._scaler is not None:
                if hasattr(self._scaler, "forward_mixed"):
                    f_calc = self._scaler.forward_mixed(f_calc, model.fractions)
                else:
                    f_calc = self._scaler(f_calc)
            F_calc = torch.abs(f_calc)

            # Differences
            delta_F_obs = F_obs_dark - data.F
            delta_F_calc = F_calc_dark - F_calc
            sigma_diff = propagate_sigma_difference(data.F_sigma, sigma_dark)

            # Combined work-set mask
            if self.use_work_set and hasattr(data, "rfree_flags"):
                mask = mask_dark & (~data.rfree_flags)
            else:
                mask = mask_dark

            total_nll = total_nll + nll_difference_amplitude(
                delta_F_obs[mask], delta_F_calc[mask], sigma_diff[mask]
            )
            n_timepoints += 1

        if self.normalize and n_timepoints > 0:
            total_nll = total_nll / n_timepoints

        return total_nll


# =========================================================================
# CollectionMLTarget
# =========================================================================


class CollectionMLTarget(Target):
    """
    Multi-timepoint maximum-likelihood amplitude target.

    For each timepoint, computes Gaussian NLL between observed and
    calculated structure factor amplitudes.

    Parameters
    ----------
    dataset_collection : DatasetCollection
    model_collection : ModelCollection
    scaler : ScalerBase, optional
        Single scaler applied to each timepoint's F_calc.
    normalize : bool
        Divide total NLL by number of matched timepoints.
    use_work_set : bool
        Compute loss only on work set.
    verbose : int
        Verbosity level.
    """

    name: str = "collection_ml_xray"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        scaler: "ScalerBase" = None,
        normalize: bool = True,
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._dataset_collection = dataset_collection
        self._model_collection = model_collection
        self.add_module("_scaler", scaler)
        self.normalize = normalize
        self.use_work_set = use_work_set

    def forward(self) -> torch.Tensor:
        dc = self._dataset_collection
        mc = self._model_collection

        total_nll = torch.tensor(0.0, device=mc.device)
        n = 0

        for tp_name in mc.timepoint_names:
            if tp_name not in dc:
                continue

            data = dc[tp_name]
            model = mc[tp_name]
            hkl = data.hkl

            f_calc = model(hkl)
            if self._scaler is not None:
                if hasattr(self._scaler, "forward_mixed"):
                    f_calc = self._scaler.forward_mixed(f_calc, model.fractions)
                else:
                    f_calc = self._scaler(f_calc)
            F_calc = torch.abs(f_calc)

            # Work-set mask
            if self.use_work_set and hasattr(data, "rfree_flags"):
                mask = ~data.rfree_flags
            else:
                mask = torch.ones(len(hkl), dtype=torch.bool, device=hkl.device)

            total_nll = total_nll + nll_xray_amplitude(
                data.F[mask], F_calc[mask], data.F_sigma[mask]
            )
            n += 1

        if self.normalize and n > 0:
            total_nll = total_nll / n

        return total_nll


# =========================================================================
# MultiModelGeometryTarget
# =========================================================================


class MultiModelGeometryTarget(Target):
    """
    Geometry restraints for the shared base models in a ModelCollection.

    Creates a ``TotalGeometryTarget`` for each base model and sums them.
    Since models are shared across timepoints, restraints only need to be
    computed once per base model (not per timepoint).

    Parameters
    ----------
    model_collection : ModelCollection
    verbose : int
        Verbosity level.
    """

    name: str = "multi_model_geometry"

    def __init__(self, model_collection: "ModelCollection", verbose: int = 0):
        super().__init__(verbose=verbose)
        self._model_collection = model_collection

        from torchref.refinement.targets.combined import TotalGeometryTarget

        self._targets = nn.ModuleList(
            [
                TotalGeometryTarget(model=m, verbose=verbose)
                for m in model_collection.base_models
            ]
        )

    def forward(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self._model_collection.device)
        for target in self._targets:
            total = total + target()
        return total

    def register_to_state(self, state):
        """
        Register each base model's geometry sub-targets individually
        into a LossState with hierarchical naming.

        Parameters
        ----------
        state : LossState
            The loss state to register targets into.
        """
        for i, target in enumerate(self._targets):
            state.register_target(
                "geometry", target, prefix=f"model_{i}"
            )
        return state

    def items(self):
        """Expose sub-targets for LossState auto-expansion."""
        result = {}
        for i, target in enumerate(self._targets):
            for sub_name, sub_target in target.items():
                result[f"model_{i}/{sub_name}"] = sub_target
        return result.items()


# =========================================================================
# MultiModelADPTarget
# =========================================================================


class MultiModelADPTarget(Target):
    """
    ADP restraints for the shared base models in a ModelCollection.

    Same pattern as MultiModelGeometryTarget but using TotalADPTarget.

    Parameters
    ----------
    model_collection : ModelCollection
    verbose : int
        Verbosity level.
    """

    name: str = "multi_model_adp"

    def __init__(self, model_collection: "ModelCollection", verbose: int = 0):
        super().__init__(verbose=verbose)
        self._model_collection = model_collection

        from torchref.refinement.targets.combined import TotalADPTarget

        self._targets = nn.ModuleList(
            [
                TotalADPTarget(model=m, verbose=verbose)
                for m in model_collection.base_models
            ]
        )

    def forward(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self._model_collection.device)
        for target in self._targets:
            total = total + target()
        return total

    def register_to_state(self, state):
        """Register per-model ADP sub-targets into LossState."""
        for i, target in enumerate(self._targets):
            state.register_target(
                "adp", target, prefix=f"model_{i}"
            )
        return state

    def items(self):
        result = {}
        for i, target in enumerate(self._targets):
            for sub_name, sub_target in target.items():
                result[f"model_{i}/{sub_name}"] = sub_target
        return result.items()


# =========================================================================
# KineticPriorTarget
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
        n = 0

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
            n += 1

        if n > 0:
            total_loss = total_loss / n

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

        sorted_names = sorted(
            self.timepoints_map, key=lambda n: self.timepoints_map[n]
        )
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
