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

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from torchref.refinement.targets.base import Target

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.scaler_base import ScalerBase


# =========================================================================
# Utility functions
# =========================================================================

_LOG_2PI = np.log(2.0 * np.pi)


def _unpack_masked_data(
    data: "ReflectionData",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extract plain tensors + validity mask from a ReflectionData call.

    Returns
    -------
    F_obs, sigma, rfree_bool, validity, centric
        All as plain tensors (not MaskedTensors).  *rfree_bool* has
        True = work, False = free.  *centric* may be None.
    """
    _, F_obs, sigma, rfree = data()
    if hasattr(F_obs, "get_mask"):
        validity = F_obs.get_mask()
        F_obs = F_obs.get_data()
        sigma = sigma.get_data() if hasattr(sigma, "get_mask") else sigma
    else:
        validity = torch.ones(len(F_obs), dtype=torch.bool, device=F_obs.device)
    centric = data.centric if hasattr(data, "centric") else None
    return F_obs, sigma, rfree.bool(), validity, centric


def _scale_fcalc(scaler, fcalc, model):
    """Apply scaler, using forward_mixed when available."""
    if scaler is None:
        return fcalc
    if hasattr(scaler, "forward_mixed") and hasattr(model, "fractions"):
        return scaler.forward_mixed(fcalc, model.fractions)
    return scaler(fcalc)


# =========================================================================
# CollectionDifferenceTarget
# =========================================================================


class CollectionDifferenceTarget(Target):
    """
    Mean-based difference target using DatasetCollection + ModelCollection.

    Computes differences relative to the **mean** across all N datasets
    (dark + timepoints), with proper error propagation accounting for the
    covariance between each dataset and the mean::

        F_mean(h) = (1/N) Σ_i F_obs_i(h)
        ΔF_obs_i  = F_obs_i - F_mean
        ΔF_calc_i = |F_calc_i| - F_calc_mean

        Var(F_i - F_mean) = σ_i²·(1 - 2/N) + (Σ_j σ_j²)/N²

    For N=2 (dark + one timepoint) this gives identical gradients to the
    direct dark-reference subtraction.  For N>2 the mean reference has
    lower noise.

    All computation is vectorized on stacked (N, n_hkl) tensors — no
    Python loops over datasets.

    Parameters
    ----------
    dataset_collection : DatasetCollection
    model_collection : ModelCollection
    scaler : ScalerBase
        Single scaler applied to all F_calc (uses ``forward_mixed``
        with per-model fractions when available).
    normalize : bool
        If True, divide total NLL by number of datasets.
    use_work_set : bool
        If True, compute loss only on the work set (rfree_flags=True).
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
        hkl = dc.hkl

        # Collect all matched dataset keys (dark + timepoints)
        all_keys = [mc.dark_key] + [n for n in mc.timepoint_names if n in dc]
        N = len(all_keys)
        if N < 2:
            return torch.tensor(0.0, device=hkl.device)

        # --- Gather per-dataset tensors ---
        F_obs_list, sigma_list, mask_list, F_calc_list = [], [], [], []

        for key in all_keys:
            data = dc[key]
            model = mc[key]

            F_obs, sigma, rfree, validity, _ = _unpack_masked_data(data)
            F_calc = torch.abs(_scale_fcalc(self._scaler, model(hkl), model))

            mask = validity & rfree if self.use_work_set else validity

            F_obs_list.append(F_obs)
            sigma_list.append(sigma)
            mask_list.append(mask)
            F_calc_list.append(F_calc)

        # --- Stack into (N, n_hkl) tensors ---
        F_obs_stack = torch.stack(F_obs_list)       # (N, n_hkl)
        sigma_stack = torch.stack(sigma_list)       # (N, n_hkl)
        mask_stack = torch.stack(mask_list)         # (N, n_hkl)
        F_calc_stack = torch.stack(F_calc_list)     # (N, n_hkl)

        # Combined mask: reflection must be valid + work-set in ALL datasets
        mask_all = mask_stack.all(dim=0)            # (n_hkl,)

        # --- Mean across datasets ---
        F_mean_obs = F_obs_stack.mean(dim=0)        # (n_hkl,)
        F_calc_mean = F_calc_stack.mean(dim=0)      # (n_hkl,)

        # --- Differences from mean: (N, n_hkl) ---
        delta_F_obs = F_obs_stack - F_mean_obs
        delta_F_calc = F_calc_stack - F_calc_mean

        # --- Error propagation: Var(F_i - F_mean) ---
        # = σ_i²·(1 - 2/N) + (Σ_j σ_j²) / N²
        sum_sigma_sq = (sigma_stack ** 2).sum(dim=0)                # (n_hkl,)
        sigma_diff_sq = sigma_stack ** 2 * (1 - 2.0 / N) + sum_sigma_sq / (N ** 2)
        sigma_diff = torch.sqrt(sigma_diff_sq.clamp(min=1e-12))    # (N, n_hkl)

        # --- Apply mask via torch.where ---
        delta_F_obs = torch.where(mask_all, delta_F_obs, torch.zeros_like(delta_F_obs))
        delta_F_calc = torch.where(mask_all, delta_F_calc, torch.zeros_like(delta_F_calc))
        sigma_diff = torch.where(mask_all, sigma_diff, torch.ones_like(sigma_diff))

        # Safe sigma clamping
        eps = torch.median(sigma_diff[:, mask_all].reshape(-1)) * 1e-1 if mask_all.any() else 1e-3
        sigma_safe = sigma_diff.clamp(min=eps)

        # --- Gaussian NLL: (N, n_hkl) ---
        diff = delta_F_obs - delta_F_calc
        nll = 0.5 * (diff / sigma_safe) ** 2 + torch.log(sigma_safe) + 0.5 * _LOG_2PI

        # NaN/Inf protection
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))

        # Sum over all datasets and reflections, normalise by total valid count
        n_valid = mask_all.sum().clamp(min=1) * N
        total_nll = (nll * mask_all).sum() / n_valid

        return total_nll


# =========================================================================
# CollectionMLTarget
# =========================================================================


class CollectionMLTarget(Target):
    """
    Multi-timepoint maximum-likelihood amplitude target.

    Computes Rice-distribution NLL (acentric) and the corresponding
    centric NLL for each timepoint, with proper validity masking and
    NaN/Inf protection.  Vectorized on stacked (N_tp, n_hkl) tensors.

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

        tp_names = [n for n in mc.timepoint_names if n in dc]
        if not tp_names:
            return torch.tensor(0.0, device=mc.device)

        # --- Gather per-timepoint tensors ---
        F_obs_list, F_calc_list, sigma_list = [], [], []
        mask_list, centric_list = [], []

        for tp_name in tp_names:
            data = dc[tp_name]
            model = mc[tp_name]
            hkl = data.hkl

            F_obs, sigma, rfree, validity, centric = _unpack_masked_data(data)
            F_calc_amp = torch.abs(_scale_fcalc(self._scaler, model(hkl), model))

            mask = validity & rfree if self.use_work_set else validity
            if centric is None:
                centric = torch.zeros(len(hkl), dtype=torch.bool, device=hkl.device)

            F_obs_list.append(F_obs)
            F_calc_list.append(F_calc_amp)
            sigma_list.append(sigma)
            mask_list.append(mask)
            centric_list.append(centric)

        # --- Stack into (N_tp, n_hkl) ---
        F_obs = torch.stack(F_obs_list)
        F_calc = torch.stack(F_calc_list)
        sigma = torch.stack(sigma_list)
        mask = torch.stack(mask_list)
        centric = torch.stack(centric_list)

        # --- Apply mask via torch.where ---
        F_obs = torch.where(mask, F_obs, torch.zeros_like(F_obs))
        F_calc = torch.where(mask, F_calc, torch.zeros_like(F_calc))
        sigma = torch.where(mask, sigma, torch.ones_like(sigma))

        # ML parameters (defaults)
        beta = sigma ** 2
        eb = beta.clamp(min=1e-6)

        # --- Acentric Rice NLL ---
        term1 = -torch.log(2 * F_obs / eb + 1e-12)
        term2 = F_obs ** 2 / eb
        term3 = F_calc ** 2 / eb
        arg_bessel = (2 * F_obs * F_calc / eb).clamp(max=1e6)
        term4 = -(torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel)
        loss_acentric = term1 + term2 + term3 + term4

        # --- Centric NLL ---
        term1_c = -0.5 * torch.log(2 / (np.pi * eb) + 1e-12)
        term2_c = F_obs ** 2 / (2 * eb)
        term3_c = F_calc ** 2 / (2 * eb)
        term4_c = -(F_obs * F_calc) / eb
        arg_exp = (-2 * F_obs * F_calc / eb).clamp(min=-80.0, max=80.0)
        term5_c = -torch.log((1 + torch.exp(arg_exp)) / 2 + 1e-12)
        loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

        # Combine
        loss = torch.where(centric, loss_centric, loss_acentric)
        loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))

        # Mean over valid reflections across all timepoints
        n_valid = mask.sum().clamp(min=1)
        total_nll = (loss * mask).sum() / n_valid

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
