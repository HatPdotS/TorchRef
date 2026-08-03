"""Collection (multi-dataset) X-ray targets.

Apply a single X-ray likelihood across a paired ``DatasetCollection`` +
``ModelCollection``.  Keys are matched automatically so each timepoint dataset
is paired with its corresponding mixed model.

All three targets subclass :class:`~torchref.refinement.targets.collection.base.CollectionXrayTarget`,
so they share the single-dataset subset/masking/R-factor contract: reflections
are selected through each member's ``data.work`` / ``data.free`` / ``data.validation``
accessors (validity mask applied, validation carved out of both), and R-factors
are reported through ``stats()`` via the one shared
:func:`~torchref.base.metrics.rfactor.rfactor_work_free` source of truth.

Targets
-------
CollectionDifferenceTarget
    Mean-based difference target (primary optimization driver).
CollectionRiceTarget
    Multi-timepoint Rice maximum-likelihood amplitude target (beta = sigma^2).
CollectionMLTarget
    Like CollectionRiceTarget but with a Luzzati/Read sigma_A term: one shared
    model-error variance ``beta`` (Luzzati ``alpha`` fixed at 1), estimated by
    maximum likelihood on the pooled free reflections of all datasets. The
    estimator is owned by the target, not the scaler.
"""

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from torchref.base.reciprocal import get_scattering_vectors
from torchref.base.targets.xray_likelihoods import complex_var_from_beta, rice_math
from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator, epsilon_from_hkl
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

from ._util import _LOG_2PI, _scale_fcalc
from .base import CollectionXrayTarget

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.scaler_base import ScalerBase


# =========================================================================
# CollectionDifferenceTarget
# =========================================================================


class CollectionDifferenceTarget(CollectionXrayTarget):
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

    This is a cross-dataset-coupled target: the per-reflection mean couples all
    datasets, so it works on aligned ``(N, n_hkl)`` stacks on the common HKL grid
    (not the flat concatenate-then-mask form the independent targets use). The
    combined mask requires a reflection to be in this target's subset in **every**
    dataset.

    Parameters
    ----------
    dataset_collection : DatasetCollection
    model_collection : ModelCollection
    scaler : ScalerBase
        Single scaler applied to all F_calc (uses ``forward_mixed``
        with per-model fractions when available).
    normalize : bool
        Unused placeholder. ``forward`` always returns the unnormalised summed
        NLL regardless of this flag.
    use_work_set : bool
        Legacy bool; superseded by ``use_set``. If True, loss on the work set.
    use_set : str, optional
        Canonical 3-way subset selector ``"work"``/``"free"``/``"val"``.
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
        use_set: str = None,
        verbose: int = 0,
    ):
        super().__init__(
            dataset_collection,
            model_collection,
            scaler=scaler,
            use_work_set=use_work_set,
            use_set=use_set,
            verbose=verbose,
        )
        self.normalize = normalize

    def forward(self) -> torch.Tensor:
        dc = self._dataset_collection
        mc = self._model_collection

        all_keys = self._keys()
        N = len(all_keys)
        if N < 2:
            return torch.tensor(0.0, device=dc.hkl.device)

        # Clear caches so a preceding no-grad stats()/get_rfactor() call cannot
        # leave a detached tensor that breaks the loss backward.
        self._reset_model_caches()

        # --- Gather per-dataset full-size tensors on the common HKL grid ---
        F_obs_list, sigma_list, mask_list, F_calc_list = [], [], [], []

        for key in all_keys:
            data = dc[key]
            model = mc[key]

            F_obs, sigma = data.get_corrected_data()
            F_calc = self._scaled_amp_full(data, model, recalc=False)
            # Subset boolean mask: validity + work/free/val selection, validation
            # carved out of both work and free.
            mask = self._subset(data).mask

            F_obs_list.append(F_obs)
            sigma_list.append(sigma)
            mask_list.append(mask)
            F_calc_list.append(F_calc)

        # --- Stack into (N, n_hkl) tensors ---
        F_obs_stack = torch.stack(F_obs_list)  # (N, n_hkl)
        sigma_stack = torch.stack(sigma_list)  # (N, n_hkl)
        mask_stack = torch.stack(mask_list)  # (N, n_hkl)
        F_calc_stack = torch.stack(F_calc_list)  # (N, n_hkl)

        # Combined mask: reflection must be in this subset in ALL datasets
        mask_all = mask_stack.all(dim=0)  # (n_hkl,)

        # --- Mean across datasets ---
        F_mean_obs = F_obs_stack.mean(dim=0)  # (n_hkl,)
        F_calc_mean = F_calc_stack.mean(dim=0)  # (n_hkl,)

        # --- Differences from mean: (N, n_hkl) ---
        delta_F_obs = F_obs_stack - F_mean_obs
        delta_F_calc = F_calc_stack - F_calc_mean

        # --- Error propagation: Var(F_i - F_mean) ---
        # = σ_i²·(1 - 2/N) + (Σ_j σ_j²) / N²
        sum_sigma_sq = (sigma_stack**2).sum(dim=0)  # (n_hkl,)
        sigma_diff_sq = sigma_stack**2 * (1 - 2.0 / N) + sum_sigma_sq / (N**2)
        sigma_diff = torch.sqrt(sigma_diff_sq.clamp(min=1e-12))  # (N, n_hkl)

        # --- Apply mask via torch.where ---
        delta_F_obs = torch.where(mask_all, delta_F_obs, torch.zeros_like(delta_F_obs))
        delta_F_calc = torch.where(
            mask_all, delta_F_calc, torch.zeros_like(delta_F_calc)
        )
        sigma_diff = torch.where(mask_all, sigma_diff, torch.ones_like(sigma_diff))

        # Safe sigma clamping
        eps = (
            torch.median(sigma_diff[:, mask_all].reshape(-1)) * 1e-1
            if mask_all.any()
            else 1e-3
        )
        sigma_safe = sigma_diff.clamp(min=eps)

        # --- Gaussian NLL: (N, n_hkl) ---
        diff = delta_F_obs - delta_F_calc
        nll = 0.5 * (diff / sigma_safe) ** 2 + torch.log(sigma_safe) + 0.5 * _LOG_2PI

        # NaN/Inf protection
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))

        # Sum over all datasets and reflections (unnormalised NLL)
        total_nll = (nll * mask_all).sum()

        return total_nll


# =========================================================================
# CollectionRiceTarget
# =========================================================================


class CollectionRiceTarget(CollectionXrayTarget):
    """
    Multi-timepoint Rice maximum-likelihood amplitude target.

    Computes Rice-distribution NLL (acentric) and the corresponding
    centric NLL for each timepoint, with proper subset masking and
    NaN/Inf protection.  The per-timepoint losses are independent, so the
    reflections and boolean masks are **concatenated across timepoints and
    masked once** (flat vectorization) rather than summed in a Python loop.

    Parameters
    ----------
    dataset_collection : DatasetCollection
    model_collection : ModelCollection
    scaler : ScalerBase, optional
        Single scaler applied to each timepoint's F_calc.
    normalize : bool
        Unused placeholder. ``forward`` always returns the unnormalised summed
        NLL regardless of this flag.
    use_work_set : bool
        Legacy bool; superseded by ``use_set``. If True, loss on the work set.
    use_set : str, optional
        Canonical 3-way subset selector ``"work"``/``"free"``/``"val"``.
    verbose : int
        Verbosity level.
    """

    name: str = "collection_rice_xray"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        scaler: "ScalerBase" = None,
        normalize: bool = True,
        use_work_set: bool = True,
        use_set: str = None,
        verbose: int = 0,
    ):
        super().__init__(
            dataset_collection,
            model_collection,
            scaler=scaler,
            use_work_set=use_work_set,
            use_set=use_set,
            verbose=verbose,
        )
        self.normalize = normalize

    def _keys(self):
        """Rice fits the timepoints only — the dark reference is excluded."""
        mc = self._model_collection
        dc = self._dataset_collection
        return [n for n in mc.timepoint_names if n in dc]

    def forward(self) -> torch.Tensor:
        dc = self._dataset_collection
        mc = self._model_collection

        tp_names = self._keys()
        if not tp_names:
            return torch.tensor(0.0, device=mc.device)

        self._reset_model_caches()

        # --- Gather per-timepoint full-size tensors, then concatenate + mask ---
        fo_parts, fc_parts, sig_parts, cen_parts, mask_parts = [], [], [], [], []
        for tp_name in tp_names:
            data = dc[tp_name]
            model = mc[tp_name]

            F_obs, sigma = data.get_corrected_data()
            F_calc = self._scaled_amp_full(data, model, recalc=False)
            centric = data.centric
            if centric is None:
                centric = torch.zeros(
                    len(F_obs), dtype=torch.bool, device=F_obs.device
                )

            fo_parts.append(F_obs)
            fc_parts.append(F_calc)
            sig_parts.append(sigma)
            cen_parts.append(centric)
            mask_parts.append(self._subset(data).mask)

        # Concatenate the data and the boolean masks, then mask the flat arrays
        # once (compact — no MaskedTensors, no per-dataset Python reduction).
        mask = torch.cat(mask_parts)
        F_obs = torch.cat(fo_parts)[mask]
        F_calc = torch.cat(fc_parts)[mask]
        sigma = torch.cat(sig_parts)[mask]
        centric = torch.cat(cen_parts)[mask]

        if F_obs.numel() == 0:
            return torch.tensor(0.0, device=mc.device)

        # ML parameters (defaults): plain Rice, beta = sigma^2
        beta = sigma**2
        eb = beta.clamp(min=1e-6)

        # --- Acentric Rice NLL ---
        term1 = -torch.log(2 * F_obs / eb + 1e-12)
        term2 = F_obs**2 / eb
        term3 = F_calc**2 / eb
        arg_bessel = (2 * F_obs * F_calc / eb).clamp(max=1e6)
        term4 = -(torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel)
        loss_acentric = term1 + term2 + term3 + term4

        # --- Centric NLL ---
        term1_c = -0.5 * torch.log(2 / (np.pi * eb) + 1e-12)
        term2_c = F_obs**2 / (2 * eb)
        term3_c = F_calc**2 / (2 * eb)
        term4_c = -(F_obs * F_calc) / eb
        arg_exp = (-2 * F_obs * F_calc / eb).clamp(min=-80.0, max=80.0)
        term5_c = -torch.log((1 + torch.exp(arg_exp)) / 2 + 1e-12)
        loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

        # Combine
        loss = torch.where(centric, loss_centric, loss_acentric)
        loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))

        # Sum over all masked reflections (unnormalised NLL)
        return loss.sum()


# =========================================================================
# CollectionMLTarget
# =========================================================================


class CollectionMLTarget(CollectionXrayTarget):
    """
    Multi-dataset maximum-likelihood σ_A (Read MLF) target.

    The collection analogue of
    :class:`~torchref.refinement.targets.xray.ml_noalpha.MLNoAlphaXrayTarget`:
    instead of ``beta = sigma**2`` (plain Rice), it uses one **shared** Luzzati
    model-error variance ``beta``.  ``beta`` is estimated by maximum likelihood on
    the **pooled** free reflections of every data–model pair and mapped back onto
    the common HKL, so a single per-reflection ``beta`` applies to every dataset.

    The estimate is owned by **this target** (a :class:`SigmaAEstimator`), not the
    scaler — the scaler owns scaling only.  The per-dataset loss is the Read MLF
    form (``mean = |Fc|``, variance ``epsilon*beta``) from
    :func:`torchref.base.targets.xray_likelihoods.rice_math`. Because
    the loss is an independent per-dataset sum, the datasets are concatenated and
    masked once (flat vectorization).  ``beta`` is detached (a constant in
    autograd); gradients reach the models only through ``F_calc``.

    Parameters
    ----------
    dataset_collection : DatasetCollection
    model_collection : ModelCollection
    scaler : ScalerBase
        Scaling layer applied to F_calc (``forward_mixed`` when available).
    normalize : bool
        Unused placeholder (kept for signature parity with the other collection
        targets, where the flag is also non-functional). TODO: remove from all three.
    use_work_set : bool
        Legacy bool; superseded by ``use_set``. If True, loss on the work set.
    use_set : str, optional
        Canonical 3-way subset selector ``"work"``/``"free"``/``"val"``.
    verbose : int
        Verbosity level.
    base_weight : float, optional
        Intrinsic X-ray up-weight applied to the summed work-set loss. Defaults
        to ``DEFAULT_BASE_WEIGHT`` (10.0) and is applied on the work set only.
    """

    name: str = "collection_ml_xray"

    # The correctly-calibrated σ_A likelihood is legitimately soft relative to
    # the geometry prior, so this collection target carries an intrinsic
    # up-weight (the single-dataset ML target exposes no such parameter).
    # TODO(weighting): stopgap — belongs in the weighting infrastructure, ideally
    # replaced by a per-cycle gradient-ratio (wxc-style) weight.
    DEFAULT_BASE_WEIGHT = 10.0

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        scaler: "ScalerBase" = None,
        normalize: bool = True,
        use_work_set: bool = True,
        use_set: str = None,
        verbose: int = 0,
        base_weight: float = None,
    ):
        super().__init__(
            dataset_collection,
            model_collection,
            scaler=scaler,
            use_work_set=use_work_set,
            use_set=use_set,
            verbose=verbose,
        )
        self.normalize = normalize
        self.base_weight = (
            self.DEFAULT_BASE_WEIGHT if base_weight is None else float(base_weight)
        )
        # The shared σ_A model-error variance is owned by the target.
        self._sigma_a = SigmaAEstimator()
        # Model-independent common-HKL geometry (multiplicity + d*²), cached.
        self._eps_common: torch.Tensor = None
        self._dss_common: torch.Tensor = None
        self._geom_key: int = None

    def _common_geom(self):
        """``(epsilon, d_star_sq)`` on the common HKL, cached per dark dataset."""
        dc = self._dataset_collection
        mc = self._model_collection
        data = dc[mc.dark_key]
        key = id(data)
        if self._eps_common is None or self._geom_key != key:
            sg = getattr(data, "spacegroup", None)
            eps = epsilon_from_hkl(data.hkl, sg)
            s = get_scattering_vectors(data.hkl, data.cell)
            dss = (torch.norm(s, dim=1) ** 2).to(eps.dtype)
            self._eps_common, self._dss_common, self._geom_key = eps, dss, key
        return self._eps_common, self._dss_common

    def forward(self) -> torch.Tensor:
        dc = self._dataset_collection
        mc = self._model_collection

        all_keys = self._keys()
        if not all_keys:
            return torch.tensor(0.0, device=mc.device)

        eps_common, dss_common = self._common_geom()
        dtype = dss_common.dtype

        # Clear any structure-factor cache populated under no_grad (e.g. by the
        # scaler's joint initialization or a preceding stats() call) so the
        # F_calc computed below carries a live graph.
        self._reset_model_caches()

        # Compute each pair's F_calc once (with grad — used by the loss), and
        # collect detached copies to pool the free reflections for ONE shared
        # beta estimate across all datasets.
        fo_parts, fc_parts, cen_parts, msk_parts = [], [], [], []
        eps_parts, dss_parts, free_parts, sig_parts = [], [], [], []
        for key in all_keys:
            data = dc[key]
            model = mc[key]

            F_obs, sig_obs = data.get_corrected_data()
            F_obs = F_obs.to(dtype)
            sig_parts.append(sig_obs.to(dtype).reshape(-1))
            F_calc = self._scaled_amp_full(data, model, recalc=False).to(dtype)
            centric = data.centric
            if centric is None:
                centric = torch.zeros(
                    len(F_obs), dtype=torch.bool, device=F_obs.device
                )

            fo_parts.append(F_obs)
            fc_parts.append(F_calc)
            cen_parts.append(centric)
            msk_parts.append(self._subset(data).mask)

            # Beta is estimated on the free set (validation excluded by data.free).
            free_parts.append(data.free.mask)
            eps_parts.append(eps_common.to(dtype))
            dss_parts.append(dss_common.to(dtype))

        # One shared (beta, epsilon) for all datasets, mapped onto the common
        # HKL via target_dss. Detached; cached until maintenance() resets it.
        _est = self._sigma_a.get(
            torch.cat(fo_parts),
            torch.cat([fc.detach() for fc in fc_parts]),  # beta needs no gradient
            torch.cat(cen_parts),
            torch.cat(eps_parts),
            torch.cat(dss_parts),
            torch.cat(free_parts),
            out_epsilon=eps_common.to(dtype),
            target_dss=dss_common,
            # Always passed, as at every other call site: it is what makes sigma_A the
            # correlation with the noise-free amplitudes rather than with the noisy data.
            sigma_obs=torch.cat(sig_parts),
        )
        # TOTAL variance: this target's likelihood does not account for sigma_obs itself.
        beta, eps = _est.beta, _est.epsilon

        # Concatenate data + boolean masks across datasets and evaluate the
        # Read-MLF loss once. beta/eps are on the common HKL, so they are tiled
        # once per dataset to align with the concatenation order.
        n_ds = len(all_keys)
        F_obs_cat = torch.cat(fo_parts)
        F_calc_cat = torch.cat(fc_parts)
        centric_cat = torch.cat(cen_parts)
        mask_cat = torch.cat(msk_parts)
        beta_cat = beta.to(F_obs_cat.dtype).repeat(n_ds)
        eps_cat = eps.to(F_obs_cat.dtype).repeat(n_ds) if eps is not None else None

        # TOTAL variance (`est.beta`, not `beta_model`): this likelihood does not
        # account for sigma_obs itself, so the measurement variance must stay inside beta.
        total = rice_math(
            F_obs_cat, F_calc_cat, complex_var_from_beta(beta_cat, eps_cat),
            centric_cat, mask=mask_cat,
        )

        # Base weight drives refinement; applied on the work set only.
        if self.use_work_set:
            total = self.base_weight * total
        return total

    def maintenance(self) -> None:
        """Invalidate the shared beta so it is re-estimated from the updated
        models on the next forward (``LossState`` calls this after each
        optimizer-step block)."""
        self._sigma_a.reset()

    def stats(self) -> Dict[str, StatEntry]:
        """Base collection X-ray stats plus shared-beta diagnostics."""
        out = super().stats()
        bb = self._sigma_a.beta_per_bin
        if bb is not None and bb.numel() > 0:
            out["beta_bin0"] = stat(bb[0].item(), VERBOSITY_STANDARD)
            out["beta_binN"] = stat(bb[-1].item(), VERBOSITY_STANDARD)
        return out
