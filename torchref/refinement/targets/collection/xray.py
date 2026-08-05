"""Collection (multi-dataset) X-ray targets.

One X-ray likelihood across a paired ``DatasetCollection`` + ``ModelCollection``, keys
matched so each timepoint dataset meets its own mixed model:
:class:`CollectionDifferenceTarget` (mean-based differences, the primary optimization
driver), :class:`CollectionRiceTarget` (per-timepoint Rice at ``beta = sigma**2``) and
:class:`CollectionMLTarget` (Rice with one shared Luzzati ``beta`` pooled over all
datasets' free reflections, owned by the target rather than the scaler).

All three inherit the single-dataset subset/masking/R-factor contract from
:class:`~torchref.refinement.targets.collection.base.CollectionXrayTarget`.
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
    Mean-based difference target over a DatasetCollection + ModelCollection.

    Differences are taken against the **mean** of all N datasets (dark +
    timepoints), with the error propagation that the dataset/mean covariance
    demands::

        F_mean(h) = (1/N) Σ_i F_obs_i(h)
        ΔF_obs_i  = F_obs_i - F_mean
        ΔF_calc_i = |F_calc_i| - F_calc_mean

        Var(F_i - F_mean) = σ_i²·(1 - 2/N) + (Σ_j σ_j²)/N²

    At N=2 the gradients are identical to direct dark-reference subtraction; above
    that the mean reference is the quieter one.

    Cross-dataset-coupled, unlike its siblings: the per-reflection mean ties all
    datasets together, so it works on aligned ``(N, n_hkl)`` stacks on the common HKL
    grid rather than the flat concatenate-then-mask form, and a reflection counts only
    if it is in this target's subset in **every** dataset.

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
        """Summed Gaussian NLL of the difference-from-mean; 0.0 if fewer than 2 sets."""
        dc = self._dataset_collection
        mc = self._model_collection

        all_keys = self._keys()
        N = len(all_keys)
        if N < 2:
            return torch.tensor(0.0, device=dc.hkl.device)

        # Clear caches so a preceding no-grad stats()/get_rfactor() call cannot
        # leave a detached tensor that breaks the loss backward.
        self._reset_model_caches()

        F_obs_list, sigma_list, mask_list, F_calc_list = [], [], [], []

        for key in all_keys:
            data = dc[key]
            model = mc[key]

            F_obs, sigma = data.get_corrected_data()
            F_calc = self._scaled_amp_full(data, model, recalc=False)
            # Validity + work/free/val selection, validation carved out of both.
            mask = self._subset(data).mask

            F_obs_list.append(F_obs)
            sigma_list.append(sigma)
            mask_list.append(mask)
            F_calc_list.append(F_calc)

        F_obs_stack = torch.stack(F_obs_list)  # (N, n_hkl)
        sigma_stack = torch.stack(sigma_list)  # (N, n_hkl)
        mask_stack = torch.stack(mask_list)  # (N, n_hkl)
        F_calc_stack = torch.stack(F_calc_list)  # (N, n_hkl)

        # A reflection must be in this subset in ALL datasets.
        mask_all = mask_stack.all(dim=0)  # (n_hkl,)

        F_mean_obs = F_obs_stack.mean(dim=0)  # (n_hkl,)
        F_calc_mean = F_calc_stack.mean(dim=0)  # (n_hkl,)

        delta_F_obs = F_obs_stack - F_mean_obs
        delta_F_calc = F_calc_stack - F_calc_mean

        # Var(F_i - F_mean) = σ_i²·(1 - 2/N) + (Σ_j σ_j²) / N²
        sum_sigma_sq = (sigma_stack**2).sum(dim=0)  # (n_hkl,)
        sigma_diff_sq = sigma_stack**2 * (1 - 2.0 / N) + sum_sigma_sq / (N**2)
        sigma_diff = torch.sqrt(sigma_diff_sq.clamp(min=1e-12))  # (N, n_hkl)

        # Mask via torch.where, not boolean indexing: no nonzero() device sync.
        delta_F_obs = torch.where(mask_all, delta_F_obs, torch.zeros_like(delta_F_obs))
        delta_F_calc = torch.where(
            mask_all, delta_F_calc, torch.zeros_like(delta_F_calc)
        )
        sigma_diff = torch.where(mask_all, sigma_diff, torch.ones_like(sigma_diff))

        # Floor sigma at 10% of its median so a zero sigma cannot blow up.
        eps = (
            torch.median(sigma_diff[:, mask_all].reshape(-1)) * 1e-1
            if mask_all.any()
            else 1e-3
        )
        sigma_safe = sigma_diff.clamp(min=eps)

        diff = delta_F_obs - delta_F_calc
        nll = 0.5 * (diff / sigma_safe) ** 2 + torch.log(sigma_safe) + 0.5 * _LOG_2PI

        # A single NaN would poison the whole gradient; 1e6 lets the step be rejected.
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))

        total_nll = (nll * mask_all).sum()

        return total_nll


# =========================================================================
# CollectionRiceTarget
# =========================================================================


class CollectionRiceTarget(CollectionXrayTarget):
    """
    Multi-timepoint Rice maximum-likelihood amplitude target.

    Rice NLL for acentrics and the folded-normal form for centrics, per timepoint.
    The per-timepoint losses are independent, so reflections and masks are
    concatenated across timepoints and masked once rather than reduced in a
    Python loop.

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
        """Summed Rice NLL over every timepoint's reflections in this subset."""
        dc = self._dataset_collection
        mc = self._model_collection

        tp_names = self._keys()
        if not tp_names:
            return torch.tensor(0.0, device=mc.device)

        self._reset_model_caches()

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

        # Mask the flat arrays once: compact, no per-dataset Python reduction.
        mask = torch.cat(mask_parts)
        F_obs = torch.cat(fo_parts)[mask]
        F_calc = torch.cat(fc_parts)[mask]
        sigma = torch.cat(sig_parts)[mask]
        centric = torch.cat(cen_parts)[mask]

        if F_obs.numel() == 0:
            return torch.tensor(0.0, device=mc.device)

        # Plain Rice: the model-error variance IS the measurement variance here.
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

        loss = torch.where(centric, loss_centric, loss_acentric)
        # A single NaN would poison the whole gradient; 1e6 lets the step be rejected.
        loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))

        return loss.sum()


# =========================================================================
# CollectionMLTarget
# =========================================================================


class CollectionMLTarget(CollectionXrayTarget):
    """
    Multi-dataset maximum-likelihood σ_A (Read MLF) target.

    The collection analogue of
    :class:`~torchref.refinement.targets.xray.ml_noalpha.MLNoAlphaXrayTarget`: instead
    of plain Rice's ``beta = sigma**2`` it uses one **shared** Luzzati model-error
    variance, fitted by maximum likelihood on the **pooled** free reflections of every
    data-model pair and mapped back onto the common HKL, so one per-reflection ``beta``
    serves all datasets. The estimator belongs to this target, not the scaler, which
    owns scaling only.

    Per-dataset loss is the Read MLF form (``mean = |Fc|``, variance ``epsilon*beta``)
    from :func:`torchref.base.targets.xray_likelihoods.rice_math`, and since those sums
    are independent the datasets are concatenated and masked once. ``beta`` is detached,
    so gradients reach the models only through ``F_calc``.

    Parameters
    ----------
    dataset_collection : DatasetCollection
    model_collection : ModelCollection
    scaler : ScalerBase
        Scaling layer applied to F_calc (``forward_mixed`` when available).
    normalize : bool
        Unused placeholder, as on the other two collection targets.
        TODO: remove from all three.
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
        """Summed Read-MLF loss over all datasets at the shared beta, work-set weighted."""
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

        # Each pair's F_calc is computed once WITH grad for the loss; detached copies
        # feed the pooled single-beta estimate.
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

            # Beta is estimated on the free set; data.free excludes validation.
            free_parts.append(data.free.mask)
            eps_parts.append(eps_common.to(dtype))
            dss_parts.append(dss_common.to(dtype))

        # One shared (beta, epsilon) for all datasets, mapped onto the common HKL via
        # target_dss. Detached; cached until maintenance() resets it.
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
        beta, eps = _est.beta, _est.epsilon

        # beta/eps live on the common HKL, so they are tiled once per dataset to line
        # up with the concatenation order.
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
