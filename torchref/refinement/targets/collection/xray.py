"""Collection (multi-dataset) X-ray targets.

Apply a single X-ray likelihood across a paired ``DatasetCollection`` +
``ModelCollection``.  Keys are matched automatically so each timepoint dataset
is paired with its corresponding mixed model.  All computation is vectorized on
stacked ``(N, n_hkl)`` tensors.

Targets
-------
CollectionDifferenceTarget
    Mean-based difference target (primary optimization driver).
CollectionMLTarget
    Multi-timepoint maximum-likelihood amplitude target (Rice, beta = sigma^2).
CollectionMLSigmaATarget
    Like CollectionMLTarget but with a Luzzati/Read sigma_A term: one shared
    alpha/beta estimated across all datasets in the (collection) scaler.
"""

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from torchref.base.targets.xray_ml_sigmaa import ml_xray_loss_beta_math
from torchref.refinement.targets.base import Target
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

from ._util import _LOG_2PI, _scale_fcalc, _unpack_masked_data

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.scaler_base import ScalerBase


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
        F_obs_stack = torch.stack(F_obs_list)  # (N, n_hkl)
        sigma_stack = torch.stack(sigma_list)  # (N, n_hkl)
        mask_stack = torch.stack(mask_list)  # (N, n_hkl)
        F_calc_stack = torch.stack(F_calc_list)  # (N, n_hkl)

        # Combined mask: reflection must be valid + work-set in ALL datasets
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

        # Sum over valid reflections across all timepoints (unnormalised NLL)
        total_nll = (loss * mask).sum()

        return total_nll


# =========================================================================
# CollectionMLSigmaATarget
# =========================================================================


class CollectionMLSigmaATarget(Target):
    """
    Multi-dataset maximum-likelihood σ_A (Read MLF) target.

    The collection analogue of
    :class:`~torchref.refinement.targets.xray.maximum_likelihood_sigmaa.MaximumLikelihoodSigmaAXrayTarget`:
    instead of ``beta = sigma**2`` (plain Rice), it uses one **shared** Luzzati
    model-error variance ``beta`` estimated by the (collection) scaler across all
    datasets (``scaler.get_beta()``).  Because the datasets share the common HKL
    set, the single per-reflection ``beta`` applies to every dataset.

    The per-dataset loss is the Read MLF form
    (``mean = |Fc|``, variance ``epsilon*beta``) from
    :func:`torchref.base.targets.xray_ml_sigmaa.ml_xray_loss_beta_math`,
    summed over datasets.  ``beta`` is detached (a constant in autograd);
    gradients reach the models only through ``F_calc``.

    Parameters
    ----------
    dataset_collection : DatasetCollection
    model_collection : ModelCollection
    scaler : ScalerBase
        Must provide ``get_beta()`` (e.g. ``CollectionScaler``).
    normalize : bool
        Unused placeholder (kept for signature parity with CollectionMLTarget).
    use_work_set : bool
        Compute loss only on the work set.
    verbose : int
        Verbosity level.
    base_weight : float, optional
        Intrinsic X-ray up-weight (see the single-dataset target). Defaults to
        ``DEFAULT_BASE_WEIGHT`` and is applied on the work set only.
    """

    name: str = "collection_ml_sigmaa_xray"

    # Mirrors the single-dataset MaximumLikelihoodSigmaAXrayTarget: the
    # correctly-calibrated σ_A likelihood is legitimately soft relative to the
    # geometry prior, so it carries an intrinsic up-weight.
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
        verbose: int = 0,
        base_weight: float = None,
    ):
        super().__init__(verbose=verbose)
        self._dataset_collection = dataset_collection
        self._model_collection = model_collection
        self.add_module("_scaler", scaler)
        self.normalize = normalize
        self.use_work_set = use_work_set
        self.base_weight = (
            self.DEFAULT_BASE_WEIGHT if base_weight is None else float(base_weight)
        )

    def forward(self) -> torch.Tensor:
        dc = self._dataset_collection
        mc = self._model_collection

        tp_names = [n for n in mc.timepoint_names if n in dc]
        # Include the dark/reference dataset too — it is a real dataset to fit.
        all_keys = [mc.dark_key] + tp_names if mc.dark_key in dc else tp_names
        if not all_keys:
            return torch.tensor(0.0, device=mc.device)

        scaler = self._scaler
        if scaler is None or not hasattr(scaler, "get_beta"):
            raise RuntimeError(
                "CollectionMLSigmaATarget requires a scaler with get_beta(); "
                f"got {type(scaler).__name__}."
            )
        # One shared (beta, epsilon) for all datasets (common HKL).
        beta, eps = scaler.get_beta()

        total = torch.tensor(0.0, device=mc.device)
        for key in all_keys:
            data = dc[key]
            model = mc[key]
            hkl = data.hkl

            F_obs, _sigma, rfree, validity, centric = _unpack_masked_data(data)
            F_calc = torch.abs(_scale_fcalc(self._scaler, model(hkl), model))
            mask = validity & rfree if self.use_work_set else validity
            if centric is None:
                centric = torch.zeros(len(hkl), dtype=torch.bool, device=hkl.device)

            b = beta.to(F_obs.dtype)
            e = eps.to(F_obs.dtype) if eps is not None else None
            total = total + ml_xray_loss_beta_math(F_obs, F_calc, b, centric, mask, e)

        # Base weight drives refinement; applied on the work set only.
        if self.use_work_set:
            total = self.base_weight * total
        return total

    def maintenance(self) -> None:
        """Invalidate the scaler's shared beta so it is re-estimated from the
        updated models on the next forward (``LossState`` calls this after each
        optimizer-step block)."""
        scaler = self._scaler
        if scaler is not None and hasattr(scaler, "reset_beta_cache"):
            scaler.reset_beta_cache()

    def stats(self) -> Dict[str, StatEntry]:
        """Report shared beta diagnostics (low/high-resolution shell values)."""
        out: Dict[str, StatEntry] = {}
        scaler = self._scaler
        bb = getattr(scaler, "beta_per_bin", None)
        if bb is not None and bb.numel() > 0:
            out["beta_bin0"] = stat(bb[0].item(), VERBOSITY_STANDARD)
            out["beta_binN"] = stat(bb[-1].item(), VERBOSITY_STANDARD)
        return out
