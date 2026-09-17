"""Collection (multi-dataset) X-ray targets.

One X-ray likelihood across a paired ``DatasetCollection`` + ``ModelCollection``, keys
matched so each timepoint dataset meets its own mixed model:

:class:`CollectionDifferenceTarget`
    Mean-based differences on amplitudes; the primary optimization driver.
:class:`CollectionDifferenceIntensityTarget`
    The same on intensities -- the whole class is one ``observable`` declaration.
:class:`CollectionMLTarget`
    Read MLF per dataset at one shared Luzzati ``beta``, pooled over every dataset's free
    reflections and owned by the target rather than the scaler. The absolute channel.

All of them get their observations, model, mask and likelihood seam from
:class:`~torchref.refinement.targets.collection.base.CollectionXrayTarget`, so each row is
a ``_per_refl`` and nothing else. The selectable set is
:data:`~torchref.refinement.targets.collection._specs.COLLECTION_XRAY_TARGETS`.

"""

from typing import TYPE_CHECKING, Dict

import torch

from torchref.base.reciprocal import get_scattering_vectors
from torchref.base.targets.xray_likelihoods import (
    complex_var_from_beta,
    gaussian_per_refl,
    rice_per_refl,
)
from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator, epsilon_from_hkl
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

from .base import CollectionSigmaALossInputs, CollectionXrayTarget

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

    #: There is no difference from a single dataset.
    min_datasets: int = 2

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

    def _loss_inputs(self, recalc: bool = False):
        """The base's stack, with the mask narrowed across datasets.

        A reflection counts only if it is in this target's subset in **every** dataset:
        the per-reflection mean ties them together, so a reflection missing from one
        member would silently shift the reference for all the others. Narrowed here
        rather than inside :meth:`_per_refl` so ``forward``'s sum and ``residuals``'
        array agree on which reflections count.
        """
        ctx = super()._loss_inputs(recalc=recalc)
        mask_all = ctx.mask.all(dim=0, keepdim=True).expand_as(ctx.mask)
        return ctx._replace(mask=mask_all)

    def _per_refl(self, ctx) -> torch.Tensor:
        """Gaussian NLL of the difference-from-mean, per reflection and unreduced."""
        N = len(ctx.keys)
        mask_all = ctx.mask[0]  # (n_hkl,) -- every row is the same after _loss_inputs

        delta_obs = ctx.obs - ctx.obs.mean(dim=0)
        delta_calc = ctx.model - ctx.model.mean(dim=0)

        # Var(F_i - F_mean) = sigma_i^2 (1 - 2/N) + (sum_j sigma_j^2) / N^2
        sum_sigma_sq = (ctx.sigma**2).sum(dim=0)
        sigma_diff_sq = ctx.sigma**2 * (1 - 2.0 / N) + sum_sigma_sq / (N**2)
        sigma_diff = torch.sqrt(sigma_diff_sq.clamp(min=1e-12))

        # Mask via torch.where, not boolean indexing: no nonzero() device sync.
        delta_obs = torch.where(mask_all, delta_obs, torch.zeros_like(delta_obs))
        delta_calc = torch.where(mask_all, delta_calc, torch.zeros_like(delta_calc))
        sigma_diff = torch.where(mask_all, sigma_diff, torch.ones_like(sigma_diff))

        # Floored on the PROPAGATED difference sigma, not the raw measurement sigma --
        # that is the quantity dividing the residual here.
        sigma_safe = sigma_diff.clamp(min=self._sigma_floor(sigma_diff, ctx.mask))

        nll = gaussian_per_refl(delta_obs, delta_calc, sigma_safe**2, var_floor=0.0)
        # A single NaN would poison the whole gradient; 1e6 lets the step be rejected.
        return torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))


class CollectionDifferenceIntensityTarget(CollectionDifferenceTarget):
    """The difference-from-mean target on **intensities** instead of amplitudes.

    A one-line row: the base reads ``I``/``sigI`` and predicts ``|F_calc|**2``, and the
    difference-from-mean algebra is observable-agnostic --
    ``Var(x_i - x_mean) = sigma_i^2 (1 - 2/N) + (sum_j sigma_j^2)/N^2`` holds for any
    quantity whose members share a mean. So the whole row is the ``observable``
    declaration, which is the point of having the axis at all.

    Prefer it over :class:`CollectionDifferenceTarget` when the difference signal is weak
    relative to the measurement error, because ``F_obs`` on a merged dataset is a
    French-Wilson posterior: strictly positive, so it reshapes exactly the weak reflections
    a small difference lives in, and it cannot represent a negative intensity at all.
    Prefer the amplitude row when the output difference *map* is the product, since the DED
    coefficients are amplitudes and keeping the loss and the map in one space is one fewer
    conversion to get wrong.

    Both rows are offered rather than one chosen: which wins is a property of a dataset's
    signal-to-noise, not something to settle once in the library.
    """

    name: str = "difference_intensity_xray"
    observable: str = "intensity"


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

    def _loss_inputs(self, recalc: bool = False):
        """The base's stack, plus the shared model-error variance this row needs.

        ``beta`` and ``epsilon`` are fitted **once** on the pooled free reflections of
        every data-model pair and mapped onto the common HKL, so one per-reflection
        variance serves every dataset -- and since they live on that common grid they
        broadcast over the dataset axis rather than being tiled.

        Detached, so gradients reach the models only through ``F_calc``. Fitted on the
        free set (``data.free`` excludes validation), and cached until
        :meth:`maintenance` resets it.
        """
        ctx = super()._loss_inputs(recalc=recalc)
        dc = self._dataset_collection
        eps_common, dss_common = self._common_geom()
        dtype = ctx.obs.dtype

        centric = dc.get_centric_flags()
        if centric is None:
            centric = torch.zeros(
                ctx.obs.shape[-1], dtype=torch.bool, device=ctx.obs.device
            )

        n_ds = len(ctx.keys)
        free = torch.cat([dc[k].free.mask for k in ctx.keys])
        est = self._sigma_a.get(
            ctx.obs.reshape(-1).to(dtype),
            # beta needs no gradient.
            ctx.model.detach().reshape(-1).to(dtype),
            centric.repeat(n_ds),
            eps_common.to(dtype).repeat(n_ds),
            dss_common.to(dtype).repeat(n_ds),
            free,
            out_epsilon=eps_common.to(dtype),
            target_dss=dss_common,
            # Always passed, as at every other call site: it is what makes sigma_A the
            # correlation with the noise-free amplitudes rather than with the noisy data.
            sigma_obs=ctx.sigma.reshape(-1).to(dtype),
        )
        return CollectionSigmaALossInputs(
            *ctx,
            centric=centric,
            beta=est.beta.to(dtype),
            epsilon=None if est.epsilon is None else est.epsilon.to(dtype),
        )

    def _per_refl(self, ctx) -> torch.Tensor:
        """Read MLF per reflection: Rice for acentrics, folded normal for centrics.

        TOTAL variance (``est.beta``, not ``beta_model``): this likelihood does not
        account for ``sigma_obs`` itself, so the measurement variance stays inside
        ``beta``.
        """
        Sigma = complex_var_from_beta(ctx.beta, ctx.epsilon)
        nll = rice_per_refl(ctx.obs, ctx.model, Sigma, ctx.centric)
        # A single NaN would poison the whole gradient; 1e6 lets the step be rejected.
        return torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))

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
