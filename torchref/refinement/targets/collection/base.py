"""Shared observation access, reduction and reporting for collection X-ray targets.

Targets declare an amplitude or intensity observable. ``_loss_inputs`` gathers
observations, predictions, uncertainties and masks on the common HKL grid;
``_per_refl`` returns unreduced losses used by both ``forward`` and ``residuals``.
Difference targets intersect member masks so each fitted reflection is present
in every dataset's selected work, free or validation subset.

R-factors use the same scaled predictions as the loss. Reporting gives the median
across datasets, with the 10/25/75/90 percentiles at higher verbosity.
"""

from typing import TYPE_CHECKING, Dict, List, NamedTuple

import torch

from torchref.base.metrics.rfactor import rfactor_work_free
from torchref.base.targets.xray_likelihoods import (
    SIGMA_FLOOR_ABS,
    SIGMA_FLOOR_FRAC,
)
from torchref.config import get_float_dtype
from torchref.refinement.targets.base import Target
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from ._util import _scale_fcalc

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.scaler_base import ScalerBase


# Percentiles reported for the per-dataset R-factor distribution.
_R_PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_R_PCT_LABELS = ("p10", "p25", "p50", "p75", "p90")


class CollectionLossInputs(NamedTuple):
    """What a collection row's :meth:`CollectionXrayTarget._per_refl` reads.

    Every tensor is ``(N, n_hkl)`` on the collection's common HKL grid, with ``N`` the
    number of matched datasets in ``keys`` order -- full size rather than compact, because
    the members are already expanded onto one grid and a compact form would need a
    different index map per dataset.

    ``mask`` has already been intersected with finiteness of ``obs`` and ``sigma``, and
    those two have been substituted where non-finite. That order matters: masking the
    *loss* is not enough, because ``torch.where`` selects the finite branch for the value
    while still backpropagating NaN through the branch it discarded. Real reflection files
    carry non-finite intensities (excluded rows, and rows French-Wilson rejected), so this
    is load-bearing rather than defensive.
    """

    obs: torch.Tensor
    model: torch.Tensor
    sigma: torch.Tensor
    mask: torch.Tensor
    keys: List[str]


class CollectionSigmaALossInputs(NamedTuple):
    """:class:`CollectionLossInputs` plus one shared model-error estimate.

    The collection twin of
    :class:`~torchref.refinement.targets.xray.sigma_a.SigmaALossInputs`. ``beta`` and
    ``epsilon`` live on the **common HKL**, shape ``(n_hkl,)``, and broadcast over the
    dataset axis: they are fitted once on the pooled free reflections of every data-model
    pair, so one per-reflection variance serves every member.

    The two shapes never mix, because each class pairs its own ``_loss_inputs`` with its
    own ``_per_refl``.
    """

    obs: torch.Tensor
    model: torch.Tensor
    sigma: torch.Tensor
    mask: torch.Tensor
    keys: List[str]
    centric: torch.Tensor = None
    beta: torch.Tensor = None
    epsilon: torch.Tensor = None


class CollectionXrayTarget(Target):
    """Base class for multi-dataset X-ray targets.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection of reflection datasets keyed by timepoint name.
    model_collection : ModelCollection
        Collection of mixed models keyed by timepoint name.
    scaler : ScalerBase, optional
        Single scaler applied to every F_calc (``forward_mixed`` when available).
    use_work_set : bool, optional
        Legacy bool; superseded by ``use_set`` when the latter is given. Default
        True (work set).
    use_set : str, optional
        Canonical 3-way subset selector ``"work"``/``"free"``/``"val"``. Takes
        precedence over ``use_work_set``; derived from it if None.
    verbose : int, optional
        Verbosity level.
    """

    name: str = "collection_xray"

    #: Which measured column this row fits: ``"amplitude"`` or ``"intensity"``. Declared
    #: rather than passed, for the same reason as the single-dataset table -- see
    #: :mod:`torchref.refinement.targets.xray.observable`.
    observable: str = "amplitude"

    #: Fewest matched datasets for the loss to mean anything. The difference targets need
    #: two (there is no difference from a single dataset); the per-dataset rows need one.
    min_datasets: int = 1

    #: Multiplies the work-set loss. Rows carrying a likelihood whose magnitude differs
    #: from its siblings' set this so the term neither swamps nor is swamped by the
    #: restraints; :meth:`CollectionTwoMomentIntensityTarget.calibrate_base_weight` fits
    #: it against a reference target's gradient norm.
    base_weight: float = 1.0

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        scaler: "ScalerBase" = None,
        use_work_set: bool = True,
        use_set: str = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._dataset_collection = dataset_collection
        self._model_collection = model_collection
        self.add_module("_scaler", scaler)
        # Canonical 3-way subset selector, mirroring XrayTarget.__init__ so the
        # loss and the reported subset never disagree.
        if use_set is None:
            use_set = "work" if use_work_set else "free"
        self.use_set = use_set
        self.use_work_set = use_set == "work"

    def _keys(self) -> List[str]:
        """Matched dataset keys this target fits: dark + present timepoints. Targets
        fitting only part of the collection override it (a target fitting only the
        excited timepoints drops the dark reference).
        """
        dc = self._dataset_collection
        mc = self._model_collection
        keys = [mc.dark_key] if mc.dark_key in dc else []
        keys += [n for n in mc.timepoint_names if n in dc]
        return keys

    def _subset(self, data):
        """Return the ``_ReflectionSubset`` view selected by ``use_set``."""
        if self.use_set == "free":
            return data.free
        if self.use_set == "val":
            return data.validation
        return data.work

    def _reset_model_caches(self) -> None:
        """Clear cached forwards on every base model. Call at the start of each
        ``forward``: a preceding no-grad R-factor evaluation can leave a detached
        tensor in the cache, which would silently kill the loss backward.
        """
        for bm in getattr(self._model_collection, "base_models", []):
            if hasattr(bm, "reset_cache"):
                bm.reset_cache()

    def _scaled_amp_full(self, data, model, recalc: bool = True) -> torch.Tensor:
        """Full-size, scaled ``|F_calc|`` for one data-model pair, on the canonical
        ASU index with anomalous mates kept distinct. ``recalc=True`` (default) is for
        the no-grad R-factor path -- it neither reuses nor leaves a cached tensor that
        would break gradient flow. The loss path passes False, after
        ``_reset_model_caches()``.
        """
        fcalc = data.structure_factors(model, recalc=recalc)
        return torch.abs(_scale_fcalc(self._scaler, fcalc, model))

    def _stack_observations(self, keys: List[str]):
        """``(obs, sigma)``, each ``(N, n_hkl)``, in this row's observable.

        Collection accessors read live dataset views and name any member missing
        an intensity column.
        """
        dc = self._dataset_collection
        if self.observable == "intensity":
            return dc.stack_I_obs(keys), dc.stack_I_sigma(keys)
        return dc.stack_F_obs(keys), dc.stack_F_sigma(keys)

    def _stack_model(self, keys: List[str], recalc: bool = False) -> torch.Tensor:
        """The model prediction, ``(N, n_hkl)``, in this row's observable.

        Default is the per-dataset scaled amplitude (squared for an intensity row). Rows
        whose prediction is not a function of one dataset at a time -- the two-moment
        model, which mixes shared components across timepoints -- override this.
        """
        dc = self._dataset_collection
        mc = self._model_collection
        amp = torch.stack(
            [self._scaled_amp_full(dc[k], mc[k], recalc=recalc) for k in keys]
        )
        return amp**2 if self.observable == "intensity" else amp

    def _sigma_floor(self, sigma: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Floor for ``sigma``, at :data:`SIGMA_FLOOR_FRAC` of its median over ``mask``.

        A merged sigma can be reported as exactly zero; unfloored, one such reflection
        dominates the whole sum. Detached, because it is a numerical safeguard rather than
        a fitted quantity -- a gradient through a median would make the loss depend on the
        ordering of near-equal sigmas.

        Taken over the fitted rows only: the members are reindexed onto a common grid, so
        the rows a dataset does not own carry filler that would move the median.
        """
        selected = sigma[mask]
        if selected.numel() == 0:
            return torch.as_tensor(1e-6, device=sigma.device, dtype=sigma.dtype)
        floor = torch.median(selected).detach() * SIGMA_FLOOR_FRAC
        return floor.clamp(min=SIGMA_FLOOR_ABS)

    def _loss_inputs(self, recalc: bool = False) -> CollectionLossInputs:
        """Gather this row's observations, model, sigma and mask -- see
        :class:`CollectionLossInputs` for the shapes and the NaN discipline.

        Rows narrow the mask here (the difference targets require a reflection to be in
        the subset of every dataset) rather than inside :meth:`_per_refl`, so that
        :meth:`forward`'s sum and :meth:`residuals`' array agree on which reflections
        count.
        """
        keys = self._keys()
        obs, sigma = self._stack_observations(keys)
        model = self._stack_model(keys, recalc=recalc)
        mask = self._dataset_collection.stack_masks(keys, use_set=self.use_set)

        obs = obs.to(model.dtype)
        sigma = sigma.to(model.dtype)

        # Sanitise into the mask BEFORE the graph, not after: see CollectionLossInputs.
        valid = torch.isfinite(obs) & torch.isfinite(sigma)
        mask = mask & valid
        obs = torch.where(valid, obs, torch.zeros_like(obs))
        sigma = torch.where(valid, sigma, torch.ones_like(sigma))

        return CollectionLossInputs(obs, model, sigma, mask, keys)

    def _per_refl(self, ctx: CollectionLossInputs) -> torch.Tensor:
        """The likelihood, per reflection and **unreduced**, shape ``(N, n_hkl)``.

        One per selectable row; no row branches. :meth:`forward` is the masked sum of
        this.
        """
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Masked sum of :meth:`_per_refl`, with ``base_weight`` on the work set only.

        Cache reset first: a preceding no-grad ``stats()`` or ``get_rfactor()`` call can
        leave a detached tensor in a base model's cache, which would silently kill the
        loss backward.
        """
        keys = self._keys()
        if len(keys) < self.min_datasets:
            return torch.zeros((), device=self._dataset_collection.hkl.device)

        self._reset_model_caches()
        ctx = self._loss_inputs(recalc=False)
        total = (self._per_refl(ctx) * ctx.mask).sum()
        # Work set only: the free-set value is a diagnostic and has to stay comparable
        # across weightings.
        if self.use_work_set and self.base_weight != 1.0:
            total = self.base_weight * total
        return total

    def residuals(self) -> torch.Tensor:
        """:meth:`_per_refl` over every reflection, ``(N, n_hkl)``, unsummed and unmasked.

        The unreduced :meth:`forward`: same observable, same model, same variance. Masked
        reflections still get a value, so the array can be used to ask *why* one was
        excluded rather than only reflecting the answer back, and non-finite values
        survive because here a NaN is a finding rather than a nuisance.
        """
        keys = self._keys()
        if len(keys) < self.min_datasets:
            dc = self._dataset_collection
            return torch.zeros((0, len(dc.hkl)), device=dc.hkl.device)
        return self._per_refl(self._loss_inputs(recalc=True))

    def get_rfactor(self) -> Dict[str, object]:
        """Per-dataset R-work / R-free plus percentile summaries.

        Each dataset's R-factor is computed with
        :func:`~torchref.base.metrics.rfactor.rfactor_work_free` on the exact
        scaled ``|F_calc|`` the loss sees, so the collection cannot disagree with
        the single-dataset targets on convention. R-factors are unweighted (no
        ``base_weight``) and scale-invariant within a dataset.

        Returns
        -------
        dict
            ``{"per_dataset": {key: (rwork, rfree)},
               "rwork_pct": {label: value}, "rfree_pct": {label: value}}``.
            The percentile dicts are empty when no dataset contributed.
        """
        dc = self._dataset_collection
        mc = self._model_collection
        per_dataset: Dict[str, tuple] = {}
        rworks: List[float] = []
        rfrees: List[float] = []
        with torch.no_grad():
            for key in self._keys():
                data = dc[key]
                model = mc[key]
                amp = self._scaled_amp_full(data, model)
                rwork, rfree = rfactor_work_free(data, amp)
                per_dataset[key] = (rwork, rfree)
                rworks.append(rwork)
                rfrees.append(rfree)
        rwork_pct = self._percentiles(rworks)
        rfree_pct = self._percentiles(rfrees)
        return {
            "per_dataset": per_dataset,
            "rwork_pct": rwork_pct,
            "rfree_pct": rfree_pct,
        }

    @staticmethod
    def _percentiles(values: List[float]) -> Dict[str, float]:
        """10/25/50/75/90 percentiles of a list of per-dataset R-factors."""
        if not values:
            return {}
        dtype = get_float_dtype()
        t = torch.tensor(values, dtype=dtype)
        q = torch.quantile(t, torch.tensor(_R_PERCENTILES, dtype=dtype))
        return {lbl: q[i].item() for i, lbl in enumerate(_R_PCT_LABELS)}

    def _n_reflections(self) -> int:
        """Total reflections in this target's subset across all datasets."""
        dc = self._dataset_collection
        return int(sum(self._subset(dc[k]).n for k in self._keys()))

    def stats(self) -> Dict[str, StatEntry]:
        """Standard collection X-ray stats: loss / n / rwork / rfree (+percentiles).

        Headline ``rwork`` / ``rfree`` are the medians of the per-dataset
        distribution (``VERBOSITY_STANDARD``); the 10/25/75/90 percentiles and the
        per-dataset values are reported at higher verbosity. Subclasses that own
        extra diagnostics (e.g. shared beta) override this and merge on top.
        """
        out: Dict[str, StatEntry] = {}
        loss = self.forward()
        out["loss"] = stat(loss.item(), VERBOSITY_STANDARD)
        out["n"] = stat(self._n_reflections(), VERBOSITY_DEBUG)

        rf = self.get_rfactor()
        rwork_pct, rfree_pct = rf["rwork_pct"], rf["rfree_pct"]
        if rwork_pct:
            out["rwork"] = stat(rwork_pct["p50"], VERBOSITY_STANDARD)
            out["rfree"] = stat(rfree_pct["p50"], VERBOSITY_STANDARD)
            for lbl in _R_PCT_LABELS:
                if lbl == "p50":
                    continue
                out[f"rwork_{lbl}"] = stat(rwork_pct[lbl], VERBOSITY_DETAILED)
                out[f"rfree_{lbl}"] = stat(rfree_pct[lbl], VERBOSITY_DETAILED)
            for key, (rw, rfr) in rf["per_dataset"].items():
                out[f"rwork_{key}"] = stat(rw, VERBOSITY_DEBUG)
                out[f"rfree_{key}"] = stat(rfr, VERBOSITY_DEBUG)
        return out
