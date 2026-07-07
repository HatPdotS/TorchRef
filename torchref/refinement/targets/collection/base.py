"""Shared base for collection (multi-dataset) X-ray targets.

``CollectionXrayTarget`` gives the multi-dataset targets the same subset and
R-factor contract as the single-dataset :class:`~torchref.refinement.targets.xray.base.XrayTarget`:

* a canonical 3-way subset selector ``use_set`` in ``{"work", "free", "val"}``
  that maps onto each member :class:`~torchref.io.datasets.reflection_data.ReflectionData`'s
  ``data.work`` / ``data.free`` / ``data.validation`` accessors (validity mask
  applied, validation carved out of both work and free);
* one shared R-factor source of truth
  (:func:`~torchref.base.metrics.rfactor.rfactor_work_free`) computed per dataset
  through the scaler's scaling, exactly what the loss sees;
* a standard ``stats()`` dict (``loss`` / ``n`` / ``rwork`` / ``rfree``) so the
  refinement logger surfaces R-factors for the collection just like it does for a
  single dataset.

Because every member of a :class:`~torchref.io.datasets.collection.DatasetCollection`
is expanded onto one common HKL grid, per-dataset R-factors form a distribution;
the headline ``rwork`` / ``rfree`` is the median and the 10/25/75/90 percentiles are
reported at higher verbosity.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from torchref.base.metrics.rfactor import rfactor_work_free
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

    # ------------------------------------------------------------------
    # Dataset / model / subset plumbing
    # ------------------------------------------------------------------

    def _keys(self) -> List[str]:
        """Matched dataset keys this target fits: dark + present timepoints.

        Overridden by targets that fit only a subset of the collection (e.g.
        :class:`CollectionRiceTarget` drops the dark reference).
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
        """Clear cached forwards on every base model.

        Called at the start of each ``forward`` so a preceding no-grad R-factor
        evaluation (which may leave a detached, grad-less tensor in the cache)
        cannot starve the loss backward.
        """
        for bm in getattr(self._model_collection, "base_models", []):
            if hasattr(bm, "reset_cache"):
                bm.reset_cache()

    def _scaled_amp_full(self, data, model, recalc: bool = True) -> torch.Tensor:
        """Full-size, scaled ``|F_calc|`` for one data-model pair.

        Uses ``hkl_for_sf()`` (signed indices) so anomalous mates get distinct
        amplitudes. ``recalc=True`` (the default, used by the no-grad R-factor
        path) forces a fresh compute so it cannot reuse — or leave behind — a
        cached tensor that would break gradient flow; the loss path passes
        ``recalc=False`` after :meth:`_reset_model_caches` has cleared the cache.
        """
        fcalc = model(data.hkl_for_sf(), recalc=recalc)
        return torch.abs(_scale_fcalc(self._scaler, fcalc, model))

    # ------------------------------------------------------------------
    # R-factor reporting (shared source of truth)
    # ------------------------------------------------------------------

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
        t = torch.tensor(values, dtype=torch.float64)
        q = torch.quantile(t, torch.tensor(_R_PERCENTILES, dtype=torch.float64))
        return {lbl: q[i].item() for i, lbl in enumerate(_R_PCT_LABELS)}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

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
