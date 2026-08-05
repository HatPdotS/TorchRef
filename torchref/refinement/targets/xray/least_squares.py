"""The two least-squares X-ray targets, differing only in who owns the scale."""

import torch
from typing import TYPE_CHECKING

from torchref.base.metrics import binwise_scale
from torchref.base.targets.xray_ls import ls_per_refl, ls_xray_loss_math

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class LeastSquaresXrayTarget(XrayTarget):
    """``--xray-mode ls``: ``L = 0.5 * sum w_i * (|F_obs| - k*|F_calc|)**2``, unit weights.

    ``k`` belongs to the attached :class:`Scaler` (per-bin scales, anisotropy, bulk
    solvent), fit separately from this target.

    The unit weights are what make this a distinct objective: at ``w_i = 1/sigma_i**2``
    this target is :class:`NLLXrayTarget` minus a parameter-independent constant, with
    **bit-identical gradients**, so ``weighting="sigma"`` would give the same refinement
    trajectory as ``--xray-mode nll`` and only report a different number. The parameter
    survives to keep the math layer's second arm reachable; it is not selectable as a mode.

    :class:`UnitWeightK1XrayTarget` below is the ``ls_wunit_k1`` row -- unit weights and a
    *self-owned* closed-form scale. The two differ in exactly one overridden hook.
    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        weighting: str = "unit",
        use_work_set: bool = True,
        verbose: int = 0,
        use_set: str = None,
        device=None,
        sigma_a=None,
    ):
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            verbose=verbose,
            use_set=use_set,
            device=device,
            sigma_a=sigma_a,
        )
        self.weighting = weighting

    def _scaled_amplitudes(
        self, F_calc: torch.Tensor, F_obs: torch.Tensor, sub
    ) -> torch.Tensor:
        """The amplitudes the LS sum sees. Here the Scaler has already scaled them.

        The one hook that distinguishes this row from ``ls_wunit_k1``.
        """
        return F_calc

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Weighted least-squares loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead of computing
            from the model.
        """
        # 5th element of get_data is the ``_ReflectionSubset`` view, not a mask.
        # F_obs/F_calc are already compact (subset-applied) so the downstream kernel
        # needs no mask.
        F_obs, F_calc, sigma, _, sub = self.get_data(fcalc=fcalc)
        return ls_xray_loss_math(
            F_obs,
            self._scaled_amplitudes(F_calc, F_obs, sub),
            sigma,
            mask=None,
            weighting=self.weighting,
        )

    def _per_refl(self, ctx) -> torch.Tensor:
        """The eager twin of :meth:`forward`'s fused kernel; see
        :meth:`~torchref.refinement.targets.xray.nll.NLLXrayTarget._per_refl`.

        Goes through :meth:`_scaled_F_calc_full` rather than
        :meth:`_scaled_amplitudes` because the two disagree for the ``ls_wunit_k1``
        row, whose closed-form scale is fit on **whatever view it is handed**. On the
        full-reflection view that would fit the scale to the free set as well, which
        is neither what the loss saw nor what the R-factor uses.
        """
        F_obs, _, sigma, _, sub = ctx
        F_calc = sub.select(self._scaled_F_calc_full())
        return ls_per_refl(F_obs, F_calc, sigma, weighting=self.weighting)


class UnitWeightK1XrayTarget(LeastSquaresXrayTarget):
    """``--xray-mode ls_wunit_k1``: Phenix-style unit weights, one global K refit per call.

    "k1" is "K_one" -- **one** K, not per-bin, as Phenix's ``update_all_scales`` fits a
    single ``k_overall`` (anisotropy off at d > 3 A); ``n_bins=1`` collapses the
    closed-form ``c[bins]`` to that scalar.

    Every forward recomputes ``c = sum|F_obs|*|F_calc| / sum|F_calc|**2`` on the work set
    and applies it to ``|F_calc|`` before the LS sum. ``c`` is ``.detach()``-ed, so the
    coordinate gradient sees it as constant for the step -- as in Phenix, which refits K
    *between* macro-cycles and freezes it within. Without the detach the loss becomes the
    envelope of all scaled losses and the gradient prefers different local minima.

    ``weighting`` is **forced** to ``"unit"``, not defaulted.

    Attach only scalers contributing ADDITIVE terms (e.g. bulk solvent) -- an overall
    ``K_overall x aniso`` multiplication would double-scale ``F_calc``.
    """

    def __init__(self, *args, **kwargs):
        kwargs["weighting"] = "unit"
        super().__init__(*args, **kwargs)
        self.n_bins = 1
        self._bins_cache: torch.Tensor = None
        self._bins_cache_dataid: int = None

    def _get_bins_cached(self) -> torch.Tensor:
        """Per-reflection bin indices, cached per ReflectionData instance. Deliberately
        *writes back* ``self.n_bins``, which ``get_bins`` may lower: ``n_bins`` is state.
        """
        dataid = id(self._data)
        if self._bins_cache is None or self._bins_cache_dataid != dataid:
            bins, n_bins = self._data.get_bins(n_bins=self.n_bins)
            device = self._data.hkl.device if hasattr(self._data, "hkl") else bins.device
            self._bins_cache = bins.to(device=device)
            self.n_bins = n_bins
            self._bins_cache_dataid = dataid
        return self._bins_cache

    def _binwise_scale(self, F_calc, F_obs, bins) -> torch.Tensor:
        """Detached closed-form scale; argument order matches ``binwise_scale``'s.
        ``weights=None`` is Phenix ``wunit`` semantics and ``valid=None`` is everything
        passed -- correct here only because the input is already a compact work-set view.
        """
        return binwise_scale(
            F_calc, F_obs, bins, valid=None, nbins=self.n_bins, weights=None
        ).detach()

    def _scaled_amplitudes(self, F_calc, F_obs, sub) -> torch.Tensor:
        # `_get_bins_cached` returns FULL-data bins, so select via the subset indices.
        bins = sub.select(self._get_bins_cached())
        return self._binwise_scale(F_calc, F_obs, bins)[bins] * F_calc

    def _scaled_F_calc_full(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Full-size ``|F_calc|`` under this target's own scale.

        The closed-form scale is fit on the **work** set and applied to **all** reflections,
        so ``R_free`` uses the same work-fit ``c`` as ``R_work`` (Phenix convention). The
        only ``_scaled_F_calc_full`` override in the family, so an edit here moves reported
        R-factors without moving any loss value.
        """
        F_calc_full = super()._scaled_F_calc_full(fcalc=fcalc)
        full_bins = self._get_bins_cached()
        work = self._data.work
        c = self._binwise_scale(
            work.select(F_calc_full), work.F, work.select(full_bins)
        )
        return c[full_bins] * F_calc_full
