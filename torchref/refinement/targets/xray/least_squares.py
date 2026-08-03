import torch
from typing import TYPE_CHECKING

from torchref.base.metrics import binwise_scale
from torchref.base.targets.xray_ls import ls_xray_loss_math

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class LeastSquaresXrayTarget(XrayTarget):
    """``--xray-mode ls``: ``L = 0.5 * sum w_i * (|F_obs| - k*|F_calc|)**2``, unit weights.

    ``k`` is owned by the attached :class:`Scaler` (per-bin scales, anisotropy, bulk
    solvent), fit separately from this target.

    **Unit weights.** ``sigma`` weighting used to be the default and was dropped as a
    duplicate: with ``w_i = 1/sigma_i**2`` this target is :class:`NLLXrayTarget` minus a
    parameter-independent constant, because
    ``nll = 0.5*d**2/sigma**2 + log sigma + 0.5*log 2pi`` over the *same*
    ``median(sigma)*0.1`` clamp. Measured on 1DAW, both reflection sets: the losses differ by
    exactly ``sum(log sigma_safe + 0.5*log 2pi)`` and the gradients are **bit-identical**, so
    ``--xray-mode ls`` and ``--xray-mode nll`` produced the same refinement trajectory and
    differed only in the number they reported. Unit weights are what make this row a distinct
    objective. ``weighting`` remains a parameter because the math layer supports both and
    ``tests/unit/refinement/test_xray_target_parity.py`` uses the sigma arm to keep that
    de-duplication argument checkable.

    ``ls_wunit_k1`` -- unit weights and a *self-owned* closed-form scale -- is
    :class:`UnitWeightK1XrayTarget` below. It used to be this same class configured with
    ``scale_mode="binwise_optimal"``, which meant two ``if self.scale_mode`` branches in two
    methods plus a factory that special-cased on the mode *name*. The two rows now differ
    only in who owns the scale, which is exactly one overridden hook.
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


class UnitWeightK1XrayTarget(LeastSquaresXrayTarget):
    """``--xray-mode ls_wunit_k1``: Phenix-style unit weights, one global K refit per call.

    The "k1" means "K_one" -- **one** K, not per-bin (Phenix's ``update_all_scales`` fits a
    single ``k_overall``, anisotropy off at d > 3 A). ``n_bins=1`` collapses the closed-form
    ``c[bins]`` to a scalar, matching Phenix exactly.

    At every forward call the closed-form optimal scale
    ``c = sum|F_obs|*|F_calc| / sum|F_calc|**2`` is computed over the work set and applied to
    ``|F_calc|`` before the LS sum. ``c`` is ``.detach()``-ed first, so the gradient w.r.t.
    atom positions treats it as a constant for that step -- matching Phenix, where
    ``update_all_scales`` refits K *before* each rigid-body LBFGS macro-cycle and freezes it
    during. Without the detach the loss is the envelope of all scaled losses and the gradient
    is biased toward different local minima than Phenix's frozen-K objective (observed on
    9RTS).

    ``weighting`` is **forced**, not defaulted: the pre-split code hardcoded ``"unit"`` inside
    the loss call while ``self.weighting`` remained a settable attribute, a discrepancy that
    was unreachable only because the factory happened to pass ``"unit"`` too.

    The caller must attach only scalers contributing ADDITIVE terms (e.g. bulk solvent) and
    **not** an overall ``K_overall x aniso`` multiplication, or ``F_calc`` is double-scaled.
    """

    def __init__(self, *args, **kwargs):
        kwargs["weighting"] = "unit"
        super().__init__(*args, **kwargs)
        self.n_bins = 1
        self._bins_cache: torch.Tensor = None
        self._bins_cache_dataid: int = None

    def _get_bins_cached(self) -> torch.Tensor:
        """Per-reflection bin indices, cached per ReflectionData instance.

        Note this *writes back* ``self.n_bins`` from ``get_bins``, which may return fewer
        bins than requested. Preserved deliberately -- ``n_bins`` is state, not a constant.
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
        """Detached closed-form scale. Argument order matches ``binwise_scale``'s.

        ``weights=None`` is unit weighting -- Phenix ``wunit`` semantics. ``valid=None``
        means all of the tensors passed, which on a compact work-set view is what is wanted.
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
        so a reported ``R_free`` uses the same work-fit ``c`` as ``R_work`` (the
        apples-to-apples Phenix convention). This is the only ``_scaled_F_calc_full``
        override in the x-ray family, and therefore the one R-factor path a change here can
        break without moving any loss value.
        """
        F_calc_full = super()._scaled_F_calc_full(fcalc=fcalc)
        full_bins = self._get_bins_cached()
        work = self._data.work
        c = self._binwise_scale(
            work.select(F_calc_full), work.F, work.select(full_bins)
        )
        return c[full_bins] * F_calc_full
