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
    """
    Least Squares target function.
    L_LS = Σ w_i * (|F_obs| - k * |F_calc|)²

    Two scale modes are supported:

    * ``scale_mode="scaler"`` (default): expects an attached ``Scaler`` that
      computes ``k * |F_calc|`` (per-bin scales, anisotropy, bulk solvent).
      The scaler is fit separately from the target.
    * ``scale_mode="binwise_optimal"``: no external scaler. At every forward
      call, compute the closed-form per-bin optimal scale
      ``c_b = Σ|F_obs|·|F_calc| / Σ|F_calc|²`` (over the work set) and
      apply ``c[bins]`` to ``|F_calc|`` before the LS sum. ``c`` is
      ``.detach()``-ed before being applied so the gradient w.r.t.
      atom positions treats ``c`` as a constant for that step — matching
      Phenix's ``ls_wunit_k1`` where ``update_all_scales`` refits ``K``
      *before* each rigid-body LBFGS macro-cycle and freezes it during.
      Without the detach, the loss is the envelope of all scaled losses
      and the gradient is biased toward different local minima than
      Phenix's frozen-K objective (observed on 9RTS).
    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        weighting: str = "sigma",
        use_work_set: bool = True,
        scale_mode: str = "scaler",
        n_bins: int = 20,
        verbose: int = 0,
        use_set: str = None,
        device=None,
    ):
        if scale_mode not in ("scaler", "binwise_optimal"):
            raise ValueError(
                f"scale_mode must be 'scaler' or 'binwise_optimal', got "
                f"{scale_mode!r}"
            )
        # In binwise_optimal mode the closed-form per-bin scale c[bins] owns
        # the overall scaling. The caller is responsible for passing only
        # scalers that contribute ADDITIVE terms (e.g. bulk solvent) and do
        # NOT apply an overall K_overall × aniso multiplication; otherwise
        # F_calc gets double-scaled.
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            verbose=verbose,
            use_set=use_set,
            device=device,
        )
        self.weighting = weighting
        self.scale_mode = scale_mode
        self.n_bins = n_bins
        self._bins_cache: torch.Tensor = None
        self._bins_cache_dataid: int = None

    def _get_bins_cached(self) -> torch.Tensor:
        """Per-reflection bin indices, cached per ReflectionData instance."""
        dataid = id(self._data)
        if self._bins_cache is None or self._bins_cache_dataid != dataid:
            bins, n_bins = self._data.get_bins(n_bins=self.n_bins)
            device = self._data.hkl.device if hasattr(self._data, "hkl") else bins.device
            self._bins_cache = bins.to(device=device)
            self.n_bins = n_bins
            self._bins_cache_dataid = dataid
        return self._bins_cache

    def _scaled_F_calc_full(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Full-size ``|F_calc|`` under the active scale.

        For ``scale_mode="scaler"`` this is just the scaler-scaled amplitude
        (base behaviour). For ``binwise_optimal`` the closed-form per-bin scale
        is fit on the **work** set and applied to **all** reflections, so a
        reported ``R_free`` uses the same work-fit ``c`` as ``R_work`` (the
        apples-to-apples Phenix convention).
        """
        F_calc_full = super()._scaled_F_calc_full(fcalc=fcalc)
        if self.scale_mode != "binwise_optimal":
            return F_calc_full
        full_bins = self._get_bins_cached()
        work = self._data.work
        c = binwise_scale(
            work.select(F_calc_full),
            work.F,
            work.select(full_bins),
            valid=None,
            nbins=self.n_bins,
            weights=None,
        ).detach()
        return c[full_bins] * F_calc_full

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute least squares loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Weighted least squares loss.
        """
        # 5th element of get_data is the ``_ReflectionSubset`` view, not a
        # mask. F_obs/F_calc are already compact (subset-applied) so the
        # downstream kernel needs no mask. For binwise_scale, the "valid"
        # selector defaults to all reflections — since we're already on the
        # work-set-restricted compact view, that's exactly what we want.
        F_obs, F_calc, sigma, _, _ = self.get_data(fcalc=fcalc)

        if self.scale_mode == "binwise_optimal":
            # bins must be the compact-aligned bins; ``_get_bins_cached``
            # returns full-data bins, so select via the work-set indices.
            sub = self._subset()
            full_bins = self._get_bins_cached()
            bins = sub.select(full_bins)
            # .detach() so the gradient w.r.t. θ treats c as a constant —
            # gives the same gradient as Phenix's "K from update_all_scales,
            # frozen during LBFGS" pattern. Without detach, c(θ) flows
            # gradients and the loss becomes an envelope objective whose
            # local minima differ from Phenix's.
            c = binwise_scale(
                F_calc, F_obs, bins,
                valid=None,
                nbins=self.n_bins,
                weights=None,  # unit weights — Phenix wunit semantics
            ).detach()
            F_calc = c[bins] * F_calc
            return ls_xray_loss_math(
                F_obs, F_calc, sigma, mask=None, weighting="unit"
            )

        return ls_xray_loss_math(
            F_obs, F_calc, sigma, mask=None, weighting=self.weighting
        )
