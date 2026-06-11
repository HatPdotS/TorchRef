"""Rice maximum-likelihood X-ray target driven by a differentiable model-error
variance ``sigma_m**2``.

This is the :class:`MaximumLikelihoodXrayTarget` Read-MLF Rice likelihood, but the
per-reflection conditional variance ``Sigma = epsilon * beta`` is built from the
*differentiable* Fisher model-error variance ``sigma_m**2`` (estimated from the
atomic B-factor distribution, reused from
:class:`~torchref.refinement.targets.xray.bhattacharyya.BhattacharyyaXrayTarget`)
instead of the free-set root-find ``beta``. Concretely ``beta := c * sigma_m**2``
with ``c`` a small, bounded, **co-refined** calibration owned by the Scaler
(``log_sigma_m_scale``).

The point: the model-error variance becomes a live, co-refinable function of the
model, usable in a single monolithic ``refine_joint`` optimization with no
macrocycle, no free-set estimation, and no ``maintenance()`` cache. Gradients reach
the B-factors through both ``F_calc`` and ``sigma_m**2``, and the calibration ``c``
is identifiable through the ``+log Sigma`` normalization term in the MLF kernel.
"""

from typing import TYPE_CHECKING, Dict

import torch

from torchref.base.targets.xray_ml_sigmaa import ml_xray_loss_beta_math
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

from .bhattacharyya import BhattacharyyaXrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class RiceSigmaMXrayTarget(BhattacharyyaXrayTarget):
    """Read-MLF Rice target with ``beta = c * sigma_m**2`` (differentiable, co-refined).

    Reuses the entire Fisher ``sigma_m`` machinery of
    :class:`BhattacharyyaXrayTarget` (cache, soft B-histogram, ``_sigma_m_sq_per_refl``)
    but, unlike that target, keeps ``sigma_m`` **in the autograd graph** and feeds
    ``c * sigma_m**2`` into the validated Read-MLF Rice likelihood
    (:func:`ml_xray_loss_beta_math`) in the slot the free-set ``beta`` normally
    occupies.

    Parameters
    ----------
    sigma_m_calib_bins : int, optional
        Degrees of freedom of the co-refined calibration ``c``. ``1`` (default) is
        a single global log-scale (safest, removes the cross-structure magnitude
        offset); ``> 1`` is a per-resolution-bin calibration (forced to the
        scaler's ``nbins``) that also absorbs a resolution-slope mismatch.

    See :class:`BhattacharyyaXrayTarget` for ``sigma_m_scale`` and ``b_grid_*``.
    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        use_work_set: bool = True,
        sigma_m_scale: float = 1.0,
        sigma_m_calib_bins: int = 1,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            sigma_m_scale=sigma_m_scale,
            verbose=verbose,
            **kwargs,
        )
        self.sigma_m_calib_bins = int(sigma_m_calib_bins)
        # Register the co-refined calibration NOW (before any optimizer snapshots
        # scaler.parameters()).
        if self._scaler is not None and hasattr(
            self._scaler, "ensure_sigma_m_calibration"
        ):
            self._scaler.ensure_sigma_m_calibration(self.sigma_m_calib_bins)

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        if not self._initialized:
            self._initialize_cache()

        F_obs, F_calc, _sigma_d, centric, sub = self.get_data(fcalc=fcalc)

        # Differentiable model-error variance (NO no_grad): gradients flow to B.
        sigma_m_sq = sub.select(self._sigma_m_sq_per_refl())

        scaler = self._scaler
        calib = sub.select(scaler.get_sigma_m_calibration()).to(F_obs.dtype)
        eps = sub.select(scaler.get_epsilon()).to(F_obs.dtype)

        beta = (calib * sigma_m_sq).clamp(min=1e-10)
        return ml_xray_loss_beta_math(F_obs, F_calc, beta, centric, epsilon=eps)

    def maintenance(self) -> None:
        """No-op: nothing to refresh between optimizer blocks.

        ``sigma_m`` and the calibration are live in the autograd graph and
        recomputed each forward — there is no free-set ``beta`` cache to
        invalidate (the whole point of this target)."""
        pass

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """Inherit the sigma_m/sigma_d diagnostics; add the calibration value."""
        base = super().stats(fcalc=fcalc)
        scaler = self._scaler
        c = getattr(scaler, "log_sigma_m_scale", None)
        if c is not None:
            with torch.no_grad():
                base["sigma_m_calib_mean"] = stat(
                    torch.exp(c).mean().item(), VERBOSITY_STANDARD
                )
        return base
