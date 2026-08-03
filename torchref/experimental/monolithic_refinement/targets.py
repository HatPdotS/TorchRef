"""Rice X-ray target driven by a differentiable, co-refined model-error variance.

EXPERIMENTAL. This is the :class:`MLNoAlphaXrayTarget` Read-MLF Rice
likelihood, but the per-reflection conditional variance ``Sigma = epsilon * beta``
is built from the *differentiable* Fisher model-error variance ``sigma_m**2``
(estimated from the atomic B-factor distribution, reused from
the retired ``BhattacharyyaXrayTarget``)
instead of the free-set root-find ``beta``. Concretely ``beta := c * sigma_m**2``
with ``c`` a small, bounded, **co-refined** calibration *owned by this target*
(``log_sigma_m_scale``).

The point: the model-error variance becomes a live, co-refinable function of the
model, usable in a single monolithic optimization with no macrocycle, no free-set
estimation, and no ``maintenance()`` cache. Gradients reach the B-factors through
both ``F_calc`` and ``sigma_m**2``, and the calibration ``c`` is identifiable
through the ``+log Sigma`` normalization term in the MLF kernel.

Self-contained (experimental): the calibration lives on the target (not the core
Scaler) and is collected into the joint optimizer by
:class:`~torchref.experimental.monolithic_refinement.refinement.MonolithicRefinement`.
"""

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn

from torchref.base.targets.xray_likelihoods import complex_var_from_beta, rice_math
from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl
from torchref.config import get_float_dtype
from torchref.refinement.model_error_estimation.sigma_m import SigmaMEstimator
from torchref.refinement.targets.xray.base import XrayTarget
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class RiceSigmaMXrayTarget(XrayTarget):
    """Read-MLF Rice target with ``beta = c * sigma_m**2`` (differentiable, co-refined).

    Owns a :class:`~torchref.refinement.model_error_estimation.sigma_m.SigmaMEstimator` (the Fisher
    machinery extracted from the retired Bhattacharyya target: cache, soft B-histogram,
    per-reflection variance) and keeps ``sigma_m`` **in the autograd graph**, feeding
    ``c * sigma_m**2`` into the validated Read-MLF Rice likelihood
    (:func:`~torchref.base.targets.xray_likelihoods.rice_math`) in the slot the
    free-set ``beta`` normally
    occupies.

    Parameters
    ----------
    sigma_m_calib_bins : int, optional
        Degrees of freedom of the co-refined calibration ``c``. ``1`` (default) is
        a single global log-scale (safest, removes the cross-structure magnitude
        offset); ``> 1`` is a per-resolution-bin calibration (forced to the
        scaler's ``nbins``) that also absorbs a resolution-slope mismatch.
    shared_log_sigma_m_scale : torch.nn.Parameter, optional
        If given, use this calibration parameter instead of creating a new one.
        Used to share one ``c`` between the work and test targets so the R-free
        statistics see the same calibration the work loss refines.

    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        use_work_set: bool = True,
        sigma_m_calib_bins: int = 1,
        shared_log_sigma_m_scale: Optional[nn.Parameter] = None,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            verbose=verbose,
            **kwargs,
        )
        # The structure-driven model-error estimator, extracted from the retired
        # Bhattacharyya target. It returns an UNSCALED variance: this target refines its
        # own calibration (`log_sigma_m_scale`), so a second scale applied underneath
        # would be degenerate with it.
        #
        # A `sigma_m_scale` constructor argument used to sit here, documented as "the
        # initial value of that calibration". It was never used: the calibration is
        # initialised to `zeros` below regardless, so the comment was false and the whole
        # thread from `--sigma-m-scale` down was dead end to end -- the factory swallowed it
        # too. Removed 2026-08 along with the CLI flag.
        self._sigma_m = SigmaMEstimator()
        self.sigma_m_calib_bins = int(sigma_m_calib_bins)
        if shared_log_sigma_m_scale is not None:
            # Share the work target's calibration (test target reads it).
            self.log_sigma_m_scale = shared_log_sigma_m_scale
        else:
            n = 1
            if self.sigma_m_calib_bins > 1 and scaler is not None:
                n = int(getattr(scaler, "nbins", 1))
            # Place on the scaler's device so it lines up with the data/F_calc.
            device = None
            if scaler is not None and getattr(scaler, "_s_half_sq", None) is not None:
                device = scaler._s_half_sq.device
            self.log_sigma_m_scale = nn.Parameter(
                torch.zeros(n, dtype=get_float_dtype(), device=device)
            )
        # Cached per-reflection multiplicity (model-independent), filled lazily.
        self._eps_cache = None

    # ------------------------------------------------------------------
    # Calibration / epsilon helpers (self-contained — no core Scaler edits)
    # ------------------------------------------------------------------

    def _calibration_per_refl(self) -> torch.Tensor:
        """Differentiable per-reflection multiplier ``c(h) = exp(log_sigma_m_scale)``."""
        c = torch.exp(self.log_sigma_m_scale.clamp(min=-15.0, max=15.0))
        n_full = len(self._scaler.bins)
        if c.numel() == 1:
            return c.expand(n_full)
        # per-bin: index by the scaler's resolution bins (differentiable in c).
        return c[self._scaler.bins]

    def _epsilon_per_refl(self) -> torch.Tensor:
        if self._eps_cache is None:
            sg = getattr(self._data, "spacegroup", None)
            self._eps_cache = epsilon_from_hkl(self._data.hkl, sg).to(
                self._scaler._s_half_sq.dtype
            )
        return self._eps_cache

    # ------------------------------------------------------------------

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        F_obs, F_calc, _sigma_d, centric, sub = self.get_data(fcalc=fcalc)

        # Differentiable model-error variance (NO no_grad): gradients flow to B. The
        # estimator deliberately does not wrap this itself -- that choice is the
        # caller's, and this caller needs the gradient.
        self._sigma_m.prepare(
            self._scaler._s_half_sq,
            self._data.get_corrected_data()[1],
            self._data.masks(),
            *self._model.get_scattering_params_iso(),
        )
        b_iso = self._model.adp()[self._model._iso_indices]
        sigma_m_sq = sub.select(self._sigma_m.sigma_m_sq(b_iso))
        calib = sub.select(self._calibration_per_refl()).to(F_obs.dtype)
        eps = sub.select(self._epsilon_per_refl()).to(F_obs.dtype)

        beta = (calib * sigma_m_sq).clamp(min=1e-10)
        return rice_math(F_obs, F_calc, complex_var_from_beta(beta, eps), centric)

    def maintenance(self) -> None:
        """No-op: nothing to refresh between optimizer blocks.

        ``sigma_m`` and the calibration are live in the autograd graph and
        recomputed each forward — there is no free-set ``beta`` cache to
        invalidate (the whole point of this target)."""
        pass

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """Inherit the sigma_m/sigma_d diagnostics; add the calibration value."""
        base = super().stats(fcalc=fcalc)
        with torch.no_grad():
            base["sigma_m_calib_mean"] = stat(
                torch.exp(self.log_sigma_m_scale).mean().item(), VERBOSITY_STANDARD
            )
        return base
