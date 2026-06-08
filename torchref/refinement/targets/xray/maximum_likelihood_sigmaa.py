"""Maximum-likelihood (Read MLF) X-ray target with Luzzati model-error variance.

Thin target: the per-resolution ``beta`` (absolute model-error variance) is
estimated by maximum likelihood and **owned by the Scaler**
(``scaler.get_beta()``), lazily cached and invalidated via this target's
:meth:`maintenance` hook. The loss is the Read MLF form (``mean = |Fc|``,
variance ``epsilon*beta``) implemented in
:mod:`torchref.base.targets.xray_ml_sigmaa`. The Luzzati ``alpha`` mean-shift was
removed after it was shown to be gauge-absorbed by the co-refined scaler.

The scaler returns ``beta``/``epsilon`` detached, so they act as constants in the
model autograd graph; gradients reach the model only through ``F_calc``.
"""

from typing import TYPE_CHECKING, Dict

import torch

from torchref.base.targets.xray_ml_sigmaa import ml_xray_loss_beta_math
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class MaximumLikelihoodSigmaAXrayTarget(XrayTarget):
    """Read MLF target; reads the ML model-error variance ``beta`` from the Scaler.

    The model mean is ``|F_calc|`` (the Luzzati ``alpha`` mean-shift was removed
    after it was shown to be gauge-absorbed by the co-refined scaler); the
    conditional variance is ``epsilon * beta``, with ``beta`` the per-shell
    Luzzati model-error variance estimated on the FREE set.
    """

    # The correctly-calibrated Read-MLF likelihood is legitimately "soft"
    # relative to the geometry prior, so it needs an intrinsic up-weight to be on
    # equal footing (empirically ~10x; the residual imbalance is a count/prior
    # effect, not a beta error -- see the floor investigation). Carry that as a
    # base weight on the target itself.
    # TODO(weighting): this is a stopgap. The base weight belongs in the weighting
    # infrastructure (LossState / component_weighting per-target base weight that
    # used to exist), not multiplied inside the target's forward. Move it back
    # once that infra is restored, ideally replaced by a principled per-cycle
    # gradient-ratio (wxc-style) weight.
    DEFAULT_BASE_WEIGHT = 10.0

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        use_work_set: bool = True,
        verbose: int = 0,
        base_weight: float = None,
        **kwargs,
    ):
        kwargs.pop("sigma_mode", None)
        kwargs.pop("sigma_m_scale", None)  # bhattacharyya-only, ignored here
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            sigma_mode="raw",
            verbose=verbose,
        )
        self.base_weight = (
            self.DEFAULT_BASE_WEIGHT if base_weight is None else float(base_weight)
        )

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        F_obs, F_calc, sigma, centric, sub = self.get_data(fcalc=fcalc)

        scaler = self._scaler
        if not hasattr(scaler, "get_beta"):
            raise RuntimeError(
                "ml_sigmaa target requires a Scaler with get_beta(); got "
                f"{type(scaler).__name__}."
            )
        # beta/epsilon: lazily estimated + cached on the scaler, detached.
        # They are full-size (per-reflection) — restrict to this subset.
        beta, eps = scaler.get_beta(fcalc)
        beta = sub.select(beta).to(F_obs.dtype)
        eps = sub.select(eps).to(F_obs.dtype) if eps is not None else None

        loss = ml_xray_loss_beta_math(F_obs, F_calc, beta, centric, epsilon=eps)
        # TODO(weighting): base weight applied inside the target as a stopgap;
        # should live in LossState/component_weighting (see class docstring).
        # Only the work set drives refinement; leave the test instance (R-free /
        # NLL_test monitoring) unscaled so reported metrics stay comparable.
        if self.use_work_set:
            loss = self.base_weight * loss
        return loss

    def maintenance(self) -> None:
        """Invalidate the scaler's cached beta so it is re-estimated from
        the updated model on the next forward. ``LossState`` calls this after
        each optimizer-step block (see ``Target.maintenance``)."""
        scaler = self._scaler
        if scaler is not None and hasattr(scaler, "reset_beta_cache"):
            scaler.reset_beta_cache()

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """Add beta diagnostics (low/high-resolution shell values)."""
        base = super().stats(fcalc=fcalc)
        scaler = self._scaler
        bb = getattr(scaler, "beta_per_bin", None)
        if bb is not None and bb.numel() > 0:
            base["beta_bin0"] = stat(bb[0].item(), VERBOSITY_STANDARD)
            base["beta_binN"] = stat(bb[-1].item(), VERBOSITY_STANDARD)
        return base
