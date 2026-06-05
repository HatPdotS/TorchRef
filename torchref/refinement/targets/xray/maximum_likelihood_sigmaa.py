"""Maximum-likelihood (Read MLF) X-ray target with Luzzati alpha/beta sigma_A.

Thin target: the per-resolution ``alpha`` (mean coupling) and ``beta`` (absolute
model-error variance) are estimated by maximum likelihood and **owned by the
Scaler** (``scaler.get_alpha_beta()``), lazily cached and invalidated via this
target's :meth:`maintenance` hook. The loss is the Read MLF form
(``mean = alpha*|Fc|``, variance ``epsilon*beta``) implemented in
:mod:`torchref.base.targets.xray_ml_sigmaa`.

The scaler returns ``alpha``/``beta``/``epsilon`` detached, so they act as
constants in the model autograd graph; gradients reach the model only through
``F_calc``.
"""

from typing import TYPE_CHECKING, Dict

import torch

from torchref.base.targets.xray_ml_sigmaa import ml_xray_loss_alpha_beta_math
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class MaximumLikelihoodSigmaAXrayTarget(XrayTarget):
    """Read MLF target; reads ML alpha/beta (sigma_A) from the Scaler."""

    # The correctly-calibrated Read-MLF/sigma_A likelihood is legitimately "soft"
    # relative to the geometry prior, so it needs an intrinsic up-weight to be on
    # equal footing (empirically ~10x; the residual imbalance is a count/prior
    # effect, not a sigma_A error -- see the floor investigation). Carry that as a
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
        F_obs, F_calc, sigma, centric, mask = self.get_data(fcalc=fcalc)

        scaler = self._scaler
        if not hasattr(scaler, "get_alpha_beta"):
            raise RuntimeError(
                "ml_sigmaa target requires a Scaler with get_alpha_beta(); got "
                f"{type(scaler).__name__}."
            )
        # alpha/beta/epsilon: lazily estimated + cached on the scaler, detached.
        alpha, beta, eps = scaler.get_alpha_beta(fcalc)
        alpha = alpha.to(F_obs.dtype)
        beta = beta.to(F_obs.dtype)
        eps = eps.to(F_obs.dtype) if eps is not None else None

        loss = ml_xray_loss_alpha_beta_math(
            F_obs, F_calc, alpha, beta, centric, mask, eps
        )
        # TODO(weighting): base weight applied inside the target as a stopgap;
        # should live in LossState/component_weighting (see class docstring).
        # Only the work set drives refinement; leave the test instance (R-free /
        # NLL_test monitoring) unscaled so reported metrics stay comparable.
        if self.use_work_set:
            loss = self.base_weight * loss
        return loss

    def maintenance(self) -> None:
        """Invalidate the scaler's cached alpha/beta so it is re-estimated from
        the updated model on the next forward. ``LossState`` calls this after
        each optimizer-step block (see ``Target.maintenance``)."""
        scaler = self._scaler
        if scaler is not None and hasattr(scaler, "reset_alpha_beta_cache"):
            scaler.reset_alpha_beta_cache()

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """Add alpha diagnostics (low/high-resolution shell values)."""
        base = super().stats(fcalc=fcalc)
        scaler = self._scaler
        ab = getattr(scaler, "alpha_per_bin", None)
        if ab is not None and ab.numel() > 0:
            base["alpha_bin0"] = stat(ab[0].item(), VERBOSITY_STANDARD)
            base["alpha_binN"] = stat(ab[-1].item(), VERBOSITY_STANDARD)
        return base
