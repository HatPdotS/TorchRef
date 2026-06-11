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

import warnings
from typing import TYPE_CHECKING, Dict, Optional

import torch

from torchref.base.targets.xray_ml_sigmaa import ml_xray_loss_beta_math
from torchref.utils.device_resolution import resolve_device
from torchref.utils.stats import VERBOSITY_STANDARD, StatEntry, stat

from .base import XrayTarget
from .gaussian import GaussianXrayTarget
from .least_squares import LeastSquaresXrayTarget
from .rice import RiceXrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class MaximumLikelihoodXrayTarget(XrayTarget):
    """Maximum-likelihood (Read MLF) X-ray target with a Luzzati σ_A model-error
    variance; reads the ML model-error variance ``beta`` from the Scaler.

    The model mean is ``|F_calc|`` (the Luzzati ``alpha`` mean-shift was removed
    after it was shown to be gauge-absorbed by the co-refined scaler); the
    conditional variance is ``epsilon * beta``, with ``beta`` the per-shell
    Luzzati model-error variance estimated on the FREE set.
    """

    # NOTE: the Read/sigma_A likelihood is legitimately "soft" relative to the
    # geometry prior and needs ~10x to be on equal footing. That up-weight now
    # lives as the transparent ``xray`` group base weight in
    # ``base_refinement.DEFAULT_GROUP_WEIGHTS`` (LossState), NOT multiplied inside
    # this target's forward — see that constant for the calibration rationale.

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        use_work_set: bool = True,
        verbose: int = 0,
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

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        F_obs, F_calc, sigma, centric, sub = self.get_data(fcalc=fcalc)

        scaler = self._scaler
        if not hasattr(scaler, "get_beta"):
            raise RuntimeError(
                "maximum-likelihood (ml) target requires a Scaler with "
                f"get_beta(); got {type(scaler).__name__}."
            )
        # beta/epsilon: lazily estimated + cached on the scaler, detached.
        # They are full-size (per-reflection) — restrict to this subset.
        beta, eps = scaler.get_beta(fcalc)
        beta = sub.select(beta).to(F_obs.dtype)
        eps = sub.select(eps).to(F_obs.dtype) if eps is not None else None

        # The data/prior balance (the ~10x x-ray up-weight) is applied by the
        # LossState ``xray`` group weight, not here — see DEFAULT_GROUP_WEIGHTS.
        return ml_xray_loss_beta_math(F_obs, F_calc, beta, centric, epsilon=eps)

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


def create_xray_target(
    data: "ReflectionData" = None,
    model: "Model" = None,
    scaler: "Scaler" = None,
    mode: str = "ml",
    use_work_set: bool = True,
    sigma_mode: str = "raw",
    sigma_m_scale: float = 1.0,
    verbose: int = 0,
    device: Optional[torch.device] = None,
) -> XrayTarget:
    """
    Factory function to create X-ray target.

    Parameters
    ----------
    data : ReflectionData
        Reference to ReflectionData object. Required for forward().
    model : Model or ModelFT, optional
        Reference to Model object for F_calc computation.
        If None, fcalc must be provided when calling forward().
    scaler : Scaler, optional
        Reference to Scaler object.
    mode : str, optional
        Target mode: 'gaussian', 'ls', 'rice', 'ml', or 'bhattacharyya'.
        Default is 'ml' (maximum-likelihood Read MLF with Luzzati σ_A). 'rice'
        is the simpler unit-variance Rice maximum-likelihood target. The legacy
        spelling 'ml_sigmaa' is accepted as a deprecated alias for 'ml'.
    use_work_set : bool, optional
        Use work set (True) or test set (False). Default is True.
    sigma_mode : str, optional
        'effective' (default) to use per-shell effective sigmas from the
        scaler (SIGMAA-style, robust), or 'raw' to use raw experimental
        sigmas from the data file.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    XrayTarget
        Appropriate XrayTarget instance.
    """
    if mode == "ml_sigmaa":
        warnings.warn(
            "X-ray mode 'ml_sigmaa' is deprecated; use 'ml' (now the "
            "maximum-likelihood Read MLF σ_A target). The former 'ml' Rice "
            "target is now 'rice'.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "ml"

    # Pin model/data/scaler onto one device before constructing the
    # target — its forward path mixes tensors from all three.
    resolve_device(model, data, scaler, device=device)

    kwargs = dict(
        data=data,
        model=model,
        scaler=scaler,
        use_work_set=use_work_set,
        sigma_mode=sigma_mode,
        verbose=verbose,
    )
    if mode == "gaussian":
        return GaussianXrayTarget(**kwargs)
    elif mode == "ls":
        return LeastSquaresXrayTarget(**kwargs)
    elif mode == "rice":
        return RiceXrayTarget(**kwargs)
    elif mode == "ml":
        return MaximumLikelihoodXrayTarget(**kwargs)
    elif mode == "bhattacharyya":
        from .bhattacharyya import BhattacharyyaXrayTarget

        return BhattacharyyaXrayTarget(
            sigma_m_scale=sigma_m_scale,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown X-ray target mode: {mode}")
