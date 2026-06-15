import warnings
from typing import TYPE_CHECKING, Optional

import torch

from torchref.base.targets.xray_ml import ml_xray_loss_math
from torchref.utils.device_resolution import resolve_device

from .base import XrayTarget
from .gaussian import GaussianXrayTarget
from .least_squares import LeastSquaresXrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class MaximumLikelihoodXrayTarget(XrayTarget):
    """
    Maximum Likelihood target function with proper centric/acentric handling.
    """

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute maximum likelihood loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Mean ML loss value.
        """
        # 5th element is the ``_ReflectionSubset`` view, not a mask. F_obs /
        # F_calc / sigma are already compact (subset-applied via sub.F etc.)
        # so the downstream kernel needs no mask.
        F_obs, F_calc, sigma, centric_flags, _ = self.get_data(fcalc=fcalc)
        return ml_xray_loss_math(F_obs, F_calc, sigma, centric_flags, mask=None)


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
        Target mode: 'gaussian', 'ls', 'ls_wunit_k1', 'ml', or
        'bhattacharyya'. Default is 'ml' (maximum-likelihood Read MLF
        with Luzzati σ_A). 'ml_sigmaa' is a deprecated alias for 'ml'.
        'ls_wunit_k1' is Phenix-style least squares with
        unit weights and a per-bin closed-form optimal scale recomputed at
        every gradient call (does not use the external scaler).
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
    # Legacy spelling — ``ml_sigmaa`` is the same as ``ml`` now.
    if mode == "ml_sigmaa":
        warnings.warn(
            "Target mode 'ml_sigmaa' is deprecated; use 'ml' instead. "
            "It now resolves to the same MaximumLikelihoodXrayTarget.",
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
    elif mode == "ls_wunit_k1":
        # Phenix-style: unit weights, SINGLE GLOBAL K recomputed every
        # gradient call. The "k1" in the target name means "K_one" — one
        # K, not per-bin (Phenix's update_all_scales fits a single
        # k_overall, aniso off at d>3Å). n_bins=1 collapses the
        # closed-form c[bins] to a single scalar — matches Phenix exactly.
        # Gradient w.r.t. θ treats c as constant via .detach() in forward.
        return LeastSquaresXrayTarget(
            weighting="unit",
            scale_mode="binwise_optimal",
            n_bins=1,
            **kwargs,
        )
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
