import numpy as np
import torch
from typing import TYPE_CHECKING

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
        F_obs, F_calc, sigma, centric_flags, mask = self.get_data(fcalc=fcalc)

        # Default parameters if not available
        alpha = torch.ones_like(F_obs)
        beta = sigma**2
        epsilon = torch.ones_like(F_obs)

        if centric_flags is None:
            centric_flags = torch.zeros_like(F_obs, dtype=torch.bool)

        F_calc_amp = torch.abs(F_calc)

        # Precompute common terms
        eb = epsilon * beta
        eb = torch.clamp(eb, min=1e-6)

        # Acentric term
        term1 = -torch.log(2 * F_obs / eb + 1e-12)
        term2 = (F_obs**2) / eb
        term3 = (alpha * F_calc_amp) ** 2 / eb

        arg_bessel = 2 * alpha * F_obs * F_calc_amp / eb
        # Clamp to prevent overflow in float32 (large arg_bessel causes issues in subsequent ops)
        arg_bessel = torch.clamp(arg_bessel, max=1e6)
        term4 = -(
            torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel
        )

        loss_acentric = term1 + term2 + term3 + term4

        # Centric term
        term1_c = -0.5 * torch.log(2 / (np.pi * eb) + 1e-12)
        term2_c = (F_obs**2) / (2 * eb)
        term3_c = (alpha * F_calc_amp) ** 2 / (2 * eb)
        term4_c = -(alpha * F_obs * F_calc_amp) / eb

        arg_exp = -2 * alpha * F_obs * F_calc_amp / eb
        # Clamp only the exp argument to prevent overflow (float32 safe range: ~[-88, 88])
        arg_exp_safe = torch.clamp(arg_exp, min=-80.0, max=80.0)
        term5_c = -torch.log((1 + torch.exp(arg_exp_safe)) / 2 + 1e-12)

        loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

        # Combine based on centric flags
        loss = torch.where(centric_flags, loss_centric, loss_acentric)

        # Replace any NaN/Inf with large finite value to maintain gradient signal
        loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))

        return (loss * mask).sum()


def create_xray_target(
    data: "ReflectionData" = None,
    model: "Model" = None,
    scaler: "Scaler" = None,
    mode: str = "gaussian",
    use_work_set: bool = True,
    sigma_mode: str = "raw",
    sigma_m_scale: float = 1.0,
    verbose: int = 0,
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
        Target mode: 'gaussian', 'ls', or 'ml'. Default is 'gaussian'.
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
    kwargs = dict(
        data=data, model=model, scaler=scaler,
        use_work_set=use_work_set, sigma_mode=sigma_mode, verbose=verbose,
    )
    if mode == "gaussian":
        return GaussianXrayTarget(**kwargs)
    elif mode == "ls":
        return LeastSquaresXrayTarget(**kwargs)
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
