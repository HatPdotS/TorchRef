import torch
from typing import Optional, TYPE_CHECKING

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
        F_obs, F_calc, sigma, centric_flags, mask = self.get_data(fcalc=fcalc)
        return ml_xray_loss_math(F_obs, F_calc, sigma, centric_flags, mask)


def create_xray_target(
    data: "ReflectionData" = None,
    model: "Model" = None,
    scaler: "Scaler" = None,
    mode: str = "gaussian",
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
    # Pin model/data/scaler onto one device before constructing the
    # target — its forward path mixes tensors from all three.
    resolve_device(model, data, scaler, device=device)

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
