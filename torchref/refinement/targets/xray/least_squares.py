import torch
from typing import TYPE_CHECKING

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
    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        weighting: str = "sigma",
        use_work_set: bool = True,
        sigma_mode: str = "raw",
        verbose: int = 0,
        use_set: str = None,
    ):
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            sigma_mode=sigma_mode,
            verbose=verbose,
            use_set=use_set,
        )
        self.weighting = weighting

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
            Mean weighted least squares loss.
        """
        F_obs, F_calc, sigma, _, _ = self.get_data(fcalc=fcalc)
        return ls_xray_loss_math(F_obs, F_calc, sigma, weighting=self.weighting)
