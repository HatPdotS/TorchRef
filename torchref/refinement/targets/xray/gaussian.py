import torch
from typing import TYPE_CHECKING

from torchref.base.targets._dispatch import use_triton
from torchref.base.targets.xray_gaussian import gaussian_xray_loss_math

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class GaussianXrayTarget(XrayTarget):
    """
    Simple Gaussian NLL target for X-ray data.

    NLL = 0.5*(F_obs - |F_calc|)²/σ² + log(σ) + 0.5*log(2π)
    """

    target_value: float = 1.0  # Ideal normalized NLL

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Mean NLL loss value.
        """
        F_obs, F_calc, sigma, _, mask = self.get_data(fcalc=fcalc)

        if use_triton(F_calc, F_obs, sigma):
            from torchref.base.targets.triton.xray_gaussian import (
                gaussian_xray_loss_math_triton,
            )
            return gaussian_xray_loss_math_triton(F_obs, F_calc, sigma, mask)
        return gaussian_xray_loss_math(F_obs, F_calc, sigma, mask)
