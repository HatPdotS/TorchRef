import numpy as np
import torch
from typing import TYPE_CHECKING

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

        F_calc_amp = torch.abs(F_calc)
        diff = F_obs - F_calc_amp

        # Avoid division by zero
        eps = torch.median(sigma) * 1e-1
        sigma_safe = torch.clamp(sigma, min=eps)

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=sigma.device, dtype=sigma.dtype)
        )
        nll = 0.5 * (diff**2) / (sigma_safe**2) + torch.log(sigma_safe) + 0.5 * log_2pi

        return (nll * mask).sum()
