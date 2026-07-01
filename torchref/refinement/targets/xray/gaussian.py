import torch
from typing import TYPE_CHECKING

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

    Attributes
    ----------
    target_value : float
        Reference value carried for the generic ``Target`` machinery. Note the
        loss returned by ``forward`` is a summed (not per-reflection normalized)
        NLL, so this value is not a tight per-reflection target.
    """

    target_value: float = 1.0

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
            Summed NLL loss on this target's set.
        """
        F_obs, F_calc, sigma, _, _ = self.get_data(fcalc=fcalc)
        return gaussian_xray_loss_math(F_obs, F_calc, sigma)
