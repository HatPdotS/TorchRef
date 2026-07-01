from typing import TYPE_CHECKING

import torch

from torchref.base.targets.xray_ml import ml_xray_loss_math

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class RiceXrayTarget(XrayTarget):
    """Rice / unit-variance maximum-likelihood X-ray target.

    The Rice-distribution amplitude likelihood with proper centric/acentric
    handling, using the experimental ``sigma`` as the only variance term
    (``beta = sigma**2``). This is the simpler ML target; the σ_A-aware
    :class:`~torchref.refinement.targets.xray.maximum_likelihood.MaximumLikelihoodXrayTarget`
    adds a per-shell Luzzati model-error variance and is the default.
    """

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the Rice maximum-likelihood loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Summed Rice ML loss on this target's set.
        """
        F_obs, F_calc, sigma, centric_flags, _ = self.get_data(fcalc=fcalc)
        return ml_xray_loss_math(F_obs, F_calc, sigma, centric_flags)
