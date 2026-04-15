import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import GeometryTarget
from ..base import von_mises_nll

if TYPE_CHECKING:
    from torchref.model.model import Model


class TorsionTarget(GeometryTarget):
    """
    Torsion angle restraint target (von Mises NLL).

    NLL = -κ*cos(φ - φ₀) + log(I₀(κ)) + log(2π)
    where κ = 1/σ²
    """

    name: str = "geometry/torsion"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=1.0, sigma=0.3)

    def forward(self) -> torch.Tensor:
        deviations_rad, sigmas_deg = self.restraints.torsion_deviations_with_sigmas()

        if len(deviations_rad) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)

        sigmas_rad = sigmas_deg * float(np.pi / 180.0)
        kappa = torch.clamp(1.0 / (sigmas_rad**2), min=1e-3, max=1e4)

        # log(I_0(kappa)) via exponentially-scaled Bessel (stable for all kappa)
        # i0e(x) = exp(-|x|) * i0(x), so log(i0(x)) = log(i0e(x)) + x
        # i0e is always > 0 for finite kappa, so no epsilon needed.
        log_i0_kappa = torch.log(torch.special.i0e(kappa)) + kappa

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=sigmas_deg.device, dtype=sigmas_deg.dtype)
        )

        # NLL = -log P(theta)
        log_prob = kappa * torch.cos(deviations_rad) - log_i0_kappa - log_2pi

        return (-log_prob).sum()

    def stats(self) -> Dict[str, StatEntry]:
        """Get torsion angle statistics."""
        deviations_rad, sigmas_deg = self.restraints.torsion_deviations_with_sigmas()

        if len(deviations_rad) == 0:
            return {}

        deviations_deg = deviations_rad * (180.0 / np.pi)
        sigmas_rad = sigmas_deg * (np.pi / 180.0)
        z_scores = deviations_rad / sigmas_rad
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(deviations_rad), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((deviations_deg**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigmas_deg.mean().item(), VERBOSITY_DEBUG),
        }
