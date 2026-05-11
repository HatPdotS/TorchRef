import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.angle import angle_math
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import GeometryTarget
from ..base import gaussian_nll

if TYPE_CHECKING:
    from torchref.model.model import Model


class AngleTarget(GeometryTarget):
    """
    Angle restraint target (Gaussian NLL).

    NLL = 0.5 * ((θ - θ₀) / σ)² + log(σ) + 0.5 * log(2π)
    """

    name: str = "geometry/angle"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=0.5)

    def forward(self) -> torch.Tensor:
        # Use the angle_math dispatcher (Triton on CUDA fp32).
        if "all" not in self.restraints.restraints["angle"]:
            self.restraints.cat_dict()
        a = self.restraints.restraints["angle"]["all"]
        idx = a["indices"]
        if idx is None or len(idx) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)
        deg2rad = float(torch.pi / 180.0)
        return angle_math(
            self.model.xyz(), idx,
            a["references"] * deg2rad,
            a["sigmas"] * deg2rad,
        )

    def stats(self) -> Dict[str, StatEntry]:
        """Get angle restraint statistics."""
        deviations_rad, sigmas_rad = self.restraints.angle_deviations()
        if len(deviations_rad) == 0:
            return {}

        # Convert to degrees for reporting
        deviations_deg = deviations_rad * (180.0 / np.pi)
        sigmas_deg = sigmas_rad * (180.0 / np.pi)
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
