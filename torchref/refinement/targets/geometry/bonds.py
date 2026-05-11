import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.bond import bond_math
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


class BondTarget(GeometryTarget):
    """
    Bond length restraint target (Gaussian NLL).

    NLL = 0.5 * ((d - d₀) / σ)² + log(σ) + 0.5 * log(2π)
    """

    name: str = "geometry/bond"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=1.0)

    def forward(self) -> torch.Tensor:
        # Use the bond_math dispatcher (Triton on CUDA fp32, eager
        # otherwise). Pulls inputs directly from the model + restraints
        # instead of going through `Restraints.bond_deviations` so the
        # Triton kernel can also own the gather + distance compute.
        if "all" not in self.restraints.restraints["bond"]:
            self.restraints.cat_dict()
        bond = self.restraints.restraints["bond"]["all"]
        idx = bond["indices"]
        if idx is None or len(idx) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)
        return bond_math(self.model.xyz(), idx, bond["references"], bond["sigmas"])

    def stats(self) -> Dict[str, StatEntry]:
        """Get bond restraint statistics."""
        deviations, sigmas = self.restraints.bond_deviations()
        if len(deviations) == 0:
            return {}

        z_scores = deviations / sigmas
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(deviations), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((deviations**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigmas.mean().item(), VERBOSITY_DEBUG),
        }
