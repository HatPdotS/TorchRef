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

from .base import ADPTarget
from ..base import adp_similarity_nll

if TYPE_CHECKING:
    from torchref.model.model import Model


class ADPSimilarityTarget(ADPTarget):
    """
    ADP Similarity restraint (SIMU in Phenix/SHELX).

    Restrains B-factors of bonded atoms to be similar.
    NLL = 0.5 * ((B_i - B_j) / σ)² + log(σ) + 0.5 * log(2π)

    Tunable parameters (as buffers):
    - _simu_sigma: float, sigma for B-factor differences (default 2.0 Å²)
    """

    name: str = "adp/simu"

    def __init__(
        self, model: "Model" = None, simu_sigma: float = 2.0, verbose: int = 0
    ):
        super().__init__(model, verbose, target_value=4.0, sigma=1.2)
        # Register simu-specific sigma as buffer (separate from base sigma)
        self.register_buffer("_simu_sigma", torch.tensor(simu_sigma))

    @property
    def simu_sigma(self) -> float:
        """Get SIMU sigma value."""
        return self._simu_sigma.item()

    @simu_sigma.setter
    def simu_sigma(self, value: float):
        """Set SIMU sigma value."""
        self._simu_sigma.fill_(value)

    def forward(self) -> torch.Tensor:
        b_diffs = self.restraints.adp_b_differences()

        if len(b_diffs) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=b_diffs.device, dtype=b_diffs.dtype)
        )
        nll = (
            0.5 * (b_diffs / self._simu_sigma) ** 2
            + torch.log(self._simu_sigma)
            + 0.5 * log_2pi
        )

        return nll.mean()

    def stats(self) -> Dict[str, any]:
        """Get SIMU restraint statistics."""
        b_diffs = self.restraints.adp_b_differences()

        if len(b_diffs) == 0:
            return {}

        b_diffs_abs = b_diffs.abs()
        z_scores = b_diffs_abs / self.simu_sigma
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "count": stat(len(b_diffs), VERBOSITY_DEBUG),
            "rms_delta_b": stat(
                torch.sqrt((b_diffs**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "mean_delta_b": stat(b_diffs_abs.mean().item(), VERBOSITY_DETAILED),
            "max_delta_b": stat(b_diffs_abs.max().item(), VERBOSITY_DETAILED),
            "mean_z": stat(z_scores.mean().item(), VERBOSITY_DEBUG),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
        }
