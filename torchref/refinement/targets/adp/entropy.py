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

if TYPE_CHECKING:
    from torchref.model.model import Model


class ADPEntropyTarget(ADPTarget):
    """
    ADP Entropy regularization target.

    Uses the model's existing adp_kl_divergence_loss or similar.
    """

    name: str = "adp/KL"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=0.5, sigma=0.5)

    def forward(self) -> torch.Tensor:
        return self.model.adp_kl_divergence_loss()

    def stats(self) -> Dict[str, any]:
        """Get KL divergence statistics."""
        adp = self.model.adp().detach()
        log_adp = torch.log(adp.clamp(min=1e-3))
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n_atoms": stat(len(adp), VERBOSITY_DEBUG),
            "mean_adp": stat(adp.mean().item(), VERBOSITY_DETAILED),
            "std_adp": stat(adp.std().item(), VERBOSITY_DETAILED),
            "min_adp": stat(adp.min().item(), VERBOSITY_DETAILED),
            "max_adp": stat(adp.max().item(), VERBOSITY_DETAILED),
            "mean_log_adp": stat(log_adp.mean().item(), VERBOSITY_DEBUG),
            "std_log_adp": stat(log_adp.std().item(), VERBOSITY_DEBUG),
        }
