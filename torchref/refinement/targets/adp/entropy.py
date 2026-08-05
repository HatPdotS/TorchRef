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
    ADP distribution regularization target.

    Penalizes the B-factor distribution for departing from a target spread: the KL
    divergence between the log-ADPs and a Gaussian at the same (detached) mean and a
    fixed ``target_log_std=0.2``, so the loss is zero at a matching spread and rises as
    the distribution tightens or broadens either way.

    Despite the class name, this is a KL divergence to a fixed-spread Gaussian, not an
    entropy term -- hence ``name = "adp/KL"``.
    """

    name: str = "adp/KL"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose)

    def forward(self) -> torch.Tensor:
        """The model's log-ADP KL divergence from the fixed-spread target Gaussian."""
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
