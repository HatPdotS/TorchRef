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

    Penalizes deviation of the B-factor (ADP) distribution from a target
    spread. ``forward()`` returns ``model.adp_kl_divergence_loss()``, the KL
    divergence between the distribution of log-ADPs and a Gaussian with the
    same (detached) mean and a fixed target log-space standard deviation
    (``target_log_std=0.2``). The loss is zero when the log-ADP spread matches
    the target and grows as the distribution becomes tighter or broader.

    Note: despite the class name "Entropy", the loss is a KL divergence to a
    fixed-spread Gaussian (hence ``name = "adp/KL"``), not an entropy term;
    the "distribution regularization" description above is the accurate one.
    """

    name: str = "adp/KL"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose)

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
