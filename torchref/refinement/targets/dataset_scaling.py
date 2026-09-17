"""Joint observed-dataset scaling target, independent of structural models."""

from typing import TYPE_CHECKING

import torch

from torchref.base.targets.dataset_scaling import dataset_scaling_loss
from torchref.refinement.targets.base import Target

if TYPE_CHECKING:
    from torchref.scaling.dataset_scaler import DatasetScaler


class DatasetScalingTarget(Target):
    """Fit a shared consensus to the scaler's training observations.

    Parameters
    ----------
    scaler : DatasetScaler
        Owner of the centered log-scale and anisotropy parameters. Its prepared
        observations exclude held-out reflections from every participating dataset.
    """

    name = "dataset_scaling"

    def __init__(self, scaler: "DatasetScaler") -> None:
        super().__init__(device=scaler.device)
        self.scaler = scaler
        self._adopt_device(scaler)

    def forward(self) -> torch.Tensor:
        """Return the dimensionless loss per independent training contrast."""
        scaler = self.scaler
        return (
            dataset_scaling_loss(
                scaler.amplitudes,
                scaler.sigmas,
                scaler.log_corrections(scaler.hkl),
                scaler.fit_mask,
            )
            / scaler.n_contrasts
        )
