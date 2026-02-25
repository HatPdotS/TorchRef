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

if TYPE_CHECKING:
    from torchref.model.model import Model


class ChiralTarget(GeometryTarget):
    """
    Chiral volume restraint target.

    Restrains the signed volume of tetrahedral chiral centers to maintain
    correct stereochemistry (R vs S configuration, L vs D amino acids).

    The chiral volume is computed as:
        V = v1 · (v2 × v3)

    where vi = position of neighbor i - position of center.

    For standard protein Cα atoms with ordering (N, C, CB):
    - L-amino acids: positive volume (~+2.5 Å³)
    - D-amino acids: negative volume (~-2.5 Å³)

    The loss function penalizes deviations from the ideal signed volume:
        NLL = 0.5 * ((V - V_ideal) / σ)² + log(σ) + 0.5 * log(2π)

    For achiral centers (volume_sign='both'), we restrain the absolute volume.
    """

    name: str = "geometry/chiral"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=0.2)

    def forward(self) -> torch.Tensor:
        xyz = self.model.xyz()
        device = xyz.device

        if "chiral" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        chiral_data = self.restraints.restraints["chiral"]
        indices = chiral_data.get("indices")

        if indices is None or len(indices) == 0:
            return torch.tensor(0.0, device=device)

        ideal_volumes = chiral_data["ideal_volumes"]
        sigmas = chiral_data["sigmas"]

        # Get atom positions: indices is (N, 4) with [center, atom1, atom2, atom3]
        pos_center = xyz[indices[:, 0]]  # (N, 3)
        pos1 = xyz[indices[:, 1]]  # (N, 3)
        pos2 = xyz[indices[:, 2]]  # (N, 3)
        pos3 = xyz[indices[:, 3]]  # (N, 3)

        # Compute vectors from center to neighbors
        v1 = pos1 - pos_center  # (N, 3)
        v2 = pos2 - pos_center  # (N, 3)
        v3 = pos3 - pos_center  # (N, 3)

        # Compute chiral volume: V = v1 · (v2 × v3)
        cross_v2_v3 = torch.cross(v2, v3, dim=-1)  # (N, 3)
        volumes = torch.sum(v1 * cross_v2_v3, dim=-1)  # (N,)

        # Handle achiral centers (ideal_volume = 0) differently
        # For achiral: restrain |V| to typical value (2.5 Å³)
        # Use torch.where unconditionally to avoid .any() GPU sync
        achiral_mask = ideal_volumes == 0
        effective_ideal = torch.where(
            achiral_mask,
            torch.full_like(ideal_volumes, 2.5),
            ideal_volumes,
        )
        effective_volumes = torch.where(achiral_mask, torch.abs(volumes), volumes)
        deviations = effective_volumes - effective_ideal

        # Gaussian NLL
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))
        nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi

        return nll.mean()

    def get_violations(self, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Get information about chiral volume violations.

        Parameters
        ----------
        threshold : float, optional
            Report deviations larger than this (Å³). Default is 0.5.

        Returns
        -------
        dict
            Dictionary with 'indices', 'volumes', 'ideal_volumes', 'deviations'.
        """
        xyz = self.model.xyz()
        device = xyz.device

        if "chiral" not in self.restraints.restraints:
            return {
                "indices": torch.tensor([], dtype=torch.long, device=device).reshape(
                    0, 4
                ),
                "volumes": torch.tensor([], device=device),
                "ideal_volumes": torch.tensor([], device=device),
                "deviations": torch.tensor([], device=device),
            }

        chiral_data = self.restraints.restraints["chiral"]
        indices = chiral_data["indices"]
        ideal_volumes = chiral_data["ideal_volumes"]

        # Compute current volumes
        pos_center = xyz[indices[:, 0]]
        pos1 = xyz[indices[:, 1]]
        pos2 = xyz[indices[:, 2]]
        pos3 = xyz[indices[:, 3]]

        v1 = pos1 - pos_center
        v2 = pos2 - pos_center
        v3 = pos3 - pos_center

        cross_v2_v3 = torch.cross(v2, v3, dim=-1)
        volumes = torch.sum(v1 * cross_v2_v3, dim=-1)

        # Compute deviations (handle achiral)
        achiral_mask = ideal_volumes == 0
        if achiral_mask.any():
            effective_ideal = torch.where(
                achiral_mask, torch.full_like(ideal_volumes, 2.5), ideal_volumes
            )
            effective_volumes = torch.where(achiral_mask, torch.abs(volumes), volumes)
            deviations = torch.abs(effective_volumes - effective_ideal)
        else:
            deviations = torch.abs(volumes - ideal_volumes)

        # Filter by threshold
        mask = deviations > threshold

        return {
            "indices": indices[mask],
            "volumes": volumes[mask],
            "ideal_volumes": ideal_volumes[mask],
            "deviations": deviations[mask],
        }

    def stats(self) -> Dict[str, any]:
        """Get chiral volume statistics."""
        xyz = self.model.xyz()
        device = xyz.device

        if "chiral" not in self.restraints.restraints:
            return {}

        chiral_data = self.restraints.restraints["chiral"]
        indices = chiral_data["indices"]
        ideal_volumes = chiral_data["ideal_volumes"]
        sigmas = chiral_data["sigmas"]

        if len(indices) == 0:
            return {}

        # Compute current volumes
        pos_center = xyz[indices[:, 0]]
        pos1 = xyz[indices[:, 1]]
        pos2 = xyz[indices[:, 2]]
        pos3 = xyz[indices[:, 3]]

        v1 = pos1 - pos_center
        v2 = pos2 - pos_center
        v3 = pos3 - pos_center

        cross_v2_v3 = torch.cross(v2, v3, dim=-1)
        volumes = torch.sum(v1 * cross_v2_v3, dim=-1)

        # Handle achiral
        achiral_mask = ideal_volumes == 0
        if achiral_mask.any():
            effective_ideal = torch.where(
                achiral_mask, torch.full_like(ideal_volumes, 2.5), ideal_volumes
            )
            effective_volumes = torch.where(achiral_mask, torch.abs(volumes), volumes)
            deviations = effective_volumes - effective_ideal
        else:
            deviations = volumes - ideal_volumes

        z_scores = deviations / sigmas
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(indices), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((deviations**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigmas.mean().item(), VERBOSITY_DEBUG),
        }
