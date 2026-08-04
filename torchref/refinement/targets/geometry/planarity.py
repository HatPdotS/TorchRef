"""Planarity restraint on the deviation of each atom from its best-fit plane."""

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


def _plane_normals(centered: torch.Tensor) -> torch.Tensor:
    """Unit plane normals (P, 3) from mean-centred coordinates (P, N, 3), by SVD.

    The smallest right singular vector is the minimum-variance direction. SVD needs no
    jitter -- backward-stable even rank-deficient, never raises on finite input -- and
    runs in the input dtype, which is enough for O(Å) coordinates. The result is
    detached; the caller must still wrap the call in ``torch.no_grad()``.
    """
    _U, _S, Vh = torch.linalg.svd(centered.detach(), full_matrices=False)
    return Vh[:, -1, :]


class PlanarityTarget(GeometryTarget):
    """
    Planarity restraint target (Gaussian NLL).

    Per planar group (aromatic rings, peptide planes), ``d_i`` is atom i's distance from
    the best-fit plane and the loss is ``0.5·(d_i/σ_i)² + log σ_i + 0.5·log 2π``.

    The plane normal is the minimum-variance direction from an SVD (see
    :func:`_plane_normals`) and is **detached**, so gradients flow through the deviation
    projection only -- as in SHELXL, Phenix and Refmac, and what avoids NaN gradients at
    exact coplanarity. Groups of <= 3 atoms are skipped: three points are coplanar by
    construction, so they carry no signal.
    """

    name: str = "geometry/planarity"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose)

    def forward(self) -> torch.Tensor:
        """Summed planarity NLL over plane-size buckets; 0.0 when there are no planes."""
        from torchref.base.targets.planarity import planarity_math
        xyz = self.model.xyz()
        device = xyz.device

        if "plane" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        # Bucketed by plane size, skipping 3-atom planes: zero signal by construction.
        plane_groups = []
        for _key, plane_data in self.restraints.restraints["plane"].items():
            indices = plane_data.get("indices")
            sigmas = plane_data.get("sigmas")
            if indices is None or len(indices) == 0:
                continue
            if indices.shape[1] <= 3:
                continue
            plane_groups.append((indices, sigmas))

        if not plane_groups:
            return torch.tensor(0.0, device=device)
        return planarity_math(xyz, plane_groups)

    def stats(self) -> Dict[str, any]:
        """Get planarity restraint statistics."""
        xyz = self.model.xyz()

        if "plane" not in self.restraints.restraints:
            return {}

        all_deviations = []
        all_sigmas = []

        for _key, plane_data in self.restraints.restraints["plane"].items():
            indices = plane_data.get("indices")
            sigmas = plane_data.get("sigmas")

            if indices is None or len(indices) == 0:
                continue

            n_atoms = indices.shape[1]
            if n_atoms <= 3:
                continue

            positions = xyz[indices]
            centroids = positions.mean(dim=1, keepdim=True)
            centered = positions - centroids

            normals = _plane_normals(centered).to(xyz.dtype)

            deviations = torch.abs(torch.einsum("paj,pj->pa", centered, normals))

            all_deviations.append(deviations.flatten())
            all_sigmas.append(sigmas.flatten())

        if not all_deviations:
            return {"n": 0, "rms_delta": 0.0, "rms_z": 0.0, "mean_sigma": 0.0}

        all_deviations = torch.cat(all_deviations)
        all_sigmas = torch.cat(all_sigmas)
        z_scores = all_deviations / all_sigmas
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(all_deviations), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((all_deviations**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(all_sigmas.mean().item(), VERBOSITY_DEBUG),
        }
