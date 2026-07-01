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
    """Compute unit plane normals from centered coordinates via SVD.

    SVD is backward-stable even for rank-deficient matrices and never raises
    on finite input, so no jitter is needed. The right singular vector with
    the smallest singular value is the plane normal (direction of minimum
    variance).

    SVD is run in the input dtype — for small (P, N, 3) matrices over
    O(Å) atom coordinates, float32 is numerically sufficient.

    The result is detached — the caller is responsible for wrapping this in
    ``torch.no_grad()``.

    Parameters
    ----------
    centered : torch.Tensor
        (P, N, 3) atoms of each plane group, mean-centred.

    Returns
    -------
    torch.Tensor
        (P, 3) plane normals (smallest-variance direction).
    """
    _U, _S, Vh = torch.linalg.svd(centered.detach(), full_matrices=False)
    return Vh[:, -1, :]


class PlanarityTarget(GeometryTarget):
    """
    Planarity restraint target (Gaussian NLL).

    For each planar group (e.g., aromatic rings, peptide planes), computes the
    distance of each atom from the best-fit plane.

    The best-fit plane normal is the smallest-variance direction of the
    centered coordinates, obtained via SVD (see ``_plane_normals``). The
    normal is detached from the computational graph so that gradients flow
    only through the deviation projection, not through the SVD. This is
    standard practice in crystallographic refinement (SHELXL, Phenix, Refmac)
    and avoids NaN gradients when atoms are exactly coplanar.

    Plane groups with <= 3 atoms are skipped since 3 coplanar points have
    zero deviation by construction and contribute no gradient signal.

    NLL = 0.5 * (d_i / σ_i)² + log(σ_i) + 0.5 * log(2π)

    where d_i is the distance of atom i from the best-fit plane.
    """

    name: str = "geometry/planarity"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose)

    def forward(self) -> torch.Tensor:
        from torchref.base.targets.planarity import planarity_math
        xyz = self.model.xyz()
        device = xyz.device

        if "plane" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        # Build (indices, sigmas) per plane-size bucket, skipping
        # 3-atom planes (zero gradient signal by construction).
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
