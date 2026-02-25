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
from ..base import gaussian_nll

if TYPE_CHECKING:
    from torchref.model.model import Model


class PlanarityTarget(GeometryTarget):
    """
    Planarity restraint target (Gaussian NLL).

    For each planar group (e.g., aromatic rings, peptide planes), computes the
    distance of each atom from the best-fit plane determined by SVD.

    The best-fit plane is found by:
    1. Computing the centroid of the atoms
    2. Centering the coordinates
    3. Finding the eigenvector with smallest eigenvalue via SVD
    4. This eigenvector is the plane normal

    NLL = 0.5 * (d_i / σ_i)² + log(σ_i) + 0.5 * log(2π)

    where d_i is the distance of atom i from the best-fit plane.

    Reference: cctbx/geometry_restraints/planarity.h
    """

    name: str = "geometry/planarity"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=0.2)

    def forward(self) -> torch.Tensor:
        xyz = self.model.xyz()
        device = xyz.device

        all_nlls = []

        if "plane" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))

        for key, plane_data in self.restraints.restraints["plane"].items():
            indices = plane_data.get("indices")
            sigmas = plane_data.get("sigmas")

            if indices is None or len(indices) == 0:
                continue

            # indices shape: (n_planes, n_atoms_per_plane)
            # sigmas shape: (n_planes, n_atoms_per_plane)

            # Gather all positions at once: (n_planes, n_atoms, 3)
            positions = xyz[indices]

            # Compute centroids: (n_planes, 1, 3)
            centroids = positions.mean(dim=1, keepdim=True)
            centered = positions - centroids  # (n_planes, n_atoms, 3)

            # BATCHED SVD
            U, S, Vh = torch.linalg.svd(centered)  # Vh: (n_planes, 3, 3)
            normals = Vh[:, -1, :]  # (n_planes, 3)

            # Batched deviation calculation
            deviations = torch.abs(torch.einsum("paj,pj->pa", centered, normals))

            # NLL calculation (all vectorized)
            nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
            all_nlls.append(nll.flatten())

        if all_nlls:
            return torch.cat(all_nlls).mean()
        return torch.tensor(0.0, device=device)

    def stats(self) -> Dict[str, any]:
        """Get planarity restraint statistics."""
        xyz = self.model.xyz()
        device = xyz.device

        if "plane" not in self.restraints.restraints:
            return {}

        all_deviations = []
        all_sigmas = []

        for key, plane_data in self.restraints.restraints["plane"].items():
            indices = plane_data.get("indices")
            sigmas = plane_data.get("sigmas")

            if indices is None or len(indices) == 0:
                continue

            # Gather all positions at once: (n_planes, n_atoms, 3)
            positions = xyz[indices]

            # Compute centroids: (n_planes, 1, 3)
            centroids = positions.mean(dim=1, keepdim=True)
            centered = positions - centroids  # (n_planes, n_atoms, 3)

            # Eigendecomposition of 3x3 covariance matrix (faster than SVD)
            cov = torch.bmm(centered.transpose(1, 2), centered)  # (n_planes, 3, 3)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # sorted ascending
            normals = eigenvectors[:, :, 0]  # (n_planes, 3)

            # Batched deviation calculation
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
