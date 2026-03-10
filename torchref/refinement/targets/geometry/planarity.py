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


class PlanarityTarget(GeometryTarget):
    """
    Planarity restraint target (Gaussian NLL).

    For each planar group (e.g., aromatic rings, peptide planes), computes the
    distance of each atom from the best-fit plane.

    The best-fit plane normal is found by eigendecomposition of the 3x3
    covariance matrix of centered coordinates (eigh). The normal is detached
    from the computational graph so that gradients flow only through the
    deviation projection, not through the eigendecomposition. This is standard
    practice in crystallographic refinement (SHELXL, Phenix, Refmac) and is
    more numerically robust than differentiating through SVD — in particular
    it avoids NaN gradients when atoms are exactly coplanar.

    Plane groups with <= 3 atoms are skipped since 3 coplanar points have
    zero deviation by construction and contribute no gradient signal.

    NLL = 0.5 * (d_i / σ_i)² + log(σ_i) + 0.5 * log(2π)

    where d_i is the distance of atom i from the best-fit plane.
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

        for _key, plane_data in self.restraints.restraints["plane"].items():
            indices = plane_data.get("indices")
            sigmas = plane_data.get("sigmas")

            if indices is None or len(indices) == 0:
                continue

            # 3 atoms are always coplanar — zero deviations, no gradient signal
            n_atoms = indices.shape[1]
            if n_atoms <= 3:
                continue

            # Gather positions: (n_planes, n_atoms, 3)
            positions = xyz[indices]

            # Center: (n_planes, n_atoms, 3)
            centroids = positions.mean(dim=1, keepdim=True)
            centered = positions - centroids

            # Plane normal via eigh on 3x3 covariance (detached — no backward
            # through the eigendecomposition, avoids NaN at degenerate eigenvalues)
            with torch.no_grad():
                cov = torch.bmm(centered.transpose(1, 2), centered)
                # Regularize to prevent ill-conditioning (atoms nearly collinear)
                jitter = 1e-6 * torch.eye(3, device=device, dtype=cov.dtype).unsqueeze(0)
                cov = cov + jitter
                _eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                normals = eigenvectors[:, :, 0]  # smallest eigenvalue

            # Deviations: gradient flows through centered only
            deviations = torch.einsum("paj,pj->pa", centered, normals)

            # NLL (squaring handles sign — no abs needed)
            nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
            all_nlls.append(nll.flatten())

        if all_nlls:
            return torch.cat(all_nlls).mean()
        return torch.tensor(0.0, device=device)

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

            cov = torch.bmm(centered.transpose(1, 2), centered)
            jitter = 1e-6 * torch.eye(3, device=xyz.device, dtype=cov.dtype).unsqueeze(0)
            cov = cov + jitter
            _eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            normals = eigenvectors[:, :, 0]

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
