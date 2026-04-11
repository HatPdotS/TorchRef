import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import GeometryTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class RamachandranTarget(GeometryTarget):
    """
    Ramachandran restraint via pre-computed NLL surfaces.

    Uses 6 residue-type-dependent NLL surfaces (general, glycine, cis-proline,
    trans-proline, pre-proline, ile/val) at 1-degree resolution.  The loss is
    computed by bilinear interpolation of the NLL surface at the current
    (phi, psi) angles.

    The surfaces store NLL = -log P(phi, psi | residue_type), so favored
    regions have low values and outlier regions have high values — consistent
    with all other geometry targets.
    """

    name: str = "geometry/ramachandran"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=2.0, sigma=1.0)

    def forward(self) -> torch.Tensor:
        xyz = self.model.xyz()
        device = xyz.device

        if not hasattr(self.restraints, "_rama_phi_indices") or self.restraints._rama_phi_indices is None:
            return torch.tensor(0.0, device=device)

        nll_surfaces = self.restraints._rama_surfaces  # (6, 360, 360)

        # Negate because restraints.torsions() uses opposite sign to IUPAC
        phi_deg = -self.restraints.torsions(self.restraints._rama_phi_indices, xyz)
        psi_deg = -self.restraints.torsions(self.restraints._rama_psi_indices, xyz)

        # Convert to grid coordinates [0, 360) with periodic wrapping
        phi_idx = (phi_deg + 180.0) % 360.0
        psi_idx = (psi_deg + 180.0) % 360.0

        # Bilinear interpolation (differentiable via fractional parts)
        phi_lo = phi_idx.detach().floor().long() % 360
        phi_hi = (phi_lo + 1) % 360
        psi_lo = psi_idx.detach().floor().long() % 360
        psi_hi = (psi_lo + 1) % 360
        phi_frac = phi_idx - phi_idx.detach().floor()
        psi_frac = psi_idx - psi_idx.detach().floor()

        # Gather 4 corner NLL values per residue
        s = self.restraints._rama_surface_type
        v00 = nll_surfaces[s, phi_lo, psi_lo]
        v01 = nll_surfaces[s, phi_lo, psi_hi]
        v10 = nll_surfaces[s, phi_hi, psi_lo]
        v11 = nll_surfaces[s, phi_hi, psi_hi]

        nll = (
            (1 - phi_frac) * (1 - psi_frac) * v00
            + (1 - phi_frac) * psi_frac * v01
            + phi_frac * (1 - psi_frac) * v10
            + phi_frac * psi_frac * v11
        )

        return nll.sum()

    def stats(self) -> Dict[str, StatEntry]:
        """Get Ramachandran restraint statistics."""
        if not hasattr(self.restraints, "_rama_phi_indices") or self.restraints._rama_phi_indices is None:
            return {}

        n = self.restraints._rama_phi_indices.shape[0]
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(n, VERBOSITY_DEBUG),
        }
