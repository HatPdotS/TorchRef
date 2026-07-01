import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.ramachandran import ramachandran_math
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
        super().__init__(model, verbose)

    def forward(self) -> torch.Tensor:
        xyz = self.model.xyz()
        if not hasattr(self.restraints, "_rama_phi_indices") or self.restraints._rama_phi_indices is None:
            return torch.tensor(0.0, device=xyz.device)
        return ramachandran_math(
            xyz,
            self.restraints._rama_phi_indices,
            self.restraints._rama_psi_indices,
            self.restraints._rama_surfaces,
            self.restraints._rama_surface_type,
        )

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
