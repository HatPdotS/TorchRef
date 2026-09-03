"""Rigid-bond (Hirshfeld / SHELX DELU) ADP restraint."""

import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.adp import adp_rigid_bond_aniso_math
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


class RigidBondTarget(ADPTarget):
    """
    Rigid Bond restraint (DELU in SHELX, Hirshfeld's rigid-bond test).

    Atoms joined by a rigid bond move together, so their mean-square displacement
    amplitudes *along the bond* should match. For anisotropic ADPs::

        z_12 = l_12^T U_1 l_12 / |l_12|²   (MSDA of atom 1 along the bond)
        z_21 = l_21^T U_2 l_21 / |l_21|²
        Δz   = z_12 - z_21  ->  0

    with ΔB = B_1 - B_2 as the isotropic proxy, and
    NLL = 0.5·(Δz/σ)² + log σ + 0.5·log 2π. Unlike SIMU
    (:class:`ADPSimilarityTarget`), which restrains the whole tensors to be
    similar, this constrains only the bond-direction component.

    References
    ----------
    - Hirshfeld, F.L. (1976). Acta Cryst. A32, 239.
    - cctbx/adp_restraints/rigid_bond.h

    Parameters
    ----------
    model : Model
        Reference to Model object.
    sigma : float, optional
        Target standard deviation for Δz, default 0.004 Å² -- ~4x looser than
        Hirshfeld's 0.001 Å² for good structures, for numerical stability and
        because Δz is noisier on mixed iso/aniso models.
    use_aniso : bool, optional
        Use the tensor calculation when the model has anisotropic ADPs. Default
        True; setting False forces the isotropic proxy.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "adp/delu"

    def __init__(
        self,
        model: "Model" = None,
        sigma: float = 0.004,
        use_aniso: bool = True,
        verbose: int = 0,
    ):
        super().__init__(model, verbose)
        self.sigma = sigma
        self.use_aniso = use_aniso

    def forward(self) -> torch.Tensor:
        """
        Summed rigid-bond NLL: directional MSDA if the model has any anisotropic
        atoms, the cheaper ΔB proxy otherwise (the two agree for isotropic atoms).
        """
        if self.use_aniso and not getattr(self.model, "_aniso_is_empty", True):
            return self._compute_aniso_rigid_bond()
        return self._compute_iso_rigid_bond()

    def _compute_iso_rigid_bond(self) -> torch.Tensor:
        """Rigid-bond NLL from ΔB: for isotropic ADPs the MSDA is B/(8π²) in every
        direction, so Δz is just ΔB/(8π²) and sigma is scaled to match.
        """
        adp = self.model.adp()
        xyz = self.model.xyz()
        device = xyz.device

        delta_z_list = []

        if "bond" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        for origin, restraint_group in self.restraints.restraints["bond"].items():
            if origin == "all":
                continue
            indices = restraint_group.get("indices")
            if indices is not None and len(indices) > 0:
                adp1 = adp[indices[:, 0]]
                adp2 = adp[indices[:, 1]]

                delta_z = (adp1 - adp2) / (8.0 * np.pi**2)
                delta_z_list.append(delta_z)

        if not delta_z_list:
            return torch.tensor(0.0, device=device)

        delta_z = torch.cat(delta_z_list, dim=0)

        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))
        nll = 0.5 * (delta_z / self.sigma) ** 2 + np.log(self.sigma) + 0.5 * log_2pi

        return nll.sum()

    def _bond_pairs(self) -> torch.Tensor:
        """Concatenate non-"all" bond restraint origins into one (N, 2) tensor."""
        chunks = []
        for origin, group in self.restraints.restraints.get("bond", {}).items():
            if origin == "all":
                continue
            idx_ = group.get("indices")
            if idx_ is not None and len(idx_) > 0:
                chunks.append(idx_)
        if chunks:
            return torch.cat(chunks, dim=0).contiguous()
        return torch.empty(0, 2, dtype=torch.long, device=self.model.xyz().device)  # dtype-ok: empty (0,2) atom-pair index tensor; PyTorch requires int64

    def _compute_aniso_rigid_bond(self) -> torch.Tensor:
        """Rigid-bond NLL from ``Δz = l^T U_1 l - l^T U_2 l`` along each bond.

        Driven by the unified per-atom U6 (:meth:`Model.adp_u6`), where isotropic
        atoms appear as lifted ``U = (B/8π²)I``, so mixed iso/aniso bonds need no
        special case and agree with :meth:`_compute_iso_rigid_bond`.
        """
        xyz = self.model.xyz()
        pairs = self._bond_pairs()
        if pairs.shape[0] == 0:
            return torch.zeros((), device=xyz.device, dtype=xyz.dtype)
        u6 = self.model.adp_u6()
        return adp_rigid_bond_aniso_math(u6, xyz, pairs, self.sigma)

    def get_delta_z_stats(self) -> Dict[str, float]:
        """
        Statistics of the Δz values, from the isotropic ΔB path regardless of
        ``use_aniso`` -- so on an anisotropic model these describe a proxy for what
        ``forward`` actually minimises, not the same quantity.

        Returns
        -------
        dict
            ``count``, ``mean``/``std``/``max``/``min``/``rms`` of |Δz|, and the
            ``mean_z``/``rms_z`` Z-scores. All zero when there are no bonds.
        """
        adp = self.model.adp()
        xyz = self.model.xyz()
        device = xyz.device

        delta_z_list = []

        if "bond" not in self.restraints.restraints:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "rms": 0.0,
                "mean_z": 0.0,
                "rms_z": 0.0,
            }

        for origin, restraint_group in self.restraints.restraints["bond"].items():
            if origin == "all":
                continue
            indices = restraint_group.get("indices")
            if indices is not None and len(indices) > 0:
                adp1 = adp[indices[:, 0]]
                adp2 = adp[indices[:, 1]]
                delta_z = (adp1 - adp2) / (8.0 * np.pi**2)
                delta_z_list.append(delta_z)

        if not delta_z_list:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "rms": 0.0,
                "mean_z": 0.0,
                "rms_z": 0.0,
            }

        delta_z_all = torch.cat(delta_z_list, dim=0)
        delta_z_abs = delta_z_all.abs()

        z_scores = delta_z_abs / self.sigma

        return {
            "count": len(delta_z_all),
            "mean": delta_z_abs.mean().item(),
            "std": delta_z_all.std().item(),
            "max": delta_z_abs.max().item(),
            "min": delta_z_abs.min().item(),
            "rms": torch.sqrt((delta_z_all**2).mean()).item(),
            "mean_z": z_scores.mean().item(),
            "rms_z": torch.sqrt((z_scores**2).mean()).item(),
        }

    def stats(self) -> Dict[str, any]:
        """Rigid-bond loss plus the Δz summary from :meth:`get_delta_z_stats`."""
        delta_z_stats = self.get_delta_z_stats()
        if delta_z_stats.get("count", 0) == 0:
            return {}

        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "count": stat(delta_z_stats["count"], VERBOSITY_DEBUG),
            "rms": stat(delta_z_stats["rms"], VERBOSITY_DETAILED),
            "mean": stat(delta_z_stats["mean"], VERBOSITY_DETAILED),
            "max": stat(delta_z_stats["max"], VERBOSITY_DETAILED),
            "rms_z": stat(delta_z_stats["rms_z"], VERBOSITY_DETAILED),
            "std": stat(delta_z_stats["std"], VERBOSITY_DEBUG),
            "min": stat(delta_z_stats["min"], VERBOSITY_DEBUG),
            "mean_z": stat(delta_z_stats["mean_z"], VERBOSITY_DEBUG),
        }
