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

from .base import ADPTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class RigidBondTarget(ADPTarget):
    """
    Rigid Bond restraint (DELU in SHELX, Hirshfeld test).

    Based on Hirshfeld's rigid bond test (Acta Cryst. A32, 239, 1976).

    For a truly rigid bond, the mean-square displacement amplitudes (MSDA)
    of the two bonded atoms along the bond direction should be equal.
    This is because in a rigid bond, the atoms move together.

    For anisotropic ADPs (U tensors)::

        z_12 = l_12^T U_1 l_12 / |l_12|²  (MSDA of atom 1 along bond)
        z_21 = l_21^T U_2 l_21 / |l_21|²  (MSDA of atom 2 along bond)
        Δz = z_12 - z_21 should be ~0

    For isotropic B-factors, the difference in B_iso is used as a proxy::

        ΔB = B_1 - B_2

    This differs from SIMU (ADPSimilarityTarget) which restrains the full
    ADP tensors to be similar. Rigid bond only restrains the component
    along the bond direction.

    Energy: E = w * Δz²
    NLL: NLL = 0.5 * (Δz / σ)² + log(σ) + 0.5 * log(2π)

    References
    ----------
    - Hirshfeld, F.L. (1976). Acta Cryst. A32, 239.
    - cctbx/adp_restraints/rigid_bond.h

    Parameters
    ----------
    model : Model
        Reference to Model object.
    sigma : float, optional
        Target standard deviation for Δz. Default is 0.004 Å².
        Hirshfeld found typical values of 0.001 Å² for good structures.
    use_aniso : bool, optional
        If True and model has anisotropic ADPs, use proper tensor calculation.
        Default is True.
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
        Compute rigid bond restraint.

        For isotropic refinement, uses B-factor differences along bonds.
        For anisotropic refinement, computes proper MSDA differences.
        """
        device = self.model.xyz().device

        # Check if model has anisotropic ADPs
        has_aniso = hasattr(self.model, "u_aniso") and self.model.u_aniso is not None

        if has_aniso and self.use_aniso:
            return self._compute_aniso_rigid_bond()
        else:
            return self._compute_iso_rigid_bond()

    def _compute_iso_rigid_bond(self) -> torch.Tensor:
        """
        Compute rigid bond restraint for isotropic B-factors.

        For isotropic ADPs, the MSDA along any direction is B/(8π²).
        So the difference in MSDA is proportional to ΔB.

        We use ΔB directly and scale sigma accordingly.
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

                # For isotropic ADPs: U_iso = B / (8π²)
                # MSDA along bond = U_iso (same in all directions)
                # Δz = (B1 - B2) / (8π²)
                delta_z = (adp1 - adp2) / (8.0 * np.pi**2)
                delta_z_list.append(delta_z)

        if not delta_z_list:
            return torch.tensor(0.0, device=device)

        delta_z = torch.cat(delta_z_list, dim=0)

        # Gaussian NLL
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))
        nll = 0.5 * (delta_z / self.sigma) ** 2 + np.log(self.sigma) + 0.5 * log_2pi

        return nll.sum()

    def _compute_aniso_rigid_bond(self) -> torch.Tensor:
        """
        Compute rigid bond restraint for anisotropic ADPs.

        For each bond:
            l = (r2 - r1) / |r2 - r1|  (unit vector along bond)
            z_12 = l^T U_1 l  (MSDA of atom 1 along bond direction)
            z_21 = l^T U_2 l  (MSDA of atom 2 along bond direction)
            Δz = z_12 - z_21

        The U tensor is symmetric 3x3, stored as 6 unique values:
            U = [[U11, U12, U13],
                 [U12, U22, U23],
                 [U13, U23, U33]]
        """
        xyz = self.model.xyz()
        device = xyz.device

        # Get anisotropic U tensors (N, 6) -> (N, 3, 3)
        u_aniso = self.model.u_aniso  # Shape: (N, 6) for U11,U22,U33,U12,U13,U23

        # Convert to full symmetric matrices
        n_atoms = u_aniso.shape[0]
        U = torch.zeros(n_atoms, 3, 3, device=device, dtype=xyz.dtype)
        U[:, 0, 0] = u_aniso[:, 0]  # U11
        U[:, 1, 1] = u_aniso[:, 1]  # U22
        U[:, 2, 2] = u_aniso[:, 2]  # U33
        U[:, 0, 1] = u_aniso[:, 3]  # U12
        U[:, 1, 0] = u_aniso[:, 3]  # U12 (symmetric)
        U[:, 0, 2] = u_aniso[:, 4]  # U13
        U[:, 2, 0] = u_aniso[:, 4]  # U13 (symmetric)
        U[:, 1, 2] = u_aniso[:, 5]  # U23
        U[:, 2, 1] = u_aniso[:, 5]  # U23 (symmetric)

        delta_z_list = []

        if "bond" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        for origin, restraint_group in self.restraints.restraints["bond"].items():
            if origin == "all":
                continue
            indices = restraint_group.get("indices")
            if indices is not None and len(indices) > 0:
                idx1 = indices[:, 0]
                idx2 = indices[:, 1]

                # Get positions and compute bond vectors
                r1 = xyz[idx1]  # (n_bonds, 3)
                r2 = xyz[idx2]  # (n_bonds, 3)
                bond_vec = r2 - r1  # (n_bonds, 3)
                # Add small epsilon to prevent division by zero gradient issues
                bond_length = torch.sqrt((bond_vec**2).sum(dim=-1, keepdim=True) + 1e-8)
                l = bond_vec / bond_length  # Unit vector along bond

                # Get U tensors for bonded atoms
                U1 = U[idx1]  # (n_bonds, 3, 3)
                U2 = U[idx2]  # (n_bonds, 3, 3)

                # Compute MSDA along bond direction: z = l^T U l
                # For batch: z = sum_ij l_i U_ij l_j
                # Using einsum: z = einsum('bi,bij,bj->b', l, U, l)
                z_12 = torch.einsum("bi,bij,bj->b", l, U1, l)
                z_21 = torch.einsum("bi,bij,bj->b", l, U2, l)

                delta_z = z_12 - z_21
                delta_z_list.append(delta_z)

        if not delta_z_list:
            return torch.tensor(0.0, device=device)

        delta_z = torch.cat(delta_z_list, dim=0)

        # Gaussian NLL
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))
        nll = 0.5 * (delta_z / self.sigma) ** 2 + np.log(self.sigma) + 0.5 * log_2pi

        return nll.sum()

    def get_delta_z_stats(self) -> Dict[str, float]:
        """
        Get statistics of Δz values for analysis.

        Returns
        -------
        dict
            Dictionary with mean, std, max, min of |Δz| values and Z-scores.
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

        # Z-scores (deviation / sigma)
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
        """
        Get rigid bond restraint statistics.

        Returns statistics including Δz values along bonds.
        """
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
