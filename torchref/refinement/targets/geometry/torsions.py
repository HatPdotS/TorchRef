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


def _von_mises_nll(deviations_rad, sigmas_deg):
    """Unimodal von Mises NLL for torsion deviations.

    Parameters
    ----------
    deviations_rad : torch.Tensor
        Wrapped angular deviations in radians.
    sigmas_deg : torch.Tensor
        Standard deviations in degrees.

    Returns
    -------
    torch.Tensor
        Per-restraint NLL values.
    """
    sigmas_rad = sigmas_deg * (np.pi / 180.0)
    kappa = torch.clamp(1.0 / (sigmas_rad**2), min=1e-3, max=1e4)

    log_i0_kappa = torch.log(torch.special.i0e(kappa)) + kappa
    log_2pi = torch.log(
        torch.tensor(2.0 * np.pi, device=kappa.device, dtype=kappa.dtype)
    )

    log_prob = kappa * torch.cos(deviations_rad) - log_i0_kappa - log_2pi
    return -log_prob


def _omega_mixture_nll(omega_rad, sigmas_deg, is_proline,
                       w_cis_proline=0.05, w_cis_general=0.0005):
    """Cis/trans von Mises mixture NLL for omega torsions.

    Models each omega angle as:

        P(ω) = w_trans · VM(ω; 180°, κ) + w_cis · VM(ω; 0°, κ)

    Since cos(ω − π) = −cos(ω), the NLL simplifies to:

        NLL = log(2π) + log I₀(κ)
              − logsumexp(log w_trans − κ cos ω,  log w_cis + κ cos ω)

    Parameters
    ----------
    omega_rad : torch.Tensor
        Current omega angles in radians.
    sigmas_deg : torch.Tensor
        Standard deviations in degrees (from monomer library, typically 5°).
    is_proline : torch.Tensor
        Boolean mask — True where the next residue is proline.
    w_cis_proline : float
        Prior weight for cis at pre-proline bonds (~5% in PDB).
    w_cis_general : float
        Prior weight for cis at non-proline bonds (~0.05% in PDB).

    Returns
    -------
    torch.Tensor
        Per-restraint NLL values.
    """
    sigmas_rad = sigmas_deg * (np.pi / 180.0)
    kappa = torch.clamp(1.0 / (sigmas_rad**2), min=1e-3, max=1e4)

    log_i0_kappa = torch.log(torch.special.i0e(kappa)) + kappa
    log_2pi = torch.log(
        torch.tensor(2.0 * np.pi, device=kappa.device, dtype=kappa.dtype)
    )
    log_norm = log_2pi + log_i0_kappa

    w_cis = torch.where(
        is_proline,
        torch.tensor(w_cis_proline, device=kappa.device, dtype=kappa.dtype),
        torch.tensor(w_cis_general, device=kappa.device, dtype=kappa.dtype),
    )
    w_trans = 1.0 - w_cis

    cos_omega = torch.cos(omega_rad)
    log_p_trans = torch.log(w_trans) - kappa * cos_omega
    log_p_cis = torch.log(w_cis) + kappa * cos_omega

    log_mixture = torch.logsumexp(torch.stack([log_p_trans, log_p_cis]), dim=0)
    return log_norm - log_mixture


class TorsionTarget(GeometryTarget):
    """
    Torsion angle restraint target.

    Handles all torsion restraints in one target:

    - **Intra-residue & disulfide torsions**: unimodal von Mises NLL
      with periodicity handling.
    - **Omega (peptide bond) torsions**: cis/trans von Mises mixture,
      so both cis and trans conformations are stable wells and the
      X-ray data decides which one to adopt.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    w_cis_proline : float, optional
        Prior weight for cis conformation at pre-proline peptide bonds.
        Default 0.05 (~5% of X-Pro bonds are cis in the PDB).
    w_cis_general : float, optional
        Prior weight for cis conformation at non-proline peptide bonds.
        Default 0.0005 (~0.05% of non-Pro bonds are cis in the PDB).
    """

    name: str = "geometry/torsion"

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        w_cis_proline: float = 0.05,
        w_cis_general: float = 0.0005,
    ):
        super().__init__(model, verbose)
        self.w_cis_proline = w_cis_proline
        self.w_cis_general = w_cis_general

    def _get_omega_data(self):
        """Get omega restraint data from the model's restraints."""
        restraints = self.restraints.restraints
        if "torsion" not in restraints or "omega" not in restraints["torsion"]:
            return None
        return restraints["torsion"]["omega"]

    def forward(self) -> torch.Tensor:
        from torchref.base.targets.torsion import torsion_omega_math
        from torchref.base.targets._dispatch import use_triton
        xyz = self.model.xyz()
        device = xyz.device
        # GPU-only scalar zero: capture-safe (no host→device copy).
        total = torch.zeros((), device=device, dtype=xyz.dtype)

        # --- Intra-residue + disulfide torsions (unimodal von Mises) ---
        # On CUDA fp32 the full dihedral + periodic wrap + von Mises NLL
        # is one Triton kernel; otherwise we fall back to the eager
        # Restraints.torsion_deviations_with_sigmas() + _von_mises_nll path.
        if use_triton(xyz):
            from torchref.base.targets.triton.torsion import (
                torsion_unimodal_full_math_triton,
            )
            if "all" not in self.restraints.restraints["torsion"]:
                self.restraints.cat_dict()
            tdata = self.restraints.restraints["torsion"]["all"]
            if len(tdata["indices"]) > 0:
                total = total + torsion_unimodal_full_math_triton(
                    xyz, tdata["indices"], tdata["references"],
                    tdata["sigmas"], tdata["periods"],
                )
        else:
            deviations_rad, sigmas_deg = self.restraints.torsion_deviations_with_sigmas()
            if len(deviations_rad) > 0:
                total = total + _von_mises_nll(deviations_rad, sigmas_deg).sum()

        # --- Omega torsions (cis/trans mixture) — dispatch through
        # torsion_omega_math, which routes to Triton on CUDA fp32.
        omega_data = self._get_omega_data()
        if omega_data is not None and len(omega_data["indices"]) > 0:
            total = total + torsion_omega_math(
                self.model.xyz(),
                omega_data["indices"],
                omega_data["sigmas"],
                omega_data["is_proline"],
                self.w_cis_proline,
                self.w_cis_general,
            )

        return total

    def stats(self) -> Dict[str, StatEntry]:
        """Get torsion angle statistics."""
        result = {}

        # --- Intra-residue + disulfide stats ---
        deviations_rad, sigmas_deg = self.restraints.torsion_deviations_with_sigmas()
        if len(deviations_rad) > 0:
            deviations_deg = deviations_rad * (180.0 / np.pi)
            sigmas_rad = sigmas_deg * (np.pi / 180.0)
            z_scores = deviations_rad / sigmas_rad
            nll = _von_mises_nll(deviations_rad, sigmas_deg)

            result["n"] = stat(len(deviations_rad), VERBOSITY_DEBUG)
            result["rms_delta"] = stat(
                torch.sqrt((deviations_deg**2).mean()).item(), VERBOSITY_DETAILED
            )
            result["rms_z"] = stat(
                torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED
            )

        # --- Omega stats ---
        omega_data = self._get_omega_data()
        if omega_data is not None and len(omega_data["indices"]) > 0:
            with torch.no_grad():
                indices = omega_data["indices"]
                is_proline = omega_data["is_proline"]
                omega_deg = self.restraints.torsions(indices)

                is_cis = torch.abs(omega_deg) < 90.0
                n_cis = int(is_cis.sum().item())

                # Deviation from nearest ideal (0° or 180°)
                dev_from_trans = torch.abs(
                    torch.remainder(omega_deg, 360.0) - 180.0
                )
                dev_from_cis = torch.abs(
                    torch.remainder(omega_deg + 180.0, 360.0) - 180.0
                )
                min_dev = torch.where(is_cis, dev_from_cis, dev_from_trans)

                result["n_omega"] = stat(len(omega_deg), VERBOSITY_DEBUG)
                result["n_cis"] = stat(n_cis, VERBOSITY_DETAILED)
                result["n_cis_proline"] = stat(
                    int((is_cis & is_proline).sum().item()), VERBOSITY_DEBUG
                )
                result["omega_rms_delta"] = stat(
                    torch.sqrt((min_dev**2).mean()).item(), VERBOSITY_DETAILED
                )

        loss = self.forward()
        result["loss"] = stat(loss.item(), VERBOSITY_STANDARD)

        return result
