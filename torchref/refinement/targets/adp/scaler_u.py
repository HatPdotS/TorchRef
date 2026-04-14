import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    stat,
)

from ..base import Target

if TYPE_CHECKING:
    from torchref.scaling.scaler_base import ScalerBase


class ScalerURegularizationTarget(Target):
    """
    Pin the isotropic component of the scaler's anisotropic U to zero.

    ``F_scaled = scale · exp(-2π²·sᵀUs) · F_calc`` couples ``U`` into the
    same Debye-Waller slot as the atomic B-factors. The *anisotropic*
    deviatoric part of U encodes real directional attenuation in the data
    and should stay free. The *isotropic* part — the trace ``U11 + U22 +
    U33`` — is a straight trade with a uniform atomic B shift and must be
    anchored so the atoms absorb the B-factor roll-off, not the scaler.

    Penalty ``(tr U)² · N_ref / 6`` — squared trace, scaled by the order
    of magnitude of the xray gradient on U. The ``/ 6`` matches the
    per-component gradient scaling the user originally requested.
    """

    name: str = "adp/scaler_U"

    def __init__(
        self,
        scaler: "ScalerBase",
        n_reflections: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        object.__setattr__(self, "_scaler", scaler)
        self._scale = float(n_reflections) / 6.0

    def forward(self) -> torch.Tensor:
        U = self._scaler.U
        trace = U[0] + U[1] + U[2]
        return self._scale * trace ** 2

    def stats(self) -> Dict[str, any]:
        U = self._scaler.U.detach()
        trace = (U[0] + U[1] + U[2]).item()
        u_diag = U[:3]
        u_off = U[3:]
        aniso_norm = torch.norm(u_diag - u_diag.mean()).item()
        return {
            "loss": stat(self._scale * trace ** 2, VERBOSITY_STANDARD),
            "tr_U": stat(trace, VERBOSITY_STANDARD),
            "U_iso": stat(trace / 3.0, VERBOSITY_DETAILED),
            "U_aniso_norm": stat(aniso_norm, VERBOSITY_DETAILED),
            "U_off_max": stat(u_off.abs().max().item(), VERBOSITY_DETAILED),
            "U_vec": stat(U.tolist(), VERBOSITY_DEBUG),
        }
