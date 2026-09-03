import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.adp import adp_simu_math, adp_simu_aniso_math
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


class ADPSimilarityTarget(ADPTarget):
    """
    ADP Similarity restraint (SIMU in Phenix/SHELX).

    Restrains bonded atoms towards similar B, at
    ``0.5·((B_i - B_j)/σ)² + log σ + 0.5·log 2π``.

    With anisotropic atoms present the restraint acts on the full U tensors through the
    unified U6 (:meth:`Model.adp_u6`): a ``B_eq`` magnitude channel reducing exactly to
    the above, plus a deviatoric channel at ``_simu_sigma_aniso``. An isotropic atom has
    zero deviatoric part, so iso<->aniso pairs need no special case; all-isotropic
    models take the Triton-accelerated B-factor path instead.

    Parameters
    ----------
    model : Model, optional
        Reference to Model object.
    simu_sigma : float, optional
        Sigma on B_eq differences (Å²). Default 2.0.
    simu_sigma_aniso : float, optional
        Sigma on deviatoric B-tensor differences (Å²), used only when anisotropic atoms
        are present. Default 1.0.
    verbose : int, optional
        Verbosity level. Default is 0.
    device : torch.device, optional
        Explicit device. When omitted, follows ``model``.
    """

    name: str = "adp/simu"

    def __init__(
        self, model: "Model" = None, simu_sigma: float = 2.0,
        simu_sigma_aniso: float = 1.0, verbose: int = 0, device=None
    ):
        super().__init__(model, verbose, device=device)
        # Buffers, not floats: both reach adp_simu_math / adp_simu_aniso_math, which
        # dispatch to Triton on CUDA float32 and need them already on the right device.
        self._register_scalar("_simu_sigma", float(simu_sigma))
        self._register_scalar("_simu_sigma_aniso", float(simu_sigma_aniso))

    @property
    def simu_sigma(self) -> float:
        """Get SIMU sigma value."""
        return self._simu_sigma.item()

    @simu_sigma.setter
    def simu_sigma(self, value: float):
        """Set SIMU sigma value."""
        self._simu_sigma.fill_(value)

    @property
    def simu_sigma_aniso(self) -> float:
        """Get SIMU deviatoric (anisotropy) sigma value."""
        return self._simu_sigma_aniso.item()

    @simu_sigma_aniso.setter
    def simu_sigma_aniso(self, value: float):
        """Set SIMU deviatoric (anisotropy) sigma value."""
        self._simu_sigma_aniso.fill_(value)

    def _get_pair_indices(self) -> torch.Tensor:
        """Non-"all" bond origins concatenated into one (N, 2) SIMU pair list. Cached
        after the first build, so a rebuilt bond list will not be picked up.
        """
        cached = getattr(self, "_simu_pair_indices_cache", None)
        if cached is not None:
            return cached
        chunks = []
        for origin, group in self.restraints.restraints.get("bond", {}).items():
            if origin == "all":
                continue
            idx_ = group.get("indices")
            if idx_ is not None and len(idx_) > 0:
                chunks.append(idx_)
        if chunks:
            cached = torch.cat(chunks, dim=0).contiguous()
        else:
            cached = torch.empty(0, 2, dtype=torch.long,  # dtype-ok: empty (0,2) atom-pair index tensor; PyTorch requires int64
                                 device=self.model.xyz().device)
        self._simu_pair_indices_cache = cached
        return cached

    def forward(self) -> torch.Tensor:
        """Summed SIMU NLL over bonded pairs; 0.0 when there are no bonds."""
        pair_indices = self._get_pair_indices()
        adp_t = self.model.adp()
        if pair_indices.shape[0] == 0:
            return torch.zeros((), device=adp_t.device, dtype=adp_t.dtype)
        # With any anisotropic atom, restrain the full U tensors; all-isotropic models
        # take the cheaper B-factor path and are numerically unchanged.
        if not getattr(self.model, "_aniso_is_empty", True):
            u6 = self.model.adp_u6()
            return adp_simu_aniso_math(
                u6, pair_indices, self._simu_sigma, self._simu_sigma_aniso
            )
        return adp_simu_math(adp_t, pair_indices, self._simu_sigma)

    def stats(self) -> Dict[str, any]:
        """Get SIMU restraint statistics."""
        b_diffs = self.restraints.adp_b_differences()

        if len(b_diffs) == 0:
            return {}

        b_diffs_abs = b_diffs.abs()
        z_scores = b_diffs_abs / self.simu_sigma
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "count": stat(len(b_diffs), VERBOSITY_DEBUG),
            "rms_delta_b": stat(
                torch.sqrt((b_diffs**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "mean_delta_b": stat(b_diffs_abs.mean().item(), VERBOSITY_DETAILED),
            "max_delta_b": stat(b_diffs_abs.max().item(), VERBOSITY_DETAILED),
            "mean_z": stat(z_scores.mean().item(), VERBOSITY_DEBUG),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
        }
