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

    Restrains B-factors of bonded atoms to be similar.
    NLL = 0.5 * ((B_i - B_j) / σ)² + log(σ) + 0.5 * log(2π)

    For anisotropic atoms the restraint acts on the full U tensors via the
    unified U6 representation (:meth:`Model.adp_u6`): a magnitude channel on
    ``B_eq`` (which reduces exactly to the isotropic restraint above) plus an
    anisotropy channel on the traceless (deviatoric) part of the B-tensor,
    scaled by ``_simu_sigma_aniso``. Iso<->aniso pairs are handled natively
    (the isotropic atom's deviatoric part is zero). When every atom is
    isotropic the original (Triton-accelerated) B-factor path is used.

    Tunable parameters (as buffers):
    - _simu_sigma: float, sigma for B_eq differences (default 2.0 Å²)
    - _simu_sigma_aniso: float, sigma for deviatoric B-tensor differences
      (default 1.0 Å²); only used when anisotropic atoms are present.
    """

    name: str = "adp/simu"

    def __init__(
        self, model: "Model" = None, simu_sigma: float = 2.0,
        simu_sigma_aniso: float = 1.0, verbose: int = 0
    ):
        super().__init__(model, verbose)
        # Register simu-specific sigma as buffer (separate from base sigma)
        self.register_buffer("_simu_sigma", torch.tensor(simu_sigma))
        self.register_buffer("_simu_sigma_aniso", torch.tensor(simu_sigma_aniso))

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
        """Concatenate non-"all" bond restraint origins into a single
        (N, 2) tensor for the SIMU pair list. Cached after first build."""
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
            cached = torch.empty(0, 2, dtype=torch.long,
                                 device=self.model.xyz().device)
        self._simu_pair_indices_cache = cached
        return cached

    def forward(self) -> torch.Tensor:
        # Use the adp_simu_math dispatcher (Triton on CUDA fp32).
        pair_indices = self._get_pair_indices()
        adp_t = self.model.adp()
        if pair_indices.shape[0] == 0:
            return torch.zeros((), device=adp_t.device, dtype=adp_t.dtype)
        # Lazily move the ``_simu_sigma`` buffer onto the model's device
        # the first time we reach here. Once moved, subsequent forwards
        # (and CUDA-Graph captures) skip the device transfer — calling
        # ``.to()`` on a CPU buffer inside a capture region triggers a
        # ``cudaErrorStreamCaptureUnsupported``.
        if (self._simu_sigma.device != adp_t.device
                or self._simu_sigma.dtype != adp_t.dtype):
            self._simu_sigma = self._simu_sigma.to(
                device=adp_t.device, dtype=adp_t.dtype,
            )
        # Anisotropic atoms present: restrain the full U tensors. Isotropic-only
        # models keep the original (Triton-accelerated) B-factor path so they
        # pay nothing and are numerically unchanged.
        if not getattr(self.model, "_aniso_is_empty", True):
            if (self._simu_sigma_aniso.device != adp_t.device
                    or self._simu_sigma_aniso.dtype != adp_t.dtype):
                self._simu_sigma_aniso = self._simu_sigma_aniso.to(
                    device=adp_t.device, dtype=adp_t.dtype,
                )
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
