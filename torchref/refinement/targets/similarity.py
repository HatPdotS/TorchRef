"""Coordinate similarity target for difference refinement.

A spike-and-slab prior on per-atom dark-to-light displacement: quadratic where the
displacement looks like noise, flat where it looks like real conformational change.
Per-atom coordinate uncertainty comes from the B-factors, ``sigma = sqrt(B/8π²)``.
"""

import torch
from typing import TYPE_CHECKING, Dict

from .base import Target
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.model.model import Model


class CoordinateSimilarityTarget(Target):
    """
    Spike-and-slab similarity restraint between dark and light models.

    Each atom is either static (its displacement is noise) or genuinely moved, and the
    loss is the negative log marginal likelihood over the two::

        L(d) = -logsumexp(-d²/(2σ²) + alpha, 0)

    with ``d = ||xyz_light - xyz_dark||`` and ``σ = sqrt(B/8π²)``. Its gradient,
    ``(d/σ²)·sigmoid(-d²/(2σ²) + alpha)``, is an L2 restraint weighted by the posterior
    probability that the atom is static: quadratic for ``d << σ``, fully plateaued for
    ``d >> σ`` so real moves go unpenalised, crossing over at ``d ~ σ·sqrt(2·alpha)``.

    Parameters
    ----------
    model_dark : Model
        Dark (ground state) model. B-factors and coordinates are detached.
    model_light : Model
        Light (excited state) model. Coordinates carry gradients.
    alpha : float, optional
        Log prior odds of the static hypothesis. Higher values mean
        stronger denoising. Default is 2.0 (crossover at ~2*sigma).
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "similarity"

    def __init__(
        self,
        model_dark: "Model" = None,
        model_light: "Model" = None,
        alpha: float = 2.0,
        verbose: int = 0,
        device=None,
    ):
        super().__init__(verbose=verbose, device=device)
        self.add_module("_model_dark", model_dark)
        self.add_module("_model_light", model_light)
        self._adopt_device(model_dark, model_light, device=device)
        self._register_scalar("_alpha", float(alpha))
        # Registered unconditionally, BEFORE the map is built: ``_build_atom_map``
        # runs only when both models are present, so without these the empty-init
        # path (the one ``load_state_dict`` uses) would have no such buffers at all.
        # ``_build_atom_map`` overwrites them rather than creating them.
        self.register_buffer(
            "_idx_dark", torch.zeros(0, dtype=torch.long, device=self.device)  # dtype-ok: index buffer for gather/index_select; PyTorch requires int64
        )
        self.register_buffer(
            "_idx_light", torch.zeros(0, dtype=torch.long, device=self.device)  # dtype-ok: index buffer for gather/index_select; PyTorch requires int64
        )
        if model_dark is not None and model_light is not None:
            self._build_atom_map()

    @property
    def model_dark(self) -> "Model":
        """Get dark model."""
        return self._model_dark

    @property
    def model_light(self) -> "Model":
        """Get light model."""
        return self._model_light

    @property
    def alpha(self) -> float:
        """Get alpha as float."""
        return self._alpha.item()

    @alpha.setter
    def alpha(self, value: float):
        """Set alpha."""
        self._alpha.fill_(value)

    def _build_atom_map(self):
        """Match atoms between the two models on
        ``(chainid, resseq, icode, name, altloc)``, overwriting the index buffers.
        Warns when nothing matches or fewer than 90% of atoms do.
        """
        import pandas as pd
        import warnings

        pdb_dark = self._model_dark.pdb.copy()
        pdb_light = self._model_light.pdb.copy()

        for df in (pdb_dark, pdb_light):
            df["_key"] = (
                df["chainid"].astype(str)
                + "_"
                + df["resseq"].astype(str)
                + "_"
                + df["icode"].astype(str).str.strip()
                + "_"
                + df["name"].astype(str).str.strip()
                + "_"
                + df["altloc"].astype(str).str.strip()
            )

        pdb_dark["_idx"] = range(len(pdb_dark))
        pdb_light["_idx"] = range(len(pdb_light))

        merged = pd.merge(
            pdb_dark[["_key", "_idx"]],
            pdb_light[["_key", "_idx"]],
            on="_key",
            suffixes=("_dark", "_light"),
        )

        n_matched = len(merged)
        n_dark = len(pdb_dark)
        n_light = len(pdb_light)

        if n_matched == 0:
            warnings.warn(
                "CoordinateSimilarityTarget: no matching atoms between "
                "dark and light models"
            )
            self.register_buffer(
                "_idx_dark", torch.zeros(0, dtype=torch.long, device=self.device)  # dtype-ok: index buffer for gather/index_select; PyTorch requires int64
            )
            self.register_buffer(
                "_idx_light", torch.zeros(0, dtype=torch.long, device=self.device)  # dtype-ok: index buffer for gather/index_select; PyTorch requires int64
            )
            return

        match_rate = n_matched / min(n_dark, n_light)
        if match_rate < 0.9:
            warnings.warn(
                f"CoordinateSimilarityTarget: only {n_matched}/{min(n_dark, n_light)} "
                f"atoms matched ({match_rate:.0%})"
            )

        if self.verbose >= 1:
            print(
                f"  Similarity target: {n_matched} matched atoms "
                f"(dark={n_dark}, light={n_light})"
            )

        # Must be on ``self.device``: as 1-D index tensors they get no
        # scalar-promotion, so a CPU-resident index against accelerator coordinates
        # costs a host sync every forward.
        self.register_buffer(
            "_idx_dark",
            torch.tensor(
                merged["_idx_dark"].values, dtype=torch.long, device=self.device  # dtype-ok: atom index tensor used for indexing; PyTorch requires int64
            ),
        )
        self.register_buffer(
            "_idx_light",
            torch.tensor(
                merged["_idx_light"].values, dtype=torch.long, device=self.device  # dtype-ok: atom index tensor used for indexing; PyTorch requires int64
            ),
        )

    def forward(self) -> torch.Tensor:
        """Spike-and-slab similarity loss, summed over matched atom pairs (0.0 if
        none matched). The dark model's coordinates and B-factors are detached.
        """
        if len(self._idx_dark) == 0:
            # ``self.device``, not ``self._alpha.device``: an empty target must
            # still hand back a loss on the refinement's device.
            return torch.tensor(0.0, device=self.device, dtype=self.dtype_float)

        xyz_dark = self._model_dark.xyz()
        xyz_light = self._model_light.xyz()

        # Dark is the frozen reference, so it is detached.
        pos_dark = xyz_dark[self._idx_dark].detach()
        pos_light = xyz_light[self._idx_light]

        delta_sq = (pos_light - pos_dark).pow(2).sum(dim=-1)

        B = self._model_dark.adp()[self._idx_dark].detach()
        sigma_sq = B / (8.0 * torch.pi**2)
        sigma_sq = torch.clamp(sigma_sq, min=1e-4)

        z_static = -0.5 * delta_sq / sigma_sq + self._alpha
        loss = -torch.logaddexp(z_static, torch.zeros_like(z_static))

        return loss.sum()

    def stats(self) -> Dict[str, StatEntry]:
        """Get similarity restraint statistics."""
        if len(self._idx_dark) == 0:
            return {}

        with torch.no_grad():
            xyz_dark = self._model_dark.xyz()
            xyz_light = self._model_light.xyz()

            pos_dark = xyz_dark[self._idx_dark]
            pos_light = xyz_light[self._idx_light]

            diff = pos_light - pos_dark
            delta_sq = (diff**2).sum(dim=-1)
            distances = torch.sqrt(delta_sq + 1e-8)

            B = self._model_dark.adp()[self._idx_dark]
            sigma_sq = B / (8.0 * torch.pi**2)
            sigma_sq = torch.clamp(sigma_sq, min=1e-4)
            sigma = torch.sqrt(sigma_sq)

            # Posterior P(static) = sigmoid(-delta^2/(2*sigma^2) + alpha)
            p_static = torch.sigmoid(-0.5 * delta_sq / sigma_sq + self._alpha)
            n_moved = (p_static < 0.5).sum().item()

            loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n_matched": stat(len(self._idx_dark), VERBOSITY_DEBUG),
            "n_moved": stat(n_moved, VERBOSITY_DETAILED),
            "rms_dist": stat(
                torch.sqrt((distances**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "mean_dist": stat(distances.mean().item(), VERBOSITY_DETAILED),
            "max_dist": stat(distances.max().item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigma.mean().item(), VERBOSITY_DETAILED),
            "alpha": stat(self._alpha.item(), VERBOSITY_DEBUG),
        }
