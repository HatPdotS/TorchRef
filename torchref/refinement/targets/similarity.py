"""
Coordinate Similarity Target for Difference Refinement

Implements a spike-and-slab prior on per-atom displacements between
dark and light models. The loss is quadratic for small displacements
(likely noise) and completely flat for large displacements (likely
genuine conformational changes).

Per-atom coordinate uncertainty sigma is derived from B-factors:
    sigma = sqrt(B / (8*pi^2))
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

    For each atom, two hypotheses are considered:

    - **Static** (prob 1-p): atom did not move, displacement is noise
    - **Moved** (prob p): atom genuinely displaced

    The loss is the negative log marginal likelihood:

        L(d) = -logsumexp(-d^2/(2*sigma^2) + alpha, 0)

    where d = ||xyz_light - xyz_dark|| and sigma = sqrt(B / (8*pi^2))
    is the per-atom coordinate uncertainty from B-factors.

    Gradient: (d / sigma^2) * sigmoid(-d^2/(2*sigma^2) + alpha)
    This is an L2 restraint weighted by the posterior probability
    that the atom is static.

    Behavior:
    - d << sigma: ~0.5 * d^2 / sigma^2  (quadratic, tight restraint)
    - d >> sigma: plateaus completely (no penalty for genuine moves)
    - Crossover at d ~ sigma * sqrt(2*alpha)

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
    ):
        super().__init__(verbose=verbose)
        self.add_module("_model_dark", model_dark)
        self.add_module("_model_light", model_light)
        self.register_buffer("_alpha", torch.tensor(alpha))
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
        """Match atoms between dark and light models by identity.

        Creates index arrays mapping corresponding atoms between the two
        models based on (chainid, resseq, icode, name, altloc) keys.
        """
        import pandas as pd
        import warnings

        pdb_dark = self._model_dark.pdb.copy()
        pdb_light = self._model_light.pdb.copy()

        # Build unique atom key
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

        # Add integer index columns
        pdb_dark["_idx"] = range(len(pdb_dark))
        pdb_light["_idx"] = range(len(pdb_light))

        # Inner merge to find matching atoms
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
                "_idx_dark", torch.zeros(0, dtype=torch.long)
            )
            self.register_buffer(
                "_idx_light", torch.zeros(0, dtype=torch.long)
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

        self.register_buffer(
            "_idx_dark",
            torch.tensor(merged["_idx_dark"].values, dtype=torch.long),
        )
        self.register_buffer(
            "_idx_light",
            torch.tensor(merged["_idx_light"].values, dtype=torch.long),
        )

    def forward(self) -> torch.Tensor:
        """Compute spike-and-slab similarity loss.

        Returns
        -------
        torch.Tensor
            Scalar mean loss over all matched atom pairs.
        """
        if len(self._idx_dark) == 0:
            device = self._alpha.device
            return torch.tensor(0.0, device=device)

        xyz_dark = self._model_dark.xyz()
        xyz_light = self._model_light.xyz()

        # Select matched atoms; detach dark (frozen reference)
        pos_dark = xyz_dark[self._idx_dark].detach()
        pos_light = xyz_light[self._idx_light]

        # Per-atom squared displacement
        delta_sq = (pos_light - pos_dark).pow(2).sum(dim=-1)

        # Per-atom sigma^2 from dark model B-factors
        B = self._model_dark.adp()[self._idx_dark].detach()
        sigma_sq = B / (8.0 * torch.pi**2)
        sigma_sq = torch.clamp(sigma_sq, min=1e-4)

        # Spike-and-slab: -logsumexp(-delta^2/(2*sigma^2) + alpha, 0)
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
