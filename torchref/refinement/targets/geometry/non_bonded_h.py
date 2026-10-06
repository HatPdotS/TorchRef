"""Non-bonded target with transient riding hydrogen VDW contacts.

Hydrogens are never stored on the model: each ``forward()`` places them from the
current heavy-atom coordinates, scores VDW repulsion on a candidate H-heavy pair
list fixed at restraint-build time, and discards them.
"""

import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.coordinates.symmetry_images import is_symmetry_image
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    stat,
)

from .non_bonded import NonBondedTarget

if TYPE_CHECKING:
    from torchref.model.model import Model
    from torchref.topology.riding import HydrogenTopology


class NonBondedHTarget(NonBondedTarget):
    """Non-bonded target with transient riding hydrogen VDW contacts.

    Drop-in replacement for :class:`NonBondedTarget`, which still computes the
    heavy-heavy loss; this adds an H-VDW term over candidate pairs derived at build
    time from the heavy-heavy list, so a forward costs only H placement and a
    vectorized distance pass. Same generalized-Gaussian NLL as the parent, including
    its sigma calibration.

    Parameters
    ----------
    model : Model, optional
        Reference to Model object.
    mode : str, optional
        Repulsion function type. Default ``'prolsq'``.
    sigma : float, optional
        Effective tolerance on the overlap (Å). Default 0.3.
    r_exp : float, optional
        Repulsion exponent. Default 4.0.
    c_rep : float or None, optional
        Legacy coefficient override; derived from ``sigma`` when None.
    buffer : float, optional
        Distance buffer (Å). Default 0.0.
    verbose : int, optional
        Verbosity level. Default 0.
    """

    name: str = "geometry/nonbonded"

    def __init__(
        self,
        model: "Model" = None,
        mode: str = "prolsq",
        sigma: float = 0.3,
        r_exp: float = 4.0,
        c_rep: "float | None" = None,
        buffer: float = 0.0,
        rebuild_threshold: float = 1.0,
        verbose: int = 0,
    ):
        super().__init__(
            model=model,
            mode=mode,
            sigma=sigma,
            r_exp=r_exp,
            c_rep=c_rep,
            buffer=buffer,
            rebuild_threshold=rebuild_threshold,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # H-VDW loss via precomputed candidates
    # ------------------------------------------------------------------

    @staticmethod
    def _h_candidates(
        xyz: torch.Tensor, h_topo: "HydrogenTopology"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[heavy | riding H]`` coordinates and the candidate pairs indexing them.

        Parameters
        ----------
        xyz : torch.Tensor
            Heavy-atom Cartesian coordinates in Å, ``(N_heavy, 3)``.
        h_topo : HydrogenTopology
            Riding topology with candidate pairs built.

        Returns
        -------
        xyz_all : torch.Tensor
            ``(N_heavy + N_h, 3)`` in Å; the hydrogens are placed from ``xyz`` on
            every call, differentiably.
        indices : torch.Tensor
            ``(P, 2)`` contiguous candidate pairs into ``xyz_all``.
        """
        from torchref.topology.riding import place_riding_hydrogens

        xyz_all = torch.cat([xyz, place_riding_hydrogens(xyz, h_topo)], dim=0)
        indices = torch.stack([h_topo.cand_idx_i, h_topo.cand_idx_j], dim=1)
        return xyz_all, indices.contiguous()

    def _h_pair_positions(
        self, xyz: torch.Tensor, h_topo: "HydrogenTopology"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both ends of every H candidate pair, ``(P, 3)`` each in Å.

        From :func:`~torchref.base.targets.nonbonded.nonbonded_pair_positions`, the
        positions the prolsq kernel scores, symmetry and lattice images included.
        """
        from torchref.base.targets.nonbonded import nonbonded_pair_positions

        xyz_all, indices = self._h_candidates(xyz, h_topo)
        return nonbonded_pair_positions(
            xyz_all,
            indices,
            h_topo.cand_symop_idx,
            h_topo.cand_cell_offset,
            *self._symmetry_tables(),
        )

    def _compute_h_vdw_loss(
        self,
        xyz: torch.Tensor,
        h_topo: "HydrogenTopology",
    ) -> torch.Tensor:
        """VDW loss over the precomputed H-heavy candidate pairs.

        Places riding hydrogens differentiably, so the gradient runs
        loss -> H_pos -> ``xyz[parent_idx]`` -> model parameters. The ``prolsq``
        mode goes through :func:`torchref.base.targets.nonbonded_heavy_math` (Triton on
        CUDA float32); the others score the positions of :meth:`_h_pair_positions`.
        """
        device = xyz.device

        n_cand = h_topo.cand_idx_i.shape[0]
        if n_cand == 0:
            return torch.tensor(0.0, device=device)

        # Fast path: prolsq goes through the dispatcher (Triton on CUDA fp32).
        if self.mode == "prolsq":
            from torchref.base.targets.nonbonded import nonbonded_heavy_math

            xyz_all, indices = self._h_candidates(xyz, h_topo)
            return nonbonded_heavy_math(
                xyz_all, indices, h_topo.cand_min_dist,
                h_topo.cand_symop_idx, h_topo.cand_cell_offset,
                *self._symmetry_tables(),
                self._c_rep, self._r_exp,
                float(self._buffer), self._sigma_vdw,
            )

        pos_i, pos_j = self._h_pair_positions(xyz, h_topo)
        actual_dist = torch.sqrt(((pos_j - pos_i) ** 2).sum(dim=-1) + 1e-8)
        min_dist = h_topo.cand_min_dist

        violations = torch.clamp(min_dist + self._buffer - actual_dist, min=0.0)

        if self.mode == "gaussian":
            sigma_val = torch.tensor(0.2, device=device, dtype=xyz.dtype)
            log_2pi = torch.log(
                torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype)
            )
            nll = (0.5 * (violations / sigma_val) ** 2
                   + torch.log(sigma_val) + 0.5 * log_2pi)
            return nll.sum()
        elif self.mode == "soft":
            threshold = 0.5
            quadratic_mask = violations <= threshold
            quadratic_energy = self._c_rep * (violations ** 2)
            linear_energy = self._c_rep * (
                2 * threshold * violations - threshold ** 2
            )
            energy = torch.where(quadratic_mask, quadratic_energy, linear_energy)
            return energy.sum()
        else:
            raise ValueError(f"Unknown non-bonded mode: {self.mode}")

    # ------------------------------------------------------------------
    # forward / stats / violations
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        """Heavy-heavy VDW loss plus the riding-hydrogen term, when H are available."""
        heavy_loss = super().forward()

        restraints = self.restraints
        if restraints is None:
            return heavy_loss
        h_topo = restraints.h_topo
        if h_topo is None or h_topo.n_hydrogens == 0 or not h_topo.has_candidates:
            return heavy_loss

        xyz = self.model.xyz()
        h_loss = self._compute_h_vdw_loss(xyz, h_topo)
        return heavy_loss + h_loss

    def get_violations(self, threshold: float = 0.0) -> Dict[str, torch.Tensor]:
        """Parent VDW violations plus ``h_*`` entries for H-involving contacts.

        H distances are taken between the positions the loss scores, symmetry and
        lattice images included.
        """
        result = super().get_violations(threshold)

        restraints = self.restraints
        if restraints is None:
            return result
        h_topo = restraints.h_topo
        if h_topo is None or h_topo.n_hydrogens == 0 or not h_topo.has_candidates:
            return result

        pos_i, pos_j = self._h_pair_positions(self.model.xyz(), h_topo)
        actual_dist = torch.norm(pos_j - pos_i, dim=-1)
        violations = torch.clamp(h_topo.cand_min_dist - actual_dist, min=0.0)

        mask = violations > threshold
        if mask.any():
            result["h_cand_idx_i"] = h_topo.cand_idx_i[mask]
            result["h_cand_idx_j"] = h_topo.cand_idx_j[mask]
            result["h_violations"] = violations[mask]
            result["h_distances"] = actual_dist[mask]
            result["h_min_distances"] = h_topo.cand_min_dist[mask]

        return result

    def stats(self) -> Dict[str, any]:
        """Get statistics including H-VDW contacts."""
        result = super().stats()

        restraints = self.restraints
        if restraints is None:
            return result
        h_topo = restraints.h_topo
        if h_topo is None or h_topo.n_hydrogens == 0 or not h_topo.has_candidates:
            return result

        pos_i, pos_j = self._h_pair_positions(self.model.xyz(), h_topo)
        actual_dist = torch.norm(pos_j - pos_i, dim=-1)
        violations = torch.clamp(h_topo.cand_min_dist - actual_dist, min=0.0)

        n_cand = h_topo.cand_idx_i.shape[0]
        n_violations = (violations > 0).sum().item()

        result["h_n_atoms"] = stat(h_topo.n_hydrogens, VERBOSITY_DETAILED)
        result["h_n_candidates"] = stat(n_cand, VERBOSITY_DETAILED)
        result["h_n_violations"] = stat(n_violations, VERBOSITY_DETAILED)

        if n_violations > 0:
            v_mask = violations > 0
            rms = torch.sqrt((violations[v_mask] ** 2).mean()).item()
            result["h_rms_violation"] = stat(rms, VERBOSITY_DETAILED)
            result["h_max_violation"] = stat(violations.max().item(), VERBOSITY_DEBUG)

        is_sym = is_symmetry_image(h_topo.cand_symop_idx, h_topo.cand_cell_offset)
        n_sym = is_sym.sum().item()
        if n_sym > 0:
            result["h_n_symmetry"] = stat(n_sym, VERBOSITY_DETAILED)

        return result
