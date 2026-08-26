"""Non-bonded target with transient riding hydrogen VDW contacts.

Hydrogens are never stored on the model: each ``forward()`` places them from the
current heavy-atom coordinates, scores VDW repulsion on a candidate H-heavy pair
list fixed at restraint-build time, and discards them.
"""

import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.config import dtypes
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    stat,
)

from .non_bonded import NonBondedTarget

if TYPE_CHECKING:
    from torchref.model.model import Model
    from torchref.restraints.hydrogen_topology import HydrogenTopology


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

    def _compute_h_vdw_loss(
        self,
        xyz: torch.Tensor,
        h_topo: "HydrogenTopology",
    ) -> torch.Tensor:
        """VDW loss over the precomputed H-heavy candidate pairs.

        Places riding hydrogens differentiably, so the gradient runs
        loss -> H_pos -> ``xyz[parent_idx]`` -> model parameters. The candidate list
        must stay sorted ASU-then-sym: the ``prolsq`` fast path hands
        ``cand_symop_idx`` straight to
        :func:`torchref.base.targets.nonbonded_heavy_math`, which relies on that
        ordering to do identity and real symmetry transforms in one pass. Other modes
        take the inline eager path below.
        """
        from torchref.restraints.hydrogen_topology import place_riding_hydrogens

        device = xyz.device

        xyz_h = place_riding_hydrogens(xyz, h_topo)
        xyz_all = torch.cat([xyz, xyz_h], dim=0)  # [heavy | H]

        n_cand = h_topo.cand_idx_i.shape[0]
        if n_cand == 0:
            return torch.tensor(0.0, device=device)

        # Fast path: prolsq goes through the dispatcher (Triton on CUDA fp32).
        if self.mode == "prolsq":
            from torchref.base.targets.nonbonded import nonbonded_heavy_math
            indices = torch.stack(
                [h_topo.cand_idx_i, h_topo.cand_idx_j], dim=1
            ).contiguous()
            return nonbonded_heavy_math(
                xyz_all, indices, h_topo.cand_min_dist,
                h_topo.cand_symop_idx, h_topo.cand_cell_offset,
                self.model.spacegroup.matrices,
                self.model.spacegroup.translations,
                self.model.cell.fractional_matrix,
                self.model.cell.inv_fractional_matrix,
                self._c_rep, self._r_exp,
                float(self._buffer), self._sigma_vdw,
            )

        # Slow path: gaussian / soft modes, inline eager.
        pos_i = xyz_all[h_topo.cand_idx_i]
        n_asu = getattr(h_topo, 'n_asu_candidates', n_cand)
        n_sym = n_cand - n_asu
        min_dist = h_topo.cand_min_dist

        if n_asu > 0:
            pos_j_asu = xyz_all[h_topo.cand_idx_j[:n_asu]]
            diff_asu = pos_j_asu - pos_i[:n_asu]
            dist_asu = torch.sqrt((diff_asu ** 2).sum(dim=-1) + 1e-8)

        if n_sym > 0:
            cell = self.model.cell
            sg = self.model.spacegroup
            sym_source = xyz_all[h_topo.cand_idx_j[n_asu:]]
            frac = cell.cartesian_to_fractional(sym_source)
            R = sg.matrices[h_topo.cand_symop_idx[n_asu:]].to(frac.dtype)
            t = sg.translations[h_topo.cand_symop_idx[n_asu:]].to(frac.dtype)
            offs = h_topo.cand_cell_offset[n_asu:].to(frac.dtype)
            frac_t = torch.bmm(R, frac.unsqueeze(-1)).squeeze(-1) + t + offs
            pos_j_sym = cell.fractional_to_cartesian(frac_t)
            diff_sym = pos_j_sym - pos_i[n_asu:]
            dist_sym = torch.sqrt((diff_sym ** 2).sum(dim=-1) + 1e-8)

        if n_asu > 0 and n_sym > 0:
            actual_dist = torch.cat([dist_asu, dist_sym])
        elif n_asu > 0:
            actual_dist = dist_asu
        else:
            actual_dist = dist_sym

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

        The H distances here ignore symmetry -- both positions are read straight out of
        ``xyz_all`` -- so symmetry-mate H contacts are reported at their intra-ASU
        separation, unlike in the loss.
        """
        from torchref.restraints.hydrogen_topology import place_riding_hydrogens

        result = super().get_violations(threshold)

        restraints = self.restraints
        if restraints is None:
            return result
        h_topo = restraints.h_topo
        if h_topo is None or h_topo.n_hydrogens == 0 or not h_topo.has_candidates:
            return result

        xyz = self.model.xyz()
        device = xyz.device
        xyz_h = place_riding_hydrogens(xyz, h_topo)
        xyz_all = torch.cat([xyz, xyz_h], dim=0)

        pos_i = xyz_all[h_topo.cand_idx_i]
        pos_j = xyz_all[h_topo.cand_idx_j]

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
        from torchref.restraints.hydrogen_topology import place_riding_hydrogens

        result = super().stats()

        restraints = self.restraints
        if restraints is None:
            return result
        h_topo = restraints.h_topo
        if h_topo is None or h_topo.n_hydrogens == 0 or not h_topo.has_candidates:
            return result

        xyz = self.model.xyz()
        device = xyz.device
        xyz_h = place_riding_hydrogens(xyz, h_topo)
        xyz_all = torch.cat([xyz, xyz_h], dim=0)

        pos_i = xyz_all[h_topo.cand_idx_i]
        pos_j = xyz_all[h_topo.cand_idx_j]

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

        n_sym = ((h_topo.cand_symop_idx != 0)
                 | (h_topo.cand_cell_offset != 0).any(dim=1)).sum().item()
        if n_sym > 0:
            result["h_n_symmetry"] = stat(n_sym, VERBOSITY_DETAILED)

        return result
