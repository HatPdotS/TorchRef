import numpy as np
import torch
from typing import TYPE_CHECKING, Dict, Tuple

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


class NonBondedTarget(GeometryTarget):
    r"""
    Non-bonded (van der Waals) restraint target using PROLSQ-style repulsion,
    parameterized as a generalized-Gaussian NLL on the overlap with
    scale :math:`\sigma`.

    Per-pair NLL (``mode='prolsq'``):

    .. math::

       \mathrm{NLL}(v) \;=\;
           \frac{v^{p}}{p\,\sigma^{p}}
           \;+\; \log\sigma
           \;+\; \tfrac{1}{2}\log(2\pi)
       \qquad
       v \;=\; \max\!\bigl(0,\, d_{\text{vdw}} + b - d\bigr)

    where :math:`p = r_\text{exp}` (default 4). The shape term
    :math:`v^p/(p\sigma^p)` is algebraically identical to the classical PROLSQ
    energy :math:`c_{\text{rep}}\,v^p` with :math:`c_{\text{rep}} =
    1/(p\,\sigma^{p})`; exposing :math:`\sigma` makes the physics legible
    (σ is an "effective tolerance" on the overlap) and puts the VDW loss on
    the same NLL footing as bond / angle / planarity.

    **Default sigma = 0.3 Å.** A practical middle ground: stiff enough
    to reliably pull medium-large clashes (~0.4–0.5 Å) out of the
    refinement but loose enough that starting from a shaken or rough
    model doesn't generate LBFGS-destabilising gradients. A 0.4 Å
    MolProbity clash sits at ~1.3σ and contributes ~0.8 NLL units; a
    0.5 Å severe clash sits at ~1.7σ and contributes ~1.9 NLL units.
    The classical PROLSQ strength ``c_rep=16, r_exp=4`` is equivalent
    to :math:`\sigma \approx 0.354\ \text{\AA}`, very close to this
    default. Set ``sigma`` explicitly for deliberate tightening (e.g.
    0.13 Å → 3σ clash, ~20 NLL units) or loosening.

    Alternative modes:

    - ``'prolsq'``: generalized-Gaussian NLL with exponent ``r_exp`` (default)
    - ``'gaussian'``: Gaussian NLL on the overlap using per-pair sigmas
    - ``'soft'``: soft repulsion with linear core outside ``threshold``

    When symmetry information is available (cell and spacegroup on the model),
    also handles contacts between ASU atoms and symmetry-related copies.
    Symmetry mate positions are recomputed on-the-fly from current ASU
    coordinates so that gradients flow to both atoms in each pair.

    Reference: cctbx/geometry_restraints/nonbonded.h, PROLSQ documentation,
    MolProbity clash criterion (Davis et al., NAR 2007).

    Parameters
    ----------
    model : Model, optional
        Reference to Model object.
    mode : str, optional
        Repulsion function type ('prolsq', 'gaussian', 'soft'). Default is 'prolsq'.
    sigma : float, optional
        Effective tolerance on the overlap in Angstroms. Default is 0.3.
    r_exp : float, optional
        Exponent of the repulsion term. Default is 4.0.
    c_rep : float, optional
        Back-door repulsion coefficient. If provided, overrides the
        sigma-derived value and the NLL becomes
        :math:`c_{\text{rep}} v^{r_\text{exp}} + \log\sigma + \tfrac{1}{2}\log(2\pi)`.
        Useful for reproducing legacy PROLSQ weights. Default is None
        (derive from ``sigma``).
    buffer : float, optional
        Distance buffer in Angstroms added to VDW radii sum. Shifts the
        repulsion onset outward so atoms feel repulsion before they clash.
        Default is 0.0.
    verbose : int, optional
        Verbosity level. Default is 0.
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
        scale: float = 10.0,
    ):
        """
        Initialize non-bonded target.

        Parameters
        ----------
        model : Model, optional
            Reference to Model object.
        mode : str, optional
            Repulsion function type ('prolsq', 'gaussian', 'soft'). Default is 'prolsq'.
        sigma : float, optional
            Effective tolerance on the overlap (Å). Default 0.3. Only
            used when ``c_rep`` is None.
        r_exp : float, optional
            Repulsion exponent. Default is 4.0.
        c_rep : float or None, optional
            Legacy coefficient override. If None (default), derived from
            ``sigma`` as ``1 / (r_exp * sigma ** r_exp)``.
        buffer : float, optional
            Distance buffer in Angstroms added to VDW radii sum. Default is 0.0.
        rebuild_threshold : float, optional
            Maximum ASU atom displacement in Angstroms since the last
            VDW pair-list build before :meth:`maintenance` triggers a
            rebuild. Default is 1.0 Å — well inside the ~2.4 Å safety
            margin of the default 6.0 Å cutoff, so newly-formed contacts
            cannot slip through the list.
        verbose : int, optional
            Verbosity level. Default is 0.
        """
        super().__init__(model, verbose, target_value=0.5, sigma=1.2)
        self.mode = mode
        self.scale = scale
        # Register sigma / r_exp / buffer as buffers so .to(device) moves them.
        self.register_buffer("_sigma_vdw", torch.tensor(float(sigma)))
        self.register_buffer("_r_exp", torch.tensor(float(r_exp)))
        self.register_buffer("_buffer", torch.tensor(float(buffer)))
        self.register_buffer(
            "_rebuild_threshold", torch.tensor(float(rebuild_threshold))
        )
        # c_rep: back-door override; by default derived from sigma so that
        # PROLSQ shape term equals v^p / (p * sigma^p).
        if c_rep is None:
            c_rep_val = 1.0 / (float(r_exp) * float(sigma) ** float(r_exp))
        else:
            c_rep_val = float(c_rep)
        self.register_buffer("_c_rep", torch.tensor(c_rep_val))

    @property
    def c_rep(self) -> float:
        """Get repulsion coefficient."""
        return self._c_rep.item()

    @c_rep.setter
    def c_rep(self, value: float):
        """Set repulsion coefficient.

        Note: this breaks the internal link ``c_rep = 1 / (r_exp * sigma**r_exp)``.
        After setting c_rep directly, ``sigma_vdw`` should be treated as
        informational only for the log(sigma) term; the shape term uses c_rep.
        """
        self._c_rep.fill_(value)

    @property
    def sigma_vdw(self) -> float:
        """Get the effective overlap tolerance sigma (Å)."""
        return self._sigma_vdw.item()

    @sigma_vdw.setter
    def sigma_vdw(self, value: float):
        """Set sigma and recompute the linked ``c_rep``."""
        self._sigma_vdw.fill_(value)
        new_c_rep = 1.0 / (self._r_exp.item() * value ** self._r_exp.item())
        self._c_rep.fill_(new_c_rep)

    @property
    def r_exp(self) -> float:
        """Get repulsion exponent."""
        return self._r_exp.item()

    @r_exp.setter
    def r_exp(self, value: float):
        """Set repulsion exponent."""
        self._r_exp.fill_(value)

    @property
    def buffer(self) -> float:
        """Get distance buffer."""
        return self._buffer.item()

    @buffer.setter
    def buffer(self, value: float):
        """Set distance buffer."""
        self._buffer.fill_(value)

    def maintenance(self) -> None:
        """Rebuild the VDW pair list if any ASU atom drifted too far.

        Fast path: one ``max().item()`` sync on the per-atom displacement
        norm between the current ASU coordinates and the snapshot taken
        at the last VDW build. If the max displacement stays within
        ``_rebuild_threshold`` we return immediately.

        Slow path (only when triggered): delegate to
        ``restraints.rebuild_vdw_restraints`` which refreshes the pair
        list using the original build kwargs and updates the snapshot.
        See :meth:`Target.maintenance` for the general contract.

        Safety invariant: the default build cutoff (6.0 Å) leaves roughly
        ``cutoff - max_vdw_sum ≈ 2.4 Å`` of slack before a previously-
        non-contact atom pair could form a new clash. Setting
        ``rebuild_threshold < 2.4 / 2 = 1.2 Å`` guarantees that no such
        pair can slip through the list — that is, a rebuild fires
        before the slack is consumed.
        """
        if self._model is None:
            return
        r = self.restraints
        if r is None:
            return
        snapshot = getattr(r, "_last_vdw_build_xyz", None)
        if snapshot is None:
            return

        with torch.no_grad():
            delta = self._model.xyz() - snapshot
            max_disp_sq = (delta * delta).sum(dim=-1).max()

        thresh_sq = self._rebuild_threshold * self._rebuild_threshold
        if max_disp_sq.item() <= thresh_sq.item():
            return  # within slack — nothing to do

        if self.verbose > 0:
            max_disp = float(max_disp_sq.item()) ** 0.5
            thresh = float(self._rebuild_threshold.item())
            print(
                f"  VDW rebuild: max drift {max_disp:.2f} Å > "
                f"threshold {thresh:.2f} Å"
            )
        r.rebuild_vdw_restraints()

    def _compute_positions(
        self, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute atom positions for all VDW pairs, handling symmetry mates.

        For intra-ASU pairs (symop=0, offset=0), the identity transform is
        applied which reduces to a direct lookup. For symmetry pairs, the mate
        position is recomputed on-the-fly through the symmetry transformation
        so that gradients flow to both atoms.

        All pairs are processed in a single vectorized pass.

        Parameters
        ----------
        xyz : torch.Tensor
            Current ASU Cartesian coordinates of shape (N, 3).

        Returns
        -------
        pos1 : torch.Tensor
            Positions of first atom in each pair (N_pairs, 3).
        pos2 : torch.Tensor
            Positions of second atom in each pair (N_pairs, 3).
        min_distances : torch.Tensor
            VDW distance threshold for each pair (N_pairs,).
        """
        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data["indices"]
        min_distances = vdw_data["min_distances"]
        symop_indices = vdw_data.get("symop_indices")
        cell_offsets = vdw_data.get("cell_offsets")

        # pos1 is always a direct ASU lookup
        pos1 = xyz[indices[:, 0]]

        has_symmetry = (
            symop_indices is not None
            and len(symop_indices) > 0
            and not (symop_indices == 0).all()
        )

        if not has_symmetry:
            # Fast path: all pairs are intra-ASU
            pos2 = xyz[indices[:, 1]]
            return pos1, pos2, min_distances

        # Unified path: apply symmetry transform to all pos2 atoms.
        # For intra-ASU pairs (symop=0, offset=0) this is the identity.
        cell = self.model.cell
        sg = self.model.symmetry

        # Gather mate source coordinates and convert to fractional
        mate_source = xyz[indices[:, 1]]  # (N_pairs, 3) -- gradients flow
        frac = cell.cartesian_to_fractional(mate_source)

        # Gather per-pair rotation matrices and translations
        R = sg.matrices[symop_indices].to(frac.dtype)       # (N_pairs, 3, 3)
        t = sg.translations[symop_indices].to(frac.dtype)   # (N_pairs, 3)
        offsets = cell_offsets.to(frac.dtype)                # (N_pairs, 3)

        # Batched symmetry transform: R @ frac + t + offset
        frac_transformed = (
            torch.bmm(R, frac.unsqueeze(-1)).squeeze(-1) + t + offsets
        )

        # Convert back to Cartesian
        pos2 = cell.fractional_to_cartesian(frac_transformed)

        return pos1, pos2, min_distances

    def forward(self) -> torch.Tensor:
        from torchref.base.targets.nonbonded import nonbonded_heavy_math
        xyz = self.model.xyz()
        device = xyz.device

        if "vdw" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data.get("indices")

        if indices is None or len(indices) == 0:
            return torch.tensor(0.0, device=device)

        sigmas = vdw_data["sigmas"]

        # The prolsq branch goes through the math dispatcher — Triton on
        # CUDA fp32, eager otherwise. Other modes (gaussian, soft) keep
        # the inline path below.
        if self.mode == "prolsq":
            return nonbonded_heavy_math(
                xyz, indices,
                vdw_data["min_distances"],
                vdw_data.get("symop_indices"),
                vdw_data.get("cell_offsets"),
                self.model.symmetry.matrices,
                self.model.symmetry.translations,
                self.model.cell.fractional_matrix,
                self.model.cell.inv_fractional_matrix,
                self._c_rep, self._r_exp,
                float(self._buffer), self._sigma_vdw,
            )

        # Compute positions (handles symmetry transparently)
        pos1, pos2, min_distances = self._compute_positions(xyz)

        # Compute actual distances with small epsilon to prevent gradient issues at d=0
        diff = pos2 - pos1
        actual_distances = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)

        # Violations: where actual distance is less than VDW sum + buffer
        violations = torch.clamp(min_distances + self._buffer - actual_distances, min=0.0)

        if self.mode == "gaussian":
            log_2pi = torch.log(
                torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype)
            )
            nll = 0.5 * (violations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
            return nll.sum()

        elif self.mode == "soft":
            threshold = 0.5  # Å - switch to linear below this
            quadratic_mask = violations <= threshold
            quadratic_energy = self._c_rep * (violations**2)
            linear_energy = self._c_rep * (2 * threshold * violations - threshold**2)
            energy = torch.where(quadratic_mask, quadratic_energy, linear_energy)
            return energy.sum()

        else:
            raise ValueError(f"Unknown non-bonded mode: {self.mode}")

    def get_violations(self, threshold: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Get information about VDW violations.

        Parameters
        ----------
        threshold : float, optional
            Only report violations greater than this (Å). Default is 0.0.

        Returns
        -------
        dict
            Dictionary with 'indices', 'violations', 'distances', 'min_distances'.
        """
        xyz = self.model.xyz()
        device = xyz.device

        if "vdw" not in self.restraints.restraints:
            return {
                "indices": torch.tensor([], dtype=torch.long, device=device).reshape(
                    0, 2
                ),
                "violations": torch.tensor([], device=device),
                "distances": torch.tensor([], device=device),
                "min_distances": torch.tensor([], device=device),
            }

        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data["indices"]

        if indices is None or len(indices) == 0:
            return {
                "indices": torch.tensor([], dtype=torch.long, device=device).reshape(
                    0, 2
                ),
                "violations": torch.tensor([], device=device),
                "distances": torch.tensor([], device=device),
                "min_distances": torch.tensor([], device=device),
            }

        pos1, pos2, min_distances = self._compute_positions(xyz)
        actual_distances = torch.norm(pos2 - pos1, dim=-1)
        violations = torch.clamp(min_distances - actual_distances, min=0.0)

        # Filter by threshold
        mask = violations > threshold

        return {
            "indices": indices[mask],
            "violations": violations[mask],
            "distances": actual_distances[mask],
            "min_distances": min_distances[mask],
        }

    def stats(self) -> Dict[str, any]:
        """Get non-bonded restraint statistics."""
        xyz = self.model.xyz()
        device = xyz.device

        if "vdw" not in self.restraints.restraints:
            return {}

        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data.get("indices")

        if indices is None or len(indices) == 0:
            return {}

        sigmas = vdw_data["sigmas"]

        pos1, pos2, min_distances = self._compute_positions(xyz)
        actual_distances = torch.norm(pos2 - pos1, dim=-1)

        # Violations: where actual distance < VDW sum
        violations = torch.clamp(min_distances - actual_distances, min=0.0)
        n_violations = (violations > 0).sum().item()

        # RMS of violations only (for those that clash)
        if n_violations > 0:
            violation_mask = violations > 0
            rms_violation = torch.sqrt((violations[violation_mask] ** 2).mean()).item()
            max_violation = violations.max().item()
        else:
            rms_violation = 0.0
            max_violation = 0.0

        loss = self.forward()

        result = {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(indices), VERBOSITY_DEBUG),
            "n_violations": stat(n_violations, VERBOSITY_DETAILED),
            "rms_violation": stat(rms_violation, VERBOSITY_DETAILED),
            "max_violation": stat(max_violation, VERBOSITY_DEBUG),
            "mean_sigma": stat(sigmas.mean().item(), VERBOSITY_DEBUG),
        }

        # Report symmetry contact count if available
        symop_indices = vdw_data.get("symop_indices")
        cell_offsets = vdw_data.get("cell_offsets")
        if symop_indices is not None and len(symop_indices) > 0:
            is_sym = (symop_indices != 0) | (cell_offsets != 0).any(dim=-1)
            n_sym = is_sym.sum().item()
            if n_sym > 0:
                result["n_symmetry"] = stat(n_sym, VERBOSITY_DETAILED)

        return result
