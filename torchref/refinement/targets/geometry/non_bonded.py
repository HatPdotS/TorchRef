"""Non-bonded (van der Waals) repulsion restraint.

Holds :class:`NonBondedTarget`, which penalizes VDW overlap including contacts
with symmetry mates. Violation counts reported by ``stats`` / ``get_violations``
deliberately exclude the ``buffer`` onset that ``forward`` penalizes.
"""

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
    Non-bonded (VDW) restraint: PROLSQ repulsion as a generalized-Gaussian NLL.

    Per-pair, with :math:`p = r_\text{exp}` and overlap
    :math:`v = \max(0, d_{\text{vdw}} + b - d)`:

    .. math::

       \mathrm{NLL}(v) = \frac{v^{p}}{p\,\sigma^{p}}
           + \log\sigma + \tfrac{1}{2}\log(2\pi)

    The shape term equals the classical PROLSQ energy :math:`c_{\text{rep}} v^p`
    at :math:`c_{\text{rep}} = 1/(p\sigma^{p})`, but :math:`\sigma` is the
    exposed knob since it reads as an overlap tolerance; the 0.3 Å default is
    near the classical ``c_rep=16, r_exp=4`` (:math:`\sigma \approx 0.354`).
    Modes: ``'prolsq'`` (above, default), ``'gaussian'`` (Gaussian NLL on the
    overlap with per-pair sigmas), ``'soft'`` (linear core past a threshold).

    With cell and spacegroup on the model, ASU-to-symmetry-mate contacts are
    included; mate positions are recomputed from current ASU coordinates each
    call, so gradients reach both atoms of a pair.

    Reference: cctbx/geometry_restraints/nonbonded.h, PROLSQ documentation,
    MolProbity clash criterion (Davis et al., NAR 2007).

    Parameters
    ----------
    model : Model, optional
        Reference to Model object.
    mode : str, optional
        One of 'prolsq', 'gaussian', 'soft'. Default is 'prolsq'.
    sigma : float, optional
        Overlap tolerance (Å), default 0.3. Stored as the ``_sigma_vdw`` buffer
        (exposed as ``sigma_vdw``); it sets the shape coefficient only when
        ``c_rep`` is None, but always supplies the ``log σ`` term. *Not* the
        base-class ``sigma`` keyword, which is inert.
    r_exp : float, optional
        Exponent of the repulsion term. Default is 4.0.
    c_rep : float, optional
        Back door for legacy PROLSQ weights: overrides the sigma-derived
        coefficient, leaving σ in the ``log`` term only. Default None.
    buffer : float, optional
        Å added to the VDW radii sum, so atoms feel repulsion before they
        clash. Default is 0.0.
    rebuild_threshold : float, optional
        Max ASU atom drift (Å) before :meth:`maintenance` rebuilds the pair
        list. Default 1.0; that method gives the bound this must respect.
    verbose : int, optional
        Verbosity level. Default is 0.
    scale : float, optional
        Stored as ``self.scale`` (default 10.0) but **not** consumed by
        ``forward()`` -- it does not scale the loss.
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
        device=None,
    ):
        """Initialize non-bonded target; see the class docstring for parameters."""
        super().__init__(model, verbose, device=device)
        self.mode = mode
        self.scale = scale
        # Tunables that reach the kernel must be buffers on the target's device
        # and float dtype: the prolsq branch hands these straight to a Triton
        # kernel, where a CPU tensor is a host pointer, not a promotable scalar.
        self._register_scalar("_sigma_vdw", float(sigma))
        self._register_scalar("_r_exp", float(r_exp))
        if c_rep is None:
            c_rep_val = 1.0 / (float(r_exp) * float(sigma) ** float(r_exp))
        else:
            c_rep_val = float(c_rep)
        self._register_scalar("_c_rep", c_rep_val)
        # Host-side, deliberately not buffers: ``buffer`` reaches the kernel as
        # a Python float (a compile-time constant there, and a device tensor
        # would cost a sync on the hot path), and ``rebuild_threshold`` is only
        # compared against an already-``.item()``-ed displacement.
        self._buffer = float(buffer)
        self._rebuild_threshold = float(rebuild_threshold)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Absorb ``_buffer`` / ``_rebuild_threshold`` from older checkpoints.

        Both are host-side floats now, so a ``strict=True`` load would reject
        them as unexpected keys; restore the values instead of dropping them.
        """
        for legacy, attr in (
            ("_buffer", "_buffer"),
            ("_rebuild_threshold", "_rebuild_threshold"),
        ):
            saved = state_dict.pop(prefix + legacy, None)
            if saved is not None:
                setattr(self, attr, float(saved.item()))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def c_rep(self) -> float:
        """Get repulsion coefficient."""
        return self._c_rep.item()

    @c_rep.setter
    def c_rep(self, value: float):
        """Set repulsion coefficient, breaking its link to ``sigma_vdw`` -- the
        shape term then uses ``c_rep`` and ``sigma_vdw`` only the log term.
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
        """Get distance buffer (host-side; see ``__init__``)."""
        return self._buffer

    @buffer.setter
    def buffer(self, value: float):
        """Set distance buffer."""
        self._buffer = float(value)

    def maintenance(self) -> None:
        """Rebuild the VDW pair list if any ASU atom drifted too far.

        Costs one ``max().item()`` sync on per-atom displacement against the
        snapshot from the last build; past ``_rebuild_threshold`` it delegates to
        ``restraints.rebuild_vdw_restraints``, which reuses the original build
        kwargs and refreshes the snapshot. See :meth:`Target.maintenance`.

        ``rebuild_threshold`` must stay under ~1.2 Å: the 6.0 Å build cutoff
        leaves only ~2.4 Å of slack over the largest VDW sum, and *both* atoms of
        a pair can move, so separation can close by twice the threshold before a
        rebuild fires. Above that, new clashes slip past the pair list unseen.
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
        if max_disp_sq.item() <= thresh_sq:
            return  # within slack — nothing to do

        if self.verbose > 0:
            max_disp = float(max_disp_sq.item()) ** 0.5
            thresh = self._rebuild_threshold
            print(
                f"  VDW rebuild: max drift {max_disp:.2f} Å > "
                f"threshold {thresh:.2f} Å"
            )
        r.rebuild_vdw_restraints()

    def _compute_positions(
        self, xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-pair ``(pos1, pos2, min_distances)`` from ASU coordinates (N, 3).

        One vectorized pass. Mate positions are recomputed through the symmetry
        transform rather than looked up, so gradients reach both atoms; intra-ASU
        pairs (symop=0, offset=0) come out as the identity.
        """
        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data["indices"]
        min_distances = vdw_data["min_distances"]
        symop_indices = vdw_data.get("symop_indices")
        cell_offsets = vdw_data.get("cell_offsets")

        pos1 = xyz[indices[:, 0]]

        has_symmetry = (
            symop_indices is not None
            and len(symop_indices) > 0
            and not (symop_indices == 0).all()
        )

        if not has_symmetry:
            # Fast path: all pairs are intra-ASU.
            pos2 = xyz[indices[:, 1]]
            return pos1, pos2, min_distances

        cell = self.model.cell
        sg = self.model.spacegroup

        mate_source = xyz[indices[:, 1]]  # (N_pairs, 3) -- gradients flow
        frac = cell.cartesian_to_fractional(mate_source)

        R = sg.matrices[symop_indices].to(frac.dtype)       # (N_pairs, 3, 3)
        t = sg.translations[symop_indices].to(frac.dtype)   # (N_pairs, 3)
        offsets = cell_offsets.to(frac.dtype)                # (N_pairs, 3)

        # R @ frac + t + offset, batched; identity for intra-ASU pairs.
        frac_transformed = (
            torch.bmm(R, frac.unsqueeze(-1)).squeeze(-1) + t + offsets
        )

        pos2 = cell.fractional_to_cartesian(frac_transformed)

        return pos1, pos2, min_distances

    def forward(self) -> torch.Tensor:
        """Summed VDW repulsion loss; 0.0 if the model has no VDW pair list."""
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
                self.model.spacegroup.matrices,
                self.model.spacegroup.translations,
                self.model.cell.fractional_matrix,
                self.model.cell.inv_fractional_matrix,
                self._c_rep, self._r_exp,
                self._buffer, self._sigma_vdw,
            )

        pos1, pos2, min_distances = self._compute_positions(xyz)

        # The epsilon keeps the sqrt gradient finite at coincident atoms.
        diff = pos2 - pos1
        actual_distances = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)

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

        Notes
        -----
        Reported violations are ``min_distances - actual_distances`` and do
        **not** include the ``buffer`` onset that ``forward()`` penalizes
        (``forward()`` uses ``min_distances + buffer - distance``). When
        ``buffer > 0`` these counts will therefore differ from the pairs the
        loss actually penalizes.
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

        mask = violations > threshold

        return {
            "indices": indices[mask],
            "violations": violations[mask],
            "distances": actual_distances[mask],
            "min_distances": min_distances[mask],
        }

    def stats(self) -> Dict[str, any]:
        """Get non-bonded restraint statistics.

        Notes
        -----
        Reported violation counts use ``min_distances - actual_distances``
        and exclude the ``buffer`` onset that ``forward()`` penalizes, so
        when ``buffer > 0`` they differ from the pairs the loss penalizes.
        """
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

        # No buffer here: stats report true clashes, not penalized onsets.
        violations = torch.clamp(min_distances - actual_distances, min=0.0)
        n_violations = (violations > 0).sum().item()

        # RMS over clashing pairs only.
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

        symop_indices = vdw_data.get("symop_indices")
        cell_offsets = vdw_data.get("cell_offsets")
        if symop_indices is not None and len(symop_indices) > 0:
            is_sym = (symop_indices != 0) | (cell_offsets != 0).any(dim=-1)
            n_sym = is_sym.sum().item()
            if n_sym > 0:
                result["n_symmetry"] = stat(n_sym, VERBOSITY_DETAILED)

        return result
