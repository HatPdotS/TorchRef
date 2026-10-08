"""
Per-member AMBER energy over an ensemble, with an optional entropy regularizer.

.. warning::

   Experimental — part of ``torchref.experimental.ensemble``. The API and
   behaviour may change or be removed without notice.

   These two targets are **not** the production ensemble Amber path. The
   refinement driver (:class:`EnsembleRefinement`) wires
   :class:`~torchref.experimental.ensemble.quasi_crystal_amber.QuasiCrystalAmberTarget`
   instead, which abandoned the per-member entropy/KL approach (see below).
   The targets here are retained for standalone / comparison use only.

- :class:`EnsembleAmberTarget` — the AMBER energy of every ensemble member
  (``N`` non-interacting copies of one chemistry), averaged. One OpenMM system is
  built for the single-copy chemistry by
  :class:`~torchref.experimental.mm.OpenMMAdapter`, and each member's coordinates
  are evaluated in it in turn.

- :class:`EnsembleAmberKLTarget` — adds the variational-Boltzmann entropy
  regularizer on top of the mean energy::

      L = (1/N) Σ_i E_amber(x_i) / kT  −  λ · Ĥ(x_1, …, x_N)

  The intent was for minimizing this to make the empirical ensemble
  approximate samples from ``p(x) ∝ exp(−E_amber(x) / kT)`` while the per-atom
  entropy surrogate ``Ĥ`` (which blows up as the spread vanishes,
  ``var → 0 ⇒ log → −∞``) resisted collapse to a single minimum. ``kT = 0``
  drops the energy term (entropy only); ``λ = 0`` drops the regularizer.

  Caveat: this per-member entropy/KL anti-collapse prior was found inadequate
  for the quasi-crystal supercell layout and was superseded — see
  :mod:`~torchref.experimental.ensemble.quasi_crystal_amber`, where physical
  crystal contacts replace the entropy term. Retained here for
  standalone / comparison use, not as the production restraint.

Hydrogens are the ensemble's own atoms, as for the single-model target: build the
ensemble with ``hydrogens="add"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import torch

from torchref.experimental.mm import OpenMMAdapter
from torchref.experimental.mm.adapter import DEFAULT_FORCEFIELD, template_coverage
from torchref.experimental.targets.amber_target import require_openmm
from torchref.refinement.targets.base import ModelTarget

from .ensemble_model import build_single_copy_model

if TYPE_CHECKING:
    from .ensemble_model import EnsembleModel


class EnsembleAmberTarget(ModelTarget):
    """Mean per-member AMBER energy over an ensemble (``N`` independent copies).

    .. warning::

       Experimental, and **not** the production ensemble Amber restraint
       (:class:`QuasiCrystalAmberTarget` is). API and behaviour may change
       without notice.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble whose members are evaluated; ``forward`` reads
        :attr:`EnsembleModel.xyz_per_member`. Must carry the hydrogens AMBER needs.
    cutoff : float
        AMBER non-bonded cutoff (Å).
    normalize_by_atoms : bool
        Report each member's energy per atom.
    residue_charges : dict, optional
        Net charge per GAFF2 residue name, overriding the dictionary's formal charges.
    restrict_to_standard : bool
        If True, leave out of the Amber system the residues no force-field XML
        template matches, so no GAFF2 parameterisation is needed. The X-ray side still
        sees the full model.
    forcefield : sequence of str
        OpenMM force-field XML files.
    charge_method : str
        antechamber charge method for GAFF2 residues ('gas' default, or 'bcc').
    verbose : int
        Verbosity.
    """

    name: str = "ensemble_amber"

    def __init__(
        self,
        model: "EnsembleModel" = None,
        cutoff: float = 5.0,
        normalize_by_atoms: bool = True,
        residue_charges: Optional[Dict[str, int]] = None,
        restrict_to_standard: bool = False,
        forcefield: Sequence[str] = DEFAULT_FORCEFIELD,
        charge_method: str = "gas",
        verbose: int = 0,
    ):
        require_openmm()
        super().__init__(model=model, verbose=verbose)
        self.restrict_to_standard = bool(restrict_to_standard)
        self._normalize = normalize_by_atoms
        self.adapter: Optional[OpenMMAdapter] = None
        if model is None:
            return
        chemistry = build_single_copy_model(model, verbose=verbose)
        atoms = (
            template_coverage(chemistry, forcefield) if restrict_to_standard else None
        )
        self.adapter = OpenMMAdapter.from_model(
            chemistry,
            atoms=atoms,
            forcefield=forcefield,
            charge_method=charge_method,
            residue_charges=residue_charges,
            cutoff=cutoff,
            hydrogens_added=getattr(model, "hydrogen_source", "keep") == "add",
            verbose=verbose,
        )

    def _member_energy(self, xyz: torch.Tensor) -> torch.Tensor:
        """One member's energy, ``xyz`` of shape ``(n_atoms_per_member, 3)`` in Å."""
        energy = self.adapter.energy(xyz)
        return energy / self.adapter.n_particles if self._normalize else energy

    def forward(self) -> torch.Tensor:
        """Mean of the per-member AMBER energies."""
        xyz = self._model.xyz_per_member
        energies = [
            self._member_energy(xyz[i]) for i in range(int(self._model.n_members))
        ]
        return torch.stack(energies).mean()

    def stats(self) -> Dict:
        from torchref.utils.stats import VERBOSITY_STANDARD, stat

        with torch.no_grad():
            loss = self.forward().item()
        return {"loss": stat(loss, VERBOSITY_STANDARD)}


class EnsembleAmberKLTarget(EnsembleAmberTarget):
    """Per-member AMBER energy + ensemble-entropy regularizer.

    .. warning::

       Experimental and superseded. The per-member entropy/KL anti-collapse
       prior was found inadequate for the quasi-crystal layout; the production
       restraint is :class:`QuasiCrystalAmberTarget` (physical crystal
       contacts, no KL term). Retained for standalone / comparison use only.
       API and behaviour may change without notice.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble whose members are evaluated.
    kT : float
        Boltzmann temperature scale (kJ/mol). Default 2.494 = 300 K. ``kT = 0``
        drops the energy term (entropy-only); the OpenMM system is still built.
    lam : float
        Coefficient on the entropy regularizer ``Ĥ``. ``lam = 0`` drops the
        regularizer; the energy term alone will collapse the ensemble.
    cutoff : float
        AMBER non-bonded cutoff (Å).
    eps : float
        Numerical floor inside ``log`` of the per-atom variance.
    normalize_by_atoms : bool
        Report each member's energy per atom.
    residue_charges : dict, optional
        Net charge per GAFF2 residue name.
    restrict_to_standard : bool
        Leave out the residues no force-field XML template matches (see
        :class:`EnsembleAmberTarget`).
    forcefield : sequence of str
        OpenMM force-field XML files.
    charge_method : str
        antechamber charge method ('gas' default).
    verbose : int
        Verbosity.
    """

    name: str = "ensemble_amber_kl"

    def __init__(
        self,
        model: "EnsembleModel" = None,
        kT: float = 2.494,
        lam: float = 1.0,
        cutoff: float = 5.0,
        eps: float = 1e-4,
        normalize_by_atoms: bool = True,
        residue_charges: Optional[Dict[str, int]] = None,
        restrict_to_standard: bool = False,
        forcefield: Sequence[str] = DEFAULT_FORCEFIELD,
        charge_method: str = "gas",
        verbose: int = 0,
    ):
        super().__init__(
            model=model,
            cutoff=cutoff,
            normalize_by_atoms=normalize_by_atoms,
            residue_charges=residue_charges,
            restrict_to_standard=restrict_to_standard,
            forcefield=forcefield,
            charge_method=charge_method,
            verbose=verbose,
        )
        self.kT = float(kT)
        self.lam = float(lam)
        self.eps = float(eps)

    def _entropy(self) -> torch.Tensor:
        """Per-atom isotropic-variance log-entropy across members.

        Sums ``log(trace(Cov_a) + eps)`` over atoms, divided by ``n_atoms``.
        """
        xyz = self._model.xyz_per_member  # (N, n_atoms, 3)
        # var across N members, summed over xyz: trace of the 3x3 covariance.
        var = xyz.var(dim=0, unbiased=False).sum(dim=-1)  # (n_atoms,)
        return torch.log(var + self.eps).mean()

    def forward(self) -> torch.Tensor:
        H = self._entropy()
        if self.kT > 0.0:
            mean_energy = super().forward()  # mean per-member AMBER energy
            loss = mean_energy / self.kT
        else:
            loss = torch.zeros((), device=H.device, dtype=H.dtype)
        return loss - self.lam * H

    def stats(self) -> Dict:
        from torchref.utils.stats import (
            VERBOSITY_DETAILED,
            VERBOSITY_STANDARD,
            stat,
        )

        with torch.no_grad():
            H = self._entropy().item()
            loss = self.forward().item()
        return {
            "loss": stat(loss, VERBOSITY_STANDARD),
            "entropy_hat": stat(H, VERBOSITY_DETAILED),
            "lam": stat(self.lam, VERBOSITY_DETAILED),
            "kT": stat(self.kT, VERBOSITY_DETAILED),
        }
