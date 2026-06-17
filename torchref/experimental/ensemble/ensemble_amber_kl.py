"""
Per-member AMBER energy over an ensemble, with an optional entropy regularizer.

Two targets, both subclasses of the single-molecule
:class:`~torchref.refinement.targets.amber_target.AmberTarget`:

- :class:`EnsembleAmberTarget` — the AMBER energy of every ensemble member
  (``N`` non-interacting copies of one chemistry), averaged. It **inherits**
  the full AmberTarget machinery (OpenMM system build via antechamber /
  ForceField, atom map, differentiable hydrogen placement, autograd bridge):
  the single-copy chemistry/topology is built once from the ensemble's
  ``_pdb_single`` and each member's coordinates are fed through the inherited
  per-conformation energy (:meth:`AmberTarget._energy`).

- :class:`EnsembleAmberKLTarget` — adds the variational-Boltzmann entropy
  regularizer on top of the mean energy::

      L = (1/N) Σ_i E_amber(x_i) / kT  −  λ · Ĥ(x_1, …, x_N)

  Minimizing this makes the empirical ensemble approximate samples from
  ``p(x) ∝ exp(−E_amber(x) / kT)`` without collapsing all members to one
  minimum — the per-atom entropy surrogate ``Ĥ`` blows up as the spread
  vanishes (``var → 0 ⇒ log → −∞``). ``kT = 0`` drops the energy term (entropy
  only); ``λ = 0`` drops the regularizer.

Hydrogen handling is **identical** to the single-molecule target: every member
is placed through the inherited :meth:`AmberTarget._place_hydrogens`
(local-frame placement, one shared implementation). There is no per-member
hydrogen logic here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch

from torchref.refinement.targets.amber_target import AMBER14_STANDARD, AmberTarget

if TYPE_CHECKING:
    from .ensemble_model import EnsembleModel


class EnsembleAmberTarget(AmberTarget):
    """Mean per-member AMBER energy over an ensemble (``N`` independent copies).

    Subclass of :class:`~torchref.refinement.targets.amber_target.AmberTarget`.
    The expensive chemistry/topology is built **once** from the ensemble's
    single-copy PDB (``EnsembleModel._pdb_single``); ``forward`` evaluates the
    inherited per-conformation energy for each member and returns the mean.
    The OpenMM system (and any antechamber parameterisation for non-standard
    residues) is built eagerly in ``__init__``.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble whose members are evaluated. ``forward`` reads
        :attr:`EnsembleModel.xyz_per_member`.
    cutoff : float
        AMBER non-bonded cutoff (Å).
    normalize_by_atoms : bool
        Report each member's energy per-atom (forwarded to the base).
    gaff2_files, residue_charges : dict, optional
        Forwarded to the base for non-standard residue parameterisation.
    restrict_to_standard : bool
        If True, drop atoms in non-:data:`AMBER14_STANDARD` residues from the
        Amber chemistry (and from each member's coordinates), so the OpenMM
        topology builds without antechamber/tleap. The X-ray side still sees
        the full model.
    charge_method : str
        antechamber charge method ('gas' default, no QM; or 'bcc').
    verbose : int
        Verbosity.
    """

    name: str = "ensemble_amber"

    def __init__(
        self,
        model: "EnsembleModel" = None,
        cutoff: float = 5.0,
        normalize_by_atoms: bool = True,
        gaff2_files=None,
        residue_charges=None,
        restrict_to_standard: bool = False,
        charge_method: str = "gas",
        verbose: int = 0,
    ):
        self.restrict_to_standard = bool(restrict_to_standard)

        # Build the single-conformation chemistry model (and the kept-atom
        # subset) BEFORE the base __init__ so the inherited build targets it.
        chem_model = None
        atom_idx_np: Optional[np.ndarray] = None
        if model is not None:
            chem_model, atom_idx_np = self._make_chem_model(model, verbose)

        super().__init__(
            model=model,
            chem_model=chem_model,
            cutoff=cutoff,
            normalize_by_atoms=normalize_by_atoms,
            gaff2_files=gaff2_files,
            residue_charges=residue_charges,
            charge_method=charge_method,
            verbose=verbose,
        )

        # Register the kept-atom indices now that nn.Module is initialised, so
        # the buffer moves with .to(device). None → use all atoms.
        if atom_idx_np is not None:
            self.register_buffer(
                "_member_atom_idx",
                torch.as_tensor(
                    atom_idx_np, dtype=torch.long, device=self._model.device
                ),
            )
        else:
            self._member_atom_idx = None

    def _make_chem_model(self, ensemble: "EnsembleModel", verbose: int):
        """Build a single-conformation :class:`Model` for the Amber chemistry.

        Returns ``(chem_model, atom_idx_np)`` where ``atom_idx_np`` indexes the
        kept atoms into the per-member atom layout (``None`` ⇒ all atoms). When
        ``restrict_to_standard`` is set, non-standard residues are dropped so
        the OpenMM topology builds without antechamber/tleap.
        """
        from .ensemble_model import build_single_copy_model

        atom_idx_np: Optional[np.ndarray] = None
        if self.restrict_to_standard:
            resnames = ensemble._pdb_single["resname"].astype(str).str.strip()
            mask = resnames.isin(AMBER14_STANDARD).values
            n_dropped = int((~mask).sum())
            if n_dropped > 0:
                if verbose > 0:
                    dropped = sorted(set(resnames[~mask].tolist()))
                    print(
                        f"[{type(self).__name__}] restrict_to_standard: dropping "
                        f"{n_dropped} atom(s) in non-standard residues: {dropped}"
                    )
                atom_idx_np = np.nonzero(mask)[0].astype(np.int64)

        chem = build_single_copy_model(ensemble, atom_idx=atom_idx_np, verbose=verbose)
        return chem, atom_idx_np

    def _member_xyz(self, i: int) -> torch.Tensor:
        """Member ``i`` coordinates ``(n_chem_atoms, 3)``, subset to kept atoms.

        The returned ordering matches ``self._chem_model.pdb`` (what the OpenMM
        atom map was built on), so it can be fed straight to
        :meth:`AmberTarget._energy`.
        """
        xyz = self._model.xyz_per_member[i]
        if self._member_atom_idx is not None:
            xyz = xyz.index_select(0, self._member_atom_idx)
        return xyz

    def forward(self) -> torch.Tensor:
        """Mean of the inherited per-conformation AMBER energy over all members."""
        n_members = int(self._model.n_members)
        energies = [self._energy(self._member_xyz(i)) for i in range(n_members)]
        return torch.stack(energies).mean()

    def stats(self) -> Dict:
        from torchref.utils.stats import VERBOSITY_STANDARD, stat

        with torch.no_grad():
            loss = self.forward().item()
        return {"loss": stat(loss, VERBOSITY_STANDARD)}


class EnsembleAmberKLTarget(EnsembleAmberTarget):
    """Per-member AMBER energy + ensemble-entropy regularizer.

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
        Forwarded to the base so the per-member energy is reported per-atom.
    gaff2_files, residue_charges : dict, optional
        Forwarded to the base for non-standard residue parameterisation.
    restrict_to_standard : bool
        Drop non-standard residues from the Amber chemistry (see
        :class:`EnsembleAmberTarget`).
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
        gaff2_files=None,
        residue_charges=None,
        restrict_to_standard: bool = False,
        charge_method: str = "gas",
        verbose: int = 0,
    ):
        super().__init__(
            model=model,
            cutoff=cutoff,
            normalize_by_atoms=normalize_by_atoms,
            gaff2_files=gaff2_files,
            residue_charges=residue_charges,
            restrict_to_standard=restrict_to_standard,
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
