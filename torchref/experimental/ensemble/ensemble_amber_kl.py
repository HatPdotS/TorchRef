"""
Variational-Boltzmann ensemble restraint via per-member AMBER energy.

Treats the ensemble as variational samples from a Boltzmann target
distribution ``p(x) ∝ exp(-E_amber(x) / kT)``. Minimizing the loss

    L = (1/N) * Σ_i E_amber(x_i) / kT  −  λ * H_hat(x_1, …, x_N)

makes the empirical ensemble approximate that target without collapsing
all members to one minimum — the entropy term ``H_hat`` keeps members
spread.

Entropy estimator
-----------------
Per-atom isotropic-Gaussian surrogate:

    H_hat = (1 / n_atoms) * Σ_a log( var_a + ε )

where ``var_a`` is the trace of the 3x3 covariance of atom ``a``
across the ``N`` members. Cheap, differentiable, and blows up when the
ensemble collapses (``var → 0`` ⇒ ``log → -∞`` ⇒ loss explodes).

Implementation notes
--------------------
One :class:`~torchref.refinement.targets.amber_target.AmberTarget` is
built once against the single-copy chemistry (``model._pdb_single``)
exposed via a thin shim model. On each forward, the shim's coordinate
function is rebound to each member's slice in turn and the resulting
N energies are summed. Topology is built once; the per-member cost is
``setPositions`` + ``getState``.

This target inherits any limitation of ``AmberTarget`` on the underlying
chemistry. Pass ``lam=0`` to skip the energy term and use only the
entropy regularizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from torch import nn

from torchref.refinement.targets.base import ModelTarget

if TYPE_CHECKING:
    from .ensemble_model import EnsembleModel


class _SingleMemberShim(nn.Module):
    """
    Minimal Model-like wrapper that exposes a single-member view of the
    ensemble as if it were a stand-alone :class:`Model`. Used to convince
    :class:`AmberTarget` we have a single-conformation model.

    Two modes:

    - **Full topology** (``atom_idx=None``): expose every atom of
      ``ensemble._pdb_single``.
    - **Restricted topology** (``atom_idx`` given): expose only the atoms
      indexed by ``atom_idx`` into ``_pdb_single``. The shim's ``xyz()``
      returns the per-member coordinates restricted to those rows. Used
      to drop non-standard residues (SO4, ligands, etc.) so AmberTarget
      can build a topology without antechamber/tleap.

    Subclasses ``nn.Module`` so :class:`ModelTarget.__init__` can register
    it via ``add_module``. We don't register the ensemble as a submodule
    (it lives on the parent target), so backward through ``self.xyz()``
    flows to the ensemble's parameters via the live view.
    """

    def __init__(
        self,
        ensemble: "EnsembleModel",
        atom_idx: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Hold a non-Parameter reference to the ensemble to keep it out of
        # the shim's nn.Module child tree (it's owned by the outer target).
        object.__setattr__(self, "_ensemble", ensemble)
        if atom_idx is None:
            self.pdb = ensemble._pdb_single
            self._atom_idx = None
        else:
            self.pdb = ensemble._pdb_single.iloc[atom_idx.cpu().numpy()].reset_index(drop=True)
            self.register_buffer("_atom_idx_buf", atom_idx.to(ensemble.device))
            self._atom_idx = self._atom_idx_buf
        self.cell = ensemble.cell
        self.spacegroup = ensemble.spacegroup
        # Member index to expose on the next xyz() call.
        self._active_member: int = 0

    # AmberTarget calls model.xyz() expecting a (n_atoms, 3) tensor.
    def xyz(self) -> torch.Tensor:
        member_xyz = self._ensemble.xyz_per_member[self._active_member]
        if self._atom_idx is None:
            return member_xyz
        return member_xyz.index_select(0, self._atom_idx)

    # Some AmberTarget code paths may probe these attributes.
    @property
    def device(self):
        return self._ensemble.device

    def strip_altlocs(self):
        # The single-copy chemistry has no altlocs (we cleared them at load).
        return self

    def update_pdb(self):
        """
        Sync ``self.pdb`` x/y/z columns to the current member's coordinates.

        ``AmberTarget._filter_pdb_for_tleap`` and similar paths call this
        so the DataFrame reflects refined positions when writing input PDB
        files for tleap. Mirrors ``Model.update_pdb`` for our duck-typed
        shim.
        """
        coords = self.xyz().detach().cpu().numpy()
        self.pdb.loc[:, ["x", "y", "z"]] = coords
        return self.pdb


class EnsembleAmberKLTarget(ModelTarget):
    """
    Per-member AMBER energy + ensemble-entropy regularizer.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble whose members are evaluated.
    kT : float
        Boltzmann temperature scale (kJ/mol). Default 2.494 = 300 K.
    lam : float
        Coefficient on the entropy regularizer ``H_hat``. ``lam=0`` drops
        the regularizer; the energy term will collapse the ensemble.
    cutoff : float
        AMBER nonbonded cutoff (Å).
    eps : float
        Numerical floor inside ``log`` of the per-atom variance.
    normalize_by_atoms : bool
        Forwarded to the underlying AmberTarget so the per-member energy
        is reported per-atom.
    gaff2_files : dict, optional
        Forwarded to AmberTarget for non-standard residues.
    residue_charges : dict, optional
        Forwarded to AmberTarget.
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
        super().__init__(model=model, verbose=verbose)
        self.kT = float(kT)
        self.lam = float(lam)
        self.eps = float(eps)
        self.restrict_to_standard = bool(restrict_to_standard)
        # Lazily-built AmberTarget against a single-member shim. We avoid
        # the OpenMM import / parameterization cost until the first forward.
        self._shim: Optional[_SingleMemberShim] = None
        self._amber_target = None
        # Default to Gasteiger (no SCF) — bonded geometry restraints are the
        # point of using Amber here, partial-charge accuracy barely matters,
        # and 'bcc' fails on multi-instance HETATM batches when sqm doesn't
        # converge (e.g. multiple SO4 ions). User can override with
        # ``charge_method='bcc'`` if desired.
        self._amber_init_kwargs = dict(
            cutoff=cutoff, normalize_by_atoms=normalize_by_atoms,
            gaff2_files=gaff2_files, residue_charges=residue_charges,
            charge_method=charge_method,
        )

    # ------------------------------------------------------------------
    # Lazy AmberTarget construction
    # ------------------------------------------------------------------

    def _ensure_amber(self) -> None:
        if self._amber_target is not None:
            return
        if self._model is None:
            raise RuntimeError("EnsembleAmberKLTarget has no model attached.")
        # Local import: AmberTarget imports openmm at construction.
        from torchref.refinement.targets.amber_target import AmberTarget, AMBER14_STANDARD

        atom_idx = None
        if self.restrict_to_standard:
            pdb = self._model._pdb_single
            resnames = pdb["resname"].astype(str).str.strip()
            standard_mask = resnames.isin(AMBER14_STANDARD).values
            n_dropped = int((~standard_mask).sum())
            if n_dropped > 0:
                dropped_set = sorted(set(resnames[~standard_mask].tolist()))
                if self.verbose > 0:
                    print(
                        f"[EnsembleAmberKL] restrict_to_standard: dropping "
                        f"{n_dropped} atom(s) in non-standard residues: "
                        f"{dropped_set}"
                    )
                atom_idx = torch.as_tensor(
                    standard_mask.nonzero()[0], dtype=torch.long,
                    device=self._model.device,
                )

        self._shim = _SingleMemberShim(self._model, atom_idx=atom_idx)
        self._amber_target = AmberTarget(
            model=self._shim,
            verbose=self.verbose,
            **self._amber_init_kwargs,
        )

    # ------------------------------------------------------------------
    # Entropy estimator
    # ------------------------------------------------------------------

    def _entropy(self) -> torch.Tensor:
        """
        Per-atom isotropic-variance log-entropy.

        Sums log(trace(Cov_a) + eps) across atoms, divided by n_atoms.
        """
        xyz = self._model.xyz_per_member  # (N, n_atoms, 3)
        # var across N members, summed over xyz: trace of the 3x3 covariance.
        var = xyz.var(dim=0, unbiased=False).sum(dim=-1)  # (n_atoms,)
        return torch.log(var + self.eps).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        H = self._entropy()
        if self.lam == 0.0 and self.kT == 0.0:
            return -H  # entropy-only mode shouldn't happen in practice
        if self.kT > 0.0:
            self._ensure_amber()
            n_members = self._model.n_members
            energies = []
            for i in range(n_members):
                self._shim._active_member = i
                energies.append(self._amber_target.forward())
            mean_energy = torch.stack(energies).mean()
            loss = mean_energy / self.kT
        else:
            loss = torch.zeros((), device=H.device, dtype=H.dtype)
        return loss - self.lam * H

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        from torchref.utils.stats import VERBOSITY_DETAILED, VERBOSITY_STANDARD, stat
        with torch.no_grad():
            H = self._entropy().item()
            loss = self.forward().item()
        return {
            "loss": stat(loss, VERBOSITY_STANDARD),
            "entropy_hat": stat(H, VERBOSITY_DETAILED),
            "lam": stat(self.lam, VERBOSITY_DETAILED),
            "kT": stat(self.kT, VERBOSITY_DETAILED),
        }
