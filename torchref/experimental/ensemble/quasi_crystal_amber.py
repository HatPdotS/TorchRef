"""
Quasi-crystal Amber target for ensemble refinement.

.. warning::

   Experimental — part of ``torchref.experimental.ensemble``. The API and
   behaviour may change or be removed without notice.

The ensemble's members are laid out as one crystal: member ``m = d·N_sym + j`` is
placed with symmetry operation ``j`` in the ``d``-th unit cell of a
``n_disorder × 1 × 1`` supercell, and the whole supercell is one OpenMM system with
PME under periodic boundaries. One evaluation covers every member, and the contacts
between members are physical crystal contacts, which are what keeps the ensemble from
collapsing -- there is no entropy or KL term.

The system is built by :class:`~torchref.experimental.mm.OpenMMAdapter` with a
:meth:`~torchref.experimental.mm.CrystalLayout.quasi_crystal` layout, from the
single-copy chemistry of the ensemble. A molecule whose symmetry copies coincide -- a
water on a two-fold axis -- is held once per site rather than stacked on itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import torch

from torchref.experimental.mm import CrystalLayout, OpenMMAdapter
from torchref.experimental.mm.adapter import DEFAULT_FORCEFIELD
from torchref.experimental.targets.amber_target import require_openmm
from torchref.refinement.targets.base import ModelTarget
from torchref.utils.stats import VERBOSITY_DETAILED, VERBOSITY_STANDARD, stat

from .ensemble_model import build_single_copy_model

if TYPE_CHECKING:
    from torchref.symmetry import Cell, SpaceGroup

    from .ensemble_model import EnsembleModel


class QuasiCrystalAmberTarget(ModelTarget):
    """Amber energy of an ensemble laid out as a ``n_disorder × 1 × 1`` supercell.

    .. warning::

       Experimental — API and behaviour may change without notice.

    This is the production ensemble Amber restraint, wired by
    :class:`EnsembleRefinement`.

    Parameters
    ----------
    model : EnsembleModel
        Ensemble with ``n_members = n_disorder * N_sym``, carrying the hydrogens AMBER
        needs (build it with ``hydrogens="add"``). Every member slot is in the
        supercell, dead ones included.
    cell : Cell
        Unit cell of the crystal.
    spacegroup : SpaceGroup
        Its space group; every operation, centring included, places one member per
        cell.
    n_disorder : int
        Unit cells along a (k ≥ 1). Must satisfy
        ``n_disorder * spacegroup.n_ops == model.n_members``.
    pme_cutoff_ang : float
        Real-space non-bonded cutoff in Å. Default 10.
    ewald_error_tolerance : float
        PME accuracy. Default 5e-4.
    normalize_per_asu : bool
        If True (default), ``forward()`` returns the supercell energy divided by
        ``n_members``, the number of asymmetric units it holds: a per-structure scale
        comparable with the per-ASU X-ray terms. If False, the supercell total.
    residue_charges : dict, optional
        Net charge per GAFF2 residue name, overriding the dictionary's formal charges.
    forcefield : sequence of str
        OpenMM force-field XML files.
    charge_method : {"gas", "bcc"}
        antechamber charges for GAFF2 residues. Default ``"gas"``.
    overlap_cutoff : float
        Distance in Å below which two copies of a molecule are one site; ``0`` keeps
        every copy.
    relax_on_init : bool
        If True, minimise the supercell with OpenMM and write the relaxed coordinates
        back into the ensemble. Default False keeps TorchRef's coordinates.
    relax_max_iterations : int
        Minimiser iteration limit.
    force_clamp : float
        Per-atom gradient clip in kJ/mol/nm. Default 10000.
    platform : str, optional
        OpenMM platform name.
    verbose : int

    Raises
    ------
    ValueError
        If ``n_disorder * N_sym`` differs from ``model.n_members``, or the ensemble
        holds fewer than half the hydrogens its dictionaries call for.
    """

    name: str = "quasi_crystal_amber"

    def __init__(
        self,
        model: "EnsembleModel",
        cell: "Cell",
        spacegroup: "SpaceGroup",
        n_disorder: int,
        pme_cutoff_ang: float = 10.0,
        ewald_error_tolerance: float = 5e-4,
        normalize_per_asu: bool = True,
        residue_charges: Optional[Dict[str, int]] = None,
        forcefield: Sequence[str] = DEFAULT_FORCEFIELD,
        charge_method: str = "gas",
        overlap_cutoff: float = 1.5,
        relax_on_init: bool = False,
        relax_max_iterations: int = 200,
        force_clamp: float = 10000.0,
        platform: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        require_openmm()
        n_sym = int(spacegroup.n_ops)
        n_members = int(model.n_members)
        if int(n_disorder) * n_sym != n_members:
            raise ValueError(
                f"n_members ({n_members}) must equal n_disorder ({n_disorder}) "
                f"* spacegroup.n_ops ({n_sym}) = {int(n_disorder) * n_sym}"
            )
        super().__init__(model=model, verbose=verbose)
        self._n_disorder = int(n_disorder)
        self._n_sym = n_sym
        self._n_members = n_members
        self._normalize_per_asu = bool(normalize_per_asu)

        chemistry = build_single_copy_model(model, verbose=verbose)
        self.adapter = OpenMMAdapter.from_model(
            chemistry,
            layout=CrystalLayout.quasi_crystal(cell, spacegroup, self._n_disorder),
            xyz=model.xyz_per_member,
            forcefield=forcefield,
            charge_method=charge_method,
            residue_charges=residue_charges,
            nonbonded="pme",
            cutoff=pme_cutoff_ang,
            ewald_error_tolerance=ewald_error_tolerance,
            overlap_cutoff=overlap_cutoff,
            hydrogens_added=getattr(model, "hydrogen_source", "keep") == "add",
            platform=platform,
            max_force=force_clamp,
            verbose=verbose,
        )
        if relax_on_init:
            self._relax_against_amber(int(relax_max_iterations))

    def _relax_against_amber(self, max_iterations: int) -> None:
        """Minimise the supercell in OpenMM and write the members back.

        Coordinates enter the ensemble's flat ``xyz`` parameter; the coordinate and
        structure-factor caches are reset so the next read sees them.
        """
        xyz = self._model.xyz_per_member.detach()
        relaxed = self.adapter.minimize(xyz, max_iterations=max_iterations)
        parameter = self._model.xyz.refinable_params
        if parameter.shape != (xyz.shape[0] * xyz.shape[1], 3):
            raise NotImplementedError(
                "relax_on_init writes plain per-atom coordinates; this ensemble's "
                "xyz is parametrised differently."
            )
        with torch.no_grad():
            parameter.copy_(relaxed.reshape(-1, 3).to(parameter))
        if hasattr(self._model.xyz, "reset_forward_cache"):
            self._model.xyz.reset_forward_cache()
        self._model.reset_cache()

    def forward(self) -> torch.Tensor:
        """Supercell energy in kJ/mol, per asymmetric unit unless disabled."""
        energy = self.adapter.energy(self._model.xyz_per_member)
        if self._normalize_per_asu:
            energy = energy / float(self._n_members)
        return energy

    def stats(self) -> Dict:
        """Return target statistics for the logging pipeline."""
        with torch.no_grad():
            loss = self.forward().item()
        return {
            "loss": stat(loss, VERBOSITY_STANDARD),
            "n_particles": stat(self.adapter.n_particles, VERBOSITY_DETAILED),
            "absent_copies": stat(
                int((~self.adapter.present).sum()), VERBOSITY_DETAILED
            ),
        }
