"""AMBER force-field energy of a TorchRef model, as a refinement target.

A thin target over :class:`~torchref.experimental.mm.OpenMMAdapter`, which builds the
OpenMM system from the model's own context -- AMBER ff14SB templates, GAFF2 for
residues those do not cover -- and owns the exchange of coordinates and forces. The
model must already carry the atoms and protonation state AMBER needs; this target adds
nothing to it. Riding hydrogens and other coordinate parametrisations receive their
forces through the model's own wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

import torch

from torchref.experimental.mm import CrystalLayout, OpenMMAdapter
from torchref.experimental.mm.adapter import DEFAULT_FORCEFIELD
from torchref.refinement.targets.base import ModelTarget
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.model.model import Model

#: Layouts :class:`AmberTarget` builds the system in.
LAYOUTS = ("isolated", "crystal")


def require_openmm() -> None:
    """Raise an ImportError naming the install command when OpenMM is absent."""
    try:
        import openmm  # noqa: F401
    except ImportError:
        raise ImportError(
            "The AMBER targets require OpenMM.\n"
            "Install with:  pip install torchref[amber]\n"
            "Or via conda:  conda install -c conda-forge openmm"
        ) from None


class AmberTarget(ModelTarget):
    """Differentiable AMBER energy of the model's coordinates.

    Parameters
    ----------
    model : Model
        Single-conformation model carrying every hydrogen and terminal atom AMBER
        needs; not modified. Hydrogens TorchRef did not generate draw a warning, and
        fewer than half of those the dictionaries call for raise ``ValueError``.
    cutoff : float
        Non-bonded cutoff in Å. Default 5.0.
    normalize_by_atoms : bool
        Divide the energy by the number of model atoms. Default True.
    layout : {"isolated", "crystal"}
        ``"isolated"``: the model alone, reaction-field electrostatics within
        ``cutoff``. ``"crystal"``: every symmetry copy of the model in the unit cell
        under periodic boundaries with PME, and the energy per asymmetric unit; a
        molecule on a special position is held once per site.
    forcefield : sequence of str
        OpenMM force-field XML files; residues none of them matches are parameterised
        with GAFF2 from their monomer dictionary (needs AmberTools).
    charge_method : {"gas", "bcc"}
        antechamber charges for GAFF2 residues: Gasteiger, or AM1-BCC (slower, can
        fail to converge).
    residue_charges : dict, optional
        Net charge per GAFF2 residue name, overriding the dictionary's formal charges.
    max_force : float
        Per-atom clip of the gradient in kJ/mol/nm. Default 10000.
    overlap_cutoff : float
        Crystal layout only: distance in Å below which two copies of a molecule are
        one site.
    platform : str, optional
        OpenMM platform name; by default the one matching the coordinates' device.
    verbose : int

    Notes
    -----
    Rebuild the target after changing atom identities, order or connectivity;
    coordinate changes need nothing. Every evaluation moves the coordinates to host
    memory and the forces back. Above ``max_force`` the gradient is clipped rather than
    the exact derivative of the energy.
    """

    name: str = "amber"

    def __init__(
        self,
        model: "Model" = None,
        cutoff: float = 5.0,
        normalize_by_atoms: bool = True,
        layout: str = "isolated",
        forcefield: Sequence[str] = DEFAULT_FORCEFIELD,
        charge_method: str = "gas",
        residue_charges: Optional[Dict[str, int]] = None,
        max_force: float = 10000.0,
        overlap_cutoff: float = 1.5,
        platform: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        require_openmm()
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        super().__init__(model=model, verbose=verbose)
        self._normalize = normalize_by_atoms
        self.layout_name = layout
        self.adapter: Optional[OpenMMAdapter] = None
        if model is None:
            return
        crystal = layout == "crystal"
        self.adapter = OpenMMAdapter.from_model(
            model,
            layout=(
                CrystalLayout.unit_cell(model.cell, model.spacegroup, cutoff)
                if crystal
                else None
            ),
            forcefield=forcefield,
            charge_method=charge_method,
            residue_charges=residue_charges,
            nonbonded="pme" if crystal else "cutoff",
            cutoff=cutoff,
            overlap_cutoff=overlap_cutoff,
            platform=platform,
            max_force=max_force,
            verbose=verbose,
        )

    def forward(self) -> torch.Tensor:
        """AMBER energy in kJ/mol per asymmetric unit, per atom if normalised."""
        if self.adapter is None:
            raise RuntimeError("AmberTarget was built without a model")
        energy = self.adapter.energy(self._model.xyz()) / self.adapter.layout.n_copies
        if self._normalize:
            energy = energy / self.adapter.n_model_atoms
        return energy

    def stats(self) -> Dict[str, "StatEntry"]:
        """Return target statistics for the logging pipeline."""
        with torch.no_grad():
            loss = self.forward().item()
        n_atoms = self.adapter.n_model_atoms
        return {
            "loss": stat(loss, VERBOSITY_STANDARD),
            "energy_kJ_mol": stat(
                loss * n_atoms if self._normalize else loss, VERBOSITY_DETAILED
            ),
            "n_atoms": stat(n_atoms, VERBOSITY_DEBUG),
            "n_copies": stat(self.adapter.layout.n_copies, VERBOSITY_DEBUG),
            "platform": stat(self.adapter.platform_name, VERBOSITY_DETAILED),
        }
