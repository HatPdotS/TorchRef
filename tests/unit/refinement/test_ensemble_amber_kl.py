"""
Unit tests for the entropy regularizer of
:class:`~torchref.experimental.ensemble.ensemble_amber_kl.EnsembleAmberKLTarget`.

These run with ``kT=0`` so the loss is the entropy term alone (the critical
anti-collapse mechanism). ``EnsembleAmberKLTarget`` now builds its OpenMM /
AMBER system eagerly in ``__init__`` (it is an ``AmberTarget`` subclass), so
even the entropy-only configuration requires OpenMM **and** AmberTools to
parameterise the 1DAW ligand (ANP) — hence the module-level gates. ANP needs an
odd net charge for a closed-shell electron count; ``-3`` (the AMP-PNP
triphosphate charge) satisfies antechamber's even-electron requirement.
"""

import os
import shutil as _sh

import pytest
import torch

openmm = pytest.importorskip("openmm")

from torchref.experimental.ensemble import EnsembleAmberKLTarget, EnsembleModel


def _ambertools_available() -> bool:
    """Check whether antechamber/tleap are available (required because the
    1DAW ligand ANP is non-standard and is parameterised via antechamber)."""
    return _sh.which("antechamber") is not None and _sh.which("tleap") is not None


requires_amber_tools = pytest.mark.skipif(
    not _ambertools_available(),
    reason="AmberTools (antechamber/tleap) not available — run under the "
           "conda env with ambertools installed for the full Amber tests.",
)

# The eager Amber build parameterises ANP, so every test in this module needs
# AmberTools (the entropy-only kT=0 path no longer dodges the build).
pytestmark = requires_amber_tools

TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "1DAW.pdb"
)

# ANP (AMP-PNP) net charge: -3 gives antechamber a closed-shell (even-electron)
# molecule. MG is an AMBER14-standard ion and needs no entry.
RESIDUE_CHARGES = {"ANP": -3}


@pytest.fixture
def ensemble() -> EnsembleModel:
    return EnsembleModel.from_single(
        TEST_PDB, n_members=8, perturb_sigma=0.5, b_const=5.0,
        seed=42, verbose=0,
    )


def test_entropy_only_mode_is_finite(ensemble):
    target = EnsembleAmberKLTarget(
        model=ensemble, lam=1.0, kT=0.0, residue_charges=RESIDUE_CHARGES, verbose=0
    )
    loss = target.forward()
    assert torch.isfinite(loss)


def test_collapsed_ensemble_yields_higher_loss(ensemble):
    target = EnsembleAmberKLTarget(
        model=ensemble, lam=1.0, kT=0.0, residue_charges=RESIDUE_CHARGES, verbose=0
    )
    loss_spread = target.forward().item()

    # Collapse all members to member 0.
    flat = ensemble.xyz.refinable_params
    view = flat.view(ensemble.n_members, ensemble.n_atoms_per_member, 3)
    member0 = view[0].clone()
    with torch.no_grad():
        for i in range(1, ensemble.n_members):
            view[i] = member0
    if hasattr(ensemble, "reset_cache"):
        ensemble.reset_cache()
    loss_collapsed = target.forward().item()

    assert loss_collapsed > loss_spread, (
        f"Collapsed ensemble (loss={loss_collapsed}) should exceed "
        f"spread (loss={loss_spread})"
    )


def test_gradient_flows_back_to_xyz(ensemble):
    target = EnsembleAmberKLTarget(
        model=ensemble, lam=1.0, kT=0.0, residue_charges=RESIDUE_CHARGES, verbose=0
    )
    ensemble.xyz.refinable_params.grad = None
    loss = target.forward()
    loss.backward()
    g = ensemble.xyz.refinable_params.grad
    assert g is not None
    assert torch.isfinite(g).all()
