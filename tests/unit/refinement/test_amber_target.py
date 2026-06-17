"""
Unit tests for the single-molecule
:class:`~torchref.refinement.targets.amber_target.AmberTarget`.

Uses a ligand-free protein (7L84) so the standard OpenMM ``Modeller`` path is
exercised — no antechamber/tleap (AmberTools) required, only OpenMM. This
closes the previous coverage gap where the single-molecule target had no unit
test (only manual scripts), which is how it was able to silently rot while the
ensemble targets were developed.
"""

import os

import pytest
import torch

openmm = pytest.importorskip("openmm")

from torchref.model.model import Model
from torchref.refinement.targets.amber_target import AmberTarget

# Ligand-free protein → standard Modeller path, no antechamber needed.
TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "7L84.pdb"
)


@pytest.fixture(scope="module")
def heavy_model() -> Model:
    """Heavy-atom, single-conformation model (OpenMM adds H internally)."""
    return Model(verbose=0, strip_H=True).load_pdb(TEST_PDB).strip_altlocs()


@pytest.fixture(scope="module")
def target(heavy_model) -> AmberTarget:
    """Built once — the OpenMM system construction is the expensive part."""
    return AmberTarget(model=heavy_model, verbose=0)


def test_build_populates_state(target, heavy_model):
    """Construction builds a usable OpenMM context + atom map + H tables."""
    assert target._context is not None
    assert target._system is not None
    assert target._n_model_atoms == len(heavy_model.pdb)
    assert target._n_omm_atoms >= target._n_model_atoms  # H added by Modeller
    # For the single molecule the chemistry model IS the model.
    assert target._chem_model is target._model
    # H-attachment tables were built (this is a protein → has H).
    assert target._h_idx is not None and target._h_idx.size > 0


def test_forward_finite_energy(target):
    """forward() returns a finite scalar energy."""
    e = target.forward()
    assert e.shape == ()
    assert torch.isfinite(e).item()


def test_backward_flows_to_xyz(heavy_model):
    """Energy gradient propagates to the model's xyz parameters and is finite."""
    target = AmberTarget(model=heavy_model, verbose=0)
    heavy_model.xyz.refinable_params.grad = None
    e = target.forward()
    e.backward()
    g = heavy_model.xyz.refinable_params.grad
    assert g is not None
    assert g.shape == (len(heavy_model.pdb), 3)
    assert torch.isfinite(g).all().item()
    assert g.abs().sum().item() > 0.0  # non-trivial forces


def test_default_charge_method_is_gas():
    """The unified base defaults to the (robust) Gasteiger charge method."""
    import inspect

    sig = inspect.signature(AmberTarget.__init__)
    assert sig.parameters["charge_method"].default == "gas"
