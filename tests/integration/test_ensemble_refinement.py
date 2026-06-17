"""
End-to-end integration test for ``EnsembleRefinement`` on 1DAW.

Asserts: validation set carved automatically, model is an EnsembleModel,
R_work decreases over macro cycles, members do not collapse to a single
configuration.
"""

import os

import pytest
import torch

from torchref.experimental.ensemble import EnsembleModel
from torchref.experimental.ensemble import EnsembleRefinement

# The module-scoped ``refinement`` fixture builds the QuasiCrystal Amber target
# eagerly (amber_weight=1.0), parameterising 1DAW's ANP ligand via GAFF2, so
# every test here needs OpenMM + AmberTools. Gated centrally in conftest.
pytestmark = pytest.mark.amber


TEST_MTZ = os.path.join(
    os.path.dirname(__file__), "..", "files", "mtz", "1DAW.mtz"
)
TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "files", "pdb", "1DAW.pdb"
)


@pytest.fixture(scope="module")
def refinement() -> EnsembleRefinement:
    torch.manual_seed(0)
    return EnsembleRefinement(
        data_file=TEST_MTZ,
        pdb=TEST_PDB,
        n_members=4,           # small N so the test stays fast
        perturb_sigma=0.01,    # symmetry-breaking only; clashes from larger values
        b_const=5.0,
        wilson_weight=0.5,
        # Amber is ON (default amber_weight=1.0) so this end-to-end test
        # exercises the real QuasiCrystal Amber path, including parameterising
        # 1DAW's ANP ligand (which is protonated from the monomer library).
        # The init OpenMM energy-minimisation is disabled: 1DAW's supercell has
        # special-position/metal clashes that make the (non-clamped) minimizer
        # diverge to NaN — a separate pre-existing amber-stability issue. The
        # differentiable forward clamps per-atom forces, so refinement is fine.
        amber_relax_on_init=False,
        amber_lam=0.0,
        amber_kT=0.0,
        val_fraction_of_free=0.5,
        xray_mode="ls",
        seed=42,
        verbose=0,
        max_res=3.0,
    )


def test_model_is_ensemble(refinement):
    assert isinstance(refinement.model, EnsembleModel)
    assert refinement.model.n_members == 4


def test_validation_set_was_generated(refinement):
    # work/free/validation are _ReflectionSubset views (.n) on ReflectionData.
    data = refinement.reflection_data
    assert int(data.validation.n) > 0
    n_total = int(data.work.n) + int(data.free.n) + int(data.validation.n)
    assert n_total == len(data)


def test_three_xray_targets_registered(refinement):
    assert refinement.xray_target_work is not None
    assert refinement.xray_target_test is not None
    assert refinement.xray_target_validation is not None
    assert refinement.xray_target_work.use_set == "work"
    assert refinement.xray_target_test.use_set == "free"
    assert refinement.xray_target_validation.use_set == "val"


def test_refine_decreases_rwork_and_keeps_ensemble_spread(refinement):
    hist = refinement.refine(macro_cycles=2)
    assert len(hist["rwork"]) == 2
    assert hist["rwork"][-1] <= hist["rwork"][0], \
        f"R_work did not decrease: {hist['rwork']}"

    # Members should not have collapsed completely.
    # perturb_sigma=0.01 → initial sum-of-variances per atom ≈ 3 * 0.01² = 3e-4
    # which is above the floor; after a couple of refinement cycles it should
    # have either grown (X-ray + entropy drives spread) or stayed similar.
    xyz = refinement.model.xyz_per_member.detach()
    var_per_atom = xyz.var(dim=0, unbiased=False).sum(dim=-1)
    assert float(var_per_atom.mean()) > 1e-5
