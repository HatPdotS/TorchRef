"""
End-to-end integration test for ``EnsembleRefinement`` on 1DAW.

Asserts: validation set carved automatically, model is an EnsembleModel,
R_work decreases over macro cycles, members do not collapse to a single
configuration.
"""

import os

import pytest
import torch

from torchref.model import EnsembleModel
from torchref.refinement import EnsembleRefinement


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
        amber_lam=0.0,         # OpenMM disabled in CI tests
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
    assert int(refinement.reflection_data.val_idx.shape[0]) > 0
    n_total = (
        int(refinement.reflection_data.work_idx.shape[0])
        + int(refinement.reflection_data.free_idx.shape[0])
        + int(refinement.reflection_data.val_idx.shape[0])
    )
    assert n_total == len(refinement.reflection_data)


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
