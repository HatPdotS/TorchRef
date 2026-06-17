"""
End-to-end integration test for ``EnsembleRefinement`` on 1DAW.

Asserts: validation set carved automatically, model is an EnsembleModel,
R_work decreases over macro cycles, members do not collapse to a single
configuration.
"""

import os
import shutil

import pytest
import torch

# This end-to-end test builds the QuasiCrystal Amber target (amber_weight=1.0),
# so it requires OpenMM. Skip the whole module if OpenMM isn't installed.
openmm = pytest.importorskip("openmm")

from torchref.experimental.ensemble import EnsembleModel
from torchref.experimental.ensemble import EnsembleRefinement


def _have_amber_tools() -> bool:
    """antechamber/tleap are needed because 1DAW's ANP ligand is non-standard
    and is parameterised via GAFF2/antechamber during the Amber build."""
    return shutil.which("antechamber") is not None and shutil.which("tleap") is not None


# The module-scoped ``refinement`` fixture builds Amber eagerly, so every test
# here transitively needs AmberTools — gate the whole module.
pytestmark = pytest.mark.skipif(
    not _have_amber_tools(),
    reason="AmberTools (antechamber/tleap) not available — run under the conda "
           "env with ambertools installed for the full Amber integration test.",
)


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
        # exercises the real QuasiCrystal Amber path. The module is gated on
        # OpenMM (and AmberTools, since 1DAW's ANP ligand is parameterised via
        # antechamber) — see the importorskip/skipif at the top of the file.
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
