"""``adp_target_mode`` gives the ADP/occupancy steps their own LossState and x-ray target.

Pins:
1. Unset, the ADP steps share the one persistent LossState.
2. Set to a different mode, :meth:`refine_adp` steps a second state whose ``"xray"`` term
   is that mode's target, while :meth:`refine_xyz` keeps the primary one; the geometry
   and ADP terms are the same objects in both, and the scaler regularizers wrap the same
   scaler, so restraints cannot diverge between the two steps.
3. ``reset_loss_state`` drops both states.
"""

import pytest

from torchref import LBFGSRefinement
from torchref.refinement.targets.xray.ml import MLXrayTarget
from torchref.refinement.targets.xray.ml_noalpha import MLNoAlphaXrayTarget

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def paths(pdb_dir, mtz_dir):
    pdb, mtz = pdb_dir / "1DAW.pdb", mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")
    return str(pdb), str(mtz)


def test_unset_shares_one_state(paths):
    pdb, mtz = paths
    ref = LBFGSRefinement(data_file=mtz, pdb=pdb, verbose=0, target_mode="ml")
    assert ref.xray_target_work_adp is None
    assert ref.adp_loss_state is ref.loss_state


def test_same_mode_shares_one_state(paths):
    pdb, mtz = paths
    ref = LBFGSRefinement(
        data_file=mtz, pdb=pdb, verbose=0, target_mode="ml", adp_target_mode="ml"
    )
    assert ref.adp_loss_state is ref.loss_state


def test_adp_steps_run_on_their_own_state(paths):
    pdb, mtz = paths
    ref = LBFGSRefinement(
        data_file=mtz,
        pdb=pdb,
        verbose=0,
        target_mode="ml",
        adp_target_mode="ml_noalpha",
    )
    main, adp = ref.loss_state, ref.adp_loss_state
    assert adp is not main
    assert type(main.targets["xray"]) is MLXrayTarget
    assert type(adp.targets["xray"]) is MLNoAlphaXrayTarget
    assert set(main.targets) == set(adp.targets)
    for name in main.targets:
        if name == "xray":
            continue
        if name.startswith("adp/scaler_"):
            assert adp.targets[name]._scaler is main.targets[name]._scaler, name
        else:
            assert adp.targets[name] is main.targets[name], name

    ref.LBFGS_DEFAULTS = {**LBFGSRefinement.LBFGS_DEFAULTS, "max_iter": 2}
    assert ref.refine_adp() is adp
    assert ref.refine_xyz() is main

    ref.reset_loss_state()
    assert ref.loss_state is not main
    assert ref.adp_loss_state is not adp
    assert ref.adp_loss_state is not ref.loss_state
