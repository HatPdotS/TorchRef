"""refine_rigid_body must not disturb the refinement it is called on.

Every cutoff rebinds the step's Refinement to a resolution-truncated data view,
which rebuilds the scaler and every target and drops the loss state. Those
rebuilds are needed for the x-ray target, whose data and scaler genuinely
change; they are collateral for the ADP and geometry targets, which are built
from the model alone and are dropped from the loss state before the optimizer
runs. `RigidBodyRefinementStep` therefore runs against a sandbox clone, and
these tests pin that the caller sees none of it.
"""
import pytest
import torch

from torchref import LBFGSRefinement


@pytest.fixture(scope="module")
def refinement(mtz_dir, pdb_dir):
    def build():
        return LBFGSRefinement(
            data_file=str(mtz_dir / "3E98.mtz"),
            pdb=str(pdb_dir / "3E98.pdb"),
            device=torch.device("cpu"),
            verbose=0,
        )
    return build


def test_component_restraint_config_survives(refinement):
    """A sigma set on a target must still be set afterwards.

    Regression: `_init_targets` rebuilt `TotalADPTarget` per cutoff with no
    restraint parameters, so this silently reverted to the ADPSimilarityTarget
    default of 2.0 and refinement continued at a restraint weight nobody chose.
    """
    ref = refinement()
    ref.adp_target["simu"].simu_sigma = 0.25
    ref.get_scales()

    ref.refine_rigid_body(iterations_per_step=10)

    assert ref.adp_target["simu"].simu_sigma == pytest.approx(0.25)


def test_custom_loss_state_weight_survives(refinement):
    """A weight registered on the LossState must still be registered afterwards.

    `adp/simu` is deliberately a key absent from DEFAULT_GROUP_WEIGHTS: a key
    that is present gets visibly overwritten, while a custom one silently
    disappears and its target falls back to the group weight.
    """
    ref = refinement()
    ref.get_scales()
    ref.complete_loss_state().set_weight("adp/simu", 0.77)

    ref.refine_rigid_body(iterations_per_step=10)

    assert ref.complete_loss_state().weights.get("adp/simu") == pytest.approx(0.77)


def test_targets_and_data_are_not_replaced(refinement):
    """Object identity, not just values -- a caller may hold its own references."""
    ref = refinement()
    ref.get_scales()
    adp, geometry, data = ref.adp_target, ref.geometry_target, ref.reflection_data

    ref.refine_rigid_body(iterations_per_step=10)

    assert ref.adp_target is adp
    assert ref.geometry_target is geometry
    assert ref.reflection_data is data


def test_refined_coordinates_still_reach_the_caller(refinement):
    """The sandbox shares the model, so the whole point still has to work."""
    ref = refinement()
    ref.get_scales()
    before = ref.model.xyz().detach().clone()

    ref.refine_rigid_body(iterations_per_step=30)

    shift = (ref.model.xyz().detach() - before).norm(dim=-1)
    assert float(shift.max()) > 0.0, "rigid body moved nothing"


def test_refinement_is_usable_afterwards(refinement):
    """A normal macrocycle must still run against the caller's own objects."""
    ref = refinement()
    ref.get_scales()

    ref.refine_rigid_body(iterations_per_step=10)
    ref.refine_scaler()
    ref.refine_adp()

    rwork, rfree = ref.get_rfactor()
    assert 0.0 < rwork < 1.0 and 0.0 < rfree < 1.0
