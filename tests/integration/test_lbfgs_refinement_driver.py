"""LBFGSRefinement driver behaviour on a deposited structure (1DAW)."""

import pytest
import torch

from torchref import LBFGSRefinement


@pytest.fixture(scope="module")
def build(mtz_dir, pdb_dir):
    def _build():
        return LBFGSRefinement(
            data_file=str(mtz_dir / "1DAW.mtz"),
            pdb=str(pdb_dir / "1DAW.pdb"),
            device=torch.device("cpu"),
            verbose=0,
        )

    return _build


@pytest.mark.integration
def test_load_state_restores_a_save_state_checkpoint(build, tmp_path):
    """A checkpoint loads onto a refinement built from the same files, metadata and
    all, and reproduces its coordinates, every scaler parameter (scale, anisotropy,
    bulk solvent) and its R-factors exactly."""
    saved = build()
    with torch.no_grad():
        saved.model.xyz.refinable_params.add_(0.03)
        for p in saved.scaler.parameters():
            p.add_(0.01)
    path = tmp_path / "refinement.pt"
    saved.save_state(str(path))

    restored = build()
    restored.load_state(str(path))

    assert torch.equal(restored.model.xyz(), saved.model.xyz())
    scaler = dict(saved.scaler.named_parameters())
    for name, value in restored.scaler.named_parameters():
        assert torch.equal(value, scaler[name]), name
    assert restored.get_rfactor() == saved.get_rfactor()


@pytest.mark.integration
def test_load_state_refuses_a_checkpoint_missing_a_parameter(build, tmp_path):
    """A parameter of the refinement that the checkpoint lacks is an error."""
    ref = build()
    state = ref.state_dict()
    del state["scaler.c_iso"]
    path = tmp_path / "partial.pt"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match="scaler.c_iso"):
        ref.load_state(str(path))


@pytest.mark.integration
def test_refine_xyz_with_xyz_frozen_leaves_coordinates_alone(build):
    """With xyz frozen the coordinate step has nothing to refine and is skipped."""
    ref = build()
    ref.model.freeze("xyz")
    before = ref.model.xyz().detach().clone()

    ref.refine_xyz()

    assert torch.equal(ref.model.xyz(), before)


@pytest.mark.integration
def test_a_second_refine_everything_still_moves_the_coordinates(build):
    """Each refine_everything call optimizes the model's current parameters, not the
    ones an earlier call's optimizer was built over."""
    ref = build()
    ref.LBFGS_DEFAULTS = dict(ref.LBFGS_DEFAULTS, max_iter=3)
    ref.refine_everything(macro_cycles=1)
    before = ref.model.xyz().detach().clone()

    ref.refine_everything(macro_cycles=1)

    assert not torch.equal(ref.model.xyz(), before)


@pytest.mark.integration
def test_refine_everything_refits_the_scale_warm(build, monkeypatch):
    """Each cycle refits the scaler from where the previous one left it: the bulk
    solvent model stays the same object and the scaler is never cold-started."""
    ref = build()
    ref.LBFGS_DEFAULTS = dict(ref.LBFGS_DEFAULTS, max_iter=3)
    solvent = ref.scaler.solvent

    def cold_start(*args, **kwargs):
        raise AssertionError("refine_everything cold-started the scaler")

    monkeypatch.setattr(ref.scaler, "initialize", cold_start)
    ref.refine_everything(macro_cycles=2)

    assert ref.scaler.solvent is solvent


@pytest.mark.integration
def test_complete_loss_state_evaluates_no_target(build):
    """complete_loss_state hands back the persistent state as it is: after an aggregate
    it does not go on to evaluate the zero-weight targets the aggregate skipped."""
    ref = build()
    state = ref.loss_state
    assert state.get_effective_weight("geometry/ramachandran") == 0.0

    def unreachable():
        raise AssertionError("complete_loss_state evaluated a target")

    state.targets["geometry/ramachandran"] = unreachable
    with torch.no_grad():
        state.aggregate()

    assert ref.complete_loss_state() is state


@pytest.mark.integration
def test_geometry_accessors_evaluate_their_component(build):
    """bond_loss, angle_loss and torsion_loss are the named geometry components."""
    ref = build()
    with torch.no_grad():
        for accessor, key in (
            (ref.bond_loss, "bond"),
            (ref.angle_loss, "angle"),
            (ref.torsion_loss, "torsion"),
        ):
            torch.testing.assert_close(accessor(), ref.geometry_target[key]())


@pytest.mark.integration
@pytest.mark.parametrize("given", ["data_file", "pdb"])
def test_one_input_file_alone_is_refused(given, mtz_dir, pdb_dir):
    """data_file and pdb come as a pair; one alone builds neither a refinement nor
    the empty shell."""
    path = {"data_file": mtz_dir / "1DAW.mtz", "pdb": pdb_dir / "1DAW.pdb"}[given]
    with pytest.raises(ValueError, match="together"):
        LBFGSRefinement(**{given: str(path)}, device=torch.device("cpu"), verbose=0)


@pytest.mark.integration
def test_path_objects_are_accepted_as_input_files(mtz_dir, pdb_dir):
    """pathlib paths load the same files as their string forms."""
    ref = LBFGSRefinement(
        data_file=mtz_dir / "1DAW.mtz",
        pdb=pdb_dir / "1DAW.pdb",
        device=torch.device("cpu"),
        verbose=0,
    )

    assert ref.data_file == str(mtz_dir / "1DAW.mtz")
    assert ref.pdb == str(pdb_dir / "1DAW.pdb")
    assert len(ref.reflection_data) > 0 and ref.model.xyz().shape[0] > 0


@pytest.mark.integration
def test_column_names_with_an_sf_mmcif_file_are_refused(cif_sf_dir, pdb_dir):
    """A column choice cannot apply to SF-mmCIF input, so it is an error rather than
    silently ignored."""
    with pytest.raises(ValueError, match="column_names"):
        LBFGSRefinement(
            data_file=str(cif_sf_dir / "1DAW-sf.cif"),
            pdb=str(pdb_dir / "1DAW.pdb"),
            column_names={"F": "FP", "SIGF": "SIGFP"},
            device=torch.device("cpu"),
            verbose=0,
        )
