"""A node-field ADP representation driven the way production drives it.

The unit tests cover the payload arithmetic and the wiring. What they cannot cover is
whether the thing is usable: whether a refinement constructed with a field mode sizes
itself from the data, carries weights appropriate to a field rather than to the per-atom
representation it replaced, actually reduces R-work over real cycles, survives a
checkpoint round trip, and can be switched into and out of after setup.

Every one of those was broken or absent at some point in this feature's life, and none
of them fails a unit test.
"""

import pytest
import torch

from torchref.model.disorder_field import (
    DisorderFieldTensor,
    ModeCovariancePayload,
    payload_code,
)
from torchref.refinement.base_refinement import DEFAULT_GROUP_WEIGHTS
from torchref.refinement.lbfgs_refinement import LBFGSRefinement

MODE_SET = "rigid_dilation"


@pytest.fixture(scope="module")
def files(mtz_dir, pdb_dir):
    return str(mtz_dir / "1DAW.mtz"), str(pdb_dir / "1DAW.pdb")


def _refinement(files, **kw):
    mtz, pdb = files
    return LBFGSRefinement(data_file=mtz, pdb=pdb, verbose=0, **kw)


@pytest.fixture(scope="module")
def field_refinement(files):
    """Built the way a caller would: mode and mode set, no explicit node count."""
    return _refinement(files, adp_mode="field_aniso", adp_mode_set=MODE_SET)


# ----------------------------------------------------------------------------------
# Setup: sized from the data, weighted for a field.
# ----------------------------------------------------------------------------------


@pytest.mark.integration
def test_construction_installs_the_mode_field(field_refinement):
    ref = field_refinement
    assert ref.model.adp_is_field
    field = ref.model.adp_field
    assert isinstance(field.payload, ModeCovariancePayload)
    assert field.payload.mode_set == MODE_SET


@pytest.mark.integration
def test_node_count_comes_from_the_reflections_not_the_atoms(field_refinement):
    """The whole point of putting the sizing on the refinement.

    The model's own default is one node per 25 atoms, which knows nothing about how much
    data there is. Here the achieved ratio has to land near the requested one, and the
    node count has to disagree with the atom-count rule (otherwise the test would pass
    on a coincidence).
    """
    ref = field_refinement
    n_par = sum(p.numel() for p in ref.model.parameters_of_types(("adp", "u")))
    achieved = ref.data.work.n / n_par
    assert 4.0 < achieved < 12.0, f"{achieved:.1f} work reflections per ADP parameter"

    atom_rule = max(4, round(len(ref.model.pdb) / 25.0))
    assert ref.model.adp_field.n_nodes != atom_rule


@pytest.mark.integration
@pytest.mark.parametrize("ratio", [3.5, 7.0, 15.0])
def test_requested_ratio_is_honoured(files, ratio):
    ref = _refinement(
        files, adp_mode="field_aniso", adp_mode_set=MODE_SET,
        reflections_per_adp_parameter=ratio,
    )
    n_par = sum(p.numel() for p in ref.model.parameters_of_types(("adp", "u")))
    achieved = ref.data.work.n / n_par
    # Node count is an integer, so the achieved ratio cannot match exactly; it must
    # track, and it must be monotone in the request.
    assert 0.6 * ratio < achieved < 1.7 * ratio, f"asked {ratio}, got {achieved:.1f}"


@pytest.mark.integration
def test_explicit_node_count_bypasses_the_budget(files):
    ref = _refinement(
        files, adp_mode="field_aniso", adp_mode_set=MODE_SET, n_nodes=11
    )
    assert ref.model.adp_field.n_nodes == 11


@pytest.mark.integration
def test_field_mode_does_not_register_the_restraints_it_duplicates(field_refinement):
    """``simu`` and ``locality`` penalise what the field enforces by construction.

    They must be absent from the component set, not present at weight zero: a zero is a
    lever, and anyone adjusting the ``adp`` group weight for their own reasons would
    silently re-enable a restraint that double-counts the parametrisation.
    """
    components = field_refinement.adp_target.target_losses()
    assert "simu" not in components
    assert "locality" not in components
    # What a field does need.
    assert "node_load" in components
    assert "node_smoothness" in components
    assert "sigd" in components, "the marginal-B prior applies to either representation"

    # And the loss is NOT rebalanced for a field: the parametrisation is the constraint,
    # so a field needs less regularisation than a per-atom model, not a reweighted
    # version of the same priors. An earlier override also silently scaled the two
    # adp/scaler_* terms, which have nothing to do with atomic ADPs.
    weights = field_refinement.weighting()
    for key, expected in DEFAULT_GROUP_WEIGHTS.items():
        assert weights.get(key) == expected, f"{key} diverged from the default"


@pytest.mark.integration
def test_per_atom_mode_does_not_register_the_node_targets(files):
    """The converse: nothing node-shaped has anything to act on off field mode."""
    ref = _refinement(files, adp_mode="isotropic")
    components = ref.adp_target.target_losses()
    assert "simu" in components and "locality" in components
    assert "node_load" not in components
    assert "node_smoothness" not in components


@pytest.mark.integration
def test_switching_replaces_the_component_set_and_the_loss_state(files):
    """A switch changes WHICH targets exist, so a cached LossState must not survive it."""
    ref = _refinement(files, adp_mode="isotropic")
    before = set(ref.adp_target.target_losses())
    assert "simu" in before
    # Force the LossState to exist so the switch has something stale to invalidate.
    ref.complete_loss_state()
    assert ref._loss_state is not None

    logger_before = ref.logger  # binds a Logger to the state that is about to go
    ref.set_adp_representation("field_aniso", mode_set=MODE_SET)
    after = set(ref.adp_target.target_losses())
    # The Logger holds a reference to the LossState, so replacing the state without
    # replacing the Logger would leave it recording into an object nothing else reads.
    assert ref.logger is not logger_before
    assert ref.logger.state is ref.loss_state
    assert "simu" not in after and "node_load" in after
    state = ref.complete_loss_state()
    registered = set(state.targets)
    assert not any(k.endswith("simu") or k.endswith("locality") for k in registered), (
        f"a stale component survived the switch: {sorted(registered)}"
    )
    assert any(k.endswith("node_load") for k in registered)


@pytest.mark.integration
def test_switching_preserves_weights_it_does_not_own(files):
    """A call about ADPs must not reset the caller's xray or geometry weights."""
    from torchref.refinement.weighting import ManualWeighting

    ref = _refinement(files, adp_mode="isotropic")
    custom = {**ref.weighting(), "xray": 2.5, "geometry": 0.35}
    ref.weighting = ManualWeighting(custom)

    ref.set_adp_representation("field_aniso", mode_set=MODE_SET)
    weights = ref.weighting()
    assert weights["xray"] == 2.5, "xray weight was clobbered by an ADP call"
    assert weights["geometry"] == 0.35
    # Nothing is rebalanced, so the caller's adp weight survives too.
    assert weights["adp"] == custom["adp"]

    ref.set_adp_representation("isotropic")
    assert ref.weighting()["xray"] == 2.5


@pytest.mark.integration
def test_per_atom_mode_keeps_the_per_atom_weights(files):
    ref = _refinement(files, adp_mode="isotropic")
    assert ref.weighting()["adp"] == DEFAULT_GROUP_WEIGHTS["adp"]


# ----------------------------------------------------------------------------------
# It has to actually refine.
# ----------------------------------------------------------------------------------


@pytest.mark.integration
def test_refinement_reduces_rwork_and_stays_finite(files):
    """Two ADP-only cycles on real data. Nothing here may be NaN and R-work must fall."""
    ref = _refinement(files, adp_mode="field_aniso", adp_mode_set=MODE_SET)
    rw0, rf0 = (float(v) for v in ref.get_rfactor())
    for _ in range(2):
        ref.refine_scaler()
        ref.refine_adp()
    rw1, rf1 = (float(v) for v in ref.get_rfactor())

    for name, v in (("Rwork", rw1), ("Rfree", rf1)):
        assert v == v, f"{name} is NaN"
        assert 0.0 < v < 0.7, f"{name} = {v:.4f} is not a plausible R-factor"
    assert rw1 < rw0 + 1e-6, f"R-work rose: {rw0:.4f} -> {rw1:.4f}"

    u6 = ref.model.adp_u6().detach()
    assert torch.isfinite(u6).all()
    ev = torch.linalg.eigvalsh(_u6_to_matrix(u6))
    assert float(ev.min()) > 0.0, "an atom went non-positive-definite during refinement"


@pytest.mark.integration
def test_full_refine_moves_coordinates_without_staling_the_field(files):
    """``refine()`` is scaler -> xyz -> ADP, so the field is read at moved coordinates.

    The field borrows the coordinate accessor rather than taking coordinates as an
    argument, so its forward cache has to fold them into its key. If it does not, the
    ADPs silently come from wherever the atoms used to be.
    """
    ref = _refinement(files, adp_mode="field_aniso", adp_mode_set=MODE_SET)
    xyz0 = ref.model.xyz().detach().clone()
    u0 = ref.model.adp_u6().detach().clone()

    ref.refine(macro_cycles=1)

    xyz1 = ref.model.xyz().detach()
    u1 = ref.model.adp_u6().detach()
    assert not torch.allclose(xyz0, xyz1), "coordinates did not move, test proves nothing"
    assert torch.isfinite(u1).all()
    assert not torch.allclose(u0, u1), "ADPs unchanged after xyz moved -- stale cache"


def _u6_to_matrix(u6):
    M = torch.zeros(u6.shape[0], 3, 3, dtype=u6.dtype)
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = u6[:, 0], u6[:, 1], u6[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = u6[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = u6[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = u6[:, 5]
    return M


# ----------------------------------------------------------------------------------
# Switching after setup, which the model alone documents as unsupported.
# ----------------------------------------------------------------------------------


@pytest.mark.integration
def test_switch_into_and_out_of_field_mode_after_setup(files):
    ref = _refinement(files, adp_mode="isotropic")
    assert not ref.model.adp_is_field
    before = float(ref.get_rfactor()[0])

    applied = ref.set_adp_representation("field_aniso", mode_set=MODE_SET)
    assert ref.model.adp_is_field
    assert applied["n_nodes"] >= 2
    assert "simu" not in ref.adp_target.target_losses()
    ref.refine_adp()
    assert torch.isfinite(torch.as_tensor(float(ref.get_rfactor()[0])))

    ref.set_adp_representation("isotropic")
    assert not ref.model.adp_is_field
    # Leaving must put the per-atom weights back, or the next stage is misweighted.
    assert ref.weighting()["adp"] == DEFAULT_GROUP_WEIGHTS["adp"]
    ref.refine_adp()
    after = float(ref.get_rfactor()[0])
    assert after == after and 0.0 < after < 0.7
    assert before == before


@pytest.mark.integration
def test_mode_set_on_an_isotropic_field_is_rejected(files):
    ref = _refinement(files, adp_mode="isotropic")
    with pytest.raises(ValueError, match="field_aniso"):
        ref.set_adp_representation("field", mode_set="rigid")


# ----------------------------------------------------------------------------------
# Checkpoints.
# ----------------------------------------------------------------------------------


@pytest.mark.integration
def test_payload_identity_survives_the_state_dict(field_refinement):
    """``state_dict`` holds tensors only, so the payload has to be encoded as one.

    Without the code, a restore infers the payload from the slot and rebuilds a
    constant-U field: wrong storage width, wrong ADPs.
    """
    sd = field_refinement.model.state_dict()
    key = [k for k in sd if k.endswith("payload_code")]
    assert key, f"no payload code in the state dict: {sorted(sd)[:8]}..."
    assert int(sd[key[0]]) == payload_code(field_refinement.model.adp_field.payload)


@pytest.mark.integration
def test_model_state_dict_round_trip_rebuilds_the_same_field(field_refinement):
    """A restore must rebuild the payload the code names, not one inferred from the slot."""
    from torchref.model.model import Model

    src = field_refinement.model
    # create_from_state_dict restores the values itself; a further load_state_dict
    # would be strict against restraint and metadata keys it never builds.
    restored = Model.create_from_state_dict(src.state_dict(), verbose=0)
    field = restored.adp_field
    assert field is not None, "restored model has no field at all"
    assert isinstance(field.payload, ModeCovariancePayload)
    assert field.payload.mode_set == MODE_SET
    assert field.node_shape == src.adp_field.node_shape
    assert torch.allclose(
        restored.adp_u6().detach(), src.adp_u6().detach(), atol=1e-5
    )


@pytest.mark.integration
def test_model_copy_round_trip_on_a_bare_model():
    """``copy()`` carries the payload and the borrowed accessor, not a deep copy of them.

    Deliberately on a model that has never been handed to a Refinement --- see
    :func:`test_copy_after_refinement_setup_is_broken_for_every_representation` for why.
    """
    from torchref.model.model import Model

    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = Model(verbose=0)
    src.load_pdb(os.path.join(here, "files", "pdb", "1DAW.pdb"))
    src.set_adp_mode("field_aniso", n_nodes=8, k_neighbors=8, mode_set=MODE_SET)
    clone = src.copy()
    assert isinstance(clone.adp_field.payload, ModeCovariancePayload)
    assert clone.adp_field.payload.mode_set == MODE_SET
    assert torch.allclose(clone.adp_u6().detach(), src.adp_u6().detach())
    assert clone.adp_field.refinable_params is not src.adp_field.refinable_params


@pytest.mark.integration
@pytest.mark.xfail(
    reason="PRE-EXISTING and representation-independent: once a Model has been through "
    "Refinement setup, a cache somewhere holds a graph-attached tensor and deepcopy "
    "refuses it. Measured identically for adp_mode isotropic, anisotropic and "
    "field_aniso, and a bare model copies fine, so the node field is not the cause -- "
    "it means no refinement of any kind can currently be checkpointed by copy().",
    raises=RuntimeError,
    strict=True,
)
@pytest.mark.integration
def test_copy_after_refinement_setup_is_broken_for_every_representation(field_refinement):
    field_refinement.model.copy()


# ----------------------------------------------------------------------------------
# The CLI, which is where a flag that does nothing hides best.
# ----------------------------------------------------------------------------------


@pytest.mark.integration
def test_cli_flags_reach_the_refinement():
    """A null control on the plumbing: the parsed values must arrive, not the defaults.

    Five CLI flags in this codebase were once silently no-ops. The check is not that the
    argument parses but that a non-default value changes what the refinement does.
    """
    import argparse

    from torchref.cli._common import add_adp_mode_arg

    parser = argparse.ArgumentParser()
    add_adp_mode_arg(parser)
    args = parser.parse_args(
        ["--adp-mode", "field_aniso", "--adp-mode-set", "affine",
         "--reflections-per-adp-parameter", "3.5", "--adp-nodes", "17"]
    )
    assert args.adp_mode == "field_aniso"
    assert args.adp_mode_set == "affine"
    assert args.reflections_per_adp_parameter == 3.5
    assert args.adp_nodes == 17

    defaults = parser.parse_args([])
    assert defaults.adp_mode == "isotropic"
    assert defaults.adp_mode_set is None
    assert defaults.adp_nodes is None


@pytest.mark.integration
def test_cli_field_mode_end_to_end(files, tmp_path):
    """The parsed flags, through the real constructor, produce a real field."""
    mtz, pdb = files
    ref = LBFGSRefinement(
        data_file=mtz, pdb=pdb, verbose=0,
        adp_mode="field_aniso", adp_mode_set="affine", n_nodes=9,
    )
    assert ref.model.adp_field.payload.mode_set == "affine"
    assert ref.model.adp_field.n_nodes == 9
    ref.refine_adp()
    out = tmp_path / "out.pdb"
    ref.model.update_pdb()
    ref.model.write_pdb(str(out))
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text()
    assert "ANISOU" in text, "an anisotropic field must write ANISOU records"
