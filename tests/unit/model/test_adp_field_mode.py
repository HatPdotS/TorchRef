"""``set_adp_mode("field")``: a third ADP representation on the existing switch.

The switch is a *conversion*, not a freeze, so entering field mode has to fit the node
values to the B it replaces and leaving has to materialise them back per atom. The
inert-until-selected property matters too: adding the mode must not perturb a model
that never asks for it.
"""

import math

import pytest
import torch

from torchref.model.disorder_field import DisorderFieldTensor
from torchref.model.model import Model
from torchref.model.parameter_wrappers import PositiveMixedTensor


@pytest.fixture(scope="module")
def pdb_path(pdb_dir):
    return str(pdb_dir / "3GR5.pdb")


def _model(pdb_path):
    model = Model(verbose=0)
    model.load_pdb(pdb_path)
    return model


@pytest.mark.unit
def test_isotropic_mode_is_untouched(pdb_path):
    """The default representation is unchanged by the new branch existing."""
    model = _model(pdb_path)
    model.set_adp_mode("isotropic")
    assert isinstance(model.adp, PositiveMixedTensor)
    assert not model.adp_is_field
    assert model.adp().shape == (len(model.pdb),)


@pytest.mark.unit
def test_unknown_mode_still_raises(pdb_path):
    model = _model(pdb_path)
    with pytest.raises(ValueError, match="field"):
        model.set_adp_mode("nonsense")


@pytest.mark.unit
def test_entering_field_mode_replaces_the_wrapper(pdb_path):
    """``adp`` becomes a field whose parameter count is set by nodes, not atoms."""
    model = _model(pdb_path)
    n_atoms = len(model.pdb)
    model.set_adp_mode("field", n_nodes=40, k_neighbors=8)

    assert model.adp_is_field
    assert isinstance(model.adp, DisorderFieldTensor)
    assert model.adp.n_nodes == 40
    assert model.adp().shape == (n_atoms,)
    # The whole point: far fewer refinable numbers than atoms.
    assert int(model.adp.refinable_params.numel()) < n_atoms


@pytest.mark.unit
def test_field_tracks_the_b_it_replaced(pdb_path):
    """The fit is against the deposited B, so it must resemble it, not restart from flat."""
    model = _model(pdb_path)
    before = model.adp().detach().clone()
    model.set_adp_mode("field", n_nodes=64, k_neighbors=8)
    after = model.adp().detach()

    spread = (before - before.mean()).pow(2).mean().sqrt()
    resid = (after - before).pow(2).mean().sqrt()
    assert resid < 0.6 * spread, f"rmse {resid:.2f} vs B spread {spread:.2f}"
    assert bool((after > 0).all())


@pytest.mark.unit
def test_more_nodes_track_more_closely(pdb_path):
    """Node count is the accuracy dial, end to end through the switch."""
    errors = []
    for n in (4, 32, 200):
        model = _model(pdb_path)
        before = model.adp().detach().clone()
        model.set_adp_mode("field", n_nodes=n, k_neighbors=8)
        errors.append(float((model.adp().detach() - before).pow(2).mean().sqrt()))
    assert errors[0] > errors[1] > errors[2], errors


@pytest.mark.unit
def test_leaving_field_mode_materialises_per_atom(pdb_path):
    """Round trip out of field mode gives back a per-atom wrapper holding its values."""
    model = _model(pdb_path)
    model.set_adp_mode("field", n_nodes=64, k_neighbors=8)
    field_values = model.adp().detach().clone()

    model.set_adp_mode("isotropic")

    assert not model.adp_is_field
    assert isinstance(model.adp, PositiveMixedTensor)
    assert torch.allclose(model.adp().detach(), field_values, atol=1e-5)


@pytest.mark.unit
def test_field_mode_collapses_anisotropic_atoms_first(pdb_path):
    """Entering from anisotropic goes through B_eq rather than dropping the U."""
    model = _model(pdb_path)
    model.set_adp_mode("anisotropic")
    assert bool(model.aniso_flag.any())

    with torch.no_grad():
        U = model.u().detach()
        beq = (8.0 * math.pi**2 / 3.0) * (U[:, 0] + U[:, 1] + U[:, 2])

    model.set_adp_mode("field", n_nodes=200, k_neighbors=8)

    assert not bool(model.aniso_flag.any()), "field mode is isotropic in this stage"
    got = model.adp().detach()
    finite = torch.isfinite(beq)
    spread = (beq[finite] - beq[finite].mean()).pow(2).mean().sqrt()
    resid = (got[finite] - beq[finite]).pow(2).mean().sqrt()
    assert resid < spread, "field ignored the equivalent isotropic B"


@pytest.mark.unit
def test_sf_indices_and_flags_stay_consistent(pdb_path):
    """Everything keyed off the iso/aniso split is refreshed, not left stale."""
    model = _model(pdb_path)
    model.set_adp_mode("field", n_nodes=32, k_neighbors=8)

    assert not bool(model.aniso_flag.any())
    assert not bool(model.pdb["anisou_flag"].to_numpy().any())
    assert int(model._iso_indices.numel()) == len(model.pdb)
    assert bool(model._aniso_is_empty)


@pytest.mark.unit
def test_get_iso_and_adp_u6_work_unmodified(pdb_path):
    """The two consumers that read ``adp()`` need no knowledge of the field."""
    model = _model(pdb_path)
    model.set_adp_mode("field", n_nodes=32, k_neighbors=8)

    xyz, adp, occ = model.get_iso()
    assert xyz.shape[0] == adp.shape[0] == occ.shape[0] == len(model.pdb)

    u6 = model.adp_u6()
    assert u6.shape == (len(model.pdb), 6)
    expected = (adp / (8.0 * math.pi**2)).detach()
    assert torch.allclose(u6[:, 0].detach(), expected, atol=1e-6)
    assert torch.allclose(u6[:, 3:].detach(), torch.zeros_like(u6[:, 3:]), atol=1e-12)


@pytest.mark.unit
def test_gradient_flows_to_the_nodes_through_adp_u6(pdb_path):
    """The path an ADP restraint would take reaches the node parameters."""
    model = _model(pdb_path)
    model.set_adp_mode("field", n_nodes=32, k_neighbors=8)
    model.adp_u6().sum().backward()
    grad = model.adp.refinable_params.grad
    assert grad is not None and bool(grad.abs().sum() > 0)


@pytest.mark.unit
def test_refine_adp_sees_the_node_parameters(pdb_path):
    """``parameters_of_types`` needs no change: the field is still the ``adp`` type."""
    model = _model(pdb_path)
    model.set_adp_mode("field", n_nodes=32, k_neighbors=8)
    params = model.parameters_of_types(["adp"])
    assert len(params) == 1
    assert params[0] is model.adp.refinable_params
    # [log B, log sigma, dx, dy, dz] -- positions are refinable by default.
    assert params[0].shape == (32, 5)
    assert model.adp.refines_positions


@pytest.mark.unit
def test_model_copy_repoints_the_accessor(pdb_path):
    """A copied field must read the COPY's coordinates, not the original's.

    ``copy()`` carries the borrowed accessor by reference, so without the re-point in
    ``Model.copy`` the two models share coordinates and moving one changes the other's
    ADPs.
    """
    model = _model(pdb_path)
    model.set_adp_mode("field", n_nodes=32, k_neighbors=8)
    clone = model.copy()

    assert clone.adp._xyz_fn.module is clone.xyz
    assert clone.adp._xyz_fn.module is not model.xyz
    before = model.adp().detach().clone()

    with torch.no_grad():
        clone.xyz.refinable_params[: len(model.pdb) // 2, 0] += 5.0
    model.adp.reset_forward_cache()

    assert torch.allclose(model.adp().detach(), before, atol=1e-9), (
        "moving the copy changed the original's ADPs -- accessor was not re-pointed"
    )


@pytest.mark.unit
def test_state_dict_round_trip_in_field_mode(pdb_path):
    """``create_from_state_dict`` rebuilds a field when the saved ``adp`` was one."""
    model = _model(pdb_path)
    model.set_adp_mode("field", n_nodes=24, k_neighbors=6)
    expected = model.adp().detach().clone()

    sd = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in model.state_dict().items()
    }
    restored = Model.create_from_state_dict(sd, verbose=0)

    assert restored.adp_is_field
    assert restored.adp.n_nodes == 24
    assert torch.allclose(restored.adp().detach(), expected, atol=1e-6)
