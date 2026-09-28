"""The hydrogen policy, and switching a loaded model between riding and atom hydrogens.

``hydrogens`` (keep / add / strip) settles the atom table at load and
``hydrogen_mode`` (atoms / riding) how its hydrogen rows are parametrised; strip with
riding is refused. The switch replaces the coordinate wrapper and nothing else:
coordinates are unchanged, the heavy-atom refinable set carries over, hydrogens leave or
rejoin the refinable set, the restraints are untouched, and the mode survives copies,
selections and state dicts.
"""

import pytest
import torch

from torchref.model.model import Model
from torchref.model.model_ft import ModelFT
from torchref.model.riding_xyz import RidingXYZTensor


@pytest.fixture
def free_model(pdb_dir):
    model = Model(verbose=0, hydrogens="add")
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model


def _n_h(model):
    return int((model.pdb["element"].str.strip() == "H").sum())


@pytest.mark.unit
def test_switch_to_riding_keeps_coordinates_and_drops_hydrogen_parameters(free_model):
    model = free_model
    before = model.xyz().detach().clone()
    n_free = model.parameters_of_types(("xyz",))[0].shape[0]
    model.set_hydrogen_mode("riding")
    assert model.hydrogen_mode == "riding"
    assert isinstance(model.xyz, RidingXYZTensor)
    assert torch.allclose(model.xyz(), before, atol=1e-4)
    assert model.parameters_of_types(("xyz",))[0].shape[0] == n_free - _n_h(model)
    assert model.xyz.n_hydrogens == _n_h(model)


@pytest.mark.unit
def test_switch_back_to_atoms_restores_per_atom_wrapper(free_model):
    model = free_model
    model.set_hydrogen_mode("riding")
    model.set_hydrogen_mode("atoms")
    assert model.hydrogen_mode == "atoms"
    assert not isinstance(model.xyz, RidingXYZTensor)
    assert model.parameters_of_types(("xyz",))[0].shape[0] == len(model.pdb)


@pytest.mark.unit
def test_restraints_survive_the_switch(free_model):
    """The atom table is unchanged, so the restraints are too; they score whatever
    coordinates they are handed, including the riding wrapper's."""
    model = free_model
    restraints = model.restraints
    model.set_hydrogen_mode("riding")
    assert model.restraints is restraints
    deviations, _ = restraints.bond_deviations(model.xyz())
    deviations.sum().backward()
    assert model.xyz.refinable_params.grad is not None


@pytest.mark.unit
def test_frozen_heavy_atoms_stay_frozen_across_the_switch(free_model):
    model = free_model
    mask = torch.zeros(len(model.pdb), dtype=torch.bool, device=model.device)
    mask[: len(model.pdb) // 2] = True
    model.xyz.update_refinable_mask(mask)
    model.set_hydrogen_mode("riding")
    full = model.xyz.full_refinable_mask
    heavy = ~torch.as_tensor((model.pdb["element"].str.strip() == "H").values, device=full.device)
    assert torch.equal(full[heavy], mask[heavy])


@pytest.mark.unit
def test_mode_survives_copy_select_and_shake(free_model):
    model = free_model
    model.set_hydrogen_mode("riding")
    dup = model.copy()
    assert dup.hydrogen_mode == "riding" and isinstance(dup.xyz, RidingXYZTensor)
    assert torch.equal(dup.xyz(), model.xyz())
    sub = model.select("resseq 10:40")
    assert isinstance(sub.xyz, RidingXYZTensor)
    assert sub.xyz.shape[0] == len(sub.pdb)
    model.shake_coords(0.05)
    assert isinstance(model.xyz, RidingXYZTensor)
    assert model.xyz.shape[0] == len(model.pdb)


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
def test_riding_mode_round_trips_through_state_dict(pdb_dir, model_class):
    kwargs = {"max_res": 3.0} if model_class is ModelFT else {}
    model = model_class(verbose=0, hydrogens="add", **kwargs)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    model.set_hydrogen_mode("riding")
    with torch.no_grad():
        model.xyz.refinable_params[0].add_(0.25)
    state = model.state_dict()
    restored = model_class.create_from_state_dict(state, device=model.device)
    assert restored.hydrogen_mode == "riding"
    assert isinstance(restored.xyz, RidingXYZTensor)
    assert torch.allclose(restored.xyz(), model.xyz(), atol=1e-5)
    assert restored.xyz.get_refinable_count() == model.xyz.get_refinable_count()


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["none", "free"])
def test_unknown_modes_are_refused(free_model, mode):
    with pytest.raises(ValueError, match="hydrogen_mode must be one of"):
        free_model.set_hydrogen_mode(mode)


@pytest.mark.unit
@pytest.mark.parametrize("hydrogens", ["keep", "add", "strip"])
@pytest.mark.parametrize("hydrogen_mode", ["atoms", "riding"])
def test_policy_matrix(pdb_dir, hydrogens, hydrogen_mode):
    """Each valid pair loads with the atom set and wrapper it names; strip+riding raises."""
    if hydrogens == "strip" and hydrogen_mode == "riding":
        with pytest.raises(ValueError, match="Nothing is left to ride"):
            Model(verbose=0, hydrogens=hydrogens, hydrogen_mode=hydrogen_mode)
        return

    model = Model(verbose=0, hydrogens=hydrogens, hydrogen_mode=hydrogen_mode)
    model.load_pdb(str(pdb_dir / "1AK5_with_H.pdb"))
    deposited = Model(verbose=0).load_pdb(str(pdb_dir / "1AK5_with_H.pdb"))

    n_h = _n_h(model)
    if hydrogens == "strip":
        assert n_h == 0
    elif hydrogens == "keep":
        assert n_h == _n_h(deposited)
    else:
        assert n_h > _n_h(deposited)
    assert isinstance(model.xyz, RidingXYZTensor) is (hydrogen_mode == "riding")
    assert model.xyz().shape[0] == len(model.pdb)


@pytest.mark.unit
def test_riding_is_refused_on_a_stripped_model(pdb_dir):
    model = Model(verbose=0, hydrogens="strip").load_pdb(str(pdb_dir / "1DAW.pdb"))
    with pytest.raises(ValueError, match="Nothing is left to ride"):
        model.set_hydrogen_mode("riding")
