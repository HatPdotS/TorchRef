"""Switching a loaded model between riding and free hydrogens.

The switch replaces the coordinate wrapper and nothing else: coordinates are
unchanged, the heavy-atom refinable set carries over, hydrogens leave or rejoin the
refinable set, the restraints keep reading the live wrapper, and the mode survives
copies, selections and state dicts.
"""

import pytest
import torch

from torchref.model.model import Model
from torchref.model.model_ft import ModelFT
from torchref.model.riding_xyz import RidingXYZTensor


@pytest.fixture
def free_model(pdb_dir):
    model = Model(verbose=0, add_hydrogens=True)
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
def test_switch_back_to_free_restores_per_atom_wrapper(free_model):
    model = free_model
    model.set_hydrogen_mode("riding")
    model.set_hydrogen_mode("free")
    assert model.hydrogen_mode == "free"
    assert not isinstance(model.xyz, RidingXYZTensor)
    assert model.parameters_of_types(("xyz",))[0].shape[0] == len(model.pdb)


@pytest.mark.unit
def test_restraints_read_the_installed_wrapper(free_model):
    model = free_model
    restraints = model.restraints
    model.set_hydrogen_mode("riding")
    assert restraints._xyz_fn is model.xyz
    with torch.no_grad():
        model.xyz.refinable_params.add_(0.1)
    assert torch.equal(restraints.xyz(), model.xyz())


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
    model = model_class(verbose=0, add_hydrogens=True, **kwargs)
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
def test_none_mode_is_refused(free_model):
    with pytest.raises(ValueError):
        free_model.set_hydrogen_mode("none")
