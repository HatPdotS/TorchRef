"""Riding-mode water completion respects the model's hydrogen generation setting."""

import pytest
import torch

from torchref import Model, ModelFT


@pytest.fixture(scope="module")
def heavy_model(pdb_dir):
    """Deposited 1DAW loaded without hydrogen generation."""
    return Model(device="cpu", verbose=0, strip_H=True, add_hydrogens=False).load_pdb(
        str(pdb_dir / "1DAW.pdb")
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
def test_riding_completes_only_waters_and_preserves_live_atoms(
    heavy_model, model_class
):
    """Water completion preserves current coordinates, ADPs, selections and links."""
    model = model_class(device="cpu", verbose=0, add_hydrogens=False, strip_H=False)
    model.load(
        lambda: (heavy_model.pdb.copy(), heavy_model.cell.data, heavy_model.spacegroup)
    )
    with torch.no_grad():
        model.xyz.refinable_params.add_(0.25)
    adp = model.adp().detach().clone()
    mask = torch.arange(len(model.pdb)) % 2 == 0
    model.xyz.update_refinable_mask(mask)
    model.adp.update_refinable_mask(mask)
    model.occupancy.freeze_all()
    before = model.xyz().detach().clone()
    links = model.ctx.links
    hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    if isinstance(model, ModelFT):
        model(hkl)
    model.ctx.add_hydrogens = True
    returned = model.set_hydrogen_mode("riding")
    assert returned is model
    is_h = torch.as_tensor(model.pdb.element.str.strip().eq("H").to_numpy())
    is_water = model.pdb.resname.str.strip().eq("HOH")
    assert int(is_h.sum()) == 2 * int(
        heavy_model.pdb.resname.str.strip().eq("HOH").sum()
    )
    assert is_water[is_h.numpy()].all()
    assert torch.allclose(model.xyz()[~is_h], before, atol=1e-5)
    assert torch.allclose(model.adp()[~is_h], adp)
    assert torch.equal(model.xyz.full_refinable_mask[~is_h], mask)
    assert torch.equal(model.adp.refinable_mask[~is_h], mask)
    assert not model.occupancy.get_refinable_atoms().any()
    assert model.ctx.links is links
    assert model.xyz.rotations.shape[0] == int(is_h.sum()) // 2
    assert model.restraints.xyz().shape == model.xyz.shape
    if isinstance(model, ModelFT):
        assert torch.isfinite(model(hkl)).all()
    restored = model_class.create_from_state_dict(model.state_dict(), device="cpu")
    assert torch.allclose(restored.xyz(), model.xyz(), atol=1e-5)


@pytest.mark.unit
def test_partial_water_is_completed_without_moving_existing_hydrogen(heavy_model):
    """A deposited oxygen with one supplied hydrogen receives just its missing partner."""
    water = (
        heavy_model.pdb[heavy_model.pdb.resname.str.strip().eq("HOH")].iloc[:1].copy()
    )
    model = heavy_model._new_model_from_df(water, strip_H=False)
    model.ctx.add_hydrogens = True
    model.set_hydrogen_mode("riding")
    model.update_pdb()
    partial = model._new_model_from_df(model.pdb.iloc[:2].copy(), strip_H=False)
    before = partial.xyz().detach().clone()
    partial.set_hydrogen_mode("riding")
    assert len(partial.pdb) == 2
    assert torch.allclose(partial.xyz(), before, atol=1e-5)
    partial.ctx.add_hydrogens = True
    partial.set_hydrogen_mode("riding")
    assert len(partial.pdb) == 3
    assert torch.allclose(partial.xyz()[:2], before, atol=1e-5)
    with torch.no_grad():
        partial.xyz.rotations.refinable_params.fill_(0.3)
    wrapper = partial.xyz
    coords = partial.xyz().detach().clone()
    partial.set_hydrogen_mode("riding")
    assert partial.xyz is wrapper
    assert torch.equal(partial.xyz(), coords)
    partial.set_hydrogen_mode("free").set_hydrogen_mode("riding")
    assert len(partial.pdb) == 3
    assert torch.allclose(partial.xyz(), coords, atol=1e-5)


@pytest.mark.unit
def test_supplied_frames_are_remapped_when_waters_are_completed(heavy_model):
    """Explicit frames for the original atom table coexist with generated water frames."""
    water = (
        heavy_model.pdb[heavy_model.pdb.resname.str.strip().eq("HOH")].iloc[:2].copy()
    )
    model = heavy_model._new_model_from_df(water, strip_H=False)
    frames = model.hydrogen_frames()
    model.ctx.add_hydrogens = True
    model.set_hydrogen_mode("riding", frames=frames)
    assert len(model.pdb) == 6
    assert model.xyz.n_hydrogens == 4
    assert model.xyz.rotations.shape == (2, 3)


@pytest.mark.unit
def test_water_completion_preserves_adp_field(heavy_model):
    """Adding water hydrogens retains the node parametrization and its parameters."""
    model = heavy_model._new_model_from_df(heavy_model.pdb.copy(), strip_H=False)
    model.set_adp_mode("field", n_nodes=8, k_neighbors=4)
    field = model.adp
    values = field().detach().clone()
    parameters = field.refinable_params
    model.ctx.add_hydrogens = True
    model.set_hydrogen_mode("riding")
    heavy = torch.as_tensor(~model.pdb.element.str.strip().eq("H").to_numpy())
    assert model.adp is field
    assert field.refinable_params is parameters
    assert torch.allclose(field()[heavy], values, atol=1e-5)
    assert field().shape == (len(model.pdb),)


@pytest.mark.integration
@pytest.mark.parametrize("generate", [False, True])
def test_refinement_targets_follow_completed_atom_table(pdb_dir, mtz_dir, generate):
    """Refinement adds water H and refreshes targets only when generation is enabled."""
    from torchref.refinement.base_refinement import Refinement

    refinement = Refinement(
        pdb=str(pdb_dir / "1DAW.pdb"),
        data_file=str(mtz_dir / "1DAW.mtz"),
        device="cpu",
        verbose=0,
        max_res=3.0,
        add_hydrogens=False,
    )
    previous = refinement.adp_target
    n_atoms = len(refinement.model.pdb)
    refinement.model.ctx.add_hydrogens = generate
    refinement.set_hydrogen_mode("riding")
    assert (refinement.adp_target is not previous) == generate
    assert (len(refinement.model.pdb) > n_atoms) == generate
    geometry = refinement.geometry_target()
    assert torch.isfinite(geometry)
    geometry.backward()
    if generate:
        gradient = refinement.model.xyz.rotations.refinable_params.grad
        assert gradient is not None and torch.isfinite(gradient).all()


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
@pytest.mark.parametrize("explicit_frames", [False, True])
def test_disabled_generation_keeps_atom_table(
    heavy_model, model_class, explicit_frames
):
    """Riding mode leaves oxygen-only waters untouched when add_hydrogens is False."""
    model = model_class(device="cpu", verbose=0, add_hydrogens=False)
    model.load(
        lambda: (heavy_model.pdb.copy(), heavy_model.cell.data, heavy_model.spacegroup)
    )
    coordinates = model.xyz().detach().clone()
    table = model.pdb.copy()
    adp, occupancy = model.adp, model.occupancy
    frames = model.hydrogen_frames() if explicit_frames else None
    model.set_hydrogen_mode("riding", frames=frames)
    assert model.pdb.equals(table)
    assert torch.equal(model.xyz(), coordinates)
    assert model.adp is adp and model.occupancy is occupancy
    assert model.xyz.n_hydrogens == 0
    assert model.xyz.rotations.shape == (0, 3)


@pytest.mark.unit
def test_stripping_prevents_water_completion(heavy_model):
    """The stripping preference takes precedence even when generation is enabled."""
    model = heavy_model._new_model_from_df(heavy_model.pdb.copy(), strip_H=True)
    model.ctx.add_hydrogens = True
    n_atoms = len(model.pdb)
    model.set_hydrogen_mode("riding")
    assert len(model.pdb) == n_atoms
    assert model.xyz.n_hydrogens == 0
