"""Water hydrogens are completed at load, and only when generation is asked for.

``hydrogens="add"`` gives every HOH its two hydrogens when the table is settled, so a
riding model starts with a water rotation per water. Switching modes afterwards never
changes the atom table: with ``hydrogens="keep"`` an oxygen-only water stays oxygen-only.
"""

import pytest
import torch

from torchref import Model, ModelFT


def _n_waters(model):
    pdb = model.pdb
    return int((pdb.resname.str.strip().eq("HOH") & pdb.element.str.strip().eq("O")).sum())


@pytest.fixture(scope="module")
def heavy_model(pdb_dir):
    """Deposited 1DAW with every hydrogen stripped."""
    return Model(device="cpu", verbose=0, hydrogens="strip").load_pdb(
        str(pdb_dir / "1DAW.pdb")
    )


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
def test_add_with_riding_completes_every_water(heavy_model, model_class):
    """Each water gets two riding hydrogens and one rotation, and the model round-trips."""
    table = heavy_model.pdb[heavy_model.pdb.resname.str.strip().eq("HOH")].copy()
    model = model_class(device="cpu", verbose=0, hydrogens="add", hydrogen_mode="riding")
    model.load(lambda: (table, heavy_model.cell.data, heavy_model.spacegroup))

    is_h = model.pdb.element.str.strip().eq("H").to_numpy()
    assert int(is_h.sum()) == 2 * len(table)
    assert model.xyz.n_hydrogens == int(is_h.sum())
    assert model.xyz.rotations.shape == (len(table), 3)
    assert model.restraints.topology.n_atoms == model.xyz.shape[0]
    heavy_rows = torch.as_tensor(~is_h)
    expected = torch.tensor(table[["x", "y", "z"]].values, dtype=model.xyz().dtype)
    assert torch.allclose(model.xyz()[heavy_rows], expected, atol=1e-5)
    if isinstance(model, ModelFT):
        hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
        assert torch.isfinite(model(hkl)).all()

    restored = model_class.create_from_state_dict(model.state_dict(), device="cpu")
    assert restored.hydrogen_mode == "riding"
    assert torch.allclose(restored.xyz(), model.xyz(), atol=1e-5)


@pytest.mark.unit
def test_partial_water_is_completed_without_moving_its_hydrogen(heavy_model):
    """A water that arrives with one hydrogen receives just its missing partner."""
    table = heavy_model.to_dataframe()
    water = table[table.resname.eq("HOH")].iloc[:1].copy()
    complete = heavy_model._derive(water, hydrogens="add")
    partial_table = complete.to_dataframe().iloc[:2].copy()

    kept = heavy_model._derive(partial_table, hydrogens="keep")
    assert len(kept.pdb) == 2
    before = kept.xyz().detach().clone()

    topped = heavy_model._derive(partial_table, hydrogens="add", hydrogen_mode="riding")
    assert len(topped.pdb) == 3
    assert torch.allclose(topped.xyz()[:2], before, atol=1e-5)

    with torch.no_grad():
        topped.xyz.rotations.refinable_params.fill_(0.3)
    wrapper = topped.xyz
    coords = topped.xyz().detach().clone()
    topped.set_hydrogen_mode("riding")
    assert topped.xyz is wrapper
    topped.set_hydrogen_mode("atoms").set_hydrogen_mode("riding")
    assert len(topped.pdb) == 3
    assert torch.allclose(topped.xyz(), coords, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
@pytest.mark.parametrize("explicit_frames", [False, True])
def test_switching_to_riding_never_changes_the_atom_table(
    heavy_model, model_class, explicit_frames
):
    """With ``hydrogens="keep"``, oxygen-only waters stay oxygen-only under riding."""
    model = model_class(device="cpu", verbose=0)
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


@pytest.mark.integration
@pytest.mark.parametrize("hydrogens", ["keep", "add"])
def test_refinement_targets_see_the_riding_waters(pdb_dir, mtz_dir, hydrogens):
    """Water rotations reach the geometry gradient when the waters were completed."""
    from torchref.refinement.base_refinement import Refinement

    refinement = Refinement(
        pdb=str(pdb_dir / "1DAW.pdb"),
        data_file=str(mtz_dir / "1DAW.mtz"),
        device="cpu",
        verbose=0,
        max_res=3.0,
        hydrogens=hydrogens,
    )
    n_atoms = len(refinement.model.pdb)
    adp_target = refinement.adp_target
    refinement.set_hydrogen_mode("riding")
    assert len(refinement.model.pdb) == n_atoms
    assert refinement.adp_target is adp_target

    geometry = refinement.geometry_target()
    assert torch.isfinite(geometry)
    geometry.backward()
    rotations = refinement.model.xyz.rotations
    assert rotations.shape[0] == (
        _n_waters(refinement.model) if hydrogens == "add" else 0
    )
    if hydrogens == "add":
        gradient = rotations.refinable_params.grad
        assert gradient is not None and torch.isfinite(gradient).all()
