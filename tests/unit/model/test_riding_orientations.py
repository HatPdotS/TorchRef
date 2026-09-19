"""Refinable hydrogen orientations preserve geometry and expose force gradients."""

import numpy as np
import pytest
import torch
from torchref import Model, ModelFT
from torchref.base.coordinates.local_frame import rotate_vectors
from torchref.config import get_float_dtype
from torchref.model.riding_xyz import RidingXYZTensor
from torchref.topology.hydrogens import _place_group, _template


@pytest.fixture(scope="module")
def oriented_model(pdb_dir):
    """A deposited protein with explicitly generated protein and water hydrogens."""
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = Model(verbose=0, device="cpu", add_hydrogens=True)
        model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    model.set_hydrogen_mode("riding")
    return model


@pytest.fixture
def two_groups(oriented_model):
    """One methyl and one water, retaining the methyl's heavy frame atoms."""
    full = oriented_model.xyz
    methyl = next(
        int(g)
        for g in torch.unique(full.torsion_group).tolist()
        if g >= 0
        and int((full.torsion_group == g).sum()) == 3
        and oriented_model.pdb.iloc[
            int(full.parent_row[full.torsion_group == g][0])
        ].element.strip()
        == "C"
    )
    water = int(full.rotation_group[full.rotation_group >= 0][0])
    selected = (full.torsion_group == methyl) | (full.rotation_group == water)
    keep = torch.zeros(full.shape[0], dtype=torch.bool)
    for rows in (full.h_row, full.parent_row, full.n1_row, full.n2_row):
        chosen = rows[selected]
        keep[chosen[chosen >= 0]] = True
    return full.select_rows(keep)


@pytest.mark.unit
def test_orientation_groups_preserve_initial_positions(oriented_model):
    """Zero rotations reproduce the deposited/generated table without atom changes."""
    model = oriented_model
    xyz = torch.as_tensor(model.pdb[["x", "y", "z"]].values, dtype=model.dtype_float)
    assert torch.allclose(model.xyz(), xyz, atol=1e-4)
    assert model.xyz.torsions.shape[0] > 0
    waters = model.pdb[model.pdb.resname.str.strip() == "HOH"]
    assert len(waters) > 0
    assert model.xyz.rotations.shape == (
        int((waters.element.str.strip() == "O").sum()),
        3,
    )


@pytest.mark.unit
def test_water_initialization_is_seeded_and_preserves_existing_direction(
    oriented_model,
):
    """Water references are reproducible and completing an O–H pair preserves its angle."""
    model = oriented_model
    xyz = model.xyz().detach().numpy()
    first = int(model.xyz._rotation_first[0])
    parent = int(model.xyz.parent_row[first])
    h_rows = model.xyz.h_row[model.xyz.rotation_group == 0].numpy()
    names = model.pdb.name.str.strip().to_numpy()
    template = _template(model.restraints.cif_dict, "HOH")
    h_names = list(names[h_rows])
    lengths = np.linalg.norm(xyz[h_rows] - xyz[parent], axis=1)

    def place(seed, present):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return _place_group(
                template,
                names[parent],
                xyz[parent],
                np.empty((0, 3)),
                0,
                h_names,
                lengths,
                present,
                xyz,
                [],
            )

    first_reference = place(1, {names[parent]: parent})
    assert np.array_equal(first_reference, place(1, {names[parent]: parent}))
    assert not np.allclose(first_reference, place(2, {names[parent]: parent}))
    completed = place(3, {names[parent]: parent, h_names[0]: int(h_rows[0])})
    assert np.allclose(completed[0], xyz[h_rows[0]], atol=1e-5)
    assert np.allclose(
        np.linalg.norm(completed[0] - completed[1]),
        np.linalg.norm(first_reference[0] - first_reference[1]),
        atol=1e-5,
    )


@pytest.mark.unit
def test_rotation_preserves_water_and_methyl_geometry(two_groups):
    """Shared rotations preserve internal distances and methyl bond angles."""
    w = two_groups
    before = w().detach()
    with torch.no_grad():
        w.torsions.refinable_params.fill_(0.8)
        w.rotations.refinable_params.copy_(
            w.rotations.refinable_params.new_tensor([[0.4, -0.2, 0.7]])
        )
    after = w()
    assert torch.equal(before[w.base_row], after[w.base_row])
    assert not torch.allclose(before[w.h_row], after[w.h_row])
    for group in (w.torsion_group, w.rotation_group):
        rows = w.h_row[group >= 0]
        parent = w.parent_row[group >= 0][0:1]
        rows = torch.cat((parent, rows))
        assert torch.allclose(
            torch.cdist(before[rows], before[rows]),
            torch.cdist(after[rows], after[rows]),
            atol=1e-5,
        )
    rows = w.h_row[w.torsion_group >= 0]
    neighbour = w.n1_row[w.torsion_group >= 0]
    assert torch.allclose(
        (before[rows] - before[neighbour]).norm(dim=-1),
        (after[rows] - after[neighbour]).norm(dim=-1),
        atol=1e-5,
    )


@pytest.mark.unit
@pytest.mark.parametrize("angle", [0.0, 0.4])
def test_orientation_gradients_match_finite_differences(two_groups, double_cpu, angle):
    """The deposited methyl/water coordinate map has correct orientation gradients."""
    w = two_groups.to(dtype=get_float_dtype())
    base = w._storage_values().detach().requires_grad_()
    torsion = w.torsions().detach().fill_(angle).requires_grad_()
    rotation = w.rotations().detach().fill_(angle).requires_grad_()
    assert torch.autograd.gradcheck(w.evaluate, (base, torsion, rotation), atol=2e-6)
    vectors = w.rigid_offset[w.rotation_group >= 0].detach()
    r = vectors.new_full(vectors.shape, angle, requires_grad=True)
    assert torch.autograd.gradgradcheck(lambda x: rotate_vectors(vectors, x), (r,))


@pytest.mark.unit
def test_xyz_optimizer_receives_and_updates_orientations(two_groups):
    """An orientation-only selection delivers both leaves to the xyz optimizer."""
    w = two_groups
    w.fix_all()
    w.refine(w.h_row)
    model = Model(device="cpu", verbose=0)
    model.xyz = w
    leaves = model.parameters_of_types(("xyz",))
    assert {id(p) for p in leaves} == {id(p) for p in w.optimization_parameters()}
    target = w.evaluate(
        w._storage_values(), w.torsions().detach() + 0.2, w.rotations().detach() + 0.2
    ).detach()
    optimizer = torch.optim.SGD(leaves, lr=0.05)
    before = w._storage_values().detach().clone()
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = (w() - target).square().sum()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] / 2
    assert torch.equal(w._storage_values(), before)
    assert w.torsions.refinable_params.abs().sum() > 0
    assert w.rotations.refinable_params.abs().sum() > 0


@pytest.mark.unit
def test_orientation_mutation_invalidates_forward_cache(two_groups):
    """Both orientation leaves participate in the coordinate cache fingerprint."""
    w = two_groups
    before = w()
    with torch.no_grad():
        w.torsions.refinable_params.add_(0.1)
    changed = w()
    assert changed is not before
    with torch.no_grad():
        w.rotations.refinable_params.add_(0.1)
    assert w() is not changed


@pytest.mark.unit
def test_copy_selection_and_checkpoint_preserve_orientations(two_groups):
    """Nonzero rotations survive copy, subset, and empty-shell state restoration."""
    w = two_groups
    with torch.no_grad():
        w.torsions.refinable_params.fill_(0.3)
        w.rotations.refinable_params.fill_(-0.2)
    expected = w().detach()
    copied = w.copy()
    assert torch.equal(copied(), expected)
    assert torch.equal(copied.rotations(), w.rotations())
    keep = torch.ones(w.shape[0], dtype=torch.bool)
    keep[w.h_row[w.torsion_group >= 0]] = False
    subset = w.select_rows(keep)
    assert torch.allclose(subset(), expected[keep], atol=1e-5)
    restored = RidingXYZTensor(device="cpu")
    restored.load_state_dict(w.state_dict())
    assert torch.equal(restored(), expected)
    with torch.no_grad():
        copied.rotations.refinable_params.add_(0.1)
    assert torch.equal(w(), expected)


@pytest.mark.unit
def test_checkpoint_without_orientation_metadata(two_groups):
    """Coordinate-only checkpoints load with fixed hydrogen orientations."""
    state = {
        name: value
        for name, value in two_groups.state_dict().items()
        if name not in ("torsion_group", "rotation_group", "virtual_reference")
        and not name.startswith(("torsions.", "rotations."))
    }
    restored = RidingXYZTensor(device="cpu")
    restored.load_state_dict(state)
    assert torch.allclose(restored(), two_groups(), atol=1e-5)
    assert restored.torsions.shape == (0,)
    assert restored.rotations.shape == (0, 3)


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
def test_model_checkpoint_restores_rotated_groups(oriented_model, model_class):
    """Model checkpoints restore orientation values, masks, and frame references."""
    source = oriented_model.copy()
    with torch.no_grad():
        source.xyz.torsions.refinable_params.fill_(0.3)
        source.xyz.rotations.refinable_params.fill_(0.2)
    source.xyz.rotations.fix(torch.arange(0, source.xyz.rotations.shape[0], 2))
    state = source.state_dict()
    restored = model_class.create_from_state_dict(state, device="cpu")
    assert torch.allclose(restored.xyz(), source.xyz(), atol=1e-5)
    assert torch.equal(
        restored.xyz.rotations.refinable_mask, source.xyz.rotations.refinable_mask
    )


@pytest.mark.unit
def test_torsion_without_second_reference_still_rotates(two_groups):
    """A fixed reference direction completes a methyl frame with one heavy bond."""
    frames = two_groups.hydrogen_frames()
    frames.n2_row[frames.torsion_group >= 0] = -1
    frames.frame_valid[frames.torsion_group >= 0] = False
    w = RidingXYZTensor(two_groups().detach(), frames)
    initial = w().detach()
    with torch.no_grad():
        w.torsions.refinable_params.fill_(0.6)
    rows = w.h_row[w.torsion_group >= 0]
    assert not torch.allclose(w()[rows], initial[rows])
    assert torch.allclose(
        torch.cdist(w()[rows], w()[rows]),
        torch.cdist(initial[rows], initial[rows]),
        atol=1e-5,
    )


@pytest.mark.unit
def test_orientation_forward_runs_on_requested_device(two_groups, any_device):
    """Coordinate and orientation gradients stay on the requested backend."""
    w = two_groups.to(any_device)
    output = w()
    output.square().sum().backward()
    assert output.device == any_device
    for p in w.optimization_parameters():
        assert p.device == any_device
        assert p.grad is not None and torch.isfinite(p.grad).all()
