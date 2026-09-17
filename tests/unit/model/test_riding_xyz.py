"""The riding coordinate wrapper: full-space face, heavy-atom storage.

Pinned here: the placed hydrogens are reproduced from the stored rows, a force on a
hydrogen reaches only the atoms that carry it, every public method speaks full atom
space while the single refinable leaf holds heavy rows only, and the wrapper survives
the conversions the model needs (to and from a plain per-atom wrapper, subsets,
copies, state dicts).
"""

import numpy as np
import pytest
import torch

from torchref.model.model import Model
from torchref.model.parameter_wrappers import MixedTensor
from torchref.model.riding_xyz import RidingXYZTensor
from torchref.topology.hydrogens import HydrogenFrames, hydrogen_frames


@pytest.fixture(scope="module")
def hydrogenated(pdb_dir):
    """1DAW with generated hydrogens, its frames, and its full coordinate table."""
    model = Model(verbose=0, add_hydrogens=True)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    frames = hydrogen_frames(model.restraints.topology)
    return model, frames, model.xyz().detach()


def _tolerance(dtype):
    return 1e-4 if dtype == torch.float32 else 1e-9


@pytest.mark.unit
def test_reconstructs_placed_hydrogens(hydrogenated):
    model, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    out = riding()
    assert out.shape == xyz.shape
    assert riding.shape == tuple(xyz.shape)
    assert float((out - xyz).norm(dim=1).max()) < _tolerance(xyz.dtype)
    assert riding.n_hydrogens == frames.n_hydrogens
    assert riding.refinable_params.shape[0] == xyz.shape[0] - frames.n_hydrogens


@pytest.mark.unit
def test_coordinate_leaf_holds_heavy_rows_with_separate_orientations(hydrogenated):
    """Heavy positions and shared orientations have separate optimizer leaves."""
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    leaves = list(riding.parameters())
    assert len(leaves) == 3
    assert leaves[0].shape == (xyz.shape[0] - frames.n_hydrogens, 3)
    assert leaves[1] is riding.torsions.refinable_params
    assert leaves[2] is riding.rotations.refinable_params
    assert riding.full_refinable_mask.sum() == leaves[0].shape[0]
    assert not riding.full_refinable_mask[riding.h_row].any()


@pytest.mark.unit
def test_gradient_through_hydrogens_lands_on_frame_atoms(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    out = riding()
    k = 5
    h_row = int(riding.h_row[k])
    (grad,) = torch.autograd.grad(out[h_row].sum(), riding.refinable_params)
    touched = set(torch.nonzero(grad.abs().sum(1) > 0).flatten().tolist())
    expected = {
        int(riding._parent_bidx[k]),
        int(riding._n1_bidx[k]),
        int(riding._n2_bidx[k]),
    }
    assert touched == expected


@pytest.mark.unit
def test_evaluate_matches_finite_differences():
    """The pure map from stored to full rows has exact gradients (float64)."""
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.5, 0.1], [-0.6, 0.8, 0.2], [-0.6, -0.8, 0.0]],
        dtype=torch.float64,
    )
    frames = HydrogenFrames(
        h_row=np.array([3, 4]),
        parent_row=np.array([0, 0]),
        n1_row=np.array([1, 1]),
        n2_row=np.array([2, 2]),
        frame_valid=np.array([True, True]),
    )
    riding = RidingXYZTensor(xyz, frames)
    base = riding.refinable_params.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(riding.evaluate, (base,), eps=1e-6, atol=1e-6)


@pytest.mark.unit
def test_masks_are_full_space_and_ignore_hydrogen_rows(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    n = xyz.shape[0]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:100] = True
    riding.update_refinable_mask(mask)
    heavy_first = int((~torch.isin(torch.arange(100), riding.h_row.cpu())).sum())
    assert riding.get_refinable_count() == heavy_first
    assert riding.full_refinable_mask.sum() == heavy_first
    riding.fix_all()
    assert riding.get_refinable_count() == 0
    assert riding.refinable_params.numel() == 0
    riding.refine_all()
    assert riding.get_refinable_count() == n - frames.n_hydrogens
    riding.fix(torch.arange(10))
    assert riding.full_refinable_mask[:10].sum() == 0


@pytest.mark.unit
def test_frozen_parent_keeps_its_hydrogens_still(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    riding.fix_all()
    before = riding().detach().clone()
    with torch.no_grad():
        riding.refinable_params.add_(1.0)  # empty leaf: nothing moves
    assert torch.equal(riding(), before)


@pytest.mark.unit
def test_rigid_motion_preserves_local_offsets(hydrogenated):
    """Writing a rotated table keeps every hydrogen riding at the same offset."""
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    offsets = riding.local_offset.clone()
    angle = torch.tensor(0.4, dtype=xyz.dtype)
    rot = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=xyz.dtype, device=xyz.device,
    )
    moved = xyz @ rot.T + torch.tensor([3.0, -1.0, 2.0], dtype=xyz.dtype, device=xyz.device)
    riding[:] = moved
    assert float((riding() - moved).norm(dim=1).max()) < 10 * _tolerance(xyz.dtype)
    assert torch.allclose(riding.local_offset, offsets, atol=10 * _tolerance(xyz.dtype))


@pytest.mark.unit
def test_assigning_a_hydrogen_row_becomes_a_new_offset(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    h = int(riding.h_row[0])
    target = xyz[h] + torch.tensor([0.3, -0.2, 0.1], dtype=xyz.dtype, device=xyz.device)
    riding[h] = target
    assert float((riding()[h] - target).norm()) < 10 * _tolerance(xyz.dtype)
    # Heavy rows untouched.
    assert float((riding()[riding.base_row] - xyz[riding.base_row]).norm(dim=1).max()) < 10 * _tolerance(xyz.dtype)


@pytest.mark.unit
def test_round_trip_with_plain_wrapper(hydrogenated):
    _, frames, xyz = hydrogenated
    plain = MixedTensor(xyz.clone(), name="xyz")
    riding = RidingXYZTensor.from_mixed_tensor(plain, frames)
    back = riding.to_mixed_tensor()
    assert isinstance(back, MixedTensor) and not isinstance(back, RidingXYZTensor)
    assert float((back() - xyz).norm(dim=1).max()) < _tolerance(xyz.dtype)
    assert back.refinable_mask.all()


@pytest.mark.unit
def test_select_rows_frees_a_hydrogen_whose_parent_is_cut(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    keep = torch.ones(xyz.shape[0], dtype=torch.bool)
    parent = int(riding.parent_row[0])
    keep[parent] = False
    sub = riding.select_rows(keep)
    assert sub.shape[0] == xyz.shape[0] - 1
    assert sub.n_hydrogens == frames.n_hydrogens - int((riding.parent_row == parent).sum())
    expected = xyz[keep.to(xyz.device)]
    assert float((sub() - expected).norm(dim=1).max()) < _tolerance(xyz.dtype)


@pytest.mark.unit
def test_copy_is_independent_and_exact(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    dup = riding.copy()
    assert torch.equal(dup(), riding())
    assert torch.equal(dup.local_offset, riding.local_offset)
    with torch.no_grad():
        dup.refinable_params.add_(1.0)
    assert not torch.equal(dup(), riding())


@pytest.mark.unit
def test_state_dict_round_trip(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    state = riding.state_dict()
    assert "h_row" in state and "local_offset" in state
    # As the model restore does: a placeholder of the right shape, values from the dict.
    placeholder = RidingXYZTensor(torch.zeros_like(xyz), frames)
    placeholder.load_state_dict(state)
    assert placeholder.shape == riding.shape
    assert torch.equal(placeholder(), riding())


@pytest.mark.unit
def test_forward_is_cached_until_parameters_move(hydrogenated):
    _, frames, xyz = hydrogenated
    riding = RidingXYZTensor(xyz, frames)
    first = riding()
    assert riding() is first
    with torch.no_grad():
        riding.refinable_params[0, 0] += 0.5
    assert riding() is not first
