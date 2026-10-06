"""Riding frames read off the bond graph, and their bookkeeping across table edits.

Every hydrogen bonded to a heavy atom gets a frame, and a parent with a single heavy
neighbour borrows a grandparent so the hydrogen turns with its torsion.
"""

import numpy as np
import pytest
import torch

from torchref.base.coordinates.local_frame import (
    frame_is_degenerate,
    local_frame_coordinates,
    place_local_frame,
)
from torchref.config import get_int_dtype
from torchref.model.model import Model
from torchref.topology.hydrogens import HydrogenFrames, hydrogen_frames


@pytest.fixture(scope="module")
def hydrogenated(pdb_dir):
    """1DAW with every missing hydrogen generated as a real atom."""
    model = Model(verbose=0, hydrogens="add")
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model


@pytest.mark.unit
def test_every_bonded_hydrogen_gets_a_frame_or_orientation(hydrogenated):
    """Hydrogens have a heavy-atom frame or an independently rotatable water group."""
    full = hydrogenated
    frames = hydrogen_frames(full.restraints.topology)
    n_h = int((full.pdb["element"].str.strip() == "H").sum())
    assert frames.n_hydrogens == n_h
    assert (frames.frame_valid | (frames.rotation_group >= 0)).all()
    assert (frames.parent_row >= 0).all()
    is_h = full.restraints.topology.atoms.is_hydrogen.cpu().numpy()
    assert not is_h[frames.parent_row].any()
    assert not is_h[frames.n1_row[frames.n1_row >= 0]].any()
    assert not is_h[frames.n2_row[frames.n2_row >= 0]].any()


@pytest.mark.unit
def test_single_neighbour_parents_borrow_the_grandparent(hydrogenated):
    """A hydroxyl or methyl hydrogen is framed on the bond it rotates about."""
    full = hydrogenated
    frames = hydrogen_frames(full.restraints.topology)
    names = full.pdb["name"].str.strip().values
    resnames = full.pdb["resname"].str.strip().values
    seen = {}
    for parent, n1, n2 in zip(frames.parent_row, frames.n1_row, frames.n2_row):
        key = (resnames[parent], names[parent])
        seen.setdefault(key, (names[n1], names[n2]))
    assert seen[("SER", "OG")] == ("CB", "CA")
    assert seen[("LYS", "NZ")] == ("CE", "CD")
    # A two-neighbour parent frames on its own neighbours.
    assert set(seen[("ALA", "CA")]) <= {"N", "C", "CB"}


@pytest.mark.unit
def test_positions_round_trip_through_their_frames(hydrogenated):
    """Placed hydrogens are reproduced exactly from heavy atoms and local offsets."""
    full = hydrogenated
    frames = hydrogen_frames(full.restraints.topology)
    xyz = full.xyz().detach().cpu()
    p, n1, n2 = xyz[frames.parent_row], xyz[frames.n1_row], xyz[frames.n2_row]
    h = xyz[frames.h_row]
    valid = torch.as_tensor(frames.frame_valid)
    assert not frame_is_degenerate(p, n1, n2)[valid].any()
    local = local_frame_coordinates(p, n1, n2, h)
    back = place_local_frame(p, n1, n2, local, valid, h - p)
    tolerance = 1e-4 if xyz.dtype == torch.float32 else 1e-9
    assert float((back - h).norm(dim=1).max()) < tolerance


@pytest.mark.unit
def test_remap_drops_orphans_and_degrades_lost_frames():
    """A hydrogen whose parent vanishes is dropped; a lost n2 leaves a rigid frame."""
    frames = HydrogenFrames(
        h_row=np.array([5, 6, 7]),
        parent_row=np.array([1, 2, 3]),
        n1_row=np.array([0, 1, 2]),
        n2_row=np.array([2, 3, 4]),
        frame_valid=np.array([True, True, True]),
    )
    # Drop atoms 2 and 7: the first hydrogen loses n2, the second loses its parent,
    # the third is gone itself.
    old_to_new = np.array([0, 1, -1, 2, 3, 4, 5, -1])
    out = frames.remap(old_to_new)
    assert out.h_row.tolist() == [4]
    assert out.parent_row.tolist() == [1]
    assert out.n2_row.tolist() == [-1]
    assert out.frame_valid.tolist() == [False]


@pytest.mark.unit
def test_tensor_round_trip_preserves_frames():
    """from_tensors reads row and group tensors back without loss."""
    rows = dict(
        h_row=[3, 4],
        parent_row=[1, 1],
        n1_row=[0, 0],
        n2_row=[2, -1],
        torsion_group=[0, -1],
        rotation_group=[-1, -1],
    )
    tensors = {k: torch.tensor(v, dtype=get_int_dtype()) for k, v in rows.items()}
    valid = torch.tensor([True, False])
    back = HydrogenFrames.from_tensors(frame_valid=valid, **tensors)
    for field, values in rows.items():
        np.testing.assert_array_equal(getattr(back, field), values)
    np.testing.assert_array_equal(back.frame_valid, valid.numpy())
