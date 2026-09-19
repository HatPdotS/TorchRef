"""Riding frames read off the bond graph, and their bookkeeping across table edits.

Every hydrogen bonded to a heavy atom gets a frame; a parent with a single heavy
neighbour borrows a grandparent so the hydrogen turns with its torsion; and the frames
planned before an insertion agree, after remapping, with frames rebuilt on the
inserted table.
"""

import numpy as np
import pytest
import torch

from torchref.base.coordinates.local_frame import (
    frame_is_degenerate,
    local_frame_coordinates,
    place_local_frame,
)
from torchref.model.model import Model
from torchref.topology.hydrogens import (
    HydrogenFrames,
    augment_atom_table_with_maps,
    hydrogen_frames,
    optimise_free_torsions,
    plan_hydrogens,
)


@pytest.fixture(scope="module")
def heavy_and_plan(pdb_dir):
    """Heavy-only 1DAW with its hydrogen plan."""
    model = Model(verbose=0, strip_H=True, add_hydrogens=False)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    restraints = model.restraints
    xyz = model.xyz().detach()
    plan = plan_hydrogens(restraints.topology, restraints.cif_dict, xyz)
    optimise_free_torsions(plan, restraints.topology, xyz)
    return model, restraints, plan


@pytest.fixture(scope="module")
def hydrogenated(heavy_and_plan):
    """The same structure with the plan inserted, plus the maps the insertion made."""
    model, restraints, plan = heavy_and_plan
    augmented, old_to_new, plan_to_new = augment_atom_table_with_maps(
        model.pdb, plan, restraints.topology
    )
    full = Model(verbose=0, strip_H=False, add_hydrogens=False)
    cell, spacegroup = model.cell.data.cpu().numpy(), model.spacegroup

    def reader():
        return augmented, cell, spacegroup

    reader.links = model.ctx.links
    full.load(reader)
    return full, old_to_new, plan_to_new


@pytest.mark.unit
def test_every_bonded_hydrogen_gets_a_frame_or_orientation(hydrogenated):
    """Hydrogens have a heavy-atom frame or an independently rotatable water group."""
    full, _, _ = hydrogenated
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
    full, _, _ = hydrogenated
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
def test_planned_frames_match_frames_rebuilt_on_the_augmented_table(
    heavy_and_plan, hydrogenated
):
    """Remapping the pre-insertion frames reproduces the post-insertion ones."""
    _, restraints, plan = heavy_and_plan
    full, old_to_new, plan_to_new = hydrogenated
    planned = hydrogen_frames(restraints.topology, plan)
    assert planned.n_planned == plan.n_hydrogens
    carried = planned.remap(old_to_new).fill_planned_rows(plan_to_new).sorted_by_row()
    rebuilt = hydrogen_frames(full.restraints.topology).sorted_by_row()
    for field in ("h_row", "parent_row", "n1_row", "n2_row"):
        np.testing.assert_array_equal(getattr(carried, field), getattr(rebuilt, field))
    np.testing.assert_array_equal(carried.frame_valid, rebuilt.frame_valid)


@pytest.mark.unit
def test_positions_round_trip_through_their_frames(hydrogenated):
    """Placed hydrogens are reproduced exactly from heavy atoms and local offsets."""
    full, _, _ = hydrogenated
    frames = hydrogen_frames(full.restraints.topology)
    xyz = full.xyz().detach().cpu()
    t = frames.to_tensors()
    p, n1, n2 = xyz[t["parent_row"]], xyz[t["n1_row"]], xyz[t["n2_row"]]
    h = xyz[t["h_row"]]
    assert not frame_is_degenerate(p, n1, n2)[t["frame_valid"]].any()
    local = local_frame_coordinates(p, n1, n2, h)
    back = place_local_frame(p, n1, n2, local, t["frame_valid"], h - p)
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
    """to_tensors / from_tensors is lossless."""
    frames = HydrogenFrames(
        h_row=np.array([3, 4]),
        parent_row=np.array([1, 1]),
        n1_row=np.array([0, 0]),
        n2_row=np.array([2, -1]),
        frame_valid=np.array([True, False]),
    )
    back = HydrogenFrames.from_tensors(**frames.to_tensors())
    for field in ("h_row", "parent_row", "n1_row", "n2_row", "frame_valid"):
        np.testing.assert_array_equal(getattr(back, field), getattr(frames, field))
