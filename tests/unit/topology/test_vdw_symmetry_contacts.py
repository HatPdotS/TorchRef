"""Symmetry contacts of the VDW pair builder, against gemmi and under lattice shifts.

Which unit cell the deposited coordinates sit in is arbitrary, so the image pairs the
builder finds must not depend on it, and they must be every contact gemmi finds with
the full space group. 6G9X (P 21 21 2, centroid at fractional x = 1.08) and 3E98
(P 1 21 1, centroid at z = 1.04) are deposited away from the origin cell; 1DAW (C 1 2 1)
runs the production builder and the riding-hydrogen candidates.
"""

from collections import defaultdict

import gemmi
import numpy as np
import pytest
import torch

from torchref.base.coordinates import is_symmetry_image, symmetry_image_positions
from torchref.base.targets.nonbonded import nonbonded_pair_positions
from torchref.config import get_float_dtype
from torchref.model.model import Model
from torchref.symmetry import SpaceGroup
from torchref.symmetry.cell import Cell
from torchref.topology import nonbonded as nb
from torchref.topology.riding import place_riding_hydrogens

pytestmark = pytest.mark.unit

CUTOFF = 6.0
# float32 positions near 100 Å against gemmi's float64 ones.
_DIST_ATOL = 1e-3
_SHIFTS = ([-1, 0, 0], [0, 0, -1], [2, -3, 1], [5, 4, -6])


def _read(path):
    """Coordinates in gemmi's atom order, the cell and the space group of one model."""
    st = gemmi.read_structure(str(path))
    atoms = [
        (ch.name, r.seqid.num, r.seqid.icode, a.name, a.altloc)
        for ch in st[0]
        for r in ch
        for a in r
    ]
    xyz = torch.tensor(
        [[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch for a in r],
        dtype=get_float_dtype(),
    )
    c = st.cell
    cpu = torch.device("cpu")
    cell = Cell([c.a, c.b, c.c, c.alpha, c.beta, c.gamma], device=cpu)
    return (
        st,
        {key: k for k, key in enumerate(atoms)},
        xyz,
        cell,
        SpaceGroup(st.spacegroup_hm, device=cpu),
    )


def _image_contacts(xyz, cell, sg):
    """Builder steps 1-4 on CPU: ``{(i, j): sorted image distances}`` for image pairs."""
    ops, offsets = nb.prefilter_symop_offsets(cell, sg, xyz, CUTOFF)
    identity = (~is_symmetry_image(ops, offsets)).nonzero()[0].item()
    lengths = torch.stack([cell.a, cell.b, cell.c])
    grid_dims = torch.clamp((lengths / CUTOFF).long(), min=1)
    _, atom_idx, combo_idx, cart = nb.assign_to_grid(
        xyz, cell, sg, ops, offsets, grid_dims
    )
    i, j, combo = nb.find_pairs_kdtree(cart, atom_idx, combo_idx, CUTOFF, identity)
    image = combo != identity
    i, j, combo = i[image], j[image], combo[image]
    partner = symmetry_image_positions(
        xyz[j],
        ops[combo],
        offsets[combo],
        sg.matrices,
        sg.translations,
        cell.fractional_matrix,
        cell.inv_fractional_matrix,
    )
    return _group(i, j, (partner - xyz[i]).norm(dim=1))


def _group(i, j, d):
    out = defaultdict(list)
    for a, b, x in zip(i.tolist(), j.tolist(), d.tolist()):
        out[(a, b)].append(x)
    return {k: sorted(v) for k, v in out.items()}


def _gemmi_contacts(st, index):
    """gemmi's symmetry and lattice contacts below ``CUTOFF``, from both ends."""
    ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate()
    search = gemmi.ContactSearch(CUTOFF)
    search.ignore = gemmi.ContactSearch.Ignore.Nothing
    search.twice = True
    search.special_pos_cutoff_sq = 0.0
    out = defaultdict(list)
    for hit in search.find_contacts(ns):
        p1, p2 = hit.partner1, hit.partner2
        image = st.cell.find_nearest_pbc_image(p1.atom.pos, p2.atom.pos, hit.image_idx)
        if image.same_asu():
            continue
        keys = [
            (
                p.chain.name,
                p.residue.seqid.num,
                p.residue.seqid.icode,
                p.atom.name,
                p.atom.altloc,
            )
            for p in (p1, p2)
        ]
        out[(index[keys[0]], index[keys[1]])].append(hit.dist)
    return {k: sorted(v) for k, v in out.items()}


def _assert_same_contacts(got, want):
    assert got.keys() == want.keys(), (
        f"{len(set(want) - set(got))} contacts missing, "
        f"{len(set(got) - set(want))} extra"
    )
    for key, distances in want.items():
        assert len(got[key]) == len(distances), key
        assert np.allclose(got[key], distances, atol=_DIST_ATOL, rtol=0), key


@pytest.fixture(scope="module", params=["6G9X.pdb", "3E98.pdb"])
def deposited(request, pdb_dir):
    st, index, xyz, cell, sg = _read(pdb_dir / request.param)
    return st, index, xyz, cell, sg, _image_contacts(xyz, cell, sg)


def test_image_contacts_match_gemmi(deposited):
    st, index, _, _, _, contacts = deposited
    _assert_same_contacts(contacts, _gemmi_contacts(st, index))


@pytest.mark.parametrize("shift", _SHIFTS)
def test_image_contacts_do_not_depend_on_the_unit_cell(deposited, shift):
    _, _, xyz, cell, sg, contacts = deposited
    translation = cell.fractional_to_cartesian(torch.tensor(shift, dtype=xyz.dtype))
    _assert_same_contacts(_image_contacts(xyz + translation, cell, sg), contacts)


def test_every_image_contact_is_listed_from_both_ends(deposited):
    contacts = deposited[-1]
    reverse = {(j, i): d for (i, j), d in contacts.items()}
    _assert_same_contacts(reverse, contacts)


@pytest.fixture(scope="module")
def model_1daw(pdb_dir):
    model = Model(verbose=0, device=torch.device("cpu"))
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model


def test_production_builder_keeps_its_contacts_under_a_lattice_shift(model_1daw):
    restraints = model_1daw.restraints
    cell, sg = model_1daw.cell, model_1daw.spacegroup
    xyz = model_1daw.xyz().detach()

    def image_distances(coords):
        vdw = nb.build_vdw_restraints_gpu(
            xyz=coords,
            vdw_radii=restraints._vdw_radii,
            cell=cell,
            sg=sg,
            topology=restraints.topology,
            exclusion_set=restraints.topology.atoms.exclusions_from_restraint_edges(),
            cutoff=CUTOFF,
            inter_residue_only=False,
        )
        tables = (
            sg.matrices,
            sg.translations,
            cell.fractional_matrix,
            cell.inv_fractional_matrix,
        )
        pos1, pos2 = nonbonded_pair_positions(
            coords,
            vdw["indices"],
            vdw["symop_indices"],
            vdw["cell_offsets"],
            *tables,
        )
        image = is_symmetry_image(vdw["symop_indices"], vdw["cell_offsets"])
        idx = vdw["indices"][image]
        return _group(idx[:, 0], idx[:, 1], (pos2 - pos1).norm(dim=1)[image])

    deposited = image_distances(xyz)
    assert sum(len(d) for d in deposited.values()) > 5000
    shift = torch.tensor([-1.0, 0.0, 0.0], dtype=xyz.dtype)
    _assert_same_contacts(
        image_distances(xyz + cell.fractional_to_cartesian(shift)), deposited
    )


def test_riding_h_candidates_are_scored_near_their_heavy_contact(model_1daw):
    """An H candidate comes from a heavy pair closer than the cutoff, so with the image
    on the right atom it lies within the cutoff plus two X-H bonds."""
    h_topo = model_1daw.restraints.h_topo
    assert h_topo is not None and h_topo.has_candidates
    cell, sg = model_1daw.cell, model_1daw.spacegroup
    xyz = model_1daw.xyz().detach()
    xyz_all = torch.cat([xyz, place_riding_hydrogens(xyz, h_topo)])
    pos_i, pos_j = nonbonded_pair_positions(
        xyz_all,
        torch.stack([h_topo.cand_idx_i, h_topo.cand_idx_j], dim=1),
        h_topo.cand_symop_idx,
        h_topo.cand_cell_offset,
        sg.matrices,
        sg.translations,
        cell.fractional_matrix,
        cell.inv_fractional_matrix,
    )
    image = is_symmetry_image(h_topo.cand_symop_idx, h_topo.cand_cell_offset)
    assert bool(image.any())
    reach = CUTOFF + 2.0 * float(h_topo.h_bond_length.max()) + _DIST_ATOL
    assert float((pos_j - pos_i).norm(dim=1).max()) < reach


def test_hydrogen_bonds_are_held_to_the_hydrogen_bond_distance(model_1daw):
    """A backbone N...O=C contact, and its riding N-H against the O, are hydrogen
    bonds: they get the ener_lib hydrogen-bond distance, not a radius sum, so the
    deposited amide hydrogen bonds carry almost no overlap."""
    from torchref.topology.nonbonded import HBOND_DISTANCE, HBOND_H_DISTANCE
    from torchref.topology.riding import candidate_contact_distances

    restraints = model_1daw.restraints
    atoms = restraints.topology.atoms
    kinds = atoms.energy_type.astype(str)
    vdw = restraints.restraints["vdw"]
    i, j = vdw["indices"].T.cpu().numpy()
    amide = ((kinds[i] == "NH1") & (kinds[j] == "O")) | (
        (kinds[i] == "O") & (kinds[j] == "NH1")
    )
    assert amide.sum() > 100
    np.testing.assert_allclose(vdw["min_distances"][amide].cpu(), HBOND_DISTANCE)

    h_topo = restraints.h_topo
    radii = torch.as_tensor(atoms.vdw_radii, dtype=get_float_dtype())
    minimum = candidate_contact_distances(h_topo, radii, atoms.hb_type)
    n_heavy = atoms.n_atoms
    cand_i, cand_j = h_topo.cand_idx_i.numpy(), h_topo.cand_idx_j.numpy()
    riding = cand_i >= n_heavy
    parent = h_topo.h_parent_idx.numpy()[np.where(riding, cand_i - n_heavy, 0)]
    heavy_j = np.where(cand_j < n_heavy, cand_j, 0)
    amide_h = riding & (cand_j < n_heavy) & (kinds[parent] == "NH1")
    amide_h &= kinds[heavy_j] == "O"
    assert amide_h.sum() > 100
    np.testing.assert_allclose(minimum[amide_h], HBOND_H_DISTANCE)

    xyz = model_1daw.xyz().detach()
    cell, sg = model_1daw.cell, model_1daw.spacegroup
    pos_i, pos_j = nonbonded_pair_positions(
        torch.cat([xyz, place_riding_hydrogens(xyz, h_topo)]),
        torch.stack([h_topo.cand_idx_i, h_topo.cand_idx_j], dim=1),
        h_topo.cand_symop_idx,
        h_topo.cand_cell_offset,
        sg.matrices,
        sg.translations,
        cell.fractional_matrix,
        cell.inv_fractional_matrix,
    )
    overlap = (minimum - (pos_j - pos_i).norm(dim=1)).clamp(min=0)
    assert float(overlap[amide_h].max()) < 0.2
