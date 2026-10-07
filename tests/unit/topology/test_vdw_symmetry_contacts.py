"""Symmetry contacts of the VDW pair builder, against gemmi and under lattice shifts.

Which unit cell the deposited coordinates sit in is arbitrary, so the image pairs the
builder finds must not depend on it, and they must be every contact gemmi finds with
the full space group. 6G9X (P 21 21 2, centroid at fractional x = 1.08) and 3E98
(P 1 21 1, centroid at z = 1.04) are deposited away from the origin cell; 1DAW (C 1 2 1)
runs the production builder and the riding-hydrogen candidates.
"""

import math
from collections import defaultdict

import gemmi
import numpy as np
import pytest
import torch

from torchref.base.coordinates import is_symmetry_image, symmetry_image_positions
from torchref.base.targets.nonbonded import nonbonded_pair_positions
from torchref.config import get_float_dtype, get_int_dtype
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
    radii = torch.as_tensor(
        restraints.topology.atoms.vdw_radii, dtype=get_float_dtype()
    )

    def image_distances(coords):
        vdw = nb.build_vdw_restraints_gpu(
            xyz=coords,
            vdw_radii=radii,
            cell=cell,
            sg=sg,
            topology=restraints.topology,
            exclusion_set=restraints.topology.atoms.exclusions_12_13_14(),
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


def test_pair_indices_take_the_configured_int_dtype(model_1daw):
    """Like the operation indices and cell offsets beside them, and every other
    restraint index, the atom indices of the pair list are the configured int dtype."""
    vdw = model_1daw.restraints.restraints["vdw"]
    assert len(vdw["indices"]) > 0
    for key in ("indices", "symop_indices", "cell_offsets"):
        assert vdw[key].dtype == get_int_dtype(), key


def test_no_atom_is_in_contact_with_its_own_image(model_1daw):
    """HOH 392 sits on a two-fold: its image is the atom itself, not a 0 A clash. Only
    an atom's genuine contacts with its own images, beyond gemmi's 0.8 A
    special-position cutoff, stay in the list."""
    vdw = model_1daw.restraints.restraints["vdw"]
    cell, sg = model_1daw.cell, model_1daw.spacegroup
    pos1, pos2 = nonbonded_pair_positions(
        model_1daw.xyz().detach(),
        vdw["indices"],
        vdw["symop_indices"],
        vdw["cell_offsets"],
        sg.matrices,
        sg.translations,
        cell.fractional_matrix,
        cell.inv_fractional_matrix,
    )
    own = vdw["indices"][:, 0] == vdw["indices"][:, 1]
    assert own.sum() > 10
    assert float((pos2 - pos1).norm(dim=1)[own].min()) > nb.SPECIAL_POSITION_CUTOFF


def test_alternates_do_not_meet_across_a_crystal_contact(pdb_dir):
    """3K7M models waters as alternates across crystal contacts (HOH 969 A against
    HOH 1049 B 1.86 A away). As inside the asymmetric unit, two different altlocs are
    never a contact, image pairs included."""
    model = Model(verbose=0, device=torch.device("cpu"))
    model.load_pdb(str(pdb_dir / "3K7M.pdb"))
    vdw = model.restraints.restraints["vdw"]
    altloc = np.char.strip(model.restraints.topology.atoms.altloc.astype(str))
    i, j = vdw["indices"].T.cpu().numpy()
    image = is_symmetry_image(vdw["symop_indices"], vdw["cell_offsets"]).cpu().numpy()
    assert (image & (altloc[i] != "") & (altloc[j] != "")).any()
    mixed = (altloc[i] != "") & (altloc[j] != "") & (altloc[i] != altloc[j])
    assert not mixed.any()


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


def _prolsq(pos1, pos2, minimum, target):
    """Per-pair PROLSQ NLL as the kernels score it, sqrt epsilon included, float64."""
    distance = torch.sqrt(((pos2 - pos1).double() ** 2).sum(dim=1) + 1e-8)
    overlap = (minimum.double() - distance).clamp(min=0)
    constant = math.log(target.sigma_vdw) + 0.5 * math.log(2.0 * math.pi)
    return target.c_rep * overlap**target.r_exp + constant


def test_a_crystal_contact_counts_once(model_1daw):
    """Both pair lists hold a crystal contact from both of its ends, so the loss takes
    half of every image pair: the intra-ASU pairs plus half the image pairs, heavy
    atoms and riding H-H contacts alike. An H-heavy candidate is listed from its
    hydrogen's end only and counts in full."""
    from torchref.refinement.targets import NonBondedHTarget, NonBondedTarget

    restraints = model_1daw.restraints
    cell, sg = model_1daw.cell, model_1daw.spacegroup
    tables = (
        sg.matrices,
        sg.translations,
        cell.fractional_matrix,
        cell.inv_fractional_matrix,
    )
    xyz = model_1daw.xyz().detach()
    target = NonBondedHTarget(model_1daw)

    vdw = restraints.restraints["vdw"]
    positions = nonbonded_pair_positions(
        xyz, vdw["indices"], vdw["symop_indices"], vdw["cell_offsets"], *tables
    )
    nll = _prolsq(*positions, vdw["min_distances"], target)
    image = is_symmetry_image(vdw["symop_indices"], vdw["cell_offsets"])
    heavy = nll[~image].sum() + 0.5 * nll[image].sum()

    h_topo = restraints.h_topo
    n_heavy = restraints.topology.n_atoms
    cand = torch.stack([h_topo.cand_idx_i, h_topo.cand_idx_j], dim=1)
    positions = nonbonded_pair_positions(
        torch.cat([xyz, place_riding_hydrogens(xyz, h_topo)]),
        cand,
        h_topo.cand_symop_idx,
        h_topo.cand_cell_offset,
        *tables,
    )
    nll = _prolsq(*positions, h_topo.cand_min_dist, target)
    both_ends = is_symmetry_image(h_topo.cand_symop_idx, h_topo.cand_cell_offset)
    both_ends &= (cand >= n_heavy).all(dim=1)
    riding = nll[~both_ends].sum() + 0.5 * nll[both_ends].sum()

    assert bool(image.any()) and bool(both_ends.any())
    with torch.no_grad():
        heavy_loss = float(NonBondedTarget(model_1daw).forward())
        total_loss = float(target.forward())
    assert heavy_loss == pytest.approx(float(heavy), rel=1e-4)
    assert total_loss == pytest.approx(float(heavy + riding), rel=1e-4)


def test_every_riding_h_pair_on_a_contact_between_residues_is_listed_once(
    model_1daw,
):
    """Each pair of riding hydrogens on the two atoms of an intra-ASU contact between
    two residues is an H-H candidate, and only once."""
    restraints = model_1daw.restraints
    h_topo = restraints.h_topo
    n_heavy = restraints.topology.n_atoms
    residue_of = restraints.topology.atoms.residue_of.tolist()
    riding = defaultdict(list)
    for h, parent in enumerate(h_topo.h_parent_idx.tolist()):
        riding[parent].append(n_heavy + h)

    vdw = restraints.restraints["vdw"]
    intra = ~is_symmetry_image(vdw["symop_indices"], vdw["cell_offsets"])
    expected = {
        frozenset((h_a, h_b))
        for a, b in vdw["indices"][intra].tolist()
        if residue_of[a] != residue_of[b]
        for h_a in riding[a]
        for h_b in riding[b]
    }

    i, j = h_topo.cand_idx_i, h_topo.cand_idx_j
    hh = ~is_symmetry_image(h_topo.cand_symop_idx, h_topo.cand_cell_offset)
    hh &= (i >= n_heavy) & (j >= n_heavy)
    listed = [frozenset(pair) for pair in zip(i[hh].tolist(), j[hh].tolist())]
    assert len(expected) > 1000
    assert len(listed) == len(set(listed))
    assert set(listed) == expected


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


def test_water_oxygens_are_hydrogen_bond_partners(model_1daw):
    """A water written as its oxygen alone is typed from the water dictionary,
    ``OH2``, a donor and an acceptor, so an amide N against it, and the N's riding
    hydrogen, are hydrogen bonds held to the ener_lib distances rather than radius
    sums, and the deposited ones carry little overlap."""
    from torchref.topology.nonbonded import HBOND_DISTANCE, HBOND_H_DISTANCE

    restraints = model_1daw.restraints
    atoms = restraints.topology.atoms
    kinds = atoms.energy_type.astype(str)
    water = restraints.topology.is_water
    assert water.sum() > 100
    assert (kinds[water] == "OH2").all()

    vdw = restraints.restraints["vdw"]
    i, j = vdw["indices"].T.cpu().numpy()
    amide_water = (kinds[i] == "NH1") & water[j]
    assert amide_water.sum() > 10
    np.testing.assert_allclose(vdw["min_distances"][amide_water], HBOND_DISTANCE)

    h_topo = restraints.h_topo
    n_heavy = atoms.n_atoms
    cand_i, cand_j = h_topo.cand_idx_i.numpy(), h_topo.cand_idx_j.numpy()
    riding = cand_i >= n_heavy
    parent = h_topo.h_parent_idx.numpy()[np.where(riding, cand_i - n_heavy, 0)]
    heavy_j = np.where(cand_j < n_heavy, cand_j, 0)
    to_water = riding & (cand_j < n_heavy) & (kinds[parent] == "NH1")
    to_water &= water[heavy_j]
    assert to_water.sum() > 100
    minimum = h_topo.cand_min_dist
    np.testing.assert_allclose(minimum[to_water], HBOND_H_DISTANCE)

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
    assert float(overlap[to_water].max()) < 0.3


def test_contacts_take_the_connected_topology_contact_distances(model_1daw):
    """Heavy pairs and riding candidates are held to the contact distances of the
    connected topology, its ener_lib radii and hydrogen-bond roles. The topology the
    restraints are constructed from carries no energy types, so its radii are the
    element radii."""
    from torchref.topology.riding import candidate_contact_distances

    restraints = model_1daw.restraints
    atoms = restraints.topology.atoms
    radii = torch.as_tensor(atoms.vdw_radii, dtype=get_float_dtype())
    vdw = restraints.restraints["vdw"]
    torch.testing.assert_close(
        vdw["min_distances"], nb.contact_distances(radii, atoms.hb_type, vdw["indices"])
    )
    h_topo = restraints.h_topo
    torch.testing.assert_close(
        h_topo.cand_min_dist, candidate_contact_distances(h_topo, radii, atoms.hb_type)
    )


@pytest.mark.gpu
def test_riding_contact_distances_take_roles_from_any_device(model_1daw, gpu_device):
    """A rebuild forms the radii and the riding candidates on CPU while ``hb_type``
    sits with the topology on the model device; the distances follow the radii."""
    from torchref.topology.riding import candidate_contact_distances

    restraints = model_1daw.restraints
    h_topo, atoms = restraints.h_topo, restraints.topology.atoms
    radii = torch.as_tensor(atoms.vdw_radii, dtype=get_float_dtype())
    on_cpu = candidate_contact_distances(h_topo, radii, atoms.hb_type)
    moved = candidate_contact_distances(h_topo, radii, atoms.hb_type.to(gpu_device))
    torch.testing.assert_close(moved, on_cpu, rtol=0, atol=0)
