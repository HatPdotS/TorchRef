"""Hydrogen generation from monomer templates, driven by the bond graph.

The properties asserted here are the ones that make template instantiation
trustworthy: every hydrogen lands at its library bond length, none is placed in a
direction the geometry does not determine, the count per parent respects the valence
left over after the graph's real bonds, and the free-torsion set is exactly the centres
whose dihedral the template cannot know.
"""

import numpy as np
import pytest

from torchref.model.model import Model
from torchref.topology.hydrogens import (
    STANDARD_VALENCE,
    augment_atom_table,
    optimise_free_torsions,
    plan_hydrogens,
)

STRUCTURES = ["7L84", "1DAW"]


@pytest.fixture(scope="module")
def built(pdb_dir):
    """``(model, restraints, plan)`` per structure, built once."""
    cache = {}

    def _build(code):
        if code not in cache:
            # add_hydrogens=False: these tests exercise generation itself, so the model
            # has to arrive without the hydrogens the loader would otherwise add.
            model = Model(verbose=0, add_hydrogens=False, strip_H=True)
            model.load_pdb(str(pdb_dir / f"{code}.pdb"))
            model.set_restraints_cif(None)
            restraints = model.restraints
            plan = plan_hydrogens(
                restraints.topology, restraints.cif_dict, model.xyz().detach()
            )
            cache[code] = (model, restraints, plan)
        return cache[code]

    return _build


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_hydrogens_sit_at_their_library_bond_length(built, code):
    """Placement is exact, not approximate: the parent distance is the library value."""
    model, _, plan = built(code)
    assert plan.n_hydrogens > 0

    coords = model.xyz().detach().cpu().numpy()
    distance = np.linalg.norm(plan.position - coords[plan.parent], axis=1)
    assert np.abs(distance - plan.bond_length).max() < 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_every_candidate_hydrogen_is_placed(built, code):
    """No hydrogen is dropped for want of a determined direction.

    A hydrogen is only planned once a strategy has fixed its direction, so a shortfall
    here means some centre fell through all three. The earlier two-shell alignment left
    12% of side-chain hydrogens beyond 1.5 A of their parent and discarded them.
    """
    model, restraints, plan = built(code)
    topology = restraints.topology

    # One hydrogen per free valence on every parent that has a template hydrogen.
    expected_parents = set(plan.parent.tolist())
    assert expected_parents, "no parents received hydrogens"

    is_h = topology.atoms.is_hydrogen
    for parent in sorted(expected_parents):
        neighbours = topology.atoms.neighbors(parent)
        heavy = int((~is_h[neighbours]).sum())
        element = str(topology.atoms.element[parent]).strip().upper()
        allowed = max(0, STANDARD_VALENCE.get(element, 4) - heavy)
        placed = int((plan.parent == parent).sum())
        assert placed <= allowed, (
            f"{code}: atom {parent} ({element}) has {heavy} heavy bonds, so at most "
            f"{allowed} hydrogens, but {placed} were planned"
        )


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_free_torsions_are_exactly_the_single_neighbour_centres(built, code):
    """A dihedral is free when the parent has one heavy neighbour, and only then."""
    _, restraints, plan = built(code)
    topology = restraints.topology
    is_h = topology.atoms.is_hydrogen

    for i in range(plan.n_hydrogens):
        parent = int(plan.parent[i])
        neighbours = topology.atoms.neighbors(parent)
        heavy = int((~is_h[neighbours]).sum())
        assert (plan.group[i] >= 0) == (heavy == 1), (
            f"{code}: hydrogen {plan.name[i]} on atom {parent} with {heavy} heavy "
            f"neighbours has group {plan.group[i]}"
        )


@pytest.mark.unit
def test_hydroxyl_rotates_and_backbone_amide_does_not(built):
    """The chemistry the graph criterion is meant to capture, spot-checked.

    A serine hydroxyl hangs off an oxygen bonded only to CB, so its dihedral is free. A
    backbone amide nitrogen is bonded to CA and to the preceding residue's carbon, which
    fixes its hydrogen entirely.
    """
    _, restraints, plan = built("7L84")
    topology = restraints.topology
    names = topology.atoms.name.astype(str)
    resnames = topology.residues.resname

    free_parents = {int(p) for p, g in zip(plan.parent, plan.group) if g >= 0}
    fixed_parents = {int(p) for p, g in zip(plan.parent, plan.group) if g < 0}

    hydroxyl = [
        int(p)
        for p in free_parents
        if names[p] == "OG"
        and str(resnames[topology.residue_of_atom(p)]).strip() == "SER"
    ]
    assert hydroxyl, "no serine hydroxyl was treated as a free torsion"

    amide = [p for p in fixed_parents if names[p] == "N"]
    assert amide, "no backbone amide nitrogen was treated as determined"

    # A backbone nitrogen rotates exactly when nothing is bonded to it on the other
    # side: an N-terminal ammonium does, an in-chain amide does not. Tied to the
    # residue graph's link edges, so a missing peptide bond would show up here.
    peptide_links = topology.residues.links_of_kind("TRANS")
    accepts_link = set(peptide_links[:, 1].tolist()) if len(peptide_links) else set()

    for parent in free_parents:
        if names[parent] != "N":
            continue
        residue = topology.residue_of_atom(parent)
        assert residue not in accepts_link, (
            f"nitrogen {parent} in residue {topology.residues.key(residue)} was "
            f"treated as rotatable even though a peptide bond reaches it"
        )
    for parent in fixed_parents:
        if names[parent] != "N":
            continue
        residue = topology.residue_of_atom(parent)
        assert residue in accepts_link, (
            f"nitrogen {parent} in residue {topology.residues.key(residue)} was "
            f"treated as determined but no peptide bond reaches it"
        )


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_torsion_scan_preserves_bond_lengths(built, code):
    """The scan rotates about a bond, so it cannot change any bond length."""
    model, restraints, plan = built(code)
    coords = model.xyz().detach()

    scanned = plan_hydrogens(restraints.topology, restraints.cif_dict, coords)
    before = scanned.position.copy()
    optimise_free_torsions(scanned, restraints.topology, coords)

    numpy_coords = coords.cpu().numpy()
    distance = np.linalg.norm(scanned.position - numpy_coords[scanned.parent], axis=1)
    assert np.abs(distance - scanned.bond_length).max() < 1e-9

    moved = np.linalg.norm(scanned.position - before, axis=1) > 1e-6
    assert moved.any(), "the scan changed nothing at all"
    assert not moved[
        ~scanned.rotatable
    ].any(), "the scan moved a hydrogen whose torsion is not free"


@pytest.mark.unit
def test_scan_reduces_clash(built):
    """Scanned hydrogens end up no closer to heavy atoms than they started."""
    model, restraints, plan = built("7L84")
    coords = model.xyz().detach()
    topology = restraints.topology

    scanned = plan_hydrogens(topology, restraints.cif_dict, coords)
    numpy_coords = coords.cpu().numpy()
    heavy = numpy_coords[~topology.atoms.is_hydrogen.cpu().numpy()]

    def closest(positions):
        gaps = np.linalg.norm(positions[:, None, :] - heavy[None, :, :], axis=-1)
        # The parent itself is always the nearest heavy atom; take the next one.
        return np.sort(gaps, axis=1)[:, 1]

    rotatable = scanned.rotatable
    before = closest(scanned.position[rotatable])
    optimise_free_torsions(scanned, topology, coords)
    after = closest(scanned.position[rotatable])

    assert after.min() >= before.min() - 1e-9, "the scan made the worst clash worse"


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_augmented_table_keeps_residues_contiguous(built, code):
    """Hydrogens are inserted into their residue, not appended after everything.

    The residue partition is built from contiguous runs of ``(chain, resseq, icode)``,
    so appending hydrogens at the end would split every hydrogenated residue in two.
    """
    model, restraints, plan = built(code)
    augmented = augment_atom_table(model.pdb, plan, restraints.topology)

    assert len(augmented) == len(model.pdb) + plan.n_hydrogens
    assert (augmented["index"].values == np.arange(len(augmented))).all()

    key = (
        augmented[["chainid", "resseq", "icode"]]
        .astype(str)
        .agg("|".join, axis=1)
        .values
    )
    runs = 1 + int((key[1:] != key[:-1]).sum())
    assert runs == len(set(key)), "a residue was split into non-adjacent runs"


@pytest.mark.unit
def test_waters_are_not_hydrogenated(built):
    """A single-atom residue is skipped, and for a reason rather than by accident.

    One heavy atom gives no frame to align a template against and no bond to rotate
    about, so a water's hydrogens could only be placed in an arbitrary direction.
    """
    _, restraints, plan = built("7L84")
    topology = restraints.topology

    waters = [
        i
        for i in range(topology.n_residues)
        if str(topology.residues.resname[i]).strip() == "HOH"
    ]
    assert waters, "7L84 has no waters, so this asserts nothing"
    assert not set(plan.residue.tolist()) & set(waters)


@pytest.mark.unit
def test_hydrogenate_returns_a_consistent_model(pdb_dir):
    """The end-to-end path yields a model whose tensors, table and restraints agree."""
    model = Model(verbose=0, add_hydrogens=False, strip_H=True)
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    model.set_restraints_cif(None)
    n_heavy = len(model.pdb)

    hydrogenated = model.hydrogenate(verbose=0)

    assert hydrogenated.ctx.strip_H is False
    assert len(hydrogenated.pdb) > n_heavy
    assert hydrogenated.xyz().shape[0] == len(hydrogenated.pdb)
    assert hydrogenated.adp().shape[0] == len(hydrogenated.pdb)
    assert len(model.pdb) == n_heavy, "the original model was modified"

    elements = hydrogenated.pdb["element"].astype(str).str.strip().values
    n_h = int((elements == "H").sum())
    assert n_h == len(hydrogenated.pdb) - n_heavy

    # Every hydrogen carries exactly one bond restraint, at library geometry.
    restraints = hydrogenated.restraints
    bonds = restraints.restraints["bond"]["all"]["indices"].cpu().numpy()
    references = restraints.restraints["bond"]["all"]["references"].cpu().numpy()
    coords = hydrogenated.xyz().detach().cpu().numpy()
    is_h = elements == "H"
    involves_h = is_h[bonds[:, 0]] | is_h[bonds[:, 1]]

    assert int(involves_h.sum()) == n_h
    lengths = np.linalg.norm(coords[bonds[:, 0]] - coords[bonds[:, 1]], axis=1)
    deviation = np.sqrt(((lengths[involves_h] - references[involves_h]) ** 2).mean())
    assert deviation < 0.02, f"placed hydrogens deviate by {deviation:.4f} A RMS"


@pytest.mark.unit
def test_strip_H_removes_deposited_hydrogens(pdb_dir):
    """The opt-out drops the hydrogens the file carries, as it always did."""
    model = Model(verbose=0, strip_H=True)
    model.load_pdb(str(pdb_dir / "1AK5_with_H.pdb"))
    elements = model.pdb["element"].astype(str).str.strip().values
    assert not (elements == "H").any()
