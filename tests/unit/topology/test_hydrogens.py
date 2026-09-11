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
    _template,
    augment_atom_table,
    optimise_free_torsions,
    plan_hydrogens,
)

# 3E98 brings HETATM selenomethionines bonded through LINK records and split side chains.
STRUCTURES = ["7L84", "1DAW", "3E98"]


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
    atoms = topology.atoms
    residues = topology.residues
    assert plan.n_hydrogens, "no hydrogens planned"

    is_h = atoms.is_hydrogen
    altlocs = np.char.strip(atoms.altloc.astype(str))
    names = np.char.strip(atoms.name.astype(str))
    checked = 0
    for residue in range(residues.n_residues):
        start, end = int(residues.atom_start[residue]), int(residues.atom_end[residue])
        template = _template(restraints.cif_dict, str(residues.resname[residue]).strip())
        if template is None:
            continue
        # Residues with altlocs plan one hydrogen per conformer; the two-sided count
        # below is for the plain case, where the graph degree is the whole story.
        if (altlocs[start:end] != "").any():
            continue
        for parent in range(start, end):
            template_h = template["h_count"].get(names[parent], 0)
            if template_h == 0:
                continue
            neighbours = atoms.neighbors(parent)
            heavy = int((~is_h[neighbours]).sum())
            element = str(atoms.element[parent]).strip().upper()
            template_heavy = len(template["heavy_adjacency"].get(names[parent], []))
            extra_bonds = max(0, heavy - template_heavy)
            expected = max(
                0,
                min(STANDARD_VALENCE.get(element, 4) - heavy, template_h - extra_bonds),
            )
            placed = int((plan.parent == parent).sum())
            assert placed == expected, (
                f"{code}: atom {parent} ({names[parent]} {element}) has {heavy} heavy "
                f"bonds against {template_heavy} in the template and {template_h} "
                f"template hydrogens, so {expected} expected, {placed} planned"
            )
            checked += 1
    assert checked > 0


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


def _row(model, chain, resseq, name, altloc=""):
    pdb = model.pdb
    mask = (
        (pdb["chainid"].astype(str) == chain)
        & (pdb["resseq"].astype(int) == resseq)
        & (pdb["name"].astype(str).str.strip() == name)
        & (pdb["altloc"].astype(str).str.strip() == altloc)
    )
    (row,) = np.nonzero(mask.values)[0]
    return int(row)


@pytest.mark.unit
def test_linked_nitrogen_keeps_one_hydrogen(built):
    """A peptide bond supplied by a LINK record displaces two of the template's three.

    MSE is a HETATM residue, so its backbone bonds come only from LINK records. MSE65
    has a split side chain: its shared N gets one hydrogen per conformer, and its
    carbonyl carbon, bonded to CA(A), CA(B), O and the next N, gets none.
    """
    model, _, plan = built("3E98")
    n73 = _row(model, "A", 73, "N")
    assert plan.name[plan.parent == n73].tolist() == ["H"]

    n65 = _row(model, "A", 65, "N")
    on_n65 = plan.parent == n65
    assert plan.name[on_n65].tolist() == ["H", "H"]
    assert sorted(plan.altloc[on_n65].tolist()) == ["A", "B"]
    assert not (plan.parent == _row(model, "A", 65, "C")).any()


@pytest.mark.unit
def test_split_side_chain_keeps_its_alpha_hydrogen(built):
    """A CA bonded to two altloc copies of CB is not saturated: one HA per conformer."""
    model, _, plan = built("3E98")
    ca_a = _row(model, "A", 65, "CA", "A")
    ca_b = _row(model, "A", 65, "CA", "B")
    assert plan.name[plan.parent == ca_a].tolist() == ["HA"]
    assert plan.name[plan.parent == ca_b].tolist() == ["HA"]


# 1U19 chain A: an acetyl cap bonded to MET1 through a LINK record. The ACE template
# is acetaldehyde-like, with a hydrogen on the carbonyl carbon that the peptide link
# displaces; the element valence alone (degree 3 of 4) would still have generated it.
_ACE_MET = """\
CRYST1   96.680   96.680  150.200  90.00  90.00  90.00 P 41          8
LINK         C   ACE A   0                 N   MET A   1     1555   1555  1.33
HETATM    1  C   ACE A   0      53.553  -7.050  35.606  1.00 47.40           C
HETATM    2  O   ACE A   0      52.916  -7.860  34.934  1.00 46.96           O
HETATM    3  CH3 ACE A   0      54.727  -7.523  36.434  1.00 47.42           C
ATOM      4  N   MET A   1      53.284  -5.731  35.670  1.00 47.11           N
ATOM      5  CA  MET A   1      52.214  -5.077  34.913  1.00 46.26           C
ATOM      6  C   MET A   1      52.674  -4.891  33.485  1.00 46.68           C
ATOM      7  O   MET A   1      53.849  -4.563  33.283  1.00 46.86           O
ATOM      8  CB  MET A   1      51.887  -3.719  35.536  1.00 45.60           C
ATOM      9  CG  MET A   1      51.426  -3.792  36.982  1.00 44.48           C
ATOM     10  SD  MET A   1      49.945  -4.797  37.183  1.00 46.06           S
ATOM     11  CE  MET A   1      48.647  -3.745  36.534  1.00 43.77           C
END
"""


@pytest.mark.unit
def test_acetyl_cap_carbon_gets_no_hydrogen(tmp_path):
    """The template's own hydrogen count, minus the link, caps the carbonyl carbon."""
    from torchref.topology.monomer.cif import find_cif_file_in_library

    if find_cif_file_in_library("ACE") is None:
        pytest.skip("ACE not in the monomer library")
    path = tmp_path / "ace_met.pdb"
    path.write_text(_ACE_MET)
    model = Model(verbose=0, add_hydrogens=False, strip_H=True)
    model.load_pdb(str(path))
    model.set_restraints_cif(None)
    restraints = model.restraints
    plan = plan_hydrogens(restraints.topology, restraints.cif_dict, model.xyz().detach())

    by_parent = {}
    for parent, name in zip(plan.parent.tolist(), plan.name.tolist()):
        by_parent.setdefault(parent, []).append(name)
    assert _row(model, "A", 0, "C") not in by_parent
    assert len(by_parent[_row(model, "A", 0, "CH3")]) == 3
    assert by_parent[_row(model, "A", 1, "N")] == ["H"]
