"""CCP4 energy types travel from the monomer template onto the atom graph.

What is pinned: the type column survives the CIF reader, link modifications retype the
atoms they change (an in-chain backbone nitrogen is an amide ``NH1``, only the
N-terminus keeps the free-amine ``NT3``), the template hydrogen count lands on every
template atom, the bundled per-type table covers every type the standard residues
use, and the graph reads its hydrogen-bond roles and contact radii from that table.
"""

import csv

import numpy as np
import pytest
import torch

from torchref import PATH_TORCHREF_DATA
from torchref.model.model import Model
from torchref.topology.hydrogens import template_atom_types
from torchref.topology.nonbonded import (
    HB_ACCEPTOR,
    HB_DONOR,
    HB_HYDROGEN,
    energy_type_table,
    vdw_radii_for_elements,
)


@pytest.fixture(scope="module")
def heavy_1daw(pdb_dir):
    model = Model(verbose=0, hydrogens="strip")
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model


@pytest.fixture(scope="module")
def hydrogenated_1daw(pdb_dir):
    model = Model(verbose=0, hydrogens="add")
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model


@pytest.mark.unit
def test_reader_keeps_type_energy(heavy_1daw):
    atoms = heavy_1daw.restraints.cif_dict["ASN"]["atoms"]
    types = dict(zip(atoms["atom_id"].str.strip(), atoms["type_energy"]))
    assert types["N"] == "NT3"
    assert types["ND2"] == "NH2"
    assert types["OD1"] == "O"
    assert types["CA"] == "CH1"


@pytest.mark.unit
def test_template_atom_types_counts_hydrogens(heavy_1daw):
    types, h_count = template_atom_types(heavy_1daw.restraints.cif_dict["ASN"])
    assert types["OXT"] == "OC"
    assert h_count["ND2"] == 2
    assert h_count["CB"] == 2
    assert h_count["N"] == 3
    assert "OD1" not in h_count


@pytest.mark.unit
def test_atom_graph_carries_types_with_link_modifications(heavy_1daw):
    """Peptide-linked backbone N is retyped NH1; the chain start stays NT3."""
    atoms = heavy_1daw.restraints.topology.atoms
    names = atoms.name.astype(str)
    resseq = heavy_1daw.pdb["resseq"].values
    chain = heavy_1daw.pdb["chainid"].astype(str).values
    resnames = heavy_1daw.pdb["resname"].str.strip().values
    is_n = names == "N"
    first_res = min(resseq[chain == chain[0]])
    n_types = atoms.energy_type[is_n]
    n_first = atoms.energy_type[is_n & (resseq == first_res) & (chain == chain[0])]
    assert set(n_first.tolist()) == {"NT3"}
    # In-chain amide N is NH1; proline's tertiary N is NH0; only chain starts stay NT3.
    assert set(n_types.tolist()) <= {"NH1", "NH0", "NT3"}
    assert (n_types == "NH1").sum() > 0.8 * is_n.sum()
    assert (atoms.energy_type[is_n & (resnames == "PRO")] == "NH0").all()
    assert (atoms.energy_type[(names == "O") & (resnames != "HOH")] == "O").all()
    assert (atoms.energy_type[(names == "CA") & (resnames != "GLY")] == "CH1").all()
    assert (atoms.energy_type[(names == "CA") & (resnames == "GLY")] == "CH2").all()


@pytest.mark.unit
def test_template_h_count_and_implicit_hydrogens(heavy_1daw):
    """A heavy-only model is missing exactly the hydrogens its templates carry."""
    atoms = heavy_1daw.restraints.topology.atoms
    names = atoms.name.astype(str)
    polymer = heavy_1daw.pdb["ATOM"].astype(str).str.strip().values == "ATOM"
    resname = heavy_1daw.pdb["resname"].str.strip().values
    counts = atoms.template_h_count.cpu().numpy()
    assert (counts[names == "CB"] >= 1).all()
    assert (counts[(names == "O") & polymer] == 0).all()
    missing = atoms.implicit_h_count().cpu().numpy()
    np.testing.assert_array_equal(missing[counts >= 0], counts[counts >= 0])
    # Waters take their two hydrogens from the water dictionary; ions have no
    # template, so their count is unknown. Every polymer atom's is known.
    assert (counts[polymer] >= 0).all()
    assert (counts[resname == "HOH"] == 2).all()
    assert (counts[resname == "MG"] == -1).all()


@pytest.mark.unit
def test_hydrogenated_model_completes_charged_amines(hydrogenated_1daw):
    """Explicit hydrogen generation fills the polymer's template hydrogen counts."""
    model = hydrogenated_1daw
    atoms = model.restraints.topology.atoms
    missing = atoms.implicit_h_count().cpu().numpy()
    polymer = model.pdb["ATOM"].astype(str).str.strip().values == "ATOM"
    assert (missing[polymer] == 0).all()
    ammonium = np.isin(atoms.energy_type, ["NT", "NT1", "NT2", "NT3", "NT4"])
    assert ammonium.any()
    assert (missing[ammonium & polymer] == 0).all()


@pytest.mark.unit
def test_bundled_table_covers_standard_residue_types(heavy_1daw):
    with open(f"{PATH_TORCHREF_DATA}/ener_lib_atoms.csv") as handle:
        rows = [r for r in csv.DictReader(l for l in handle if not l.startswith("#"))]
    table = {row["type"] for row in rows}
    used = set(heavy_1daw.restraints.topology.atoms.energy_type.tolist()) - {""}
    assert used and used <= table
    by_type = {row["type"]: row for row in rows}
    assert by_type["NH1"]["hb_type"] == "D"
    assert by_type["O"]["hb_type"] == "A"
    assert by_type["OH1"]["hb_type"] == "B"
    assert float(by_type["CH3"]["vdwh_radius"]) > float(by_type["CH3"]["vdw_radius"])


@pytest.mark.unit
def test_atom_graph_reads_hydrogen_bond_roles_from_the_table(hydrogenated_1daw):
    """ener_lib's donor/acceptor column by type; a hydrogen takes its parent's donor."""
    atoms = hydrogenated_1daw.restraints.topology.atoms
    roles = atoms.hb_type.cpu().numpy()
    kinds = atoms.energy_type.astype(str)
    names = atoms.name.astype(str)
    assert (roles[kinds == "NH1"] == HB_DONOR).all()
    assert (roles[kinds == "O"] == HB_ACCEPTOR).all()
    assert (roles[kinds == "OH1"] == HB_DONOR | HB_ACCEPTOR).all()
    assert (roles[np.isin(kinds, ["CH1", "CH2", "CH3"])] == 0).all()
    # The templates type every hydrogen H (no role); an amide H bonds as its N does.
    assert (roles[names == "H"] == HB_HYDROGEN).all()
    assert (roles[names == "HA"] == 0).all()


@pytest.mark.unit
def test_contact_radii_follow_energy_types(heavy_1daw, hydrogenated_1daw):
    """Typed radii, with the hydrogens folded in only where a hydrogenated graph lacks
    them: a heavy-only graph gets its hydrogens back as riding atoms."""
    table = energy_type_table()
    atoms = heavy_1daw.restraints.topology.atoms
    kinds = atoms.energy_type.astype(str)
    radii = atoms.vdw_radii
    for kind in ("CH1", "CH2", "CH3", "NH1", "O"):
        np.testing.assert_allclose(radii[kinds == kind], table[kind][1])
    untyped = kinds == ""
    assert untyped.any()
    np.testing.assert_allclose(
        radii[untyped], vdw_radii_for_elements(atoms.element[untyped])
    )

    # One residue's hydrogens removed from the hydrogenated graph: its atoms, and
    # only those, take the radius with their hydrogens folded in.
    full = hydrogenated_1daw.restraints.topology.atoms
    is_h = full.is_hydrogen.cpu().numpy()
    residue_of = full.residue_of.cpu().numpy()
    lysine = residue_of[np.nonzero(full.resname.astype(str) == "LYS")[0][0]]
    keep = ~(is_h & (residue_of == lysine))
    remap = torch.full((full.n_atoms,), -1, dtype=torch.long)
    remap[torch.as_tensor(keep)] = torch.arange(int(keep.sum()))
    stripped = full.subset(remap, torch.arange(int(residue_of.max()) + 1))
    kinds = stripped.energy_type.astype(str)
    folded = stripped.implicit_h_count().cpu().numpy() > 0
    in_lysine = stripped.residue_of.cpu().numpy() == lysine
    assert (folded & in_lysine).sum() >= 7
    expected = [table[kind][2] for kind in kinds[folded]]
    np.testing.assert_allclose(stripped.vdw_radii[folded], expected)
    bare = ~folded & (kinds != "")
    expected = [table[kind][1] for kind in kinds[bare]]
    np.testing.assert_allclose(stripped.vdw_radii[bare], expected)
