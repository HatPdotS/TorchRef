"""CCP4 energy types travel from the monomer template onto the atom graph.

What is pinned: the type column survives the CIF reader, link modifications retype the
atoms they change (an in-chain backbone nitrogen is an amide ``NH1``, only the
N-terminus keeps the free-amine ``NT3``), the template hydrogen count lands on every
template atom, and the bundled per-type table covers every type the standard residues
use.
"""

import csv

import numpy as np
import pytest

from torchref import PATH_TORCHREF_DATA
from torchref.model.model import Model
from torchref.topology.hydrogens import template_atom_types


@pytest.fixture(scope="module")
def heavy_1daw(pdb_dir):
    model = Model(verbose=0, strip_H=True, add_hydrogens=False)
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
    single_atom = np.isin(heavy_1daw.pdb["resname"].str.strip().values, ["HOH", "MG"])
    counts = atoms.template_h_count.cpu().numpy()
    assert (counts[names == "CB"] >= 1).all()
    assert (counts[(names == "O") & polymer] == 0).all()
    missing = atoms.implicit_h_count().cpu().numpy()
    np.testing.assert_array_equal(missing[counts >= 0], counts[counts >= 0])
    # Waters and ions have no template, so their count is unknown; every polymer
    # atom's is known.
    assert (counts[polymer] >= 0).all()
    assert (counts[single_atom] == -1).all()


@pytest.mark.unit
def test_hydrogenated_model_completes_charged_amines(pdb_dir):
    """Explicit hydrogen generation fills the polymer's template hydrogen counts."""
    model = Model(verbose=0, add_hydrogens=True)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
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
