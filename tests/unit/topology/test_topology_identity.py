"""The identity half of a topology: built from an atom table, no dictionaries needed.

A node-only topology must describe the same atoms and residues as the connected one the
restraint build produces, select the same atoms a direct reading of the table would,
and insert hydrogens exactly where the table-level insertion does.
"""

import numpy as np
import pytest
import torch

from torchref.io.pdb import PDBReader
from torchref.model.model import Model
from torchref.topology.hydrogens import augment_atom_table_with_maps, plan_hydrogens
from torchref.topology.topology import IDENTITY_COLUMNS, Topology

STRUCTURES = ["1DAW", "7L84", "1AK5_with_H"]


def _table(pdb_dir, code):
    df, _, _ = PDBReader(verbose=0).read(str(pdb_dir / f"{code}.pdb"))()
    return df.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"]).reset_index(
        drop=True
    )


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_node_topology_matches_the_connected_one(pdb_dir, code):
    """Atoms and residues agree with the restraint build's topology, edges aside."""
    model = Model(verbose=0).load_pdb(str(pdb_dir / f"{code}.pdb"))
    connected = model.restraints.topology
    nodes = Topology.from_table(model.pdb)

    assert connected.connected and not nodes.connected
    assert nodes.atoms.bonds.n_edges == 0
    for field in ("name", "element", "altloc"):
        np.testing.assert_array_equal(
            getattr(nodes.atoms, field), getattr(connected.atoms, field)
        )
    assert torch.equal(nodes.atoms.residue_of.cpu(), connected.atoms.residue_of.cpu())
    for field in ("chain", "resseq", "icode", "resname", "atom_start", "atom_end"):
        np.testing.assert_array_equal(
            getattr(nodes.residues, field), getattr(connected.residues, field)
        )


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_columns_round_trip(pdb_dir, code):
    nodes = Topology.from_table(_table(pdb_dir, code))
    again = Topology.from_columns(nodes.columns())
    for key, value in nodes.columns().items():
        np.testing.assert_array_equal(again.columns()[key], value)
    assert set(nodes.columns()) == set(IDENTITY_COLUMNS)


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_selection_matches_the_table(pdb_dir, code):
    """Every keyword and operator, checked against the same condition on the table."""
    df = _table(pdb_dir, code)
    nodes = Topology.from_table(df)
    water = df.resname == "HOH"
    blank = df.altloc.astype(str).str.strip() == ""
    cases = {
        "all": np.ones(len(df), bool),
        "chain A": df.chainid == "A",
        "resseq 10": df.resseq == 10,
        "resseq 10:40": df.resseq.between(10, 40),
        "resname hoh": water,
        "name ca": df.name == "CA",
        "element c": df.element.str.strip().str.capitalize() == "C",
        "altloc A": df.altloc == "A",
        "not resname HOH": ~water,
        "chain A and not resname HOH": (df.chainid == "A") & ~water,
        "name CA or name CB": df.name.isin(["CA", "CB"]),
        "not (resname HOH or element H)": ~(water | (df.element.str.strip() == "H")),
        "(chain A and resseq 1:50) or (resname HOH and not name O)": (
            (df.chainid == "A") & df.resseq.between(1, 50)
        )
        | (water & (df.name != "O")),
        "NOT resname HOH AND name CA": ~water & (df.name == "CA"),
        "resseq 1:20 or resseq 30:40 and name N": df.resseq.between(1, 20)
        | (df.resseq.between(30, 40) & (df.name == "N")),
        "not not name CA": df.name == "CA",
    }
    for selection, expected in cases.items():
        got = nodes.select(selection).numpy()
        np.testing.assert_array_equal(got, np.asarray(expected), err_msg=selection)
    assert not nodes.select("altloc A").numpy()[blank.to_numpy()].any()


@pytest.mark.unit
@pytest.mark.parametrize(
    "selection", ["", "chain", "resname HOH and", "(name CA", "name CA)", "bogus X"]
)
def test_malformed_selections_raise(pdb_dir, selection):
    nodes = Topology.from_table(_table(pdb_dir, "1DAW"))
    with pytest.raises(ValueError):
        nodes.select(selection)


@pytest.mark.unit
def test_water_and_polymer_masks(pdb_dir):
    df = _table(pdb_dir, "1DAW")
    nodes = Topology.from_table(df)
    np.testing.assert_array_equal(nodes.is_water, (df.resname == "HOH").to_numpy())
    np.testing.assert_array_equal(nodes.is_polymer, (df.ATOM == "ATOM").to_numpy())
    np.testing.assert_array_equal(
        nodes.atoms.is_hydrogen.cpu().numpy(), (df.element.str.strip() == "H").to_numpy()
    )


@pytest.mark.unit
@pytest.mark.parametrize("code", ["1DAW", "7L84"])
def test_hydrogen_insertion_matches_the_table_insertion(pdb_dir, code):
    """Same row maps and the same identity, row for row, as the table-level insertion."""
    model = Model(verbose=0, hydrogens="strip").load_pdb(str(pdb_dir / f"{code}.pdb"))
    restraints = model.ctx.build_restraints(model.xyz(), nonbonded=False, verbose=0)
    plan = plan_hydrogens(restraints.topology, restraints.cif_dict, model.xyz().detach())
    assert plan.n_hydrogens > 0

    augmented, old_to_new, plan_to_new = augment_atom_table_with_maps(
        model.pdb, plan, restraints.topology
    )
    nodes, source, old_to_new_t, plan_to_new_t = Topology.from_table(
        model.pdb
    ).with_hydrogens(plan)

    np.testing.assert_array_equal(old_to_new_t, old_to_new)
    np.testing.assert_array_equal(plan_to_new_t, plan_to_new)
    np.testing.assert_array_equal(source[old_to_new], np.arange(len(model.pdb)))
    np.testing.assert_array_equal(source[plan_to_new], plan.parent)
    expected = Topology.from_table(augmented)
    for key, value in expected.columns().items():
        np.testing.assert_array_equal(nodes.columns()[key], value, err_msg=key)
    np.testing.assert_array_equal(nodes.residues.atom_start, expected.residues.atom_start)


@pytest.mark.unit
def test_padded_strings_read_like_clean_ones(pdb_dir):
    """Whitespace around names, residue names, elements and icodes is not identity."""
    df = _table(pdb_dir, "1DAW")
    padded = df.copy()
    for column in ("name", "resname", "element", "icode"):
        padded[column] = " " + padded[column].astype(str) + " "
    clean, noisy = Topology.from_table(df), Topology.from_table(padded)
    for key, value in clean.columns().items():
        np.testing.assert_array_equal(noisy.columns()[key], value, err_msg=key)


@pytest.mark.unit
@pytest.mark.parametrize("dropped", [["icode"], ["altloc"], ["ATOM"], ["charge"], ["element"]])
def test_optional_columns_fall_back_to_defaults(pdb_dir, dropped):
    df = _table(pdb_dir, "1DAW")
    full = Topology.from_table(df)
    reduced = Topology.from_table(df.drop(columns=dropped))
    assert reduced.n_atoms == full.n_atoms
    defaults = {"icode": "", "altloc": " ", "ATOM": False, "charge": 0, "element": ""}
    column = {"ATOM": "is_hetatm"}.get(dropped[0], dropped[0])
    assert (reduced.columns()[column] == defaults[dropped[0]]).all()
    if dropped[0] in ("charge", "element"):
        np.testing.assert_array_equal(reduced.residues.atom_start, full.residues.atom_start)


@pytest.mark.unit
def test_charges_are_coerced(pdb_dir):
    df = _table(pdb_dir, "1DAW")
    df["charge"] = df["charge"].astype(object)
    df.loc[::7, "charge"] = "junk"
    df.loc[1, "charge"] = 2
    charge = Topology.from_table(df).atoms.charge
    assert charge.dtype == np.int64 and charge[0] == 0 and charge[1] == 2
