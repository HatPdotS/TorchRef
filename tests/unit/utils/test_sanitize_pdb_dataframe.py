"""What ``sanitize_pdb_dataframe`` renumbers before a model is written, and what not.

``Model.write_pdb`` and ``Model.write_cif`` pass every table through it. A HETATM
residue that repeats an atom identifier ``(chainid, resseq, icode, name, altloc)`` --
unnumbered waters, a copied ligand -- gets a new resseq as a whole. Nothing else
changes: an insertion code keeps residues 90 and 90A apart, and ATOM records are never
renumbered. Tables come from 1DAW (protein, ANP 340, MG 341-342, waters 350-634).
"""

import pandas as pd
import pytest

from torchref.io import pdb
from torchref.utils import sanitize_pdb_dataframe

ATOM_KEY = ["chainid", "resseq", "icode", "name", "altloc"]


@pytest.fixture(scope="module")
def table(pdb_dir):
    """1DAW as the PDB reader's atom table."""
    df, _, _ = pdb.read(str(pdb_dir / "1DAW.pdb"))()
    return df


def _resseq(df):
    return df["resseq"].to_numpy().tolist()


@pytest.mark.unit
def test_a_clean_model_is_untouched(table):
    pd.testing.assert_frame_equal(sanitize_pdb_dataframe(table), table)


@pytest.mark.unit
def test_an_insertion_code_pair_is_untouched(table):
    """GLY 90 and GLY 90A share a name and a number; the insertion code separates them."""
    inserted = table.copy()
    inserted.loc[inserted["resseq"] == 91, ["resseq", "icode"]] = [90, "A"]
    assert inserted.duplicated(["chainid", "resseq", "name", "altloc"]).any()

    pd.testing.assert_frame_equal(sanitize_pdb_dataframe(inserted), inserted)


@pytest.mark.unit
def test_a_copied_ligand_is_renumbered_whole(table):
    """A second ANP 340 becomes one residue at the next free number."""
    anp = table[table["resname"] == "ANP"]
    df = pd.concat([table, anp], ignore_index=True)

    out = sanitize_pdb_dataframe(df)

    assert _resseq(out.iloc[: len(table)]) == _resseq(table)
    assert set(out["resseq"].iloc[len(table) :]) == {table["resseq"].max() + 1}


@pytest.mark.unit
def test_unnumbered_waters_become_one_residue_each(table):
    df = table.copy()
    water = (df["resname"] == "HOH").to_numpy()
    df.loc[water, "resseq"] = 0

    out = sanitize_pdb_dataframe(df)

    assert not out.duplicated(ATOM_KEY).any()
    assert out.loc[water, "resseq"].nunique() == water.sum()
    assert _resseq(out.loc[~water]) == _resseq(table.loc[~water])
    assert (df.loc[water, "resseq"] == 0).all(), "the input table was modified"


@pytest.mark.unit
def test_a_water_colliding_with_a_polymer_residue_moves(table):
    """A water numbered 90 collides with GLY 90's O; the water moves, though it is first."""
    water = table[table["resname"] == "HOH"].iloc[[0]].assign(resseq=90)
    df = pd.concat([water, table], ignore_index=True)

    out = sanitize_pdb_dataframe(df)

    assert out["resseq"].iloc[0] == table["resseq"].max() + 1
    assert _resseq(out.iloc[1:]) == _resseq(table)


@pytest.mark.unit
def test_atom_records_are_never_renumbered(table):
    """A duplicated polymer residue is left as it is rather than torn apart."""
    df = pd.concat([table, table[table["resseq"] == 90]], ignore_index=True)

    out = sanitize_pdb_dataframe(df)

    assert _resseq(out) == _resseq(df)
