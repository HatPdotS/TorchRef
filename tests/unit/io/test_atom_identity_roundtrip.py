"""Atom identity survives every coordinate writer.

Residues 10 and 10A differ only in their insertion code: column 27 of an ATOM, HETATM
or ANISOU record, ``pdbx_PDB_ins_code`` in mmCIF. Each test writes a deposited
structure renumbered to carry insertion codes and reads the file back with gemmi, an
independent reader, so a writer that drops, moves or changes the case of that one
character fails here. 7L84 is the base because it carries ANISOU records, which repeat
the identity columns, as well as altlocs and waters.

Residues 10/10A share a name, so losing the code would merge them into one residue;
23/23A/23B have different names.
"""

import gemmi
import pytest

from torchref.io import cif, pdb
from torchref.model.model import Model

BASE = "7L84"

#: Deposited resseq -> the (resseq, icode) written in its place.
RENUMBER = {11: (10, "A"), 24: (23, "A"), 25: (23, "B")}

RECORDS = ("ATOM", "HETATM", "ANISOU")


def _rewrite_with_insertion_codes(source, destination):
    """Copy a PDB file, renumbering :data:`RENUMBER`; only columns 23-27 change."""
    out = []
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith(RECORDS):
            resseq = int(line[22:26])
            if resseq in RENUMBER:
                new_resseq, icode = RENUMBER[resseq]
                line = f"{line[:22]}{new_resseq:>4d}{icode}{line[27:]}"
        out.append(line)
    destination.write_text("".join(out))


def _residues(model):
    """``(chain, seqnum, icode, resname, n_atoms)`` of every residue in a gemmi model."""
    return [
        (chain.name, res.seqid.num, res.seqid.icode, res.name, len(res))
        for chain in model
        for res in chain
    ]


def _read(path):
    """Residues and the number of anisotropic atoms in the first model, per gemmi."""
    model = gemmi.read_structure(str(path))[0]
    n_aniso = sum(
        atom.aniso.nonzero() for chain in model for res in chain for atom in res
    )
    return _residues(model), n_aniso


@pytest.fixture(scope="module")
def inserted(pdb_dir, tmp_path_factory):
    """Path of the renumbered PDB file."""
    path = tmp_path_factory.mktemp("icode") / f"{BASE}_icode.pdb"
    _rewrite_with_insertion_codes(pdb_dir / f"{BASE}.pdb", path)
    return path


@pytest.fixture(scope="module")
def table(inserted):
    """The renumbered file as the PDB reader's atom table."""
    df, _, _ = pdb.read(str(inserted))()
    return df


@pytest.mark.unit
def test_the_rewrite_produced_insertion_codes(inserted, table):
    """Guard the fixture: if the rewrite silently failed the rest proves nothing."""
    residues, n_aniso = _read(inserted)
    seqids = {(num, icode) for _, num, icode, _, _ in residues}
    assert {(10, " "), (10, "A"), (23, " "), (23, "A"), (23, "B")} <= seqids
    assert n_aniso > 0
    assert set(table["icode"]) == {"", "A", "B"}
    assert table.loc[table["icode"] == "A", "anisou_flag"].any()


@pytest.mark.unit
def test_pdb_write_puts_the_identity_in_columns_7_to_27(inserted, table, tmp_path):
    """ATOM, HETATM and ANISOU records carry the input's columns 1-27 byte for byte."""
    out = tmp_path / "out.pdb"
    pdb.write(table, str(out))

    def identities(path):
        return [
            line[:27]
            for line in path.read_text().splitlines()
            if line.startswith(RECORDS)
        ]

    assert identities(out) == identities(inserted)
    assert all(
        len(line) == 80
        for line in out.read_text().splitlines()
        if line.startswith(RECORDS)
    )


@pytest.mark.unit
def test_pdb_write_round_trips_insertion_codes(inserted, table, tmp_path):
    out = tmp_path / "out.pdb"
    pdb.write(table, str(out))

    assert _read(out) == _read(inserted)
    back, _, _ = pdb.read(str(out))()
    assert back["icode"].tolist() == table["icode"].tolist()


@pytest.mark.unit
def test_write_multi_model_round_trips_insertion_codes(inserted, table, tmp_path):
    out = tmp_path / "multi.pdb"
    pdb.write_multi_model([table, table], str(out))

    structure = gemmi.read_structure(str(out))
    expected, _ = _read(inserted)
    assert len(structure) == 2
    assert all(_residues(model) == expected for model in structure)


@pytest.fixture(scope="module")
def model(inserted):
    """The renumbered file loaded as a Model, every atom kept."""
    model = Model(verbose=0, hydrogens="keep")
    model.load_pdb(str(inserted))
    return model


@pytest.mark.unit
def test_model_write_pdb_round_trips_insertion_codes(inserted, model, tmp_path):
    """Through Model.write_pdb, which sanitizes the table before writing it."""
    out = tmp_path / "model.pdb"
    model.write_pdb(str(out))

    assert _read(out) == _read(inserted)


@pytest.mark.unit
def test_model_write_cif_round_trips_insertion_codes(inserted, model, tmp_path):
    """The same through mmCIF, insertion codes in their original (upper) case."""
    out = tmp_path / "model.cif"
    model.write_cif(str(out))

    assert _read(out) == _read(inserted)
    back, _, _ = cif.read_model(str(out))()
    assert set(back["icode"]) == {"", "A", "B"}


@pytest.mark.unit
def test_blank_chain_atoms_survive_pdb_to_cif(pdb_dir, tmp_path):
    """Atoms with a blank chain ID, which the PDB reader reads as NaN, are written."""
    source = tmp_path / "blank_chain.pdb"
    lines = []
    for line in (pdb_dir / "1DAW.pdb").read_text().splitlines(keepends=True):
        if line.startswith(RECORDS) and line[17:20] == "HOH":
            line = f"{line[:21]} {line[22:]}"
        lines.append(line)
    source.write_text("".join(lines))
    table, _, _ = pdb.read(str(source))()
    assert table["chainid"].fillna("").eq("").sum() == 285

    out = tmp_path / "blank_chain.cif"
    cif.write_model(table, str(out))

    residues, _ = _read(out)
    assert sum(n_atoms for *_, n_atoms in residues) == len(table)
    assert sum(resname == "HOH" for _, _, _, resname, _ in residues) == 285
