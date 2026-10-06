"""PDB records and fields through load_as_dataframe and write, on deposited files."""

import gemmi
import pytest

from torchref.io import pdb

ATOM_RECORDS = ("ATOM", "HETATM", "ANISOU")


def _write_edited(pdb_dir, tmp_path, edit):
    """Write 1DAW.pdb with ``edit`` applied to each line; drop lines it maps to None."""
    lines = (pdb_dir / "1DAW.pdb").read_text().splitlines()
    edited = [e for e in (edit(line) for line in lines) if e is not None]
    path = tmp_path / "edited.pdb"
    path.write_text("\n".join(edited) + "\n")
    return str(path)


@pytest.mark.unit
def test_blank_element_columns_raise(pdb_dir, tmp_path):
    path = _write_edited(
        pdb_dir,
        tmp_path,
        lambda line: line[:76] if line.startswith(ATOM_RECORDS) else line,
    )
    with pytest.raises(ValueError, match="no element in columns 77-78"):
        pdb.load_as_dataframe(path)


@pytest.mark.unit
def test_last_atom_is_read_without_an_end_record(pdb_dir, tmp_path):
    path = _write_edited(
        pdb_dir,
        tmp_path,
        lambda line: line if line.startswith(("CRYST1", "ATOM", "HETATM")) else None,
    )
    assert len(pdb.load_as_dataframe(path)) == 3051


@pytest.mark.unit
def test_models_are_numbered(pdb_dir, tmp_path):
    single = pdb.load_as_dataframe(str(pdb_dir / "1DAW.pdb"))
    path = str(tmp_path / "two_models.pdb")
    pdb.write_multi_model([single, single], path)

    both = pdb.load_as_dataframe(path)

    assert set(single["model_num"]) == {1}
    assert both.groupby("model_num").size().to_dict() == {1: 3051, 2: 3051}


@pytest.mark.unit
def test_anisou_records_match_atoms_of_their_own_model(pdb_dir, tmp_path):
    lines = (pdb_dir / "7L84.pdb").read_text().splitlines()
    atoms = [line for line in lines if line.startswith(ATOM_RECORDS)]
    header = [line for line in lines if line.startswith("CRYST1")]
    models = [[f"MODEL     {n:>4}", *atoms, "ENDMDL"] for n in (1, 2)]
    path = tmp_path / "two_models.pdb"
    path.write_text("\n".join(header + models[0] + models[1] + ["END"]) + "\n")

    table = pdb.load_as_dataframe(str(path))

    assert len(table) == 2 * 2305
    assert table.groupby("model_num").size().to_dict() == {1: 2305, 2: 2305}
    assert table.groupby("model_num")["anisou_flag"].sum().to_dict() == {
        1: 1209,
        2: 1209,
    }


@pytest.mark.unit
def test_anisou_fields_are_read_over_all_seven_columns(pdb_dir, tmp_path):
    table = pdb.load_as_dataframe(str(pdb_dir / "4BX9.pdb"))
    atom = table.index[table["anisou_flag"]][0]
    # -10 and -12.3456 Å² fill their 7-column fields: '-100000', '-123456'.
    table.loc[atom, ["u12", "u13"]] = [-10.0, -12.3456]
    path = str(tmp_path / "wide_anisou.pdb")
    pdb.write(table, path)

    u = ["u11", "u22", "u33", "u12", "u13", "u23"]
    back = pdb.load_as_dataframe(path)
    assert back.loc[atom, u].tolist() == table.loc[atom, u].tolist()


#: 1DAW's last atom record, a water.
LAST_ATOM = "HETATM 3052 "


@pytest.mark.unit
@pytest.mark.parametrize(
    "serial, resseq, expected",
    [("A0000", "A000", (100000, 10000)), ("*****", " 634", (3051, 634))],
    ids=["hybrid-36", "unparsable-serial"],
)
def test_overflowing_serials_and_residue_numbers_are_read(
    pdb_dir, tmp_path, serial, resseq, expected
):
    path = _write_edited(
        pdb_dir,
        tmp_path,
        lambda line: (
            line[:6] + serial + line[11:22] + resseq + line[26:]
            if line.startswith(LAST_ATOM)
            else line
        ),
    )
    table = pdb.load_as_dataframe(path)
    assert tuple(table[["serial", "resseq"]].iloc[-1]) == expected


@pytest.mark.unit
def test_overflowing_serials_and_residue_numbers_are_written_in_hybrid_36(
    pdb_dir, tmp_path
):
    table = pdb.load_as_dataframe(str(pdb_dir / "1DAW.pdb"))
    table.loc[table.index[-1], ["serial", "resseq"]] = [100000, 10000]
    path = str(tmp_path / "large.pdb")
    pdb.write(table, path)

    with open(path) as f:
        last = [line for line in f if line.startswith(("ATOM", "HETATM"))][-1]
    assert (last[6:11], last[22:26]) == ("A0000", "A000")
    back = pdb.load_as_dataframe(path)
    assert tuple(back[["serial", "resseq"]].iloc[-1]) == (100000, 10000)
    water = gemmi.read_structure(path)[0][-1][-1]
    assert (water[0].serial, water.seqid.num) == (100000, 10000)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["3-", "-3"])
def test_charges_are_read_in_either_sign_order(pdb_dir, tmp_path, field):
    path = _write_edited(
        pdb_dir,
        tmp_path,
        lambda line: line[:78] + field if line.startswith(LAST_ATOM) else line,
    )
    assert pdb.load_as_dataframe(path)["charge"].iloc[-1] == -3


@pytest.mark.unit
def test_charges_are_written_digit_first(pdb_dir, tmp_path):
    table = pdb.load_as_dataframe(str(pdb_dir / "1DAW.pdb"))
    table.loc[table.index[-1], "charge"] = -1
    path = str(tmp_path / "charged.pdb")
    pdb.write(table, path)

    with open(path) as f:
        last = [line for line in f if line.startswith(("ATOM", "HETATM"))][-1]
    assert last[78:80] == "1-"
    assert pdb.load_as_dataframe(path)["charge"].iloc[-1] == -1
