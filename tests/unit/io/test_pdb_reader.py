"""Which PDB records and fields reach the atom table, read against 1DAW."""

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
    with pytest.raises(ValueError, match="blank element field"):
        pdb.load_as_dataframe(path)
