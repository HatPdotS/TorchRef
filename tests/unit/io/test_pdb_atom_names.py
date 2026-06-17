"""Regression tests for PDB atom-name writing.

Both PDB writers used to truncate 4-character atom names to their last 3
characters (``name[-3:]``), corrupting names such as ``HD11`` -> ``D11``.
The fix justifies names into PDB columns 13-16 without truncation. See
TORCHREF_AUDIT.md cluster 3.
"""

import pandas as pd
import pytest

from torchref.io import pdb


def _atom_df():
    """Minimal DataFrame with a 4-char H name (+ANISOU) and a 2-char C name."""
    return pd.DataFrame(
        {
            "ATOM": ["ATOM", "ATOM"],
            "serial": [1, 2],
            "name": ["HD11", "CA"],
            "altloc": ["", ""],
            "resname": ["LEU", "LEU"],
            "chainid": ["A", "A"],
            "resseq": [1, 1],
            "icode": ["", ""],
            "x": [1.234, 2.345],
            "y": [2.0, 3.0],
            "z": [3.0, 4.0],
            "occupancy": [1.0, 1.0],
            "tempfactor": [20.0, 20.0],
            "element": ["H", "C"],
            "charge": [0, 0],
            "anisou_flag": [True, False],
            "u11": [0.01, 0.0],
            "u22": [0.01, 0.0],
            "u33": [0.01, 0.0],
            "u12": [0.0, 0.0],
            "u13": [0.0, 0.0],
            "u23": [0.0, 0.0],
        }
    )


def _atom_name_field(line: str) -> str:
    """PDB atom name occupies columns 13-16 (0-indexed slice 12:16)."""
    return line[12:16]


@pytest.mark.unit
def test_format_pdb_atom_name_no_truncation():
    from torchref.io.pdb import _format_pdb_atom_name

    assert _format_pdb_atom_name("HD11", "H") == "HD11"  # 4-char preserved
    assert _format_pdb_atom_name("1HG2", "H") == "1HG2"
    assert _format_pdb_atom_name("CA", "C").strip() == "CA"  # single-letter elem
    assert _format_pdb_atom_name("FE", "FE") == "FE  "  # two-letter elem, col 13
    # All fields are exactly the 4-column width.
    for n, e in [("HD11", "H"), ("CA", "C"), ("FE", "FE"), ("N", "N")]:
        assert len(_format_pdb_atom_name(n, e)) == 4


@pytest.mark.unit
def test_write_preserves_4char_name(tmp_path):
    out = tmp_path / "out.pdb"
    pdb.write(_atom_df(), str(out))

    atom_lines = [l for l in out.read_text().splitlines() if l.startswith("ATOM")]
    anisou_lines = [l for l in out.read_text().splitlines() if l.startswith("ANISOU")]

    assert _atom_name_field(atom_lines[0]) == "HD11"  # not "D11"/" D11"
    assert _atom_name_field(atom_lines[1]).strip() == "CA"
    # ANISOU record for the same atom must carry the same 4-char name.
    assert len(anisou_lines) == 1
    assert _atom_name_field(anisou_lines[0]) == "HD11"


@pytest.mark.unit
def test_write_multi_model_preserves_4char_name(tmp_path):
    out = tmp_path / "multi.pdb"
    pdb.write_multi_model([_atom_df(), _atom_df()], str(out))

    atom_lines = [l for l in out.read_text().splitlines() if l.startswith("ATOM")]
    assert len(atom_lines) == 4  # 2 atoms x 2 models
    assert all(
        _atom_name_field(l) == "HD11" for l in atom_lines if "HD11" in l or "D11" in l
    )
    assert any(_atom_name_field(l) == "HD11" for l in atom_lines)


@pytest.mark.unit
def test_write_then_read_roundtrips_name(tmp_path):
    out = tmp_path / "rt.pdb"
    pdb.write(_atom_df(), str(out))

    df_back, _cell, _sg = pdb.read(str(out))()
    names = set(df_back["name"].astype(str).str.strip())
    assert "HD11" in names
    assert "CA" in names
