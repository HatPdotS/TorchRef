"""``torchref.strip-altlocs`` on a deposited structure."""

import sys

import pytest

from torchref.cli import strip_altlocs
from torchref.io import pdb

pytestmark = pytest.mark.integration


def test_microheterogeneous_residue_keeps_one_conformer(pdb_dir, tmp_path, monkeypatch):
    """Alternates with different residue names are conformers of one residue.

    1DAW's GLU A40 has altlocs A and B over five side-chain atoms; with the B atoms
    renamed GLN, the output still holds one conformer of A40.
    """
    lines = (pdb_dir / "1DAW.pdb").read_text().splitlines(keepends=True)
    renamed = 0
    for i, line in enumerate(lines):
        if line.startswith("ATOM") and line[16:26] == "BGLU A  40":
            lines[i] = line[:17] + "GLN" + line[20:]
            renamed += 1
    assert renamed == 5
    source, output = tmp_path / "microheterogeneous.pdb", tmp_path / "stripped.pdb"
    source.write_text("".join(lines))
    monkeypatch.setattr(
        sys, "argv", ["torchref.strip-altlocs", str(source), str(output)]
    )

    assert strip_altlocs.main() == 0

    atoms = pdb.load_as_dataframe(str(output))
    residue = atoms[(atoms["chainid"] == "A") & (atoms["resseq"] == 40)]
    assert len(residue) == 9
    assert not residue["name"].duplicated().any()
    assert set(residue["resname"]) == {"GLU"}
    assert len(atoms) == 3046
