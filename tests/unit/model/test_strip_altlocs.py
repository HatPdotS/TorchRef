"""``strip_altlocs`` compares conformers within one residue, insertion code included.

Residues 100 and 100A are different residues even when both carry an altloc label, so
neither may be dropped as the other's losing conformer. Within one residue the
conformer with the higher occupancy survives, whatever residue name it carries.
"""

import numpy as np
import pandas as pd
import pytest

from torchref.io.pdb import PDBReader
from torchref.model.model import Model


@pytest.fixture(scope="module")
def table(pdb_dir):
    df, cell, sg = PDBReader(verbose=0).read(str(pdb_dir / "1DAW.pdb"))()
    return df, np.asarray(cell), sg


def _model(df, cell, sg):
    return Model(verbose=0, device="cpu").load(lambda: (df, cell, sg))


@pytest.fixture(scope="module")
def baseline(table):
    """Atoms left after stripping 1DAW as deposited; it has altlocs of its own."""
    return _model(*table).strip_altlocs().n_atoms


def _residue_rows(df, resseq):
    return df.index[(df.chainid == "A") & (df.resseq == resseq)]


@pytest.mark.unit
def test_insertion_coded_residues_are_not_conformers_of_each_other(table, baseline):
    """100 (altloc A) and 100A (altloc B) are two residues; both survive.

    Same residue name on purpose: only the insertion code tells them apart.
    """
    df, cell, sg = table
    df = df.copy()
    first, second = _residue_rows(df, 10), _residue_rows(df, 11)
    df.loc[first, ["altloc", "occupancy"]] = ["A", 0.6]
    df.loc[second, ["resseq", "icode", "altloc", "occupancy", "resname"]] = [
        10,
        "A",
        "B",
        0.4,
        df.loc[first[0], "resname"],
    ]

    model = _model(df, cell, sg)
    stripped = model.strip_altlocs()

    assert stripped.n_atoms == baseline
    assert (stripped.ctx.topology.atoms.altloc == " ").all()
    residues = stripped.ctx.topology.residues
    keys = {residues.key(r) for r in range(residues.n_residues)}
    assert {("A", 10, ""), ("A", 10, "A")} <= keys


@pytest.mark.unit
def test_the_higher_occupancy_conformer_survives(table, baseline):
    """A real two-conformer residue keeps its better conformer and its shared atoms."""
    df, cell, sg = table
    rows = _residue_rows(df, 20)
    side = rows[~df.loc[rows, "name"].isin(["N", "CA", "C", "O"]).to_numpy()]
    minor = df.loc[side].copy()
    minor[["altloc", "occupancy"]] = ["B", 0.3]
    minor[["x", "y", "z"]] += 0.5
    df = df.copy()
    df.loc[side, ["altloc", "occupancy"]] = ["A", 0.7]
    df = pd.concat([df.loc[: rows[-1]], minor, df.loc[rows[-1] + 1 :]]).reset_index(
        drop=True
    )

    model = _model(df, cell, sg)
    stripped = model.strip_altlocs()

    assert model.n_atoms > baseline
    assert stripped.n_atoms == baseline
    kept = stripped.to_dataframe()
    kept = kept[(kept.chainid == "A") & (kept.resseq == 20)]
    original = df[(df.chainid == "A") & (df.resseq == 20) & (df.altloc != "B")]
    np.testing.assert_allclose(
        kept[["x", "y", "z"]].to_numpy(),
        original[["x", "y", "z"]].to_numpy(),
        atol=1e-4,
    )


@pytest.mark.unit
def test_microheterogeneity_keeps_one_residue(table):
    """Alternates with different residue names at one position are alternates too."""
    df, cell, sg = table
    rows = _residue_rows(df, 30)
    other = df.loc[rows].copy()
    other[["altloc", "occupancy", "resname"]] = ["B", 0.35, "XAA"]
    df = df.copy()
    df.loc[rows, ["altloc", "occupancy"]] = ["A", 0.65]
    df = pd.concat([df.loc[: rows[-1]], other, df.loc[rows[-1] + 1 :]]).reset_index(
        drop=True
    )

    stripped = _model(df, cell, sg).strip_altlocs()

    kept = stripped.to_dataframe()
    kept = kept[(kept.chainid == "A") & (kept.resseq == 30)]
    assert len(kept) == len(rows)
    assert (kept.resname != "XAA").all()
