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


@pytest.fixture
def microheterogeneous_model(table):
    """Deposited THR A30 with a higher-occupancy ALA alternate at the same position."""
    df, cell, sg = table
    rows = _residue_rows(df, 30)
    other = df.loc[rows].copy()
    other = other[other.name.str.strip().isin(["N", "CA", "C", "O", "CB"])].copy()
    assert len(other) == 5
    other[["altloc", "occupancy", "resname"]] = ["B", 0.65, "ALA"]
    df = df.copy()
    df.loc[rows, ["altloc", "occupancy"]] = ["A", 0.35]
    df = pd.concat([df.loc[: rows[-1]], other, df.loc[rows[-1] + 1 :]]).reset_index(
        drop=True
    )
    return _model(df, cell, sg)


def test_microheterogeneity_identity_and_current_occupancy(microheterogeneous_model):
    """Each alternate retains its chemical name and selection through model derivation."""
    model = microheterogeneous_model
    query = "chain A and resseq 30 and resname ALA"
    assert model.get_selection_mask(query).sum().item() == 5
    cols = model.ctx.topology.columns()
    at_position = (cols["chain"] == "A") & (cols["resseq"] == 30)
    assert len(set(model.ctx.topology.atoms.residue_of[at_position].tolist())) == 1
    assert (cols["resname"][at_position & (cols["altloc"] == "B")] == "ALA").all()
    groups = model.ctx._residue_groups(with_altloc=True)
    assert len(groups[("ALA", 30, "A", "B")]) == 5
    assert len(groups[("THR", 30, "A", "A")]) > 5
    np.testing.assert_allclose(
        model.occupancy().detach().numpy()[at_position & (cols["altloc"] == "B")], 0.65
    )
    for derived in (
        model.copy(),
        Model.create_from_state_dict(model.state_dict(), device="cpu", verbose=0),
    ):
        assert derived.get_selection_mask(query).sum().item() == 5
    selected = model.select(query)
    assert selected.n_atoms == 5
    assert (selected.to_dataframe().resname == "ALA").all()
    stripped = model.strip_altlocs().to_dataframe()
    kept = stripped[(stripped.chainid == "A") & (stripped.resseq == 30)]
    assert len(kept) == 5
    assert (kept.resname == "ALA").all()
    assert (kept.altloc.str.strip() == "").all()


def test_microheterogeneity_restraint_templates(microheterogeneous_model):
    """The ALA alternate has its own methyl template, independently of THR."""
    model = microheterogeneous_model
    connected = model.restraints.topology
    cols = connected.columns()
    for identity, altloc, energy, h_count in (
        ("THR", "A", "CH1", 1),
        ("ALA", "B", "CH3", 3),
    ):
        cb = np.nonzero(
            (cols["chain"] == "A")
            & (cols["resseq"] == 30)
            & (cols["altloc"] == altloc)
            & (cols["name"] == "CB")
        )[0]
        assert len(cb) == 1
        row = cb[0]
        assert connected.resname_of_atom(row) == identity
        assert connected.atoms.energy_type[row] == energy
        assert connected.atoms.template_h_count[row].item() == h_count
    b_rows = set(
        np.nonzero(
            (cols["chain"] == "A") & (cols["resseq"] == 30) & (cols["altloc"] == "B")
        )[0]
    )
    bonds = connected.atoms.bonds.indices.cpu().numpy()
    own_bonds = {
        tuple(sorted(cols["name"][[a, b]]))
        for a, b in bonds
        if a in b_rows and b in b_rows
    }
    assert own_bonds == {
        tuple(sorted(pair))
        for pair in [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")]
    }
    assert (cols["resname"][list(b_rows)] == "ALA").all()


@pytest.mark.parametrize("suffix", ["pdb", "cif"])
def test_microheterogeneity_writers(microheterogeneous_model, tmp_path, suffix):
    """Coordinate writers retain both chemical identities and the winning identity."""
    import gemmi

    model = microheterogeneous_model
    for current, stripped in ((model, False), (model.strip_altlocs(), True)):
        path = tmp_path / (("stripped" if stripped else "alternates") + "." + suffix)
        getattr(current, "write_" + suffix)(str(path))
        structure = gemmi.read_structure(str(path))
        rows = [
            (res.name, atom.altloc)
            for chain in structure[0]
            if chain.name == "A"
            for res in chain
            if res.seqid.num == 30
            for atom in res
        ]
        if stripped:
            assert len(rows) == 5 and all(name == "ALA" for name, _ in rows)
        else:
            assert sum(name == "ALA" and alt == "B" for name, alt in rows) == 5
            assert any(name == "THR" and alt == "A" for name, alt in rows)


def test_microheterogeneity_shared_atoms_take_winning_name(microheterogeneous_model):
    """Shared backbone atoms become part of the retained chemical residue on stripping."""
    model = microheterogeneous_model
    df = model.to_dataframe()
    at_position = (df.chainid == "A") & (df.resseq == 30)
    backbone = df.name.str.strip().isin(["N", "CA", "C", "O"])
    df.loc[at_position & backbone & (df.altloc == "A"), ["altloc", "occupancy"]] = [
        "",
        1.0,
    ]
    df = df[~(at_position & backbone & (df.altloc == "B"))].reset_index(drop=True)
    shared = _model(df, np.asarray(model.cell.data.cpu()), model.spacegroup.hm)
    kept = shared.strip_altlocs().to_dataframe()
    kept = kept[(kept.chainid == "A") & (kept.resseq == 30)]
    assert len(kept) == 5
    assert (kept.resname == "ALA").all()
    connected = shared.restraints.topology
    columns = connected.columns()
    cb_b = np.nonzero(
        (columns["chain"] == "A")
        & (columns["resseq"] == 30)
        & (columns["altloc"] == "B")
        & (columns["name"] == "CB")
    )[0][0]
    neighbours = connected.atoms.neighbors(int(cb_b)).tolist()
    assert [columns["name"][row] for row in neighbours] == ["CA"]
