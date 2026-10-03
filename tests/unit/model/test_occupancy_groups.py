"""Occupancy grouping on load, and its round trip through a checkpoint.

A sharing group never spans two residues, residues are ``(chain, resseq, icode)`` so
100 and 100A are two, and loading keeps every deposited occupancy except where the
atoms of one altloc conformer disagree (a conformer is one group by design). A
checkpoint restores the grouping it was saved with, whatever the occupancies have
become.
"""

import numpy as np
import pytest
import torch

from torchref.io.pdb import PDBReader
from torchref.model.model import Model

pytestmark = pytest.mark.unit

GLY = ("N", "CA", "C", "O")
MET = ("N", "CA", "C", "O", "CB", "CG", "SD", "CE")
THR = ("N", "CA", "C", "O", "CB", "OG1", "CG2")
ALA = ("N", "CA", "C", "O", "CB")

#: ``(resname, resseq, icode, [(atom name, altloc, occupancy), ...])``, chain A.
INSERTION_CODES = [
    ("GLY", 99, "", [(n, " ", 1.0) for n in GLY]),
    ("GLY", 100, "", [(n, "A", 0.7) for n in GLY] + [(n, "B", 0.3) for n in GLY]),
    ("GLY", 100, "A", [(n, "A", 0.4) for n in GLY] + [(n, "B", 0.6) for n in GLY]),
    ("GLY", 101, "", [(n, " ", 0.8) for n in GLY]),
    ("GLY", 101, "A", [(n, " ", 0.5) for n in GLY]),
    ("MET", 102, "", [(n, " ", 0.38 if n == "SD" else 1.0) for n in MET]),
]

#: THR and ALA alternates at one position, between two glycines.
MICROHETEROGENEITY = [
    ("GLY", 29, "", [(n, " ", 1.0) for n in GLY]),
    ("THR", 30, "", [(n, "A", 0.35) for n in THR]),
    ("ALA", 30, "", [(n, "B", 0.65) for n in ALA]),
    ("GLY", 31, "", [(n, " ", 1.0) for n in GLY]),
]

STRUCTURES = ["3E98", "5BOV", "6G9X"]


def _write_pdb(path, residues):
    """Write ``residues`` as a P1 PDB file and return the deposited occupancies."""
    lines = ["CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1           1"]
    occupancies = []
    for resname, resseq, icode, atoms in residues:
        for name, altloc, occupancy in atoms:
            serial = len(occupancies) + 1
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s}{altloc}{resname} A{resseq:4d}{icode:1s}"
                f"   {0.8 * serial:8.3f}{5.0:8.3f}{5.0:8.3f}{occupancy:6.2f}{20.0:6.2f}"
                f"          {name[0]:>2s}"
            )
            occupancies.append(occupancy)
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return np.asarray(occupancies)


def _load(path):
    return Model(verbose=0, device="cpu").load_pdb(str(path))


def _residues_per_group(model):
    """Number of distinct topology residues in each sharing group."""
    groups = model.occupancy.expansion_mask.cpu().numpy()
    residue = model.ctx.topology.atoms.residue_of.cpu().numpy()
    pairs = np.unique(np.stack([groups, residue]), axis=1)
    return np.bincount(pairs[0])


def _deposited(path):
    """The reader's table with the rows a model drops removed, and its occupancies."""
    table, _, _ = PDBReader(verbose=0).read(str(path))()
    table = table.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"])
    table = table.reset_index(drop=True)
    return table, table["occupancy"].clip(0, 1).to_numpy()


def _disagreeing_conformers(table):
    """Row lists of altloc conformers whose deposited occupancies are not uniform."""
    altloc = table["altloc"].astype(str).str.strip()
    out = []
    for _, residue in table[altloc != ""].groupby(["chainid", "resseq", "icode"]):
        if residue["altloc"].nunique() < 2:
            continue
        for _, conformer in residue.groupby("altloc"):
            if conformer["occupancy"].nunique() > 1:
                out.append(conformer.index.to_numpy())
    return out


def test_insertion_codes_and_ungrouped_atoms_keep_their_occupancies(tmp_path):
    """100/100A stay apart, and atoms left ungrouped never join another residue."""
    deposited = _write_pdb(tmp_path / "icode.pdb", INSERTION_CODES)
    model = _load(tmp_path / "icode.pdb")

    np.testing.assert_allclose(model.occupancy().detach().numpy(), deposited, atol=1e-5)
    assert (_residues_per_group(model) == 1).all()
    # 100 and 100A: two conformers each; 99, 101 and 101A: one group each; MET 102's
    # atoms disagree, so one group per atom.
    assert model.occupancy.collapsed_shape == (2 + 2 + 3 + len(MET),)
    pairs = [[rows.tolist() for rows in pair] for pair in model.ctx.altloc_pairs]
    assert pairs == [
        [[4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19]],
    ]


def test_microheterogeneous_alternates_are_one_residues_conformers(tmp_path):
    """THR A and ALA B at one position are linked conformers that sum to 1."""
    deposited = _write_pdb(tmp_path / "micro.pdb", MICROHETEROGENEITY)
    model = _load(tmp_path / "micro.pdb")

    ((residue, labels, conformers),) = model.ctx.altloc_residues()
    assert model.ctx.topology.residues.key(residue) == ("A", 30, "")
    assert labels == ["A", "B"]
    assert (len(conformers["A"]), len(conformers["B"])) == (len(THR), len(ALA))
    np.testing.assert_allclose(model.occupancy().detach().numpy(), deposited, atol=1e-5)

    model.occupancy[torch.tensor(conformers["B"])] = 0.9
    occupancy = model.occupancy().detach()
    total = occupancy[conformers["A"][0]] + occupancy[conformers["B"][0]]
    assert total.item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("code", STRUCTURES)
def test_loading_keeps_every_deposited_occupancy(pdb_dir, code):
    """Values survive the load; only a disagreeing conformer collapses to one value."""
    table, deposited = _deposited(pdb_dir / f"{code}.pdb")
    model = _load(pdb_dir / f"{code}.pdb")
    loaded = model.occupancy().detach().numpy()

    assert (_residues_per_group(model) == 1).all()
    shared = np.zeros(len(table), dtype=bool)
    for rows in _disagreeing_conformers(table):
        shared[rows] = True
        assert np.ptp(loaded[rows]) < 1e-6
    np.testing.assert_allclose(loaded[~shared], deposited[~shared], atol=1e-5)


@pytest.mark.parametrize("code", STRUCTURES)
def test_checkpoint_round_trip_keeps_grouping_and_values(pdb_dir, code, tmp_path):
    """``create_from_state_dict``, and ``load_state`` into an empty model, restore
    groups and values."""
    model = _load(pdb_dir / f"{code}.pdb")
    path = tmp_path / "model.pt"
    model.save_state(str(path))
    from_file = Model(verbose=0, device="cpu")
    from_file.load_state(str(path))
    from_dict = Model.create_from_state_dict(
        model.state_dict(), device="cpu", verbose=0
    )

    for restored in (from_dict, from_file):
        assert torch.equal(
            restored.occupancy.expansion_mask, model.occupancy.expansion_mask
        )
        assert torch.equal(
            restored.occupancy.refinable_mask, model.occupancy.refinable_mask
        )
        assert torch.equal(restored.occupancy(), model.occupancy())


def test_checkpoint_restores_the_saved_grouping_after_values_move(pdb_dir):
    """Occupancies moved across the sharing deadband restore into the saved groups.

    With every occupancy at 1.0 a fresh load would pool far more atoms than the
    saved groups hold.
    """
    model = _load(pdb_dir / "6G9X.pdb")
    model.occupancy[:] = 1.0

    restored = Model.create_from_state_dict(model.state_dict(), device="cpu", verbose=0)

    assert torch.equal(
        restored.occupancy.expansion_mask, model.occupancy.expansion_mask
    )
    assert torch.equal(restored.occupancy(), model.occupancy())
    assert (
        restored.occupancy.get_refinable_count()
        == model.occupancy.get_refinable_count()
    )
