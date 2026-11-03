"""C-beta dihedral restraints: two per L amino acid, Phenix's targets and sign.

The ``cbeta`` torsion origin holds ``N-C-CA-CB`` and ``C-N-CA-CB`` for each conformer
with all four atoms, and is part of the ``all`` group the torsion target reads.
"""

import numpy as np
import pytest
import torch

from torchref.model.model import Model

pytestmark = pytest.mark.unit


def _deposited(path):
    model = Model(verbose=0)
    model.load_pdb(str(path))
    return model.xyz().detach(), model.restraints


def _names(restraints, rows):
    names = np.asarray(restraints.topology.columns()["name"]).astype(str)
    return names[rows.cpu().numpy()]


def test_two_cbeta_dihedrals_per_amino_acid(pdb_dir):
    """Every residue with N, CA, C and CB gets both dihedrals; glycine gets none."""
    _, restraints = _deposited(pdb_dir / "1DAW.pdb")
    group = restraints.restraints["torsion"]["cbeta"]
    names = _names(restraints, group["indices"])
    assert {tuple(row) for row in names} == {
        ("N", "C", "CA", "CB"),
        ("C", "N", "CA", "CB"),
    }

    columns = restraints.topology.columns()
    resname = np.asarray(columns["resname"]).astype(str)
    is_cb = np.asarray(columns["name"]).astype(str) == "CB"
    assert len(group["indices"]) == 2 * int(is_cb.sum())
    assert "GLY" not in set(resname[group["indices"][:, 3].cpu().numpy()])

    owner = resname[group["indices"][:, 3].cpu().numpy()]
    branched = np.isin(owner, ["VAL", "THR", "ILE"])
    n_first = names[:, 0] == "N"
    refs = group["references"].cpu().numpy()
    assert np.allclose(refs[branched & n_first], 123.4)
    assert np.allclose(refs[branched & ~n_first], -122.0)
    assert torch.all(group["sigmas"] == 2.5)
    assert torch.all(group["periods"] == 1)


def test_cbeta_rows_are_in_the_torsion_target_group(pdb_dir):
    """The torsion ``all`` group carries the ``cbeta`` rows alongside ``intra``."""
    _, restraints = _deposited(pdb_dir / "1DAW.pdb")
    torsion = restraints.restraints["torsion"]
    members = [o for o in ("intra", "disulfide", "cbeta") if o in torsion]
    n = sum(len(torsion[o]["indices"]) for o in members)
    assert len(torsion["all"]["indices"]) == n


def test_phenix_refined_model_sits_on_the_cbeta_targets(pdb_dir):
    """6G9X was refined with Phenix's C-beta restraints: rms deviation under σ."""
    xyz, restraints = _deposited(pdb_dir / "6G9X.pdb")
    group = restraints.restraints["torsion"]["cbeta"]
    measured = restraints.torsions(group["indices"], xyz)
    delta = torch.remainder(measured - group["references"] + 180.0, 360.0) - 180.0
    assert float(delta.square().mean().sqrt()) < 2.0


def test_inverted_ca_gets_no_cbeta_dihedrals(pdb_dir, tmp_path):
    """A CB mirrored through N-CA-C is D-handed and left to the chiral term."""
    lines = (pdb_dir / "1DAW.pdb").read_text().splitlines()
    arg19 = {
        line[12:16]: i
        for i, line in enumerate(lines)
        if line.startswith("ATOM") and line[17:26] == "ARG A  19"
    }
    pos = {
        k: np.array([float(lines[arg19[k]][c : c + 8]) for c in (30, 38, 46)])
        for k in (" N  ", " CA ", " C  ", " CB ")
    }
    normal = np.cross(pos[" N  "] - pos[" CA "], pos[" C  "] - pos[" CA "])
    normal /= np.linalg.norm(normal)
    cb = pos[" CB "] - 2.0 * np.dot(pos[" CB "] - pos[" CA "], normal) * normal
    row = arg19[" CB "]
    lines[row] = lines[row][:30] + "".join(f"{v:8.3f}" for v in cb) + lines[row][54:]
    path = tmp_path / "inverted.pdb"
    path.write_text("\n".join(lines) + "\n")

    _, restraints = _deposited(path)
    columns = restraints.topology.columns()
    (cb_row,) = np.flatnonzero(
        (columns["chain"] == "A")
        & (columns["resseq"] == 19)
        & (columns["name"] == "CB")
    )
    cbeta = restraints.restraints["torsion"]["cbeta"]["indices"]
    assert not bool((cbeta[:, 3] == int(cb_row)).any())
