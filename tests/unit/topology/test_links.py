"""Covalent links in the bond graph: counted once, and the same from PDB and mmCIF.

A repeated LINK record, or a bond emitted once per altloc conformer between two shared
atoms, used to inflate an atom's graph degree. Hydrogen generation reads that degree as
the number of heavy partners, so a Schiff-base nitrogen listed in two identical LINK
records lost its hydrogen and a CA with a split side chain lost its HA.
"""

import numpy as np
import pytest

from torchref.model.model import Model

# 3E98 chain A, LEU72 - MSE73: the MSE is a HETATM residue, so its peptide bonds come
# only from LINK records.
_LEU_MSE = """\
CRYST1   53.841   88.114   60.963  90.00 107.92  90.00 P 1 21 1      4
{links}
ATOM    206  N   LEU A  72      25.385   1.315  55.882  1.00 51.26           N
ATOM    207  CA  LEU A  72      24.279   2.045  56.497  1.00 52.28           C
ATOM    208  C   LEU A  72      24.644   2.581  57.879  1.00 51.78           C
ATOM    209  O   LEU A  72      24.190   3.667  58.275  1.00 52.62           O
ATOM    210  CB  LEU A  72      23.061   1.126  56.613  1.00 52.98           C
ATOM    211  CG  LEU A  72      21.688   1.747  56.752  1.00 56.56           C
ATOM    212  CD1 LEU A  72      21.410   2.644  55.536  1.00 55.74           C
ATOM    213  CD2 LEU A  72      20.665   0.619  56.876  1.00 54.03           C
HETATM  214  N   MSE A  73      25.451   1.824  58.619  1.00 50.52           N
HETATM  215  CA  MSE A  73      25.888   2.255  59.952  1.00 51.25           C
HETATM  216  C   MSE A  73      26.955   3.368  59.879  1.00 51.87           C
HETATM  217  O   MSE A  73      26.968   4.263  60.731  1.00 52.14           O
HETATM  218  CB  MSE A  73      26.424   1.077  60.754  1.00 50.55           C
HETATM  219  CG  MSE A  73      25.351   0.118  61.312  1.00 56.44           C
HETATM  220 SE   MSE A  73      26.197  -1.503  62.054  0.75 51.93          SE
HETATM  221  CE  MSE A  73      27.068  -0.667  63.543  1.00 59.19           C
END
"""
_LINK = "LINK         C   LEU A  72                 N   MSE A  73     1555   1555  1.33"


def _row(model, chain, resseq, name, altloc=""):
    pdb = model.pdb
    mask = (
        (pdb["chainid"].astype(str) == chain)
        & (pdb["resseq"].astype(int) == resseq)
        & (pdb["name"].astype(str).str.strip() == name)
        & (pdb["altloc"].astype(str).str.strip() == altloc)
    )
    (row,) = np.nonzero(mask.values)[0]
    return int(row)


def _load(path):
    model = Model(verbose=0, add_hydrogens=False, strip_H=True)
    model.load_pdb(str(path)) if str(path).endswith(".pdb") else model.load_cif(str(path))
    model.set_restraints_cif(None)
    return model


@pytest.mark.unit
def test_repeated_link_record_contributes_one_edge(tmp_path):
    """The parser keeps both records; the graph carries one bond and one restraint."""
    path = tmp_path / "dup_link.pdb"
    path.write_text(_LEU_MSE.format(links=_LINK + "\n" + _LINK))
    model = _load(path)
    restraints = model.restraints
    atoms = restraints.topology.atoms

    assert len(restraints.links) == 2
    link_rows = atoms.bonds.origin("link")
    assert len(link_rows) == 1
    n_mse = _row(model, "A", 73, "N")
    assert int(atoms.degree(n_mse)) == 2  # CA and the previous C


@pytest.mark.unit
def test_shared_atom_degree_counts_each_partner_once(pdb_dir):
    """A bond between two blank-altloc atoms is one edge however many conformers emit it."""
    model = _load(pdb_dir / "3E98.pdb")
    atoms = model.restraints.topology.atoms
    # MSE65 has split CA/CB/CG/SE/CE; its C is bonded to CA(A), CA(B), O and ARG66 N.
    assert int(atoms.degree(_row(model, "A", 65, "C"))) == 4

    model = _load(pdb_dir / "3A5V.pdb")
    atoms = model.restraints.topology.atoms
    # CYS53 has a split side chain: N sees C(prev) and CA; CA sees N, C, CB(A), CB(B).
    assert int(atoms.degree(_row(model, "A", 53, "N"))) == 2
    assert int(atoms.degree(_row(model, "A", 53, "CA"))) == 4


def _link_identities(model):
    atoms = model.restraints.topology.atoms
    pdb = model.pdb
    key = lambda i: (
        str(pdb["chainid"].iloc[i]),
        int(pdb["resseq"].iloc[i]),
        str(pdb["icode"].iloc[i]).strip(),
        str(pdb["name"].iloc[i]).strip(),
        str(pdb["altloc"].iloc[i]).strip(),
    )
    return {frozenset((key(int(a)), key(int(b)))) for a, b in atoms.bonds.origin("link")}


@pytest.mark.unit
@pytest.mark.parametrize("code", ["1DAW", "2DQ6", "3A5V", "3E98", "5BOV", "6G9X"])
def test_link_edges_agree_between_pdb_and_cif(pdb_dir, cif_dir, code):
    """mmCIF ``_struct_conn`` yields the link edges the PDB LINK records do."""
    from_pdb = _load(pdb_dir / f"{code}.pdb")
    from_cif = _load(cif_dir / f"{code}.cif")
    assert from_cif.ctx.links is not None and len(from_cif.ctx.links) > 0
    assert _link_identities(from_cif) == _link_identities(from_pdb)
