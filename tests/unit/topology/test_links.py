"""Covalent links in the bond graph: counted once, and the same from PDB and mmCIF.

A repeated LINK record, or a bond emitted once per altloc conformer between two shared
atoms, used to inflate an atom's graph degree. Hydrogen generation reads that degree as
the number of heavy partners, so a Schiff-base nitrogen listed in two identical LINK
records lost its hydrogen and a CA with a split side chain lost its HA.
"""

import numpy as np
import pytest

from torchref.model.model import Model

# 3E98 chain A, LEU72 - MSE73, the MSE a HETATM residue.
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
    model = Model(verbose=0, hydrogens="strip")
    model.load_pdb(str(path)) if str(path).endswith(".pdb") else model.load_cif(str(path))
    model.ctx.set_cif_path(None)
    return model


@pytest.mark.unit
def test_repeated_link_record_contributes_one_edge(tmp_path):
    """The parser keeps both records; the graph carries one bond and one restraint."""
    path = tmp_path / "dup_link.pdb"
    text = _LEU_MSE.format(links=_LINK + "\n" + _LINK)
    # MSE renumbered 75: no peptide link forms, so the C-N bond comes from LINK alone.
    path.write_text(text.replace("MSE A  73", "MSE A  75"))
    model = _load(path)
    restraints = model.restraints
    atoms = restraints.topology.atoms

    assert len(restraints.links) == 2
    link_rows = atoms.bonds.origin("link")
    assert len(link_rows) == 1
    n_mse = _row(model, "A", 75, "N")
    assert int(atoms.degree(n_mse)) == 2  # CA and the previous C


@pytest.mark.unit
def test_link_record_duplicating_a_peptide_bond_is_dropped(tmp_path):
    """The HETATM MSE is peptide-linked, and its LINK record adds no second bond."""
    path = tmp_path / "peptide_link.pdb"
    path.write_text(_LEU_MSE.format(links=_LINK))
    model = _load(path)
    atoms = model.restraints.topology.atoms
    c_leu, n_mse = _row(model, "A", 72, "C"), _row(model, "A", 73, "N")

    assert "link" not in atoms.bonds.origin_bounds
    assert atoms.bonds.origin("peptide").tolist() == [[c_leu, n_mse]]
    assert int(atoms.degree(n_mse)) == 2


# 6G9X chain A, GLU41 and SER45 across a real chain break (C-N 8.0 A), the SER
# renumbered 42 so that the two are numbered consecutively.
_CHAIN_BREAK = """\
CRYST1  106.765  200.539   64.561  90.00  90.00  90.00 P 21 21 2     8
ATOM    264  N   GLU A  41     102.334  20.809  17.251  1.00128.98           N
ATOM    265  CA  GLU A  41     102.508  21.761  16.154  1.00130.11           C
ATOM    266  C   GLU A  41     103.882  22.437  16.164  1.00124.73           C
ATOM    267  O   GLU A  41     104.829  21.942  16.780  1.00119.88           O
ATOM    268  CB  GLU A  41     101.404  22.806  16.190  1.00130.16           C
ATOM    269  CG  GLU A  41     100.029  22.196  15.983  1.00131.42           C
ATOM    270  CD  GLU A  41      98.954  23.222  15.651  1.00135.88           C
ATOM    271  OE1 GLU A  41      99.195  24.444  15.804  1.00137.19           O
ATOM    272  OE2 GLU A  41      97.856  22.798  15.231  1.00136.18           O
ATOM    273  N   SER A  42     109.145  23.023  22.168  1.00100.36           N
ATOM    274  CA  SER A  42     109.791  22.385  23.315  1.00 96.39           C
ATOM    275  C   SER A  42     108.891  22.403  24.550  1.00103.70           C
ATOM    276  O   SER A  42     108.166  23.370  24.782  1.00105.34           O
ATOM    277  CB  SER A  42     111.120  23.073  23.634  1.00 99.98           C
ATOM    278  OG  SER A  42     111.609  22.683  24.910  1.00100.07           O
END
"""


@pytest.mark.unit
def test_consecutive_numbers_across_a_chain_break_get_no_peptide_link(tmp_path):
    """Sequence-adjacent numbering is not enough: the C-N distance must be a bond."""
    path = tmp_path / "chain_break.pdb"
    path.write_text(_CHAIN_BREAK)
    model = _load(path)
    topology = model.restraints.topology

    assert len(topology.residues.links_of_kind("TRANS")) == 0
    assert "peptide" not in topology.atoms.bonds.origin_bounds
    assert int(topology.atoms.degree(_row(model, "A", 42, "N"))) == 1
    assert not topology.is_polymer.any()


@pytest.mark.unit
def test_hetatm_amino_acid_is_peptide_linked(pdb_dir):
    """3E98's MSE65 is a HETATM residue, linked like any other.

    GLU64-MSE65-ARG66 carry peptide bonds, angles, planes, omega, phi/psi and the
    Ramachandran pair; MSE65 is patched for both links and counts as polymer, and the
    LINK records for those two bonds add nothing.
    """
    model = _load(pdb_dir / "3E98.pdb")
    restraints = model.restraints
    entries = restraints.restraints
    topology = restraints.topology
    c64, n65 = _row(model, "A", 64, "C"), _row(model, "A", 65, "N")
    c65, n66 = _row(model, "A", 65, "C"), _row(model, "A", 66, "N")

    bonds = {tuple(row) for row in entries["bond"]["peptide"]["indices"].tolist()}
    assert {(c64, n65), (c65, n66)} <= bonds
    assert "link" not in topology.atoms.bonds.origin_bounds
    angles = entries["angle"]["peptide"]["indices"].tolist()
    omega = entries["torsion"]["omega"]["indices"].tolist()
    planes = entries["plane"]["4_atoms"]["indices"].tolist()
    for c, n in ((c64, n65), (c65, n66)):
        assert sum({c, n} <= set(row) for row in angles) >= 3
        assert any(row[1:3] == [c, n] for row in omega)
        assert any({c, n} <= set(row) for row in planes)

    phi = entries["torsion"]["phi"]["indices"].tolist()
    psi = entries["torsion"]["psi"]["indices"].tolist()
    assert any(row[:2] == [c64, n65] for row in phi)
    assert any(row[0] == n65 and row[3] == n66 for row in psi)
    rama_phi = restraints._rama_phi_indices.tolist()
    assert any(row[:2] == [c64, n65] for row in rama_phi)

    mse65 = int(topology.atoms.residue_of[n65])
    assert topology.residues.template_key[mse65] == "MSE:DEL-HN1+DEL-OXT"
    assert topology.is_polymer[n65]


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
    if "link" not in atoms.bonds.origin_bounds:
        return set()
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


def _restraint_groups(restraints):
    """``(name, group)`` for every restraint group, ``all`` excluded."""
    entries = restraints.restraints
    groups = [
        (f"{edge_type}/{origin}", group)
        for edge_type in ("bond", "angle", "torsion")
        for origin, group in entries[edge_type].items()
        if origin != "all"
    ]
    groups.append(("chiral", entries["chiral"]))
    groups += [(f"plane/{key}", group) for key, group in entries["plane"].items()]
    return groups


@pytest.mark.unit
@pytest.mark.parametrize("code", ["3A5V", "7L84"])
def test_altloc_conformers_emit_shared_restraints_once(pdb_dir, code):
    """A restraint over shared atoms is emitted once; none joins two conformations.

    Hydrogens are kept: 7L84's split residues share their backbone hydrogens.
    """
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / f"{code}.pdb"))
    restraints = model.restraints
    altloc = restraints.topology.atoms.altloc

    for name, group in _restraint_groups(restraints):
        n_rows = len(group["indices"])
        columns = [group[prop].reshape(n_rows, -1).tolist() for prop in sorted(group)]
        keys = [tuple(tuple(column[i]) for column in columns) for i in range(n_rows)]
        assert len(set(keys)) == n_rows, f"{name}: {n_rows - len(set(keys))} repeats"

        labels = [set(altloc[row]) - {" "} for row in group["indices"].tolist()]
        assert all(len(found) < 2 for found in labels), name


@pytest.mark.unit
def test_split_sg_disulfide_restrains_each_conformer(pdb_dir):
    """3A5V CYS53 has two SG conformers, each bonded to CYS21's SG.

    Each bond carries its own two CB-SG-SG angles and CB-SG-SG-CB torsion, with CB
    from that SG's conformer, and the CB(B)..SG21 1-3 pair takes no repulsion.
    """
    model = _load(pdb_dir / "3A5V.pdb")
    restraints = model.restraints
    cb21, sg21 = _row(model, "A", 21, "CB"), _row(model, "A", 21, "SG")

    for alt in ("A", "B"):
        cb53, sg53 = _row(model, "A", 53, "CB", alt), _row(model, "A", 53, "SG", alt)
        for edge_type, expected in (
            ("angle", {(cb21, sg21, sg53), (sg21, sg53, cb53)}),
            ("torsion", {(cb21, sg21, sg53, cb53)}),
        ):
            rows = restraints.restraints[edge_type]["disulfide"]["indices"].tolist()
            found = [tuple(row) for row in rows if {sg21, sg53} <= set(row)]
            assert sorted(found) == sorted(expected), (edge_type, alt)

    cb53b = _row(model, "A", 53, "CB", "B")
    vdw = restraints.restraints["vdw"]["indices"].tolist()
    assert not any({a, b} == {cb53b, sg21} for a, b in vdw)
