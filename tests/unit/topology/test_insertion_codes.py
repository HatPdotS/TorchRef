"""Residues distinguished only by an insertion code.

A deposited structure may number two residues 100 and 100A. They are different residues
with different chemistry, and the only thing separating them is the insertion code. The
topology keys residues on ``(chain, resseq, icode)`` for that reason, and every builder
-- intra-residue matching and the peptide links alike -- works on those residues, so
each inserted residue gets its own geometry and the chain runs through the insertion.

No bundled structure has an insertion code, so the case is synthesised here rather
than shipped as another data file: the rewrite is then visible, and it is obvious that
nothing but the numbering changed.
"""

import pytest

from torchref.model.model import Model
from torchref.topology import build_topology
from torchref.topology.topology import Topology

#: Base structure: chain A, no altlocs, no insertion codes anywhere.
BASE = "3GR5"

#: The three consecutive residues collapsed onto one sequence number. Their real
#: identities differ (SER, LEU, GLU), so the restraints of the second and third are
#: distinguishable from the first's rather than being duplicates of it.
STRETCH = (23, 24, 25)


def _rewrite_with_insertion_codes(source, destination):
    """Copy a PDB, renumbering ``STRETCH`` as ``N``, ``NA``, ``NB``.

    Only columns 23-27 change -- the sequence number and the insertion code. Every
    atom, coordinate and residue name is untouched, and the residues stay in file order,
    so they remain contiguous exactly as a real insertion would be.

    Returns
    -------
    tuple of tuple
        The ``(resseq, icode)`` pairs written, in order.
    """
    first = STRETCH[0]
    codes = ["", "A", "B"]
    mapping = {old: (first, codes[i]) for i, old in enumerate(STRETCH)}

    out = []
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            resseq = int(line[22:26])
            if resseq in mapping:
                new_seq, icode = mapping[resseq]
                line = f"{line[:22]}{new_seq:>4d}{icode:1s}{line[27:]}"
        out.append(line)
    destination.write_text("".join(out))
    return tuple((first, code) for code in codes)


@pytest.fixture(scope="module")
def inserted(pdb_dir, tmp_path_factory):
    """``(topology, restraints, expected keys, model)`` for the rewritten file."""
    path = tmp_path_factory.mktemp("icode") / f"{BASE}_icode.pdb"
    expected = _rewrite_with_insertion_codes(pdb_dir / f"{BASE}.pdb", path)

    model = Model(verbose=0, hydrogens="strip")
    model.load_pdb(str(path))
    model.ctx.set_cif_path(None)
    restraints = model.restraints

    topology = build_topology(
        Topology.from_table(model.pdb),
        restraints.cif_dict,
        model.xyz().detach(),
        link_dict=restraints.link_dict,
        link_list=restraints.link_list,
        links=restraints.links,
        verbose=0,
    )
    return topology, restraints, expected, model


def _tuples(indices):
    return {tuple(int(v) for v in row) for row in indices.cpu().numpy()}


@pytest.mark.unit
def test_the_rewrite_actually_produced_insertion_codes(inserted):
    """Guard the fixture: if the rewrite silently failed the rest proves nothing."""
    _, _, _, model = inserted
    icodes = model.pdb["icode"].astype(str).str.strip().values
    assert set(icodes[icodes != ""]) == {"A", "B"}


@pytest.mark.unit
def test_the_graph_keeps_them_apart(inserted):
    """Three residue nodes, one per insertion code."""
    topology, _, expected, _ = inserted
    residues = topology.residues

    found = [
        residues.key(i)
        for i in range(residues.n_residues)
        if (int(residues.resseq[i]), str(residues.icode[i]).strip())
        in {(seq, code) for seq, code in expected}
    ]
    assert len(found) == 3, f"expected three inserted residues, found {found}"
    assert len({key[2] for key in found}) == 3, "insertion codes were not distinguished"


def _inserted_residue_indices(topology, expected):
    wanted = {(seq, code) for seq, code in expected}
    return [
        i
        for i in range(topology.n_residues)
        if (int(topology.residues.resseq[i]), str(topology.residues.icode[i]).strip())
        in wanted
    ]


def _bonds_within(edges, topology, residue):
    start = int(topology.residues.atom_start[residue])
    end = int(topology.residues.atom_end[residue])
    return {e for e in edges if all(start <= int(a) < end for a in e)}


@pytest.mark.unit
def test_the_inserted_residues_get_their_own_intra_restraints(inserted):
    """Each of the three carries bonds of its own, not just the first.

    Under the merged grouping only the first residue's template matched, so the second
    and third had no intra-residue geometry at all.
    """
    topology, _, expected, _ = inserted

    inserted_residues = [
        i
        for i in range(topology.n_residues)
        if (
            int(topology.residues.resseq[i]),
            str(topology.residues.icode[i]).strip(),
        )
        in {(seq, code) for seq, code in expected}
    ]

    intra = topology.atoms.bonds.origin("intra").cpu().numpy()
    for residue in inserted_residues:
        start = int(topology.residues.atom_start[residue])
        end = int(topology.residues.atom_end[residue])
        own = [
            row
            for row in intra
            if start <= int(row[0]) < end and start <= int(row[1]) < end
        ]
        assert own, (
            f"residue {topology.residues.key(residue)} "
            f"({topology.residues.resname[residue]}) has no intra-residue bonds"
        )


@pytest.mark.unit
def test_the_inserted_stretch_is_peptide_linked(inserted):
    """An insertion-code step is a sequence step, so the chain is not broken.

    ``find_peptide_links`` allows a ``resseq`` difference of 0 precisely for this: 100
    to 100A is consecutive. Without it the inserted residues would float free of the
    chain.
    """
    topology, _, expected, _ = inserted

    inserted_residues = {
        i
        for i in range(topology.n_residues)
        if (
            int(topology.residues.resseq[i]),
            str(topology.residues.icode[i]).strip(),
        )
        in {(seq, code) for seq, code in expected}
    }

    links = topology.residues.links_of_kind("TRANS")
    internal = [
        pair
        for pair in links
        if int(pair[0]) in inserted_residues and int(pair[1]) in inserted_residues
    ]
    assert len(internal) == 2, (
        f"expected two peptide links inside the three inserted residues, got "
        f"{len(internal)}"
    )


def _atom(topology, residue, name):
    rows = topology.residues.atom_rows(residue)
    return next(r for r in rows if str(topology.atoms.name[r]).strip() == name)


@pytest.mark.unit
def test_peptide_edges_run_through_the_insertion(inserted):
    """C(i)-N(i+1) bonds, and the phi/psi torsions, exist at every insertion-code step.

    The peptide-link builders pair residues along the residue graph's links, so 23 to
    23A and 23A to 23B are linked like any other step, and the middle residue gets
    both its phi and its psi.
    """
    topology, _, expected, _ = inserted
    residues = _inserted_residue_indices(topology, expected)
    peptide = topology.atoms.bonds.tuple_set("peptide")
    phi = topology.atoms.torsions.tuple_set("phi")
    psi = topology.atoms.torsions.tuple_set("psi")

    for first, second in zip(residues, residues[1:]):
        c, n = _atom(topology, first, "C"), _atom(topology, second, "N")
        assert (min(c, n), max(c, n)) in {tuple(sorted(e)) for e in peptide}, (
            f"no peptide bond from {topology.residues.key(first)} to "
            f"{topology.residues.key(second)}"
        )
    middle = residues[1]
    assert any(row[1] == _atom(topology, middle, "N") for row in phi)
    assert any(row[0] == _atom(topology, middle, "N") for row in psi)
