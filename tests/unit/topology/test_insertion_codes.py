"""Residues distinguished only by an insertion code.

A deposited structure may number two residues 100 and 100A. They are different residues
with different chemistry, and the only thing separating them is the insertion code. The
topology keys residues on ``(chain, resseq, icode)`` for that reason; the restraint
builders key on ``(chain, resseq)`` alone, which merges them into one residue whose
atom names then collide, so the name-to-index map keeps the first of each and every
restraint belonging to the later residues is silently lost.

No bundled structure has an insertion code, so the case is synthesised here rather
than shipped as another data file: the rewrite is then visible, and it is obvious that
nothing but the numbering changed.
"""

import pytest

from torchref.model.model import Model
from torchref.topology import build_topology

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

    model = Model(verbose=0, strip_H=True, add_hydrogens=False)
    model.load_pdb(str(path))
    model.set_restraints_cif(None)
    restraints = model.restraints

    topology = build_topology(
        model.pdb,
        restraints.cif_dict,
        link_dict=restraints.link_dict,
        link_list=restraints.link_list,
        links=restraints.links,
        xyz=model.xyz().detach(),
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


@pytest.mark.unit
def test_the_builders_merge_them(inserted):
    """The comparison only means something if the old grouping really does merge.

    ``PreprocessedPDB`` groups on ``(chain, resseq)``, so the three residues become one
    with three sets of backbone atom names.
    """
    from torchref.topology.builders import PreprocessedPDB

    _, _, expected, model = inserted
    preprocessed = PreprocessedPDB(model.pdb)

    merged = [
        i
        for i in range(preprocessed.n_residues)
        if int(preprocessed.residue_resseqs[i]) == expected[0][0]
    ]
    assert len(merged) == 1, "the builders did not merge the inserted residues"
    assert preprocessed.has_duplicate_atoms(merged[0]), (
        "the merged residue should carry duplicate atom names, which is what makes the "
        "name-to-index map lose the later residues"
    )


def _legacy_intra_bonds(model, restraints):
    """Intra-residue bonds as the ``(chain, resseq)``-keyed builder produces them.

    ``restraints.py`` now builds from the topology, so it cannot serve as the
    baseline -- it *is* the graph. ``BondRestraintBuilder`` is the original path, still
    keying residues on ``(chain, resseq)``, which is the behaviour under test.
    """
    import torch

    from torchref.topology.builders import BondRestraintBuilder

    built = BondRestraintBuilder(verbose=0).build(
        model.pdb, restraints.cif_dict, torch.device("cpu")
    )
    if not built:
        return set()
    return {tuple(int(v) for v in row) for row in built["indices"].cpu().numpy()}


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
def test_the_legacy_grouping_loses_the_later_residues(inserted):
    """The merged residue gets restraints for its first component only.

    This is the defect keying on ``(chain, resseq, icode)`` fixes. The three residues
    become one, their backbone atom names collide, the name-to-index map keeps the first
    of each, and the second and third end up with no intra-residue geometry at all.
    """
    topology, restraints, expected, model = inserted
    legacy = _legacy_intra_bonds(model, restraints)
    residues = _inserted_residue_indices(topology, expected)
    assert len(residues) == 3

    counts = [len(_bonds_within(legacy, topology, r)) for r in residues]
    assert counts[0] > 0, "even the first component lost its bonds; check the fixture"
    assert counts[1:] == [0, 0], (
        f"the legacy grouping was expected to lose the second and third residues, "
        f"but found {counts} bonds in them"
    )


@pytest.mark.unit
def test_the_graph_finds_what_the_legacy_grouping_lost(inserted):
    """Every inserted residue gets its own bonds, and the graph is a strict superset.

    Localised, not merely larger: the bonds the graph adds all lie inside the inserted
    residues, so this is the insertion-code fix rather than a general difference.
    """
    topology, restraints, expected, model = inserted
    legacy = _legacy_intra_bonds(model, restraints)
    graph = topology.atoms.bonds.tuple_set("intra")
    residues = _inserted_residue_indices(topology, expected)

    for residue in residues:
        assert _bonds_within(
            graph, topology, residue
        ), f"residue {topology.residues.key(residue)} has no intra-residue bonds"

    gained = graph - legacy
    assert gained, "the graph found nothing the legacy grouping missed"

    inserted_set = set(residues)
    stray = [
        edge
        for edge in gained
        if not ({topology.residue_of_atom(a) for a in edge} & inserted_set)
    ]
    assert not stray, (
        f"{len(stray)} gained bonds lie outside the inserted residues, so the "
        f"difference is not localised to the insertion codes: {stray[:3]}"
    )


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
