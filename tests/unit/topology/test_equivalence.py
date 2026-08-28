"""The topology graph must reproduce the restraint builders' edge sets exactly.

Compares as **sets** of atom-index tuples, per edge type and per origin. Row order is
deliberately not compared: the graph lays edges out in a canonical order, whereas the
old storage concatenated origins in Python ``set`` iteration order and so varied
between processes.
"""

import pytest
import torch

from torchref.model.model import Model
from torchref.topology import build_topology

#: Structures chosen to cover the cases the builders treat specially: alternative
#: conformations, pre-existing hydrogens, disulfides, multi-compound ligands, and a
#: metal plus a nucleotide analogue.
STRUCTURES = [
    "7L84",
    "1AK5_with_H",
    "3VRJ",
    "3A5V",
    "1DAW",
]


def _tuples(indices: torch.Tensor) -> set:
    """Rows of an index tensor as a set of int tuples."""
    return {tuple(int(v) for v in row) for row in indices.cpu().numpy()}


def _current_edges(restraints) -> dict:
    """``{edge type: {origin: set of tuples}}`` from the existing restraint storage."""
    out = {}
    for rtype in ("bond", "angle", "torsion"):
        out[rtype] = {}
        for origin in restraints.restraints[rtype].keys():
            if origin == "all":
                continue
            group = restraints.restraints[rtype][origin]
            if group is None or group.get("indices") is None:
                continue
            out[rtype][origin] = _tuples(group["indices"])

    out["chiral"] = {}
    chiral = restraints.restraints.get("chiral")
    if chiral is not None and chiral.get("indices") is not None:
        out["chiral"]["intra"] = _tuples(chiral["indices"])

    out["plane"] = {}
    for key in restraints.restraints["plane"].keys():
        group = restraints.restraints["plane"][key]
        if group is None or group.get("indices") is None:
            continue
        out["plane"][key] = _tuples(group["indices"])
    return out


@pytest.fixture(scope="module")
def built(request, pdb_dir):
    """``(topology, current edge sets)`` for one structure, built once per module."""
    cache = {}

    def _build(code):
        if code not in cache:
            path = pdb_dir / f"{code}.pdb"
            if not path.exists():
                pytest.skip(f"{code}.pdb not bundled")
            model = Model(verbose=0)
            model.load_pdb(str(path))
            model.set_restraints_cif(None)
            restraints = model.restraints
            topology = build_topology(
                model.pdb,
                restraints.cif_dict,
                link_dict=getattr(restraints, "link_dict", None),
                link_list=getattr(restraints, "link_list", None),
                links=restraints.links,
                xyz=model.xyz().detach(),
                verbose=0,
            )
            cache[code] = (topology, _current_edges(restraints))
        return cache[code]

    return _build


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
@pytest.mark.parametrize("edge_type", ["bond", "angle", "torsion", "chiral"])
def test_edge_sets_match_builders(built, code, edge_type):
    """Every origin of every edge type holds exactly the builders' index tuples."""
    topology, current = built(code)
    block = topology.edge_block(edge_type)
    graph = {origin: block.tuple_set(origin) for origin in block.origins()}
    expected = current.get(edge_type, {})

    assert set(graph) == set(expected), (
        f"{code} {edge_type}: origins differ -- "
        f"graph {sorted(graph)} vs builders {sorted(expected)}"
    )
    for origin in sorted(expected):
        missing = expected[origin] - graph[origin]
        extra = graph[origin] - expected[origin]
        assert not missing and not extra, (
            f"{code} {edge_type}/{origin}: {len(missing)} edges the builders "
            f"produced are absent from the graph, {len(extra)} are only in the graph"
        )


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_plane_sets_match_builders(built, code):
    """Planes match per atom count, pooling the origins the graph splits them into."""
    topology, current = built(code)
    graph = {}
    for size, block in topology.atoms.planes.items():
        graph.setdefault(f"{size}_atoms", set()).update(block.tuple_set())
    expected = current["plane"]

    assert set(graph) == set(expected), (
        f"{code} planes: size groups differ -- "
        f"graph {sorted(graph)} vs builders {sorted(expected)}"
    )
    for key in sorted(expected):
        assert graph[key] == expected[key], (
            f"{code} plane/{key}: "
            f"{len(expected[key] - graph[key])} missing, "
            f"{len(graph[key] - expected[key])} extra"
        )


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_exclusions_reproduce_current_set(built, code):
    """The restraint-edge exclusions equal what the non-bonded term is given today."""
    topology, _ = built(code)
    from_edges = topology.atoms.exclusions_from_restraint_edges()

    expected = set()
    for edge_type, cols in (("bond", (0, 1)), ("angle", (0, 2)), ("torsion", (0, 3))):
        block = topology.edge_block(edge_type)
        for row in block.indices.cpu().numpy():
            a, b = int(row[cols[0]]), int(row[cols[1]])
            if a != b:
                expected.add((min(a, b), max(a, b)))

    assert from_edges == expected


@pytest.mark.unit
@pytest.mark.parametrize("code", STRUCTURES)
def test_connectivity_exclusions_are_a_superset(built, code):
    """Connectivity-derived exclusions cover the restraint-derived ones, and then some.

    The difference is the defect the connectivity path fixes: a pair that is 1-3 or 1-4
    bonded but whose angle or torsion the monomer library does not restrain is currently
    not excluded, so the non-bonded term pushes it apart.
    """
    topology, _ = built(code)
    from_edges = topology.atoms.exclusions_from_restraint_edges()
    from_bonds = topology.atoms.exclusions_12_13_14()

    assert from_edges <= from_bonds, (
        f"{code}: {len(from_edges - from_bonds)} restraint-derived exclusions are "
        f"not reachable within three bonds, which should be impossible"
    )


@pytest.mark.unit
@pytest.mark.parametrize("code", ["7L84", "3VRJ"])
def test_adjacency_matches_bond_block(built, code):
    """Every bond appears in both atoms' neighbour lists, and nothing else does."""
    topology, _ = built(code)
    atoms = topology.atoms

    from_adjacency = set()
    for i in range(atoms.n_atoms):
        for j in atoms.neighbors(i).cpu().tolist():
            from_adjacency.add((min(i, j), max(i, j)))

    from_block = set()
    for a, b in atoms.bonds.indices.cpu().numpy():
        from_block.add((min(int(a), int(b)), max(int(a), int(b))))

    assert from_adjacency == from_block
    assert int(atoms.degree().sum()) == 2 * atoms.bonds.n_edges


@pytest.mark.unit
def test_layout_is_reproducible(built, pdb_dir):
    """Two builds of one structure lay the blocks out identically.

    The canonical order removes the process-to-process variation the old storage had,
    where origins were concatenated in ``set`` iteration order.
    """
    topology_a, _ = built("7L84")

    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    model.set_restraints_cif(None)
    restraints = model.restraints
    topology_b = build_topology(
        model.pdb,
        restraints.cif_dict,
        link_dict=getattr(restraints, "link_dict", None),
        link_list=getattr(restraints, "link_list", None),
        links=restraints.links,
        xyz=model.xyz().detach(),
        verbose=0,
    )

    for edge_type in ("bond", "angle", "torsion", "chiral"):
        a = topology_a.edge_block(edge_type)
        b = topology_b.edge_block(edge_type)
        assert a.origin_bounds == b.origin_bounds, edge_type
        assert torch.equal(a.indices, b.indices), edge_type


@pytest.mark.unit
def test_origin_slices_share_storage(built):
    """A per-origin view is a slice of the block, not a copy."""
    topology, _ = built("7L84")
    block = topology.atoms.bonds
    origin = block.origins()[0]
    view = block.origin(origin)

    assert view.data_ptr() == block.indices.data_ptr()
    saved = int(block.indices[0, 0])
    block.indices[0, 0] = saved + 1000
    assert int(view[0, 0]) == saved + 1000
    block.indices[0, 0] = saved


@pytest.mark.unit
def test_residue_identity_includes_insertion_code(built):
    """Residue nodes are keyed on ``(chain, resseq, icode)``.

    The restraint builders group on ``(chain, resseq)`` alone, which merges residue 100
    with 100A and leaves the inserted residue without intra-residue restraints.
    """
    topology, _ = built("7L84")
    residues = topology.residues
    keys = [residues.key(i) for i in range(residues.n_residues)]

    assert len(keys) == len(set(keys)), "residue identity is not unique"
    assert all(len(k) == 3 for k in keys)


@pytest.mark.unit
def test_residue_join_recovers_per_atom_identity(built):
    """Per-atom residue name is recovered via ``residue_of``, not stored per atom."""
    topology, _ = built("7L84")

    for atom in (0, topology.n_atoms // 2, topology.n_atoms - 1):
        residue = topology.residue_of_atom(atom)
        assert topology.resname_of_atom(atom) == topology.residues.resname[residue]
        assert (
            topology.residues.atom_start[residue]
            <= atom
            < topology.residues.atom_end[residue]
        )


@pytest.mark.unit
def test_connectivity_exclusions_match_brute_force():
    """The vectorised path-walk agrees with a plain breadth-first reference.

    ``test_connectivity_exclusions_are_a_superset`` alone would also pass for an
    implementation that returned every pair, so the walk is pinned against an
    independent one on a graph small enough to enumerate: a six-ring with two
    substituents, which exercises the ring closure and the 1-4 wrap-around.
    """
    import numpy as np

    from torchref.topology import EdgeBlock
    from torchref.topology.atom_graph import AtomGraph

    bonds = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),  # six-ring
        (0, 6),  # substituent on 0
        (6, 7),  # and one more out
    ]
    n = 8

    def block(rows, arity, edge_type):
        return EdgeBlock.from_origins(
            {"intra": np.asarray(rows, dtype=np.int64).reshape(-1, arity)},
            arity,
            edge_type,
        )

    graph = AtomGraph(
        name=np.array([f"A{i}" for i in range(n)]),
        element=np.array(["C"] * n),
        altloc=np.array([" "] * n),
        residue_of=torch.zeros(n, dtype=torch.int64),
        bonds=block(bonds, 2, "bond"),
        angles=EdgeBlock.empty(3),
        torsions=EdgeBlock.empty(4),
        chirals=EdgeBlock.empty(4),
    )

    adjacency = {i: set() for i in range(n)}
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)

    expected = set()
    for start in range(n):
        frontier = {start}
        seen = {start}
        for _ in range(3):
            frontier = {j for i in frontier for j in adjacency[i]} - seen
            seen |= frontier
            for other in frontier:
                expected.add((min(start, other), max(start, other)))

    assert graph.exclusions_12_13_14() == expected
