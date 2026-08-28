"""Subsetting and copying a topology.

``subset`` reindexes what survives instead of rebuilding, so the properties that matter
are the ones a plausible-but-wrong implementation would break: that every surviving edge
keeps the *same atoms* it had before, that an edge touching a removed atom is gone
entirely rather than left dangling, that the blocks stay canonically ordered without a
re-sort, and that the residue level stays consistent with the atom level.
"""

import numpy as np
import pytest
import torch

from torchref.model.model import Model
from torchref.topology import EdgeBlock

KEYED_TYPES = ("bond", "angle", "torsion", "chiral")


@pytest.fixture(scope="module")
def topology(pdb_dir):
    """A topology with altlocs, disulfides, peptide links and hydrogens."""
    model = Model(verbose=0, add_hydrogens=False, strip_H=True)
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    model.set_restraints_cif(None)
    return model.restraints.topology


def _named_edges(topology, edge_type):
    """Edges as tuples of ``(chain, resseq, icode, atom name)``, identity not index.

    Comparing on identity rather than index is the whole point: an index-based check
    passes trivially after a remap, whereas this catches a remap that points an edge at
    the wrong atom.
    """
    residues = topology.residues
    names = topology.atoms.name.astype(str)
    out = set()
    block = topology.edge_block(edge_type)
    for row in block.indices.cpu().numpy():
        out.add(
            tuple(
                (*residues.key(topology.residue_of_atom(int(a))), names[int(a)])
                for a in row
            )
        )
    return out


@pytest.mark.unit
def test_subset_of_everything_is_the_same_graph(topology):
    """Keeping every atom changes nothing -- the identity case."""
    whole = topology.subset(torch.ones(topology.n_atoms, dtype=torch.bool))

    assert whole.n_atoms == topology.n_atoms
    assert whole.n_residues == topology.n_residues
    for edge_type in KEYED_TYPES:
        assert torch.equal(
            whole.edge_block(edge_type).indices,
            topology.edge_block(edge_type).indices,
        )
        assert (
            whole.edge_block(edge_type).origin_bounds
            == topology.edge_block(edge_type).origin_bounds
        )


@pytest.mark.unit
def test_surviving_edges_keep_the_same_atoms(topology):
    """Every edge in the subset joins exactly the atoms it joined before.

    Checked on residue-and-name identity, so a remap that silently shifted an index
    would fail here even though the shapes still looked right.
    """
    residues = topology.residues
    chain_a = np.array(
        [
            str(residues.chain[topology.residue_of_atom(atom)]) == "A"
            for atom in range(topology.n_atoms)
        ]
    )
    assert chain_a.any() and not chain_a.all() or chain_a.all()

    keep = torch.zeros(topology.n_atoms, dtype=torch.bool)
    keep[: topology.n_atoms // 2] = True
    reduced = topology.subset(keep)

    for edge_type in KEYED_TYPES:
        after = _named_edges(reduced, edge_type)
        before = _named_edges(topology, edge_type)
        assert after <= before, (
            f"{edge_type}: the subset invented {len(after - before)} edges that were "
            f"not in the original"
        )


@pytest.mark.unit
def test_an_edge_dies_with_any_of_its_atoms(topology):
    """No edge survives that touches a removed atom.

    A bond to an atom that is gone is not a bond, and leaving it would point an index
    outside the graph.
    """
    keep = torch.ones(topology.n_atoms, dtype=torch.bool)
    keep[5] = False
    keep[100] = False
    reduced = topology.subset(keep)

    for edge_type in KEYED_TYPES:
        block = reduced.edge_block(edge_type)
        if block.n_edges == 0:
            continue
        assert int(block.indices.max()) < reduced.n_atoms
        assert int(block.indices.min()) >= 0

    # Precisely: the edges lost are exactly those that used a dropped atom.
    dropped = {5, 100}
    for edge_type in KEYED_TYPES:
        original = topology.edge_block(edge_type).indices.cpu().numpy()
        touching = sum(1 for row in original if dropped & set(int(a) for a in row))
        assert (
            reduced.edge_block(edge_type).n_edges
            == topology.edge_block(edge_type).n_edges - touching
        ), f"{edge_type}: wrong number of edges dropped"


@pytest.mark.unit
def test_blocks_stay_canonically_ordered(topology):
    """Subsetting needs no re-sort, because the remap is monotone on survivors.

    If this fails the block is no longer canonical, and the ``all`` group stops being a
    contiguous span -- which is what makes it a view rather than a copy.
    """
    keep = torch.zeros(topology.n_atoms, dtype=torch.bool)
    keep[::2] = True
    reduced = topology.subset(keep)

    for edge_type in KEYED_TYPES:
        block = reduced.edge_block(edge_type)
        for origin in block.origins():
            rows = block.origin(origin).cpu().numpy()
            if len(rows) < 2:
                continue
            order = np.lexsort(
                tuple(rows[:, c] for c in reversed(range(rows.shape[1])))
            )
            assert (
                order == np.arange(len(rows))
            ).all(), f"{edge_type}/{origin} is no longer lexicographically sorted"
        # bounds must remain contiguous and cover the block
        spans = sorted(block.origin_bounds.values())
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        if spans:
            assert spans[0][0] == 0 and spans[-1][1] == block.n_edges


@pytest.mark.unit
def test_residue_level_stays_consistent_with_the_atoms(topology):
    """Atom ranges, residue count and links all agree after subsetting."""
    keep = torch.zeros(topology.n_atoms, dtype=torch.bool)
    keep[: topology.n_atoms // 3] = True
    reduced = topology.subset(keep)

    residues = reduced.residues
    assert residues.n_residues == reduced.n_residues
    # Ranges partition the atoms, in order, with no gaps.
    assert int(residues.atom_start[0]) == 0
    assert int(residues.atom_end[-1]) == reduced.n_atoms
    assert (residues.atom_start[1:] == residues.atom_end[:-1]).all()

    # residue_of agrees with the ranges it is supposed to index.
    for residue in range(residues.n_residues):
        rows = range(int(residues.atom_start[residue]), int(residues.atom_end[residue]))
        for row in rows:
            assert reduced.residue_of_atom(row) == residue

    # No link edge points outside the surviving residues.
    if len(residues.link_pairs):
        assert residues.link_pairs.max() < residues.n_residues
        assert residues.link_pairs.min() >= 0


@pytest.mark.unit
def test_dropping_a_whole_residue_drops_its_links(topology):
    """A residue with no atoms left is gone, and so is any link that reached it."""
    residues = topology.residues
    linked = None
    for pair in residues.links_of_kind("TRANS"):
        linked = int(pair[0])
        break
    assert linked is not None, "7L84 has peptide links"

    keep = torch.ones(topology.n_atoms, dtype=torch.bool)
    for row in range(int(residues.atom_start[linked]), int(residues.atom_end[linked])):
        keep[row] = False
    reduced = topology.subset(keep)

    assert reduced.n_residues == topology.n_residues - 1
    before = len(residues.links_of_kind("TRANS"))
    after = len(reduced.residues.links_of_kind("TRANS"))
    assert after < before, "a link to the removed residue survived"


@pytest.mark.unit
def test_subset_accepts_indices_as_well_as_a_mask(topology):
    """Integer indices give the same result as the equivalent mask."""
    indices = torch.arange(0, topology.n_atoms, 3)
    mask = torch.zeros(topology.n_atoms, dtype=torch.bool)
    mask[indices] = True

    by_index = topology.subset(indices)
    by_mask = topology.subset(mask)

    assert by_index.n_atoms == by_mask.n_atoms
    for edge_type in KEYED_TYPES:
        assert torch.equal(
            by_index.edge_block(edge_type).indices,
            by_mask.edge_block(edge_type).indices,
        )


@pytest.mark.unit
def test_subset_ignores_the_order_of_the_indices(topology):
    """A shuffled index list yields the topology's own atom order.

    Deliberate: the edge blocks stay canonical only under a monotone relabelling, so
    honouring a caller's order would silently leave them unsorted.
    """
    indices = torch.arange(0, topology.n_atoms, 5)
    shuffled = indices[torch.randperm(len(indices))]

    assert torch.equal(
        topology.subset(indices).atoms.bonds.indices,
        topology.subset(shuffled).atoms.bonds.indices,
    )


@pytest.mark.unit
def test_subset_of_nothing_is_an_error(topology):
    """An empty selection is a mistake, not an empty topology."""
    with pytest.raises(ValueError, match="no atoms"):
        topology.subset(torch.zeros(topology.n_atoms, dtype=torch.bool))


@pytest.mark.unit
def test_copy_shares_nothing(topology):
    """A copy has equal contents and independent storage."""
    duplicate = topology.copy()

    assert duplicate.n_atoms == topology.n_atoms
    assert duplicate.n_residues == topology.n_residues
    for edge_type in KEYED_TYPES:
        original = topology.edge_block(edge_type)
        copied = duplicate.edge_block(edge_type)
        assert torch.equal(copied.indices, original.indices)
        assert copied.indices.data_ptr() != original.indices.data_ptr()

    saved = int(topology.atoms.bonds.indices[0, 0])
    duplicate.atoms.bonds.indices[0, 0] = saved + 13
    assert int(topology.atoms.bonds.indices[0, 0]) == saved

    duplicate.residues.resname[0] = "XXX"
    assert topology.residues.resname[0] != "XXX"


@pytest.mark.unit
def test_copy_rebuilds_its_own_adjacency(topology):
    """``neighbors`` on a copy reads the copy's bonds, not the original's."""
    duplicate = topology.copy()
    atom = int(topology.atoms.bonds.indices[0, 0])
    assert torch.equal(duplicate.neighbors(atom), topology.neighbors(atom))
    assert (
        duplicate.atoms._adj_indices.data_ptr()
        != topology.atoms._adj_indices.data_ptr()
    )


@pytest.mark.unit
def test_subset_adjacency_matches_its_own_bonds(topology):
    """The reduced graph's adjacency is rebuilt, not carried over stale."""
    keep = torch.zeros(topology.n_atoms, dtype=torch.bool)
    keep[: topology.n_atoms // 2] = True
    reduced = topology.subset(keep)

    from_adjacency = set()
    for atom in range(reduced.n_atoms):
        for other in reduced.neighbors(atom).cpu().tolist():
            from_adjacency.add((min(atom, other), max(atom, other)))

    from_block = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in reduced.atoms.bonds.indices.cpu().numpy()
    }
    assert from_adjacency == from_block
    assert int(reduced.atoms.degree().sum()) == 2 * reduced.atoms.bonds.n_edges


@pytest.mark.unit
def test_empty_block_subsets_to_empty():
    """An edge type with no edges survives subsetting without special-casing."""
    block = EdgeBlock.empty(3)
    remap = torch.arange(10, dtype=torch.int64)
    reduced = block.subset(remap)
    assert reduced.n_edges == 0
    assert reduced.arity == 3
