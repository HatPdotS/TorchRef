"""The restraint storage: plain dict, views into the edge blocks, no per-access work.

The geometry targets read ``restraints[edge_type][origin][property]`` on every
iteration, so the properties tested here are load-bearing rather than cosmetic: the
mapping must be a plain dict of already-materialised tensors, the per-origin entries
must alias the contiguous blocks rather than copy them, and none of that may come
undone on a device move or a copy.
"""

import pytest
import torch

from torchref.model.model import Model
from torchref.utils.caching import ParameterFingerprint

KEYED_TYPES = ("bond", "angle", "torsion")

#: The surface the geometry targets rely on. Guards the contract from drifting: 'phi'
#: and 'psi' are conformationally free and must NOT acquire a reference value or sigma,
#: and 'omega' must keep the proline flag its own target reads.
EXPECTED_PROPERTIES = {
    ("bond", "all"): {"indices", "references", "sigmas"},
    ("bond", "intra"): {"indices", "references", "sigmas"},
    ("angle", "all"): {"indices", "references", "sigmas"},
    ("torsion", "all"): {"indices", "references", "sigmas", "periods"},
    ("torsion", "phi"): {"indices", "periods"},
    ("torsion", "psi"): {"indices", "periods"},
    ("torsion", "omega"): {
        "indices",
        "references",
        "sigmas",
        "periods",
        "is_proline",
    },
}


@pytest.fixture(scope="module")
def restraints(pdb_dir):
    """Restraints for a structure with altlocs, disulfides and peptide links."""
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    model.set_restraints_cif(None)
    return model.restraints


@pytest.mark.unit
def test_restraints_is_a_plain_dict(restraints):
    """Not an accessor object that has to be constructed per access."""
    assert type(restraints.restraints) is dict
    assert restraints.restraints is restraints.restraints


@pytest.mark.unit
def test_access_allocates_nothing(restraints):
    """Every level of the lookup returns the same object each time.

    This is what makes the read cheap: three dict lookups and no construction. The
    previous storage built a fresh accessor, then a fresh per-type accessor, then a
    fresh dict of six string-keyed buffer lookups, on every call.
    """
    first = restraints.restraints["bond"]["all"]
    second = restraints.restraints["bond"]["all"]
    assert first is second
    assert first["indices"] is second["indices"]


@pytest.mark.unit
@pytest.mark.parametrize("edge_type", KEYED_TYPES)
def test_origin_entries_alias_the_block(restraints, edge_type):
    """Per-origin indices are slices of one block, not copies of it."""
    block = restraints.topology.edge_block(edge_type)
    entries = restraints.restraints[edge_type]

    for origin, bounds in block.origin_bounds.items():
        indices = entries[origin]["indices"]
        assert indices.data_ptr() == block.indices[bounds[0] : bounds[1]].data_ptr()
        assert indices.shape[0] == bounds[1] - bounds[0]


@pytest.mark.unit
@pytest.mark.parametrize("edge_type", KEYED_TYPES)
def test_all_group_is_a_view(restraints, edge_type):
    """The combined group the targets read is a span of the block, not a concatenation.

    The block layout deliberately keeps each type's ``all`` members adjacent so this
    holds; ``torsion`` is the one that needs it, since its group is only ``intra`` plus
    ``disulfide``.
    """
    block = restraints.topology.edge_block(edge_type)
    combined = restraints.restraints[edge_type]["all"]["indices"]
    assert combined.data_ptr() == block.indices.data_ptr()


@pytest.mark.unit
def test_in_place_block_edit_is_visible_through_every_entry(restraints):
    """Shared storage means a block edit needs no invalidation to be seen."""
    block = restraints.topology.atoms.bonds
    entries = restraints.restraints["bond"]
    origin = block.origins()[0]

    saved = int(block.indices[0, 0])
    try:
        block.indices[0, 0] = saved + 7
        assert int(entries[origin]["indices"][0, 0]) == saved + 7
        assert int(entries["all"]["indices"][0, 0]) == saved + 7
    finally:
        block.indices[0, 0] = saved


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(EXPECTED_PROPERTIES))
def test_expected_properties_present(restraints, key):
    """Each group carries exactly the properties its consumers expect."""
    edge_type, origin = key
    group = restraints.restraints[edge_type][origin]
    assert set(group) == EXPECTED_PROPERTIES[key]


@pytest.mark.unit
def test_cat_dict_is_idempotent(restraints):
    """Repeated calls leave the restraint counts alone.

    The previous implementation registered ``'all'`` as an origin when it wrote the
    combined group, so a second call concatenated the group into itself and doubled
    every bond, angle and torsion -- a silent 2x on the geometry weight. Deriving the
    group as a span of the block makes that unrepresentable.
    """
    before = {
        edge_type: restraints.restraints[edge_type]["all"]["indices"].shape[0]
        for edge_type in KEYED_TYPES
    }
    restraints.cat_dict()
    restraints.cat_dict()
    after = {
        edge_type: restraints.restraints[edge_type]["all"]["indices"].shape[0]
        for edge_type in KEYED_TYPES
    }
    assert before == after


@pytest.mark.unit
def test_entries_survive_a_device_apply(restraints):
    """A ``.to()`` re-slices the entries instead of leaving them stale or duplicated.

    ``DeviceMixin``'s walk recurses into dicts, so without the ``_apply`` override each
    view would be moved on its own and become an independent tensor.
    """
    before = restraints.restraints["bond"]["all"]["indices"].clone()
    n_before = {
        t: restraints.restraints[t]["all"]["indices"].shape[0] for t in KEYED_TYPES
    }

    restraints.to(torch.device("cpu"))

    block = restraints.topology.atoms.bonds.indices
    after = restraints.restraints["bond"]["all"]["indices"]
    assert after.data_ptr() == block.data_ptr(), "entries no longer alias the block"
    assert torch.equal(after, before)
    assert {
        t: restraints.restraints[t]["all"]["indices"].shape[0] for t in KEYED_TYPES
    } == n_before
    assert restraints.restraints["vdw"].get("indices") is not None


@pytest.mark.unit
def test_blocks_are_untouched_by_a_refinement_step(restraints):
    """The edge tensors are constants; nothing in a loss evaluation may mutate them."""
    blocks = [restraints.topology.edge_block(t).indices for t in KEYED_TYPES]
    fingerprint = ParameterFingerprint(blocks)

    loss = restraints.nll_bonds().sum() + restraints.nll_angles().sum()
    loss.backward()

    assert fingerprint.matches(
        [restraints.topology.edge_block(t).indices for t in KEYED_TYPES]
    )


@pytest.mark.unit
def test_rebuilding_entries_reslices_onto_the_current_blocks(restraints):
    """Re-deriving the entries produces fresh views of the same blocks.

    This is the operation ``_apply`` and ``copy`` both rely on, and the one that has to
    stay cheap: it re-slices rather than recomputing anything.

    ``RestraintsNew.copy`` is not exercised here because it cannot run at all -- it is
    ``deepcopy``, which walks the *borrowed* ``_xyz_fn`` wrapper, whose cache holds a
    graph-attached tensor once ``xyz()`` has been evaluated. Verified to fail
    identically at the commit before this change, so it is pre-existing rather than a
    regression, and it is reached only through ``Model.copy`` on a model whose lazy
    restraints have already been built.
    """
    block = restraints.topology.atoms.bonds.indices
    before = restraints.restraints["bond"]["all"]["indices"].clone()

    restraints._rebuild_entries()

    after = restraints.restraints["bond"]["all"]["indices"]
    assert after.data_ptr() == block.data_ptr()
    assert torch.equal(after, before)
    for origin, bounds in restraints.topology.atoms.bonds.origin_bounds.items():
        entry = restraints.restraints["bond"][origin]["indices"]
        assert entry.shape[0] == bounds[1] - bounds[0]
    assert restraints.restraints["vdw"].get("indices") is not None
