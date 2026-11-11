"""ModelContext.from_sequence: a context from chains of residues, before coordinates.

The atoms and their order are the ones a deposited model of the same sequence has, the
restraints over them are the ones the file-built context gets, and nothing in the
restraint graph waits for coordinates.
"""

import numpy as np
import pytest
import torch

from torchref.model import Model
from torchref.model.context import THREE_TO_ONE, ModelContext
from torchref.topology import Topology
from torchref.topology.sequence import ONE_TO_THREE


def _tensors(entries, path=()):
    """``{path: tensor}`` over a nested restraint dict."""
    out = {}
    for key, value in entries.items():
        if isinstance(value, dict):
            out.update(_tensors(value, path + (key,)))
        elif torch.is_tensor(value):
            out[path + (key,)] = value
    return out


def _geometry(restraints):
    """The restraint entries that are graph and ideal values, pair list aside."""
    return _tensors({k: v for k, v in restraints.restraints.items() if k != "vdw"})


@pytest.fixture(scope="module")
def deposited(pdb_dir):
    """1DAW's protein chain, one conformer: what its sequence alone describes."""
    model = Model(device=torch.device("cpu")).load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model.select(
        "not resname HOH and not resname MG and not resname ANP"
    ).strip_altlocs()


@pytest.fixture(scope="module")
def from_sequence(deposited):
    """The context 1DAW's sequence gives, numbered as deposited, and its values."""
    ((chain, sequence),) = deposited.ctx.chain_sequences
    first = int(deposited.ctx.topology.residues.resseq.min())
    return ModelContext.from_sequence(
        {chain: sequence},
        deposited.cell,
        deposited.spacegroup,
        device=torch.device("cpu"),
        start=first,
    )


@pytest.mark.unit
def test_atoms_are_the_deposited_ones_in_the_deposited_order(deposited, from_sequence):
    ctx, _ = from_sequence
    expected = deposited.ctx.topology.columns()
    built = ctx.topology.columns()
    for key in ("name", "element", "chain", "resseq", "icode", "resname", "altloc"):
        np.testing.assert_array_equal(built[key], expected[key])
    assert not ctx.topology.connected
    assert not ctx.topology.atoms.is_hydrogen.any()


@pytest.mark.unit
def test_sequence_round_trips(deposited, from_sequence):
    ctx, _ = from_sequence
    assert ctx.chain_sequences == deposited.ctx.chain_sequences


@pytest.mark.unit
def test_restraints_are_the_file_built_ones(deposited, from_sequence):
    """Over the deposited coordinates the two contexts restrain the same edges to the
    same values, and the sequence-built one does so without coordinates too."""
    ctx, _ = from_sequence
    xyz = deposited.xyz().detach()
    expected = _geometry(deposited.restraints)

    with_coordinates = _geometry(ctx.build_restraints(xyz, nonbonded=False))
    assert with_coordinates.keys() == expected.keys()
    for path, tensor in expected.items():
        assert torch.equal(with_coordinates[path], tensor), path

    without = _geometry(ctx.build_restraints())
    assert without.keys() == expected.keys()
    for path, tensor in expected.items():
        assert torch.equal(without[path], tensor), path


@pytest.mark.unit
def test_a_build_without_coordinates_is_returned_not_stored(from_sequence):
    """It has no pair list, so it must not become the context's restraints, and it
    cannot grow one later."""
    ctx, _ = from_sequence
    ctx.restraints = None
    restraints = ctx.build_restraints()

    assert ctx.restraints is None
    assert "indices" not in (restraints.restraints.get("vdw") or {})
    with pytest.raises(RuntimeError, match="before initial build"):
        restraints.rebuild_vdw_restraints(torch.zeros(ctx.n_atoms, 3))


@pytest.mark.unit
def test_values_are_placeholders_without_coordinates(from_sequence):
    ctx, values = from_sequence
    assert len(values) == ctx.n_atoms
    assert np.isnan(values.xyz).all() and values.xyz.shape == (ctx.n_atoms, 3)
    assert (values.b == 20.0).all()
    assert (values.occupancy == 1.0).all()
    assert not values.aniso.any() and np.isnan(values.u).all()
    assert ctx.initialized


@pytest.mark.unit
def test_link_modes_default_to_the_coordinate_free_ones_and_are_inherited(
    from_sequence,
):
    ctx, values = from_sequence
    assert (ctx.peptide_links, ctx.disulfides) == ("sequence", "open")
    assert (ctx.copy().peptide_links, ctx.copy().disulfides) == ("sequence", "open")
    derived, _ = ctx.derive(ctx.topology.copy(), values)
    assert (derived.peptide_links, derived.disulfides) == ("sequence", "open")

    state = ctx.state()
    restored, _ = ModelContext.from_state(
        state, dtype=torch.float32, device=torch.device("cpu")
    )
    assert (restored.peptide_links, restored.disulfides) == ("sequence", "open")
    assert "peptide_links" not in state and "disulfides" not in state


@pytest.mark.unit
def test_a_file_loaded_context_still_detects(deposited):
    assert (deposited.ctx.peptide_links, deposited.ctx.disulfides) == (
        "distance",
        "detect",
    )
    with pytest.raises(ValueError, match="peptide_links='sequence'"):
        deposited.ctx.build_restraints()


@pytest.mark.unit
def test_link_modes_survive_a_checkpoint(deposited):
    model = deposited.copy()
    model.ctx.disulfides = "open"
    restored = Model.create_from_state_dict(
        model.state_dict(), device=torch.device("cpu")
    )
    assert (restored.ctx.peptide_links, restored.ctx.disulfides) == (
        "distance",
        "open",
    )


@pytest.mark.unit
def test_no_crystal_is_needed_for_the_graph():
    ctx, _ = ModelContext.from_sequence({"A": "ACDEFGHIK"})
    assert ctx.cell is None and ctx.spacegroup is None
    restraints = ctx.build_restraints()
    assert len(restraints.topology.residues.links_of_kind("TRANS")) == 8


@pytest.mark.unit
def test_a_context_without_a_crystal_derives():
    ctx, values = ModelContext.from_sequence({"A": "AG"})
    derived, derived_values = ctx.derive(ctx.topology.copy(), values)

    assert derived.cell is None and derived.n_atoms == ctx.n_atoms
    assert len(derived_values) == len(values)
    assert (derived.peptide_links, derived.disulfides) == ("sequence", "open")


@pytest.mark.unit
def test_several_chains_names_and_waters():
    """A string is one-letter codes, a list residue names; waters stand alone, are
    HETATM and stay out of the sequences."""
    ctx, _ = ModelContext.from_sequence(
        {"A": "GAV", "B": ["MSE", "ALA"], "W": ["HOH"] * 3},
        start={"B": 10},
    )
    columns = ctx.topology.columns()

    assert ctx.chain_sequences == [("A", "GAV"), ("B", "MA")]
    assert sorted(set(columns["resseq"][columns["chain"] == "B"])) == [10, 11]
    assert sorted(set(columns["resseq"][columns["chain"] == "W"])) == [1, 2, 3]
    water = columns["chain"] == "W"
    assert columns["name"][water].tolist() == ["O"] * 3
    assert columns["is_hetatm"][water].all() and not columns["is_hetatm"][~water].any()

    links = ctx.build_restraints().topology.residues.links_of_kind("TRANS")
    assert len(links) == 3


@pytest.mark.unit
def test_terminal_oxygen_is_kept_on_the_last_residue_only_and_only_if_asked():
    with_oxt, _ = ModelContext.from_sequence({"A": "AGA"})
    columns = with_oxt.topology.columns()
    assert columns["resseq"][columns["name"] == "OXT"].tolist() == [3]

    without, _ = ModelContext.from_sequence({"A": "AGA"}, oxt=False)
    assert "OXT" not in without.topology.columns()["name"]
    assert without.n_atoms == with_oxt.n_atoms - 1


@pytest.mark.unit
def test_what_a_sequence_cannot_give_is_refused():
    with pytest.raises(ValueError, match="places hydrogens from coordinates"):
        ModelContext.from_sequence({"A": "AG"}, hydrogens="add")
    with pytest.raises(ValueError, match="one-letter code 'B'"):
        ModelContext.from_sequence({"A": "ABG"})
    with pytest.raises(ValueError, match="no residues"):
        ModelContext.from_sequence({"A": ""})
    with pytest.raises(ValueError, match="disulfides must be one of"):
        ModelContext.from_sequence({"A": "AG"}, disulfides="closed")


@pytest.mark.unit
def test_one_letter_codes_agree_with_the_sequence_reader():
    """Every letter expands to a residue that reads back as that letter."""
    assert all(THREE_TO_ONE[name] == letter for letter, name in ONE_TO_THREE.items())


@pytest.mark.unit
def test_topology_from_sequence_is_the_context_topology(from_sequence, deposited):
    ctx, _ = from_sequence
    ((chain, sequence),) = deposited.ctx.chain_sequences
    first = int(ctx.topology.residues.resseq.min())
    topology = Topology.from_sequence({chain: sequence}, start=first)
    for key, value in ctx.topology.columns().items():
        np.testing.assert_array_equal(topology.columns()[key], value)
