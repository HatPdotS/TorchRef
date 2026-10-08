"""OpenMM topology built from a model's own identity and bonds.

Pins that OpenMM atom ``k`` is model row ``rows[k]`` with the model's name, element,
residue and chain; that the bonds are the restraint topology's, minus coordination
bonds to metals; that molecules are whole; and the hydrogen policy: a warning whenever
TorchRef did not add the hydrogens, an error below half of those the dictionaries
call for.
"""

import warnings

import numpy as np
import pytest

from torchref.experimental.mm.topology import (
    MIN_HYDROGEN_FRACTION,
    build_asu_topology,
    check_hydrogens,
)
from torchref.model.model import Model

pytestmark = pytest.mark.openmm


@pytest.fixture(scope="module")
def protein(pdb_dir):
    """7L84 with TorchRef's hydrogens: protein and waters, standard residues only."""
    return (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "7L84.pdb"))
        .strip_altlocs()
    )


@pytest.fixture(scope="module")
def ligand_model(pdb_dir):
    """1DAW with hydrogens: ANP, two Mg ions and their LINK records."""
    return (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "1DAW.pdb"))
        .strip_altlocs()
    )


def test_atoms_follow_model_rows(protein):
    """Every model atom appears once, in row order, with its identity."""
    asu = build_asu_topology(protein)
    topology, particles, origin = asu.to_openmm()
    columns = protein.ctx.topology.columns()
    assert np.array_equal(asu.rows, np.arange(protein.n_atoms))
    assert np.array_equal(particles, np.arange(protein.n_atoms))
    atoms = list(topology.atoms())
    assert [a.name for a in atoms] == list(columns["name"])
    assert [a.residue.name for a in atoms] == list(columns["resname"])
    assert [a.residue.chain.id for a in atoms] == list(columns["chain"])
    symbols = np.char.capitalize(np.char.strip(columns["element"].astype(str)))
    assert [a.element.symbol for a in atoms] == list(symbols)
    assert len(origin) == topology.getNumResidues() == asu.n_residues


def test_bonds_are_the_restraint_bonds(protein):
    """OpenMM's bonds are exactly the restraint topology's bonds."""
    asu = build_asu_topology(protein)
    topology, _, _ = asu.to_openmm()
    restraint = protein.restraints.topology.atoms.bonds.indices.cpu().numpy()
    expected = {tuple(sorted(map(int, b))) for b in restraint}
    built = {tuple(sorted((a.index, b.index))) for a, b in topology.bonds()}
    assert built == expected


def test_metal_coordination_is_not_a_bond(ligand_model):
    """1DAW's Mg LINK records are restraints, not force-field bonds."""
    asu = build_asu_topology(ligand_model)
    symbols = asu.symbol
    assert "Mg" in set(symbols)
    assert not np.isin(symbols[asu.bonds], ["Mg"]).any()
    graph = ligand_model.restraints.topology.atoms
    start, end = graph.bonds.origin_bounds["link"]
    links = graph.bonds.indices[start:end].cpu().numpy()
    assert np.isin(graph.symbols[links], ["Mg"]).any()


def test_molecules_are_whole(ligand_model):
    """Each water, ion and ligand is its own molecule; a chain is one molecule."""
    asu = build_asu_topology(ligand_model)
    residue_of = asu.residue_of
    for r in range(asu.n_residues):
        assert len(set(asu.molecule_of[residue_of == r])) == 1
    waters = np.flatnonzero(asu.residue_name == "HOH")
    water_molecules = asu.molecule_of[asu.residue_start[waters]]
    assert len(set(water_molecules)) == len(waters)


def test_selection_must_hold_whole_molecules(ligand_model):
    """Dropping ANP is allowed; taking half a residue is not."""
    columns = ligand_model.ctx.topology.columns()
    keep = columns["resname"] != "ANP"
    asu = build_asu_topology(ligand_model, atoms=keep)
    assert "ANP" not in set(asu.residue_name)
    assert np.array_equal(asu.rows, np.flatnonzero(keep))
    with pytest.raises(ValueError, match="splits a molecule"):
        build_asu_topology(ligand_model, atoms=columns["name"] != "CA")


def test_hydrogens_added_by_torchref_are_silent(protein):
    graph = protein.restraints.topology.atoms
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        present, expected = check_hydrogens(graph, np.arange(protein.n_atoms), True)
    assert expected > 0 and present >= 0.95 * expected


def test_kept_hydrogens_always_warn(protein):
    graph = protein.restraints.topology.atoms
    with pytest.warns(UserWarning, match="TorchRef did not add"):
        check_hydrogens(graph, np.arange(protein.n_atoms), False)


def test_heavy_atom_model_is_rejected(pdb_dir):
    model = (
        Model(verbose=0, device="cpu", hydrogens="strip")
        .load_pdb(str(pdb_dir / "7L84.pdb"))
        .strip_altlocs()
    )
    graph = model.restraints.topology.atoms
    with pytest.raises(ValueError, match="AMBER needs every one"):
        with pytest.warns(UserWarning):
            check_hydrogens(graph, np.arange(model.n_atoms), False)


def test_partially_deposited_hydrogens_are_rejected(pdb_dir):
    """1AK5 as deposited carries about a quarter of its hydrogens."""
    model = (
        Model(verbose=0, device="cpu", hydrogens="keep")
        .load_pdb(str(pdb_dir / "1AK5_with_H.pdb"))
        .strip_altlocs()
    )
    graph = model.restraints.topology.atoms
    with pytest.raises(ValueError, match="AMBER needs every one"):
        with pytest.warns(UserWarning):
            check_hydrogens(graph, np.arange(model.n_atoms), False)
    assert MIN_HYDROGEN_FRACTION == 0.5


def test_alternate_conformations_are_rejected(pdb_dir):
    model = Model(verbose=0, device="cpu").load_pdb(str(pdb_dir / "1DAW.pdb"))
    if not (model.ctx.topology.atoms.altloc != " ").any():
        pytest.skip("1DAW lost its alternate conformations")
    with pytest.raises(ValueError, match="strip_altlocs"):
        build_asu_topology(model)
