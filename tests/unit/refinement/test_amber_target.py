"""AMBER evaluates the model's complete coordinates and returns atomic gradients."""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from torchref.experimental.targets.amber_target import AmberTarget
from torchref.model.model import Model

pytestmark = pytest.mark.openmm
TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "7L84.pdb"
)


@pytest.fixture(scope="module")
def protein():
    """A deposited, hydrogenated protein with methyl orientation parameters."""
    model = Model(verbose=0, device="cpu", hydrogens="add").load_pdb(TEST_PDB)
    model = model.strip_altlocs()
    model.set_hydrogen_mode("riding")
    return model


@pytest.fixture(scope="module")
def target(protein):
    """Build the AMBER system once per module."""
    return AmberTarget(model=protein, verbose=0)


def test_build_preserves_model(protein):
    """Target construction neither changes model atoms nor replaces coordinates."""
    before = protein.to_dataframe()
    xyz = protein.xyz
    positions = xyz().detach().clone()
    target = AmberTarget(model=protein)
    pd.testing.assert_frame_equal(protein.to_dataframe(), before)
    assert protein.xyz is xyz
    assert torch.equal(protein.xyz(), positions)
    assert target.adapter.n_particles == protein.n_atoms
    assert np.array_equal(target.adapter.particles, np.arange(protein.n_atoms))


def test_forward_is_the_adapter_energy_per_atom(target, protein):
    loss = target.forward()
    expected = target.adapter.energy(protein.xyz()) / protein.n_atoms
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert set(target.stats()) >= {"loss", "energy_kJ_mol", "platform"}


def test_forward_and_orientation_gradients(target, protein):
    """AMBER hydrogen forces reach the model's methyl torsion parameters."""
    loss = target.forward()
    leaves = protein.xyz.optimization_parameters()
    gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
    assert loss.shape == () and torch.isfinite(loss)
    for leaf, gradient in zip(leaves, gradients):
        if leaf.numel():
            assert gradient is not None and torch.isfinite(gradient).all()
    torsion_grad = torch.autograd.grad(
        target.forward(), protein.xyz.torsions.refinable_params
    )[0]
    assert torsion_grad.abs().sum() > 0


def test_crystal_layout_is_per_asymmetric_unit(protein):
    """The crystal layout holds every symmetry copy and reports one copy's share."""
    crystal = AmberTarget(
        model=protein, layout="crystal", normalize_by_atoms=False, cutoff=8.0
    )
    n_ops = int(protein.spacegroup.n_ops)
    assert crystal.adapter.layout.n_copies == n_ops
    total = crystal.adapter.energy(protein.xyz())
    assert crystal.forward().item() == pytest.approx(total.item() / n_ops, rel=1e-6)
    gradient = torch.autograd.grad(crystal.forward(), protein.xyz.refinable_params)[0]
    assert torch.isfinite(gradient).all()


def test_missing_hydrogens_are_not_added():
    """A heavy-atom model is rejected and left untouched."""
    model = (
        Model(verbose=0, device="cpu", hydrogens="strip")
        .load_pdb(TEST_PDB)
        .strip_altlocs()
    )
    before = model.to_dataframe()
    wrapper = model.xyz
    with pytest.raises(ValueError, match="AMBER needs every one"):
        with pytest.warns(UserWarning, match="TorchRef did not add"):
            AmberTarget(model=model)
    pd.testing.assert_frame_equal(model.to_dataframe(), before)
    assert model.xyz is wrapper


def test_deposited_hydrogens_are_used_with_a_warning():
    """A file's own hydrogens are used as they are, with a warning every time.

    7L84 carries hydrogens on the protein but not on its waters: more than half of
    what the dictionaries call for, so it passes the count, and the waters are then
    reported as lacking hydrogens rather than handed to GAFF2.
    """
    model = (
        Model(verbose=0, device="cpu", hydrogens="keep")
        .load_pdb(TEST_PDB)
        .strip_altlocs()
    )
    with pytest.warns(UserWarning, match="TorchRef did not add"):
        with pytest.raises(ValueError, match=r"lacking hydrogens.*HOH"):
            AmberTarget(model=model)


@pytest.fixture(scope="module")
def water_target(pdb_dir):
    """Two nearby deposited waters with TorchRef-generated rotating hydrogens."""
    model = Model(verbose=0, device="cpu").load_pdb(str(pdb_dir / "1DAW.pdb"))
    table = model.to_dataframe()
    waters = table[table.resname.str.strip().eq("HOH")].copy()
    coords = waters[["x", "y", "z"]].to_numpy()
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = model._derive(
            waters.iloc[sorted([i, j])].copy(), hydrogens="add", hydrogen_mode="riding"
        )
    return AmberTarget(model=model, normalize_by_atoms=False)


def test_water_rotation_changes_amber_energy(water_target):
    """Rotating water H changes AMBER energy while oxygens remain fixed."""
    target = water_target
    model = target._model
    rotations = model.xyz.rotations.refinable_params
    before = rotations.detach().clone()
    positions = model.xyz().detach().clone()
    try:
        initial_energy = target.forward().detach()
        with torch.no_grad():
            rotations[0] += rotations.new_tensor([0.2, -0.3, 0.4])
        energy = target.forward()
        grad = torch.autograd.grad(energy, rotations)[0]
        assert not torch.allclose(energy, initial_energy)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert torch.equal(
            model.xyz()[model.xyz.base_row], positions[model.xyz.base_row]
        )
    finally:
        with torch.no_grad():
            rotations.copy_(before)


def test_torchref_hydrogenation_prepares_compatible_protein():
    """TorchRef can prepare all model hydrogens before AMBER construction."""
    model = (
        Model(verbose=0, device="cpu", hydrogens="strip")
        .load_pdb(TEST_PDB)
        .strip_altlocs()
        .hydrogenate()
    )
    model.set_hydrogen_mode("riding")
    positions = model.xyz().detach().clone()
    target = AmberTarget(model=model)
    assert target.adapter.n_particles == model.n_atoms
    assert torch.equal(model.xyz(), positions)
    assert torch.isfinite(target.forward())


def test_partial_terminal_hydrogens_preserve_h1_alias(protein):
    """Completing an existing terminal H1 adds H2/H3 without an equivalent H."""
    table = protein.to_dataframe()
    first = table.iloc[0]
    residue = (
        (table.chainid == first.chainid)
        & (table.resseq == first.resseq)
        & (table.icode == first.icode)
    )
    missing = residue & table.name.str.strip().isin(["H2", "H3"])
    partial = protein._derive(table.loc[~missing].copy(), hydrogens="keep")
    prepared = partial.hydrogenate()
    result = prepared.to_dataframe()
    first_residue = result[
        (result.chainid == first.chainid) & (result.resseq == first.resseq)
    ]
    names = set(first_residue.name.str.strip())
    assert {"H1", "H2", "H3"} <= names
    assert "H" not in names
    assert prepared.n_atoms == protein.n_atoms
    target = AmberTarget(model=prepared)
    assert torch.isfinite(target.forward())


def test_unknown_layout_is_refused(protein):
    with pytest.raises(ValueError, match="layout must be one of"):
        AmberTarget(model=protein, layout="supercell")
