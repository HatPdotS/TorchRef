"""AMBER evaluates the model's complete coordinates and returns atomic gradients."""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from torchref.config import get_int_dtype
from torchref.experimental.targets.amber_target import AmberTarget, _OpenMMAMBERFunction
from torchref.model.model import Model

pytestmark = pytest.mark.openmm
TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "7L84.pdb"
)


@pytest.fixture(scope="module")
def protein():
    """A deposited, hydrogenated protein with methyl orientation parameters."""
    model = Model(verbose=0, device="cpu", add_hydrogens=True).load_pdb(TEST_PDB)
    model = model.strip_altlocs()
    model.set_hydrogen_mode("riding")
    return model


@pytest.fixture(scope="module")
def target(protein):
    """Build the expensive AMBER context once per module."""
    return AmberTarget(model=protein, verbose=0)


def test_build_preserves_model(protein):
    """Target construction neither changes model atoms nor replaces coordinates."""
    before = protein.pdb.copy(deep=True)
    xyz = protein.xyz
    positions = xyz().detach().clone()
    target = AmberTarget(model=protein)
    pd.testing.assert_frame_equal(protein.pdb, before)
    assert protein.xyz is xyz
    assert torch.equal(protein.xyz(), positions)
    assert target._n_omm_atoms == target._n_model_atoms == len(protein.pdb)
    assert np.array_equal(np.sort(target._model_to_omm), np.arange(len(protein.pdb)))
    h1 = int(np.flatnonzero(protein.pdb.name.str.strip().eq("H1"))[0])
    assert h1 in protein.xyz.h_row.tolist()


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


def test_every_position_and_force_has_model_order(target, protein):
    """All H positions are supplied live, with force sign, units and normalization."""
    import openmm.unit as unit

    xyz = protein.xyz().detach().clone().requires_grad_()
    h_rows = torch.tensor(np.flatnonzero(protein.pdb.element.str.strip().eq("H")))
    with torch.no_grad():
        xyz[h_rows] += xyz.new_tensor([0.003, -0.002, 0.001])
    energy = target._energy(xyz)
    gradient = torch.autograd.grad(energy, xyz)[0]
    state = target._context.getState(getPositions=True, getForces=True, getEnergy=True)
    pos = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    np.testing.assert_allclose(
        pos[target._model_to_omm], xyz.detach().numpy() * 0.1, atol=1e-7
    )
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
    )
    scale = np.minimum(
        10000 / np.maximum(np.linalg.norm(forces, axis=1, keepdims=True), 1e-10), 1
    )
    expected = (
        -forces[target._model_to_omm] * scale[target._model_to_omm] * 0.1 / len(xyz)
    )
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=3e-6, atol=1e-5)
    assert gradient[h_rows].abs().sum() > 0
    expected_energy = state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    ) / len(xyz)
    assert energy.item() == pytest.approx(expected_energy, rel=1e-6)


def test_permuted_positions_and_gradients(target, protein):
    """The cached inverse map handles arbitrary model/OpenMM atom permutations."""
    xyz = protein.xyz().detach().clone().requires_grad_()
    reference = target._energy(xyz)
    reference_grad = torch.autograd.grad(reference, xyz)[0]
    perm = torch.arange(len(xyz) - 1, -1, -1)
    inverse = torch.argsort(perm)
    original = target._omm_to_model.clone()
    try:
        target._omm_to_model = inverse[original].to(get_int_dtype())
        permuted = xyz.detach()[perm].requires_grad_()
        loss = target._energy(permuted)
        grad = torch.autograd.grad(loss, permuted)[0]
        assert torch.allclose(loss, reference)
        assert torch.allclose(grad, reference_grad[perm])
    finally:
        target._omm_to_model = original


def test_missing_hydrogens_are_not_added():
    """Disabled generation leaves a heavy-only model untouched on AMBER rejection."""
    model = (
        Model(verbose=0, device="cpu", strip_H=True, add_hydrogens=False)
        .load_pdb(TEST_PDB)
        .strip_altlocs()
    )
    before = model.pdb.copy(deep=True)
    wrapper = model.xyz
    with pytest.raises(ValueError, match="Prepare missing atoms"):
        AmberTarget(model=model)
    pd.testing.assert_frame_equal(model.pdb, before)
    assert model.xyz is wrapper


@pytest.fixture(scope="module")
def water_target(pdb_dir):
    """Two nearby deposited waters with TorchRef-generated rotating hydrogens."""
    model = Model(verbose=0, device="cpu", add_hydrogens=False).load_pdb(
        str(pdb_dir / "1DAW.pdb")
    )
    waters = model.pdb[model.pdb.resname.str.strip().eq("HOH")].copy()
    coords = waters[["x", "y", "z"]].to_numpy()
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    model = model._new_model_from_df(waters.iloc[sorted([i, j])].copy(), strip_H=False)
    model.ctx.add_hydrogens = True
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model.set_hydrogen_mode("riding")
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


def test_unclipped_water_rotation_derivative(water_target):
    """AMBER's force bridge agrees with an orientation finite difference in float32."""
    target = water_target
    wrapper = target._model.xyz
    base = wrapper._storage_values().detach()
    torsions = wrapper.torsions().detach()
    rotation = wrapper.rotations().detach().requires_grad_()

    def energy(rot):
        xyz = wrapper.evaluate(base, torsions, rot)
        return _OpenMMAMBERFunction.apply(
            target._compose_full_omm_xyz(xyz), target._context, float("inf")
        )

    gradient = torch.autograd.grad(energy(rotation), rotation)[0]
    direction = gradient.detach() / gradient.norm()
    step = 1e-3
    finite = (
        energy(rotation + step * direction) - energy(rotation - step * direction)
    ) / (2 * step)
    assert finite.item() == pytest.approx(
        (gradient * direction).sum().item(), rel=0.015, abs=0.1
    )


def test_atom_count_change_requires_rebuild(target, protein):
    """A target rejects coordinates from an expanded or reduced atom table."""
    with pytest.raises(ValueError, match="Atom count changed"):
        target._energy(protein.xyz()[:-1])


@pytest.mark.parametrize("ligand_instances", [False, True])
def test_gaff2_mapping_preserves_residue_instances(water_target, ligand_instances):
    """Residue and atom permutations cannot exchange two identical molecules."""
    import openmm as mm
    import openmm.app as app

    model = water_target._model
    source = list(water_target._topology.residues())
    keys = list(
        dict.fromkeys(
            model.pdb[["chainid", "resseq", "icode"]].itertuples(index=False, name=None)
        )
    )
    topology = app.Topology()
    chain = topology.addChain("renumbered")
    positions = []
    expected = []
    for residue in reversed(source):
        dest = topology.addResidue(residue.name, chain, str(len(positions) + 10))
        mapped = {}
        for atom in reversed(list(residue.atoms())):
            mapped[atom.index] = topology.addAtom(atom.name, atom.element, dest)
            row = int(water_target._omm_to_model[atom.index])
            expected.append(row)
            positions.append(model.xyz()[row].detach().numpy() * 0.1)
        for a, b in water_target._topology.bonds():
            if a.index in mapped and b.index in mapped:
                topology.addBond(mapped[a.index], mapped[b.index])
    candidate = AmberTarget()
    candidate._chem_model = model
    candidate._topology = topology
    candidate._system = mm.System()
    for _ in expected:
        candidate._system.addParticle(1)
    candidate._tleap_residue_map = True
    candidate._gaff2_residue_keys = list(reversed(keys)) if ligand_instances else []
    candidate._tleap_pos_nm = np.asarray(positions)
    if ligand_instances:
        candidate._tleap_pos_nm += 1000
    candidate._build_atom_map()
    assert candidate._omm_to_model.tolist() == expected


def test_duplicate_particle_mapping_is_rejected(water_target):
    """A duplicate source index cannot silently send two forces to one atom."""
    candidate = AmberTarget()
    candidate._chem_model = water_target._model
    candidate._topology = water_target._topology
    candidate._system = water_target._system
    candidate._source_model_rows = np.zeros(water_target._n_model_atoms, dtype=np.int32)
    with pytest.raises(ValueError, match="not one-to-one"):
        candidate._build_atom_map()


def test_supercell_gather_keeps_live_hydrogens(water_target):
    """The ensemble path transfers all transformed hydrogen rows unchanged."""
    from torchref.experimental.ensemble.quasi_crystal_amber import (
        QuasiCrystalAmberTarget,
    )

    candidate = QuasiCrystalAmberTarget.__new__(QuasiCrystalAmberTarget)
    torch.nn.Module.__init__(candidate)
    n_atoms = water_target._n_model_atoms
    candidate._n_members = 2
    candidate._n_model_per_member = n_atoms
    candidate._omm_to_model = torch.arange(n_atoms - 1, -1, -1, dtype=get_int_dtype())
    xyz = water_target._model.xyz().detach().unsqueeze(0).repeat(2, 1, 1) * 0.1
    xyz[1] += 0.3
    xyz.requires_grad_()
    result = candidate._compose_full_omm_xyz(xyz).reshape(2, n_atoms, 3)
    assert torch.equal(result.flip(1), xyz)
    weights = torch.arange(result.numel(), dtype=xyz.dtype).reshape_as(result)
    gradient = torch.autograd.grad((result * weights).sum(), xyz)[0]
    assert torch.equal(gradient, weights.flip(1))


@pytest.mark.parametrize("use_reference_dtype", [False, True])
def test_bridge_preserves_input_dtype(water_target, use_reference_dtype, double_cpu):
    """The OpenMM boundary preserves configured reference and primary dtypes."""
    from torchref.config import get_float_dtype

    dtype = (
        get_float_dtype() if use_reference_dtype else water_target._model.xyz().dtype
    )
    xyz = water_target._model.xyz().detach().to(dtype=dtype).requires_grad_()
    loss = water_target._energy(xyz)
    gradient = torch.autograd.grad(loss, xyz)[0]
    assert loss.dtype == gradient.dtype == xyz.dtype
    assert torch.isfinite(gradient).all()


def test_bridge_follows_coordinate_device(water_target, any_device):
    """Coordinates and gradients stay on the caller's device across the CPU bridge."""
    xyz = water_target._model.xyz().detach().to(any_device).requires_grad_()
    loss = water_target._energy(xyz)
    gradient = torch.autograd.grad(loss, xyz)[0]
    assert loss.device == gradient.device == xyz.device
    assert torch.isfinite(gradient).all()


def test_torchref_hydrogenation_prepares_compatible_protein():
    """TorchRef can prepare all model hydrogens before AMBER construction."""
    model = (
        Model(verbose=0, device="cpu", strip_H=True, add_hydrogens=False)
        .load_pdb(TEST_PDB)
        .strip_altlocs()
        .hydrogenate()
    )
    model.set_hydrogen_mode("riding")
    positions = model.xyz().detach().clone()
    target = AmberTarget(model=model)
    assert target._n_omm_atoms == len(model.pdb)
    assert torch.equal(model.xyz(), positions)
    assert torch.isfinite(target.forward())


def test_context_initialization_uses_live_model(water_target, monkeypatch):
    """Backend template coordinates cannot replace the model's live positions."""
    candidate = AmberTarget()
    candidate._chem_model = water_target._model
    candidate._source_model_rows = water_target._omm_to_model.cpu().numpy()
    stale_positions = np.zeros_like(water_target._pos_buf)
    monkeypatch.setattr(
        candidate,
        "_build_omm_system",
        lambda params: (water_target._system, water_target._topology, stale_positions),
    )
    captured = []
    monkeypatch.setattr(
        candidate, "_build_context", lambda positions: captured.append(positions.copy())
    )
    candidate._build()
    expected = (
        water_target._compose_full_omm_xyz(water_target._model.xyz())
        .detach()
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(captured[0], expected)
    np.testing.assert_array_equal(candidate._pos_buf, expected)


def test_partial_terminal_hydrogens_preserve_h1_alias(protein):
    """Completing an existing terminal H1 adds H2/H3 without an equivalent H."""
    pdb = protein.pdb
    first = pdb.iloc[0]
    residue = (
        (pdb.chainid == first.chainid)
        & (pdb.resseq == first.resseq)
        & (pdb.icode == first.icode)
    )
    missing = residue & pdb.name.str.strip().isin(["H2", "H3"])
    partial = protein._new_model_from_df(pdb.loc[~missing].copy(), strip_H=False)
    prepared = partial.hydrogenate()
    first_residue = prepared.pdb[
        (prepared.pdb.chainid == first.chainid) & (prepared.pdb.resseq == first.resseq)
    ]
    names = set(first_residue.name.str.strip())
    assert {"H1", "H2", "H3"} <= names
    assert "H" not in names
    assert len(prepared.pdb) == len(protein.pdb)
    target = AmberTarget(model=prepared)
    assert torch.isfinite(target.forward())
