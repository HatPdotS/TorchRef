"""Energies and forces cross between TorchRef and OpenMM through one explicit map.

Pins the bridge: positions reach OpenMM in nm at the particles the map names, the
gradient is minus OpenMM's force in kJ/mol/Å clipped per particle, dtype and device of
the caller are kept, riding-hydrogen parameters receive their share, and the context
is only created when something is evaluated.
"""

import numpy as np
import pytest
import torch

from torchref.experimental.mm import CrystalLayout, OpenMMAdapter
from torchref.model.model import Model

pytestmark = pytest.mark.openmm


@pytest.fixture(scope="module")
def protein(pdb_dir):
    model = (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "7L84.pdb"))
        .strip_altlocs()
    )
    model.set_hydrogen_mode("riding")
    return model


@pytest.fixture(scope="module")
def adapter(protein):
    """On the Reference platform, so a second OpenMM evaluation repeats the first."""
    return OpenMMAdapter.from_model(protein, platform="Reference")


@pytest.fixture(scope="module")
def waters(pdb_dir):
    """Two nearby deposited waters with TorchRef-generated rotating hydrogens."""
    model = Model(verbose=0, device="cpu").load_pdb(str(pdb_dir / "1DAW.pdb"))
    table = model.to_dataframe()
    water = table[table.resname.str.strip().eq("HOH")]
    coords = water[["x", "y", "z"]].to_numpy()
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    i, j = np.unravel_index(np.argmin(distance), distance.shape)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return model._derive(
            water.iloc[sorted([i, j])].copy(), hydrogens="add", hydrogen_mode="riding"
        )


def test_map_is_the_identity_for_a_whole_isolated_model(adapter, protein):
    assert adapter.n_particles == protein.n_atoms
    assert np.array_equal(adapter.particles, np.arange(protein.n_atoms))
    assert adapter._context is None


def test_positions_forces_units_and_sign(adapter, protein):
    """OpenMM sees model coordinates in nm; the gradient is -F in kJ/mol/Å, clipped."""
    import openmm.unit as unit

    xyz = protein.xyz().detach().clone()
    hydrogens = torch.as_tensor(protein.ctx.topology.atoms.is_hydrogen.cpu().numpy())
    with torch.no_grad():
        xyz[hydrogens] += xyz.new_tensor([0.003, -0.002, 0.001])
    xyz.requires_grad_()
    energy = adapter.energy(xyz)
    gradient = torch.autograd.grad(energy, xyz)[0]
    state = adapter.context.getState(getPositions=True, getForces=True, getEnergy=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.testing.assert_allclose(positions, xyz.detach().numpy() * 0.1, atol=1e-7)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
    )
    norms = np.linalg.norm(forces, axis=1, keepdims=True)
    clipped = forces * np.minimum(adapter.max_force / np.maximum(norms, 1e-10), 1.0)
    np.testing.assert_allclose(gradient.numpy(), -clipped * 0.1, rtol=1e-6, atol=1e-6)
    reference = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert energy.item() == pytest.approx(reference, rel=1e-6)
    assert gradient[hydrogens].abs().sum() > 0


def test_energy_and_forces_matches_autograd_below_the_clip(protein):
    """The diagnostic forces equal the autograd gradient wherever no clip applies.

    On the deterministic Reference platform, so the two evaluations agree to the
    float32 cast of the gradient.
    """
    adapter = OpenMMAdapter.from_model(protein, platform="Reference")
    xyz = protein.xyz().detach().clone().requires_grad_()
    energy, forces = adapter.energy_and_forces(xyz)
    gradient = torch.autograd.grad(adapter.energy(xyz), xyz)[0].numpy()
    small = np.linalg.norm(forces[0], axis=1) * 10 < adapter.max_force
    np.testing.assert_allclose(-forces[0][small], gradient[small], rtol=1e-6, atol=1e-6)
    assert energy == pytest.approx(adapter.energy(xyz).item(), rel=1e-6)


def test_group_energies_sum_to_the_total(adapter, protein):
    xyz = protein.xyz().detach()
    groups = adapter.group_energies(xyz)
    assert {"HarmonicBondForce", "HarmonicAngleForce", "NonbondedForce"} <= set(groups)
    assert sum(groups.values()) == pytest.approx(adapter.energy(xyz).item(), rel=1e-5)


def test_riding_parameters_receive_gradients(adapter, protein):
    """AMBER forces on hydrogens reach the methyl torsion parameters."""
    gradient = torch.autograd.grad(
        adapter.energy(protein.xyz()), protein.xyz.torsions.refinable_params
    )[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_atom_count_change_requires_rebuild(adapter, protein):
    with pytest.raises(ValueError, match="rebuild"):
        adapter.energy(protein.xyz()[:-1])


def test_water_rotation_derivative(waters):
    """The force bridge agrees with a finite difference in the orientation, float32."""
    adapter = OpenMMAdapter.from_model(waters, max_force=float("inf"))
    wrapper = waters.xyz
    base = wrapper._storage_values().detach()
    torsions = wrapper.torsions().detach()
    rotation = wrapper.rotations().detach().requires_grad_()

    def energy(rot):
        return adapter.energy(wrapper.evaluate(base, torsions, rot))

    gradient = torch.autograd.grad(energy(rotation), rotation)[0]
    direction = gradient.detach() / gradient.norm()
    step = 1e-3
    finite = (
        energy(rotation + step * direction) - energy(rotation - step * direction)
    ) / (2 * step)
    assert finite.item() == pytest.approx(
        (gradient * direction).sum().item(), rel=0.015, abs=0.1
    )


@pytest.mark.parametrize("use_reference_dtype", [False, True])
def test_bridge_preserves_dtype(waters, use_reference_dtype, double_cpu):
    from torchref.config import get_float_dtype

    adapter = OpenMMAdapter.from_model(waters)
    dtype = get_float_dtype() if use_reference_dtype else waters.xyz().dtype
    xyz = waters.xyz().detach().to(dtype=dtype).requires_grad_()
    loss = adapter.energy(xyz)
    gradient = torch.autograd.grad(loss, xyz)[0]
    assert loss.dtype == gradient.dtype == xyz.dtype
    assert torch.isfinite(gradient).all()


def test_bridge_follows_coordinate_device(waters, any_device):
    adapter = OpenMMAdapter.from_model(waters)
    xyz = waters.xyz().detach().to(any_device).requires_grad_()
    loss = adapter.energy(xyz)
    gradient = torch.autograd.grad(loss, xyz)[0]
    assert loss.device == gradient.device == xyz.device
    assert torch.isfinite(gradient).all()


def test_minimize_lowers_the_energy_and_keeps_shape(waters):
    adapter = OpenMMAdapter.from_model(waters, platform="Reference")
    xyz = waters.xyz().detach()
    relaxed = adapter.minimize(xyz, max_iterations=50)
    assert relaxed.shape == xyz.shape and relaxed.dtype == xyz.dtype
    assert adapter.energy(relaxed).item() <= adapter.energy(xyz).item()


def test_heavy_atom_model_is_rejected_before_openmm(pdb_dir):
    model = (
        Model(verbose=0, device="cpu", hydrogens="strip")
        .load_pdb(str(pdb_dir / "7L84.pdb"))
        .strip_altlocs()
    )
    with pytest.raises(ValueError, match="AMBER needs every one"):
        with pytest.warns(UserWarning):
            OpenMMAdapter.from_model(model)


def test_pme_needs_a_periodic_layout(waters):
    with pytest.raises(ValueError, match="non-periodic"):
        OpenMMAdapter.from_model(
            waters, nonbonded="pme", layout=CrystalLayout.isolated()
        )


def test_particle_gather_gradient_is_exact():
    """The row gather's copy-backward equals the gradient of plain indexing."""
    from torchref.experimental.mm.adapter import _TakeRows

    values = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    index = torch.tensor([7, 0, 3, 11, 5], dtype=torch.int64)
    assert torch.autograd.gradcheck(lambda v: _TakeRows.apply(v, index), (values,))


def test_whole_isolated_model_skips_the_gather(adapter):
    assert adapter._identity
