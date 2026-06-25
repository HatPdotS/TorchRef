"""
Unit tests for
:class:`~torchref.experimental.ensemble.supercell.SupercellLayout`.

Covers the pure tensor math: identity sym op + n_disorder=1 round-trip, tile
shift, sym op application, 3GR5 P 6_5 2 2 shape sanity, and gradient flow.
"""

import pytest
import torch

from torchref.experimental.ensemble.supercell import SupercellLayout


def _identity_layout(n_disorder: int = 1) -> SupercellLayout:
    """Identity sym op on a 10×10×10 cubic cell."""
    return SupercellLayout(
        cell=torch.eye(3) * 10.0,
        sym_rotations=torch.eye(3).unsqueeze(0),
        sym_translations=torch.zeros(1, 3),
        n_disorder=n_disorder,
    )


def test_identity_passthrough():
    """Identity sym op + n_disorder=1 must reproduce input exactly."""
    layout = _identity_layout(n_disorder=1)
    xyz = torch.randn(1, 5, 3)
    out = layout.compute_member_positions(xyz)
    assert out.shape == (1, 5, 3)
    assert torch.allclose(out, xyz, atol=1e-6)


def test_tile_shift_only():
    """n_disorder=3, identity sym op: tile d gets a shift of d·a_vec."""
    layout = _identity_layout(n_disorder=3)
    # 1 atom at (1, 2, 3); replicate to N=3 disorder copies.
    one = torch.tensor([[[1.0, 2.0, 3.0]]])
    xyz = one.expand(3, 1, 3).contiguous()
    out = layout.compute_member_positions(xyz)
    expected = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],   # d=0, no shift
            [[11.0, 2.0, 3.0]],  # d=1, shifted by 10 along a
            [[21.0, 2.0, 3.0]],  # d=2, shifted by 20 along a
        ]
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_sym_op_rotation():
    """N_sym=2 with a C2-about-z rotation. Member 0 = identity, member 1 = C2.

    C2 sends (x, y, z) -> (-x, -y, z). On a 10x10x10 cubic cell the
    fractional and Cartesian rotations are identical.
    """
    cell = torch.eye(3) * 10.0
    R = torch.stack(
        [
            torch.eye(3),
            torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        ]
    )
    t = torch.zeros(2, 3)
    layout = SupercellLayout(
        cell=cell, sym_rotations=R, sym_translations=t, n_disorder=1
    )
    assert layout.n_sym == 2
    assert layout.n_members == 2

    # Same input coords (1, 2, 3) on both members.
    one = torch.tensor([[[1.0, 2.0, 3.0]]])
    xyz = one.expand(2, 1, 3).contiguous()
    out = layout.compute_member_positions(xyz)
    expected = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],     # identity
            [[-1.0, -2.0, 3.0]],   # C2-z
        ]
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_sym_op_with_translation():
    """Sym op with a non-zero fractional translation (e.g., 2_1 screw axis)."""
    cell = torch.eye(3) * 10.0
    R = torch.eye(3).unsqueeze(0)
    t = torch.tensor([[0.5, 0.0, 0.0]])  # half-cell shift along a (fractional)
    layout = SupercellLayout(
        cell=cell, sym_rotations=R, sym_translations=t, n_disorder=1
    )
    xyz = torch.tensor([[[1.0, 2.0, 3.0]]])
    out = layout.compute_member_positions(xyz)
    # +0.5 along a in fractional = +5 in Cartesian (cell length 10).
    expected = torch.tensor([[[6.0, 2.0, 3.0]]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_supercell_vectors_property():
    """supercell_vectors should scale axis a by n_disorder, keep b, c."""
    cell = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 30.0],
        ]
    ).T  # columns are lattice vectors
    R = torch.eye(3).unsqueeze(0)
    t = torch.zeros(1, 3)
    layout = SupercellLayout(
        cell=cell, sym_rotations=R, sym_translations=t, n_disorder=4
    )
    sv = layout.supercell_vectors
    assert torch.allclose(sv[:, 0], cell[:, 0] * 4.0)
    assert torch.allclose(sv[:, 1], cell[:, 1])
    assert torch.allclose(sv[:, 2], cell[:, 2])


def test_gradient_flows_to_model_xyz():
    """Gradient on supercell positions must propagate to model_xyz."""
    layout = _identity_layout(n_disorder=2)
    xyz = torch.randn(2, 4, 3, requires_grad=True)
    out = layout.compute_member_positions(xyz)
    out.sum().backward()
    assert xyz.grad is not None
    assert torch.isfinite(xyz.grad).all()
    assert (xyz.grad != 0).all()


def test_3gr5_layout_dims():
    """For 3GR5 (P 6_5 2 2 → N_sym=12) with n_disorder=8: N=96 members."""
    from torchref.symmetry import Cell, SpaceGroup

    # 3GR5 hexagonal cell (approximate).
    cell = Cell([85.7, 85.7, 141.4, 90.0, 90.0, 120.0]).fractional_matrix.cpu()
    sg = SpaceGroup("P 65 2 2")
    layout = SupercellLayout(
        cell=cell,
        sym_rotations=sg.matrices.cpu(),
        sym_translations=sg.translations.cpu(),
        n_disorder=8,
    )
    assert layout.n_sym == 12
    assert layout.n_members == 96

    xyz = torch.randn(96, 50, 3)
    out = layout.compute_member_positions(xyz)
    assert out.shape == (96, 50, 3)
    assert torch.isfinite(out).all()


def test_invalid_shapes_raise():
    """Construction with wrong-shaped tensors should error clearly."""
    R = torch.eye(3).unsqueeze(0)
    t = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="cell must be"):
        SupercellLayout(
            cell=torch.eye(4), sym_rotations=R, sym_translations=t, n_disorder=1
        )
    with pytest.raises(ValueError, match="sym_rotations must be"):
        SupercellLayout(
            cell=torch.eye(3),
            sym_rotations=torch.eye(3),  # missing leading n_sym axis
            sym_translations=t,
            n_disorder=1,
        )
    with pytest.raises(ValueError, match="sym_translations must be"):
        SupercellLayout(
            cell=torch.eye(3),
            sym_rotations=R,
            sym_translations=torch.zeros(2, 3),  # n_sym mismatch
            n_disorder=1,
        )
    with pytest.raises(ValueError, match="n_disorder"):
        SupercellLayout(
            cell=torch.eye(3),
            sym_rotations=R,
            sym_translations=t,
            n_disorder=0,
        )


def test_member_count_mismatch_raises():
    """compute_member_positions should error if model_xyz N doesn't match."""
    layout = _identity_layout(n_disorder=3)  # n_members = 3
    bad = torch.randn(5, 4, 3)
    with pytest.raises(ValueError, match="does not match layout"):
        layout.compute_member_positions(bad)


# --------------------------------------------------------------------------
# Supercell System replication
# --------------------------------------------------------------------------

openmm = pytest.importorskip("openmm")


def _build_minimal_template(n_atoms: int = 4):
    """Synthetic 4-atom OpenMM System: 3 bonds + 2 angles + 1 torsion + NB.

    Small enough to test replication counts exactly, large enough that each
    force class is exercised.
    """
    import openmm.unit as u_omm

    sys = openmm.System()
    for _ in range(n_atoms):
        sys.addParticle(12.0 * u_omm.amu)  # carbon mass for testing

    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15 * u_omm.nanometer, 1e5 * u_omm.kilojoule_per_mole / u_omm.nanometer ** 2)
    bonds.addBond(1, 2, 0.15 * u_omm.nanometer, 1e5 * u_omm.kilojoule_per_mole / u_omm.nanometer ** 2)
    bonds.addBond(2, 3, 0.15 * u_omm.nanometer, 1e5 * u_omm.kilojoule_per_mole / u_omm.nanometer ** 2)
    sys.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 1.911, 500.0)
    angles.addAngle(1, 2, 3, 1.911, 500.0)
    sys.addForce(angles)

    torsions = openmm.PeriodicTorsionForce()
    torsions.addTorsion(0, 1, 2, 3, 3, 0.0, 8.0)
    sys.addForce(torsions)

    nb = openmm.NonbondedForce()
    for _ in range(n_atoms):
        nb.addParticle(0.0, 0.3, 0.5)  # neutral, σ=0.3 nm, ε=0.5
    # Standard 1-2/1-3 exceptions (no nonbonded between bonded neighbours).
    nb.addException(0, 1, 0.0, 1.0, 0.0)
    nb.addException(1, 2, 0.0, 1.0, 0.0)
    nb.addException(2, 3, 0.0, 1.0, 0.0)
    nb.addException(0, 2, 0.0, 1.0, 0.0)
    nb.addException(1, 3, 0.0, 1.0, 0.0)
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
    nb.setCutoffDistance(0.5 * u_omm.nanometer)
    sys.addForce(nb)

    return sys


def test_supercell_system_particle_count():
    """Particles = n_members × n_template."""
    from torchref.experimental.ensemble.supercell import (
        _replicate_to_supercell_system,
    )

    template = _build_minimal_template(n_atoms=4)
    layout = SupercellLayout(
        cell=torch.eye(3) * 10.0,
        sym_rotations=torch.eye(3).unsqueeze(0),
        sym_translations=torch.zeros(1, 3),
        n_disorder=5,
    )
    new = _replicate_to_supercell_system(template, layout)
    assert new.getNumParticles() == 5 * 4


def test_supercell_system_force_counts():
    """Each replicated force has n_members × template force count entries."""
    from torchref.experimental.ensemble.supercell import (
        _replicate_to_supercell_system,
    )

    template = _build_minimal_template(n_atoms=4)
    n_members = 6
    layout = SupercellLayout(
        cell=torch.eye(3) * 10.0,
        sym_rotations=torch.eye(3).unsqueeze(0),
        sym_translations=torch.zeros(1, 3),
        n_disorder=n_members,
    )
    new = _replicate_to_supercell_system(template, layout)

    # Map force type to handle from the new System.
    forces = {type(f).__name__: f for f in new.getForces()}
    assert "HarmonicBondForce" in forces
    assert "HarmonicAngleForce" in forces
    assert "PeriodicTorsionForce" in forces
    assert "NonbondedForce" in forces

    assert forces["HarmonicBondForce"].getNumBonds() == n_members * 3
    assert forces["HarmonicAngleForce"].getNumAngles() == n_members * 2
    assert forces["PeriodicTorsionForce"].getNumTorsions() == n_members * 1
    # 4 particles × n_members; 5 exceptions × n_members.
    assert forces["NonbondedForce"].getNumParticles() == n_members * 4
    assert forces["NonbondedForce"].getNumExceptions() == n_members * 5


def test_supercell_system_pme_and_box():
    """NonbondedForce uses PME with the supercell PBC vectors."""
    from torchref.experimental.ensemble.supercell import (
        _replicate_to_supercell_system,
    )

    template = _build_minimal_template(n_atoms=4)
    layout = SupercellLayout(
        cell=torch.eye(3) * 10.0,
        sym_rotations=torch.eye(3).unsqueeze(0),
        sym_translations=torch.zeros(1, 3),
        n_disorder=3,
    )
    new = _replicate_to_supercell_system(template, layout, pme_cutoff_nm=0.8)

    nb = next(f for f in new.getForces() if isinstance(f, openmm.NonbondedForce))
    assert nb.getNonbondedMethod() == openmm.NonbondedForce.PME
    # cutoff comes back as a Quantity; compare in nm.
    import openmm.unit as u_omm
    assert abs(nb.getCutoffDistance().value_in_unit(u_omm.nanometer) - 0.8) < 1e-9

    # Supercell PBC: k × a along axis a, b/c unchanged.  In nm: 3 × 1.0 nm = 3.0 nm.
    a_vec, b_vec, c_vec = new.getDefaultPeriodicBoxVectors()
    assert abs(a_vec[0].value_in_unit(u_omm.nanometer) - 3.0) < 1e-6  # k × 10 Å → 3 nm
    assert abs(b_vec[1].value_in_unit(u_omm.nanometer) - 1.0) < 1e-6
    assert abs(c_vec[2].value_in_unit(u_omm.nanometer) - 1.0) < 1e-6


def test_supercell_system_atom_indices_offset_correctly():
    """Bond indices in member m should sit in [m·n_template, (m+1)·n_template)."""
    from torchref.experimental.ensemble.supercell import (
        _replicate_to_supercell_system,
    )

    template = _build_minimal_template(n_atoms=4)
    n_members = 5
    layout = SupercellLayout(
        cell=torch.eye(3) * 10.0,
        sym_rotations=torch.eye(3).unsqueeze(0),
        sym_translations=torch.zeros(1, 3),
        n_disorder=n_members,
    )
    new = _replicate_to_supercell_system(template, layout)

    bonds = next(f for f in new.getForces() if isinstance(f, openmm.HarmonicBondForce))
    # Template has 3 bonds: (0,1), (1,2), (2,3). Member m's bonds should be
    # at indices 3m, 3m+1, 3m+2 with atom indices offset by m·4.
    for m in range(n_members):
        for k_in_template, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
            p1, p2, _length, _K = bonds.getBondParameters(m * 3 + k_in_template)
            assert p1 == m * 4 + a, (
                f"member {m} bond {k_in_template}: p1 expected {m*4+a}, got {p1}"
            )
            assert p2 == m * 4 + b


def test_supercell_runs_an_energy_eval():
    """Smoke-test: build a Context on the supercell System and get an energy.

    Positions: 4-atom chain (one along a small Cartesian path) replicated
    via the layout's compute_member_positions. Verifies the System is
    self-consistent (no OpenMM validation error) and an energy comes out.
    """
    from torchref.experimental.ensemble.supercell import (
        _replicate_to_supercell_system,
    )
    import openmm.unit as u_omm

    template = _build_minimal_template(n_atoms=4)
    layout = SupercellLayout(
        cell=torch.eye(3) * 10.0,
        sym_rotations=torch.eye(3).unsqueeze(0),
        sym_translations=torch.zeros(1, 3),
        n_disorder=3,
    )
    new = _replicate_to_supercell_system(template, layout, pme_cutoff_nm=0.5)

    # Build positions: 4-atom chain in Å, replicated to 3 members.
    chain = torch.tensor([[0.0, 0.0, 0.0],
                          [1.5, 0.0, 0.0],
                          [3.0, 0.0, 0.0],
                          [4.5, 0.0, 0.0]])
    xyz = chain.unsqueeze(0).expand(3, 4, 3).contiguous()
    import numpy as np

    pos_ang = layout.compute_member_positions(xyz)  # (N, n_atoms, 3)
    # setPositions expects flat (N·n_atoms, 3).
    pos_nm = (pos_ang / 10.0).reshape(-1, 3).cpu().numpy().astype(np.float64)

    integrator = openmm.VerletIntegrator(1.0 * u_omm.femtoseconds)
    # CPU platform avoids GPU dependency in unit tests.
    platform = openmm.Platform.getPlatformByName("CPU")
    ctx = openmm.Context(new, integrator, platform)
    ctx.setPositions(pos_nm)
    state = ctx.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(u_omm.kilojoule_per_mole)
    # Energy must be finite (positive or negative, but not NaN/inf).
    assert energy == energy  # NaN check
    assert abs(energy) < 1e8, f"energy = {energy} kJ/mol unexpectedly large"
