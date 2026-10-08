"""
Construction tests for :class:`~torchref.experimental.ensemble.quasi_crystal_amber.QuasiCrystalAmberTarget`.

Pins that the supercell holds every member under its symmetry operation, with a
PME box equal to the unit cell for one cell along a, that the N = n_disorder * N_sym
rule is enforced, that the energy is finite with bounded gradients once the
special-position water is held once per site, and the per-ASU normalisation.
"""

import os

import pytest
import torch

from torchref.io.datasets import ReflectionData
from torchref.experimental.ensemble import EnsembleModel
from torchref.experimental.ensemble.quasi_crystal_amber import (
    QuasiCrystalAmberTarget,
)

# The build runs GAFF2 for 3GR5's SO4, so this module needs OpenMM + AmberTools.
# Gated centrally in conftest.
pytestmark = pytest.mark.amber


# 3GR5 is altloc-free (verified) and the production reference structure,
# so it's used for the actual Amber construction tests. 1DAW is the smallest
# altloc-bearing PDB in tests/files/; it's used in a dedicated test below
# to verify that EnsembleModel.from_single strips altlocs by default.
TEST_PDB_AMBER = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "3GR5.pdb"
)
TEST_MTZ_AMBER = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "mtz", "3GR5.mtz"
)
TEST_PDB_ALTLOC = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "1DAW.pdb"
)


@pytest.fixture(scope="module")
def small_setup():
    """N=12 ensemble of 3GR5 (P 6_5 2 2 → N_sym=12 → n_disorder=1), hydrogenated.

    3GR5 carries SO4 (net charge -2 from its dictionary) that antechamber
    parameterises with Gasteiger charges.
    """
    data = ReflectionData(verbose=0)
    data.load_mtz(TEST_MTZ_AMBER)
    ens = EnsembleModel.from_single(
        TEST_PDB_AMBER,
        n_members=12,
        perturb_sigma=0.01,
        b_const=5.0,
        seed=0,
        verbose=0,
        hydrogens="add",
    )
    ens.cell = data.cell
    ens.spacegroup = data.spacegroup
    return ens, data


def test_construction_succeeds_n_disorder_1(small_setup):
    """Smallest valid case: n_members = 12 = 1 * N_sym."""
    ens, data = small_setup
    target = QuasiCrystalAmberTarget(
        model=ens,
        cell=data.cell,
        spacegroup=data.spacegroup,
        n_disorder=1,
        charge_method="gas",
        verbose=0,
    )
    assert target._n_members == 12
    assert target._n_sym == 12
    assert target._n_disorder == 1
    adapter = target.adapter
    assert adapter.system.getNumParticles() == adapter.n_particles
    assert adapter.layout.n_copies == 12
    # HOH 224 sits on a two-fold: six of its twelve copies coincide with others.
    assert (~adapter.present).sum() == 6
    assert adapter.n_particles == 12 * ens.n_atoms_per_member - 6 * 3


def test_n_members_must_match_layout(small_setup):
    """n_disorder * N_sym must equal model.n_members."""
    ens, data = small_setup
    with pytest.raises(ValueError, match="must equal n_disorder"):
        QuasiCrystalAmberTarget(
            model=ens,
            cell=data.cell,
            spacegroup=data.spacegroup,
            n_disorder=2,  # would need n_members=24; ens has 12
                charge_method="gas",
            verbose=0,
        )


def test_construction_pme_box_matches_supercell(small_setup):
    """Periodic box equals the small cell when n_disorder=1."""
    ens, data = small_setup
    target = QuasiCrystalAmberTarget(
        model=ens,
        cell=data.cell,
        spacegroup=data.spacegroup,
        n_disorder=1,
        charge_method="gas",
        verbose=0,
    )
    import openmm.unit as u_omm

    box = target.adapter.system.getDefaultPeriodicBoxVectors()
    cell_matrix_ang = data.cell.fractional_matrix.cpu().numpy()
    # Each box vector matches the corresponding column of B (Å → nm).
    for i in range(3):
        for k in range(3):
            got = box[i][k].value_in_unit(u_omm.nanometer)
            expected = float(cell_matrix_ang[k, i]) / 10.0
            assert abs(got - expected) < 1e-5, (
                f"box[{i}][{k}]: expected {expected}, got {got}"
            )


def test_forward_returns_finite_energy_with_gradient(small_setup):
    """forward() returns a protein-scale energy and a bounded gradient.

    3GR5's HOH 224 sits on a two-fold axis; held once per site, it does not stack
    on its own copy. Generated hydrogens of neighbouring copies can still clash, and
    the per-atom clip (10000 kJ/mol/nm) bounds the gradient there.
    """
    ens, data = small_setup
    target = QuasiCrystalAmberTarget(
        model=ens,
        cell=data.cell,
        spacegroup=data.spacegroup,
        n_disorder=1,
        charge_method="gas",
        verbose=0,
    )
    ens.xyz.refinable_params.grad = None
    energy = target.forward()
    assert energy.ndim == 0, f"energy must be scalar; got shape {energy.shape}"
    assert torch.isfinite(energy), f"energy = {float(energy.detach())} not finite"
    assert abs(energy.item()) < 1e6, f"energy per ASU = {energy.item()} kJ/mol"
    energy.backward()
    g = ens.xyz.refinable_params.grad
    assert g is not None, "no gradient on ensemble xyz"
    assert torch.isfinite(g).all(), "non-finite gradient entries"
    assert float(g.abs().max()) < 1000.0, (
        f"max |gradient| = {float(g.abs().max())} unexpectedly large; "
        "force-clamp may not be in effect"
    )
    assert (g.abs().sum(dim=-1) > 0).any(), "all atoms have zero gradient"


def test_forward_per_asu_normalization(small_setup):
    """``forward()`` returns supercell-energy / n_members when
    ``normalize_per_asu=True`` (default).

    Build two identical targets, one with normalization on and one off, run
    forward on each, and assert the ratio matches ``n_members`` (= the number
    of ASU copies in the supercell, = 12 for the n_disorder=1 P6_5 2 2 case).
    """
    ens, data = small_setup

    target_per_asu = QuasiCrystalAmberTarget(
        model=ens, cell=data.cell, spacegroup=data.spacegroup,
        n_disorder=1, charge_method="gas",
        normalize_per_asu=True, verbose=0,
    )
    target_total = QuasiCrystalAmberTarget(
        model=ens, cell=data.cell, spacegroup=data.spacegroup,
        n_disorder=1, charge_method="gas",
        normalize_per_asu=False, verbose=0,
    )
    with torch.no_grad():
        e_per_asu = float(target_per_asu.forward())
        e_total = float(target_total.forward())
    n_members = target_per_asu._n_members
    assert n_members == 12, f"P6_5 2 2 supercell at n_disorder=1: expected 12, got {n_members}"
    # Allow tiny numerical drift from two separate OpenMM contexts.
    ratio = e_total / e_per_asu
    assert abs(ratio - n_members) / n_members < 1e-6, (
        f"normalize_per_asu=True should divide total by n_members={n_members}; "
        f"got ratio e_total/e_per_asu = {ratio}"
    )


def test_altloc_stripping_at_ensemble_creation():
    """EnsembleModel.from_single must drop alternate conformations from
    the per-member atom layout — required so OpenMM topology / FFT / Amber
    don't see double-counted atoms. Uses 1DAW (which has altlocs A and B)."""
    ens = EnsembleModel.from_single(
        TEST_PDB_ALTLOC,
        n_members=2,
        perturb_sigma=0.0,
        b_const=5.0,
        seed=0,
        verbose=0,
    )
    altloc = ens._pdb_single["altloc"].astype(str).str.strip()
    # No row should carry a non-blank altloc after stripping.
    assert (altloc == "").all(), (
        f"_pdb_single still has altlocs: {altloc.unique().tolist()}"
    )
