"""
Construction tests for :class:`~torchref.experimental.ensemble.quasi_crystal_amber.QuasiCrystalAmberTarget`.

This file covers only the *construction* path — verifying that the
single-copy template build → sym + tile replication → supercell Context
build runs end-to-end without errors, atom counts come out as expected, and
the N % N_sym validation fires when it should. Forward-path tests come in a
follow-up increment once forward is implemented.
"""

import os

import pytest
import torch

# OpenMM is required for the target; skip the module entirely if it isn't
# available in the test env.
openmm = pytest.importorskip("openmm")

from torchref.io.datasets import ReflectionData
from torchref.experimental.ensemble import EnsembleModel
from torchref.experimental.ensemble.quasi_crystal_amber import (
    QuasiCrystalAmberTarget,
)


def _have_amber_tools() -> bool:
    """Check whether antechamber/tleap are available (required when the
    template build runs GAFF2 for non-standard residues like 3GR5's SO4)."""
    import shutil as _sh
    return _sh.which("antechamber") is not None and _sh.which("tleap") is not None


requires_amber_tools = pytest.mark.skipif(
    not _have_amber_tools(),
    reason="AmberTools (antechamber/tleap) not available — run under the "
           "conda env with ambertools installed for the full Amber tests.",
)


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
    """N=12 ensemble of 3GR5 (P 6_5 2 2 → N_sym=12 → n_disorder=1).

    3GR5 carries an SO4 ligand (charge -2) that antechamber parametrises
    via Gasteiger charges. Module-scoped because the antechamber + tleap
    pipeline is the slow part.
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
    )
    ens.cell = data.cell
    ens.spacegroup = data.spacegroup
    return ens, data


@requires_amber_tools
def test_construction_succeeds_n_disorder_1(small_setup):
    """Smallest valid case: n_members = 12 = 1 * N_sym."""
    ens, data = small_setup
    target = QuasiCrystalAmberTarget(
        model=ens,
        cell=data.cell,
        spacegroup=data.spacegroup,
        n_disorder=1,
        residue_charges={"SO4": -2},
        charge_method="gas",
        verbose=0,
    )
    assert target._n_members == 12
    assert target._n_sym == 12
    assert target._n_disorder == 1
    assert target._system.getNumParticles() == target._n_omm_total
    assert target._n_omm_total == 12 * target._n_omm_per_member


@requires_amber_tools
def test_n_members_must_match_layout(small_setup):
    """n_disorder * N_sym must equal model.n_members."""
    ens, data = small_setup
    with pytest.raises(ValueError, match="must equal n_disorder"):
        QuasiCrystalAmberTarget(
            model=ens,
            cell=data.cell,
            spacegroup=data.spacegroup,
            n_disorder=2,  # would need n_members=24; ens has 12
            residue_charges={"SO4": -2},
            charge_method="gas",
            verbose=0,
        )


@requires_amber_tools
def test_construction_pme_box_matches_supercell(small_setup):
    """Periodic box equals the small cell when n_disorder=1."""
    ens, data = small_setup
    target = QuasiCrystalAmberTarget(
        model=ens,
        cell=data.cell,
        spacegroup=data.spacegroup,
        n_disorder=1,
        residue_charges={"SO4": -2},
        charge_method="gas",
        verbose=0,
    )
    import openmm.unit as u_omm

    box = target._system.getDefaultPeriodicBoxVectors()
    cell_matrix_ang = data.cell.fractional_matrix.cpu().numpy()
    # Each box vector matches the corresponding column of B (Å → nm).
    for i in range(3):
        for k in range(3):
            got = box[i][k].value_in_unit(u_omm.nanometer)
            expected = float(cell_matrix_ang[k, i]) / 10.0
            assert abs(got - expected) < 1e-5, (
                f"box[{i}][{k}]: expected {expected}, got {got}"
            )


@requires_amber_tools
def test_forward_returns_finite_energy_with_gradient(small_setup):
    """forward() returns a finite energy and a *bounded* gradient.

    The energy value itself can be huge for any crystal with special-position
    atoms (e.g. 3GR5's HOH 224 sits on a 2-fold axis → 12 sym-mates overlap
    in the supercell → astronomical LJ). What matters for refinement is that
    the gradient stays bounded — which it does, because the autograd
    Function clamps per-atom forces (10000 kJ/mol/nm). Adam's per-parameter
    normalization then handles the initial step out of the clash gracefully.
    """
    ens, data = small_setup
    target = QuasiCrystalAmberTarget(
        model=ens,
        cell=data.cell,
        spacegroup=data.spacegroup,
        n_disorder=1,
        residue_charges={"SO4": -2},
        charge_method="gas",
        verbose=0,
    )
    ens.xyz.refinable_params.grad = None
    energy = target.forward()
    assert energy.ndim == 0, f"energy must be scalar; got shape {energy.shape}"
    assert torch.isfinite(energy), f"energy = {float(energy.detach())} not finite"
    energy.backward()
    g = ens.xyz.refinable_params.grad
    assert g is not None, "no gradient on ensemble xyz"
    assert torch.isfinite(g).all(), "non-finite gradient entries"
    # The clamp produces O(1) per-coord gradients — bounded regardless of the
    # raw energy value. If the gradient is huge here, the clamp is broken.
    assert float(g.abs().max()) < 100.0, (
        f"max |gradient| = {float(g.abs().max())} unexpectedly large; "
        "force-clamp may not be in effect"
    )
    assert (g.abs().sum(dim=-1) > 0).any(), "all atoms have zero gradient"


@requires_amber_tools
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
        n_disorder=1, residue_charges={"SO4": -2}, charge_method="gas",
        normalize_per_asu=True, verbose=0,
    )
    target_total = QuasiCrystalAmberTarget(
        model=ens, cell=data.cell, spacegroup=data.spacegroup,
        n_disorder=1, residue_charges={"SO4": -2}, charge_method="gas",
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
