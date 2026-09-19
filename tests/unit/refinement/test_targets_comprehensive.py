"""
Comprehensive unit tests for targets module.

These tests focus on individual target classes with mock/minimal data
to achieve higher coverage of the targets module.
"""

import numpy as np
import pytest
import torch

# =============================================================================
# Base Target Tests
# =============================================================================


@pytest.mark.unit
class TestBaseTarget:
    """Test base Target class functionality."""

    def test_target_initialization_empty(self):
        """Test empty initialization for state_dict loading."""
        from torchref.refinement.targets import Target

        target = Target()
        assert target.verbose == 0
        assert isinstance(target, torch.nn.Module)

    def test_target_initialization_with_verbose(self):
        """Test initialization with verbose setting."""
        from torchref.refinement.targets import Target

        target = Target(verbose=2)
        assert target.verbose == 2

    def test_target_forward_not_implemented(self):
        """Test that forward raises NotImplementedError."""
        from torchref.refinement.targets import Target

        target = Target()
        with pytest.raises(NotImplementedError):
            target.forward()


# =============================================================================
# X-ray Target Tests
# =============================================================================


@pytest.mark.unit
class TestXrayTargetBase:
    """Test XrayTarget base class."""

    def test_xray_target_initialization(self):
        """Test XrayTarget initialization."""
        from torchref.refinement.targets import XrayTarget

        target = XrayTarget()
        assert target._model is None
        assert target._data is None
        assert target._scaler is None


@pytest.mark.unit
class TestNLLXrayTarget:
    """Test NLLXrayTarget (the sigma-weighted Gaussian NLL, ``--xray-mode nll``)."""

    def test_gaussian_target_initialization(self):
        """Test NLLXrayTarget initialization."""
        from torchref.refinement.targets import NLLXrayTarget

        target = NLLXrayTarget()
        assert target._model is None
        assert target._data is None

        # NLL can be negative depending on normalization


@pytest.mark.unit
class TestRiceXrayTarget:
    """Test RiceXrayTarget."""

    def test_rice_target_initialization(self):
        """Test RiceXrayTarget initialization."""
        from torchref.refinement.targets import RiceXrayTarget

        target = RiceXrayTarget()
        assert target._model is None
        assert target._data is None


# =============================================================================
# Geometry Target Tests
# =============================================================================


@pytest.mark.unit
class TestGeometryTargetBase:
    """Test GeometryTarget base class."""

    def test_geometry_target_initialization(self):
        """Test GeometryTarget initialization."""
        from torchref.refinement.targets import GeometryTarget

        target = GeometryTarget()
        assert target._model is None


@pytest.mark.unit
class TestBondTarget:
    """Test BondTarget."""

    def test_bond_target_initialization(self):
        """Test BondTarget initialization."""
        from torchref.refinement.targets import BondTarget

        target = BondTarget()
        assert target._model is None


@pytest.mark.unit
class TestAngleTarget:
    """Test AngleTarget."""

    def test_angle_target_initialization(self):
        """Test AngleTarget initialization."""
        from torchref.refinement.targets import AngleTarget

        target = AngleTarget()
        assert target._model is None


@pytest.mark.unit
class TestTorsionTarget:
    """Test TorsionTarget."""

    def test_torsion_target_initialization(self):
        """Test TorsionTarget initialization."""
        from torchref.refinement.targets import TorsionTarget

        target = TorsionTarget()
        assert target._model is None


@pytest.mark.unit
class TestPlanarityTarget:
    """Test PlanarityTarget."""

    def test_planarity_target_initialization(self):
        """Test PlanarityTarget initialization."""
        from torchref.refinement.targets import PlanarityTarget

        target = PlanarityTarget()
        assert target._model is None


@pytest.mark.unit
class TestChiralTarget:
    """Test ChiralTarget."""

    def test_chiral_target_initialization(self):
        """Test ChiralTarget initialization."""
        from torchref.refinement.targets import ChiralTarget

        target = ChiralTarget()
        assert target._model is None


@pytest.mark.unit
class TestNonBondedTarget:
    """Test NonBondedTarget."""

    def test_nonbonded_target_initialization(self):
        """Test NonBondedTarget initialization."""
        from torchref.refinement.targets import NonBondedTarget

        target = NonBondedTarget()
        assert target._model is None


@pytest.mark.unit
class TestTotalGeometryTarget:
    """Test TotalGeometryTarget."""

    def test_total_geometry_target_initialization(self):
        """Test TotalGeometryTarget initialization."""
        from torchref.refinement.targets import TotalGeometryTarget

        target = TotalGeometryTarget()
        assert target._model is None


# =============================================================================
# ADP Target Tests
# =============================================================================


@pytest.mark.unit
class TestADPTargetBase:
    """Test ADPTarget base class."""

    def test_adp_target_initialization(self):
        """Test ADPTarget initialization."""
        from torchref.refinement.targets import ADPTarget

        target = ADPTarget()
        assert target._model is None


@pytest.mark.unit
class TestRigidBondTarget:
    """Test RigidBondTarget (DELU restraint)."""

    def test_aniso_path_runs_and_routes_grad_to_u(self, pdb_dir):
        """The anisotropic DELU path actually executes and feeds gradient to the
        U tensors. Regression for the dead ``hasattr(model, "u_aniso")`` gate,
        which made the aniso branch unreachable so DELU silently used the iso
        ΔB proxy even for anisotropic models. Bond pairs are injected directly to
        avoid building the full monomer-library restraints in the unit env."""
        from torchref.model.model_ft import ModelFT
        from torchref.refinement.targets.adp.rigid_bond import RigidBondTarget

        m = ModelFT()
        m.load_pdb(str(pdb_dir / "7L84.pdb"))
        m.set_adp_mode("anisotropic")
        assert not m._aniso_is_empty  # genuinely anisotropic

        tgt = RigidBondTarget(m)
        aniso_idx = m.aniso_flag.nonzero(as_tuple=True)[0][:6]
        pairs = torch.stack([aniso_idx[:-1], aniso_idx[1:]], dim=1)
        tgt._bond_pairs = lambda: pairs  # bypass restraint building

        loss = tgt.forward()  # routes to the aniso path (aniso atoms present)
        assert torch.isfinite(loss)
        m.zero_grad(set_to_none=True)
        loss.backward()
        g = m.u.refinable_params.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.unit
class TestADPSigdTarget:
    """Test ADPSigdTarget, the shifted inverse-gamma ADP distribution prior.

    These exercise ``adp_sigd_math`` directly rather than through a Model: the
    kernel is the whole of the restraint, and the properties below are what make
    it a correct replacement for the log-normal KL term it superseded. Two of
    them (monotonicity in spread, finiteness for uniform B) are regressions
    against defects in that old term.
    """

    def test_sigd_target_initialization(self):
        """Defaults are M&M's alpha and an unshifted distribution."""
        from torchref.refinement.targets import ADPSigdTarget

        target = ADPSigdTarget()
        assert target._model is None
        assert target.alpha == pytest.approx(3.5)
        assert target.b_shift == pytest.approx(0.0)

    def test_matches_inverse_gamma_nll(self):
        """The kernel equals scipy's inverse-gamma NLL, offset at the mode."""
        from scipy import stats as sps

        from torchref.base.targets.adp import adp_sigd_math

        alpha = 3.5
        rng = np.random.default_rng(0)
        b = torch.tensor(
            np.exp(rng.standard_normal(5000) * 0.4) * 30.0, dtype=torch.float64
        )
        a = torch.tensor(alpha, dtype=torch.float64)
        s0 = torch.tensor(0.0, dtype=torch.float64)

        beta = float(b.mean()) * (alpha - 1.0)
        mode = beta / (alpha + 1.0)
        expected = -sps.invgamma.logpdf(
            b.numpy(), alpha, scale=beta
        ).sum() + sps.invgamma.logpdf(mode, alpha, scale=beta) * len(b)
        assert float(adp_sigd_math(b, a, s0)) == pytest.approx(expected, rel=1e-10)

    def test_monotonically_increasing_in_spread(self):
        """The loss must never reward spreading the B distribution out.

        Regression: the log-normal KL this replaced had a negative slope below
        its 0.2 target, so it actively widened an over-tight distribution.
        """
        from torchref.base.targets.adp import adp_sigd_math

        a = torch.tensor(3.5, dtype=torch.float64)
        s0 = torch.tensor(0.0, dtype=torch.float64)
        rng = np.random.default_rng(0)
        base = rng.standard_normal(200000)

        losses = []
        for sigma in (0.10, 0.20, 0.30, 0.38, 0.45, 0.55, 0.70):
            b = torch.tensor(np.exp(base * sigma), dtype=torch.float64)
            b = b * 30.0 / b.mean()  # hold the mean fixed, vary only the spread
            losses.append(float(adp_sigd_math(b, a, s0)) / len(b))

        assert all(hi > lo for lo, hi in zip(losses, losses[1:])), losses

    def test_finite_for_uniform_b(self):
        """Uniform B is finite and differentiable.

        Regression: the old KL divided by std(log B) and returned +inf here,
        which LBFGS rejected as non-finite, leaving the B-factors stuck uniform.
        """
        from torchref.base.targets.adp import adp_sigd_math

        alpha = 3.5
        a = torch.tensor(alpha, dtype=torch.float64)
        s0 = torch.tensor(0.0, dtype=torch.float64)
        b = torch.full((500,), 30.0, dtype=torch.float64, requires_grad=True)

        loss = adp_sigd_math(b, a, s0)
        assert torch.isfinite(loss)

        # Closed form for a uniform distribution:
        #   (alpha+1) log((alpha+1)/(alpha-1)) - 2   per atom
        expected = (alpha + 1.0) * np.log((alpha + 1.0) / (alpha - 1.0)) - 2.0
        assert float(loss.detach()) / 500 == pytest.approx(expected)

        loss.backward()
        assert torch.isfinite(b.grad).all()

    def test_scale_invariant(self):
        """Scaling every B leaves the loss unchanged.

        beta tracks the detached mean, so the term restrains the distribution's
        shape only and can never drive the overall B level up or down.
        """
        from torchref.base.targets.adp import adp_sigd_math

        a = torch.tensor(3.5, dtype=torch.float64)
        s0 = torch.tensor(0.0, dtype=torch.float64)
        rng = np.random.default_rng(0)
        b = torch.tensor(
            np.exp(rng.standard_normal(2000) * 0.4) * 30.0, dtype=torch.float64
        )

        assert float(adp_sigd_math(b * 7.0, a, s0)) == pytest.approx(
            float(adp_sigd_math(b, a, s0)), rel=1e-10
        )

    def test_gradient_pushes_toward_the_mode(self):
        """Descent raises a B below the mode and lowers one above it."""
        from torchref.base.targets.adp import adp_sigd_math

        alpha = 3.5
        a = torch.tensor(alpha, dtype=torch.float64)
        s0 = torch.tensor(0.0, dtype=torch.float64)
        # With beta from the detached mean, the mode is (alpha-1)/(alpha+1) of it.
        mode = 30.0 * (alpha - 1.0) / (alpha + 1.0)
        b = torch.tensor(
            [0.4 * mode, 3.0 * mode], dtype=torch.float64, requires_grad=True
        )
        # Pad so the mean (and hence beta) is pinned near 30 regardless of the two
        # probe atoms, isolating the per-atom gradient direction.
        pad = torch.full((2000,), 30.0, dtype=torch.float64)
        adp_sigd_math(torch.cat([b, pad]), a, s0).backward()

        assert b.grad[0] < 0  # below the mode -> descent increases B
        assert b.grad[1] > 0  # above the mode -> descent decreases B


# =============================================================================
# R-factor Tests
# =============================================================================


@pytest.mark.unit
class TestRfactorCalculations:
    """Test R-factor calculation functions."""

    def test_get_rfactors_basic(self):
        """Test basic R-factor calculation."""
        from torchref.base.math_torch import get_rfactors

        fobs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        fcalc = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], dtype=torch.float32)

        # Create rfree mask (1 reflection in test set)
        rfree_mask = torch.tensor([True, True, True, True, False], dtype=torch.bool)

        r_work, r_free = get_rfactors(fobs, fcalc, rfree_mask)

        # Both should be small since fcalc is close to fobs
        assert r_work < 0.2
        # r_free only has one reflection

    def test_get_rfactors_perfect_fit(self):
        """Test R-factor with perfect fit."""
        from torchref.base.math_torch import get_rfactors

        fobs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        fcalc = fobs.clone()  # Perfect fit

        rfree_mask = torch.tensor([True, True, True, True, False], dtype=torch.bool)

        r_work, r_free = get_rfactors(fobs, fcalc, rfree_mask)

        assert r_work < 0.001  # Should be ~0

    def test_bin_wise_rfactors(self):
        """Test bin-wise R-factor calculation."""
        from torchref.base.math_torch import bin_wise_rfactors

        n_refl = 100
        n_bins = 5

        fobs = torch.rand(n_refl) + 1.0
        fcalc = fobs * (1 + 0.1 * torch.randn(n_refl))
        # Note: rfree=True means work set (not free set)
        rfree_mask = torch.rand(n_refl) > 0.1

        # Ensure all bins are represented
        bins = torch.arange(n_refl) % n_bins

        r_work_bins, r_free_bins = bin_wise_rfactors(fobs, fcalc, rfree_mask, bins)

        # Should have results for each bin
        assert len(r_work_bins) == n_bins
        assert len(r_free_bins) == n_bins


# =============================================================================
# Loss Function Tests
# =============================================================================


# =============================================================================
# Helper Function Tests
# =============================================================================
