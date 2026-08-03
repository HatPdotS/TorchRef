"""Tests for the experimental monolithic_refinement module.

The Read-MLF Rice likelihood driven by a differentiable, co-refined model-error
variance ``beta = c * sigma_m**2`` (no free-set beta, no maintenance cache), plus
the MonolithicRefinement driver that co-refines everything in one joint step:

- target construction / subclassing / no-op maintenance
- reduction: the beta slot is the same rice_math kernel
- end-to-end on 1DAW: work/test share one calibration; gradients reach BOTH the
  model B-factors and the co-refined calibration; the calibration is identifiable
  (settles finite); and it actually moves in the monolithic refine_joint step.
"""

import pytest
import torch

from torchref.base.targets.xray_likelihoods import complex_var_from_beta, rice_math


@pytest.mark.unit
class TestRiceSigmaMTarget:
    def test_importable(self):
        from torchref.experimental.monolithic_refinement import (  # noqa: F401
            MonolithicRefinement,
            RiceSigmaMXrayTarget,
        )

    def test_owns_a_sigma_m_estimator(self):
        """The Bhattacharyya target it used to subclass is gone; the estimator survived.

        `RiceSigmaMXrayTarget` inherited from `BhattacharyyaXrayTarget` purely to borrow
        the Fisher sigma_m machinery. That machinery is now
        `torchref.refinement.model_error_estimation.sigma_m.SigmaMEstimator` and the Bhattacharyya loss
        was deleted, so ownership replaces inheritance.
        """
        from torchref.refinement.model_error_estimation.sigma_m import SigmaMEstimator
        from torchref.experimental.monolithic_refinement.targets import (
            RiceSigmaMXrayTarget,
        )

        t = RiceSigmaMXrayTarget()
        assert isinstance(t._sigma_m, SigmaMEstimator)
        # constructible with no data/model, which the other tests here rely on
        assert t._sigma_m.ready is False

    def test_owns_calibration_parameter(self):
        from torchref.experimental.monolithic_refinement import RiceSigmaMXrayTarget

        t = RiceSigmaMXrayTarget()
        assert hasattr(t, "log_sigma_m_scale")
        assert t.log_sigma_m_scale.requires_grad
        assert t.log_sigma_m_scale.numel() == 1  # global by default

    def test_maintenance_is_noop(self):
        from torchref.experimental.monolithic_refinement import RiceSigmaMXrayTarget

        RiceSigmaMXrayTarget().maintenance()  # must not raise / touch any cache


@pytest.mark.unit
class TestBetaSlot:
    def test_beta_slot_is_ml_kernel(self):
        """beta = c * sigma_m**2 folded into the kernel == applying c first."""
        n = 300
        g = torch.Generator().manual_seed(1)
        F_obs = torch.rand(n, generator=g, dtype=torch.float64) * 100
        F_calc = torch.rand(n, generator=g, dtype=torch.float64) * 80 + 1
        centric = torch.rand(n, generator=g) < 0.3
        eps = torch.ones(n, dtype=torch.float64)
        sigma_m_sq = torch.rand(n, generator=g, dtype=torch.float64) * 50 + 1
        c = 7.0
        a = rice_math(
            F_obs, F_calc, complex_var_from_beta(c * sigma_m_sq, eps), centric
        )
        b = rice_math(
            F_obs, F_calc,
            complex_var_from_beta((c * sigma_m_sq).clamp(min=1e-10), eps), centric,
        )
        assert a.item() == pytest.approx(b.item(), rel=1e-12, abs=1e-9)


@pytest.mark.integration
class TestMonolithicOnRealData:
    @pytest.fixture(scope="class")
    def refinement(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "1DAW.pdb"
        mtz = mtz_dir / "1DAW.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("1DAW fixture not present")
        from torchref.experimental.monolithic_refinement import MonolithicRefinement

        return MonolithicRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0)

    def test_targets_are_rice_sigma_m(self, refinement):
        from torchref.experimental.monolithic_refinement import RiceSigmaMXrayTarget

        assert isinstance(refinement.xray_target_work, RiceSigmaMXrayTarget)
        assert isinstance(refinement.xray_target_test, RiceSigmaMXrayTarget)

    def test_work_and_test_share_calibration(self, refinement):
        # one c, so R-free stats see the same calibration the work loss refines
        assert (
            refinement.xray_target_work.log_sigma_m_scale
            is refinement.xray_target_test.log_sigma_m_scale
        )

    def test_forward_finite(self, refinement):
        assert torch.isfinite(refinement.xray_target_work.forward())

    def test_gradient_reaches_bfactors(self, refinement):
        target = refinement.xray_target_work
        b = target._model.adp.refinable_params
        if b.grad is not None:
            b.grad.zero_()
        target.forward().backward()
        assert b.grad is not None and torch.isfinite(b.grad).all()
        assert b.grad.abs().sum() > 0

    def test_gradient_reaches_calibration(self, refinement):
        target = refinement.xray_target_work
        c = target.log_sigma_m_scale
        if c.grad is not None:
            c.grad.zero_()
        target.forward().backward()
        assert c.grad is not None and torch.isfinite(c.grad).all()
        assert c.grad.abs().sum() > 0

    def test_calibration_identifiable(self, refinement):
        """Optimizing c alone settles finite (the +log Sigma term pins it)."""
        target = refinement.xray_target_work
        c = target.log_sigma_m_scale
        with torch.no_grad():
            c.zero_()
        opt = torch.optim.Adam([c], lr=0.2)
        losses = []
        for _ in range(40):
            opt.zero_grad()
            loss = target.forward()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert torch.isfinite(c).all()
        assert abs(c.item()) < 15.0
        assert losses[-1] <= losses[0] + 1e-6


@pytest.mark.integration
class TestMonolithicCorefine:
    def test_calibration_moves_in_refine_joint(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "1DAW.pdb"
        mtz = mtz_dir / "1DAW.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("1DAW fixture not present")
        from torchref.experimental.monolithic_refinement import MonolithicRefinement

        ref = MonolithicRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0)
        c0 = ref.xray_target_work.log_sigma_m_scale.detach().clone()
        rw0, _ = ref.get_rfactor()
        ref.refine_monolithic(n_steps=3)
        c1 = ref.xray_target_work.log_sigma_m_scale.detach().clone()
        rw1, _ = ref.get_rfactor()
        assert not torch.allclose(c0, c1)  # error-model calib co-refined
        assert torch.isfinite(c1).all()
        # monolithic step is productive (doesn't diverge); allow LBFGS noise.
        assert rw1 <= rw0 + 1e-3
        assert torch.isfinite(ref.xray_target_work.forward())
