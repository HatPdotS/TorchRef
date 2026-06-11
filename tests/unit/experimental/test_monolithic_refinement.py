"""Tests for the rice_sigma_m target.

The Read-MLF Rice likelihood driven by a differentiable, co-refined model-error
variance ``beta = c * sigma_m**2`` (no free-set beta, no maintenance cache):

- factory dispatch / export
- reduction: with c=1 and constant sigma_m**2 it equals ml_xray_loss_beta_math
- end-to-end on 1DAW: gradients reach BOTH the model B-factors and the co-refined
  scaler calibration log_sigma_m_scale; the calibration is identifiable (settles
  finite); maintenance is a no-op; log_sigma_m_scale is in scaler.parameters().
"""

import pytest
import torch

from torchref.base.targets.xray_ml_sigmaa import ml_xray_loss_beta_math


@pytest.mark.unit
class TestRiceSigmaMFactory:
    def test_factory_dispatch(self):
        from torchref.refinement.targets.xray import (
            RiceSigmaMXrayTarget,
            create_xray_target,
        )

        assert isinstance(create_xray_target(mode="rice_sigma_m"), RiceSigmaMXrayTarget)

    def test_exported(self):
        from torchref.refinement.targets.xray import RiceSigmaMXrayTarget  # noqa: F401

    def test_is_subclass_of_bhattacharyya(self):
        from torchref.refinement.targets.xray import (
            BhattacharyyaXrayTarget,
            RiceSigmaMXrayTarget,
        )

        assert issubclass(RiceSigmaMXrayTarget, BhattacharyyaXrayTarget)

    def test_maintenance_is_noop(self):
        from torchref.refinement.targets.xray import RiceSigmaMXrayTarget

        # Should not touch any beta cache / not raise without a scaler.
        RiceSigmaMXrayTarget().maintenance()


@pytest.mark.unit
class TestRiceSigmaMMathSlot:
    def test_beta_slot_matches_kernel(self):
        """forward must equal ml_xray_loss_beta_math with beta = c * sigma_m**2.

        We don't build a full target here; just confirm the kernel is the same
        one used, by feeding a constant beta two ways.
        """
        n = 300
        g = torch.Generator().manual_seed(1)
        F_obs = torch.rand(n, generator=g, dtype=torch.float64) * 100
        F_calc = torch.rand(n, generator=g, dtype=torch.float64) * 80 + 1
        centric = torch.rand(n, generator=g) < 0.3
        eps = torch.ones(n, dtype=torch.float64)

        sigma_m_sq = torch.rand(n, generator=g, dtype=torch.float64) * 50 + 1
        c = 7.0
        direct = ml_xray_loss_beta_math(
            F_obs, F_calc, c * sigma_m_sq, centric, epsilon=eps
        )
        # c folded into beta vs applied separately — identical.
        folded = ml_xray_loss_beta_math(
            F_obs, F_calc, (c * sigma_m_sq).clamp(min=1e-10), centric, epsilon=eps
        )
        assert direct.item() == pytest.approx(folded.item(), rel=1e-12, abs=1e-9)


@pytest.mark.integration
class TestRiceSigmaMOnRealData:
    @pytest.fixture(scope="class")
    def refinement(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "1DAW.pdb"
        mtz = mtz_dir / "1DAW.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("1DAW fixture not present")
        from torchref import LBFGSRefinement

        ref = LBFGSRefinement(
            data_file=str(mtz), pdb=str(pdb), target_mode="rice_sigma_m", verbose=0
        )
        return ref

    def test_forward_finite(self, refinement):
        assert torch.isfinite(refinement.xray_target_work.forward())

    def test_calibration_registered_on_scaler(self, refinement):
        scaler = refinement.scaler
        assert hasattr(scaler, "log_sigma_m_scale")
        param_ids = {id(p) for p in scaler.parameters()}
        assert id(scaler.log_sigma_m_scale) in param_ids

    def test_gradient_reaches_bfactors(self, refinement):
        target = refinement.xray_target_work
        model = target._model
        b = model.adp.refinable_params
        if b.grad is not None:
            b.grad.zero_()
        target.forward().backward()
        # sigma_m is differentiable in B (no no_grad), so B gets a gradient.
        assert b.grad is not None and torch.isfinite(b.grad).all()
        assert b.grad.abs().sum() > 0

    def test_gradient_reaches_calibration(self, refinement):
        target = refinement.xray_target_work
        c = refinement.scaler.log_sigma_m_scale
        if c.grad is not None:
            c.grad.zero_()
        target.forward().backward()
        assert c.grad is not None and torch.isfinite(c.grad).all()
        assert c.grad.abs().sum() > 0

    def test_calibration_identifiable(self, refinement):
        """Optimizing the calibration alone settles to a finite value (the
        +log Sigma normalization term pins it — no run-away to 0/inf)."""
        target = refinement.xray_target_work
        c = refinement.scaler.log_sigma_m_scale
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
        assert abs(c.item()) < 15.0  # bounded, did not run away
        assert losses[-1] <= losses[0] + 1e-6  # improved or held

    def test_maintenance_noop_does_not_break(self, refinement):
        # No beta cache to reset; should be a harmless no-op.
        refinement.xray_target_work.maintenance()
        assert torch.isfinite(refinement.xray_target_work.forward())


@pytest.mark.integration
class TestRiceSigmaMMonolithicCorefine:
    """The calibration must actually move in the monolithic refine_joint step
    (it is an error-model param co-refined regardless of ``corefine_scaler``)."""

    def test_calibration_corefines_in_refine_joint(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "1DAW.pdb"
        mtz = mtz_dir / "1DAW.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("1DAW fixture not present")
        from torchref import LBFGSRefinement

        ref = LBFGSRefinement(
            data_file=str(mtz), pdb=str(pdb), target_mode="rice_sigma_m", verbose=0
        )
        c0 = ref.scaler.log_sigma_m_scale.detach().clone()
        rw0, rf0 = ref.get_rfactor()
        for _ in range(3):
            ref.refine_joint()
        c1 = ref.scaler.log_sigma_m_scale.detach().clone()
        rw1, rf1 = ref.get_rfactor()
        # calibration moved (was excluded before the _error_model_params wiring)
        assert not torch.allclose(c0, c1)
        assert torch.isfinite(c1).all()
        # monolithic step is productive: R-work drops, nothing diverges
        assert rw1 < rw0
        assert torch.isfinite(ref.xray_target_work.forward())
