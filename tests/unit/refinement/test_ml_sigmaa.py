"""Tests for the 'ml' target (Read MLF with ML model-error variance beta).

- factory dispatch / thin target
- loss math: exact reduction to the unit-variance ML target at beta=eps=1
- gradient isolation (beta is a detached constant)
- end-to-end on 1DAW: beta owned by the target's SigmaAEstimator, maintenance
  resets the cache, gradients reach the model.
"""

import pytest
import torch

from torchref.base.targets.xray_ml import _ml_xray_loss_math_eager
from torchref.base.targets.xray_ml_sigmaa import ml_xray_loss_beta_math


@pytest.mark.unit
class TestMaximumLikelihoodXrayTarget:
    def test_init_thin(self):
        from torchref.refinement.targets import MaximumLikelihoodXrayTarget

        target = MaximumLikelihoodXrayTarget()
        assert target._model is None and target._data is None and target._scaler is None
        # σ_A (beta) is owned by the target via a SigmaAEstimator; the scaler
        # owns scaling only.
        from torchref.base.targets.xray_ml_sigmaa import SigmaAEstimator

        assert isinstance(target._sigma_a, SigmaAEstimator)

    def test_factory_dispatch(self):
        from torchref.refinement.targets import (
            MaximumLikelihoodXrayTarget,
            create_xray_target,
        )

        assert isinstance(
            create_xray_target(mode="ml"), MaximumLikelihoodXrayTarget
        )

    def test_factory_ml_sigmaa_alias_deprecated(self):
        """'ml_sigmaa' still resolves to the 'ml' target, with a warning."""
        from torchref.refinement.targets import (
            MaximumLikelihoodXrayTarget,
            create_xray_target,
        )

        with pytest.warns(DeprecationWarning):
            target = create_xray_target(mode="ml_sigmaa")
        assert isinstance(target, MaximumLikelihoodXrayTarget)

    def test_factory_unknown_mode_raises(self):
        from torchref.refinement.targets import create_xray_target

        with pytest.raises(ValueError):
            create_xray_target(mode="not_a_mode")

    def test_factory_default_is_ml(self):
        """'ml' (the σ_A Read MLF target) is the promoted default."""
        from torchref.refinement.targets import (
            MaximumLikelihoodXrayTarget,
            create_xray_target,
        )

        assert isinstance(create_xray_target(), MaximumLikelihoodXrayTarget)

    def test_lbfgs_default_target_mode(self):
        import inspect

        from torchref import LBFGSRefinement

        sig = inspect.signature(LBFGSRefinement.__init__)
        assert sig.parameters["target_mode"].default == "ml"


@pytest.mark.unit
class TestCollectionTargetsRelocated:
    """Collection targets are exported from refinement.targets and re-exported
    from kinetic.targets for back-compat."""

    def test_exported_from_refinement_targets(self):
        from torchref.refinement.targets import (  # noqa: F401
            CollectionDifferenceTarget,
            CollectionMLTarget,
            CollectionRiceTarget,
            MultiModelADPTarget,
            MultiModelGeometryTarget,
        )

    def test_kinetic_backcompat_reexports(self):
        from torchref.experimental.kinetic.targets import (  # noqa: F401
            CollectionRiceTarget,
        )
        from torchref.experimental.kinetic.targets import CollectionMLTarget as KinCML
        from torchref.experimental.kinetic.targets import (  # noqa: F401
            KineticPriorTarget,
            _scale_fcalc,
        )
        from torchref.refinement.targets import CollectionMLTarget as RefCML

        assert RefCML is KinCML

    def test_collection_ml_base_weight(self):
        from torchref.refinement.targets import CollectionMLTarget

        assert CollectionMLTarget.DEFAULT_BASE_WEIGHT == 10.0
        # maintenance hook present (resets the target's own shared beta)
        assert hasattr(CollectionMLTarget, "maintenance")


@pytest.mark.unit
class TestBetaMath:
    def _inputs(self, n=400, seed=0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        F_obs = torch.rand(n, generator=g, dtype=dtype) * 100
        F_calc = torch.rand(n, generator=g, dtype=dtype) * 80 + 1
        centric = torch.rand(n, generator=g) < 0.3
        mask = torch.rand(n, generator=g) < 0.9
        return F_obs, F_calc, centric, mask

    def test_reduces_to_unit_variance_ml(self):
        """beta=1, eps=1 ⇒ Sigma=1 ⇒ unit-variance MLF (== ml at sigma=1)."""
        F_obs, F_calc, centric, mask = self._inputs()
        one = torch.ones_like(F_obs)
        new = ml_xray_loss_beta_math(F_obs, F_calc, one, centric, mask, one)
        old = _ml_xray_loss_math_eager(F_obs, F_calc, one, centric, mask)
        assert new.item() == pytest.approx(old.item(), rel=1e-9, abs=1e-6)

    def test_gradient_isolation(self):
        n = 200
        g = torch.Generator().manual_seed(3)
        F_obs = torch.rand(n, generator=g, dtype=torch.float64) * 100
        F_calc = (
            torch.rand(n, generator=g, dtype=torch.float64) * 80 + 1
        ).requires_grad_(True)
        beta = torch.full((n,), 30.0, dtype=torch.float64)  # detached constant
        eps = torch.ones(n, dtype=torch.float64)
        centric = torch.zeros(n, dtype=torch.bool)
        mask = torch.ones(n, dtype=torch.bool)
        loss = ml_xray_loss_beta_math(F_obs, F_calc, beta, centric, mask, eps)
        loss.backward()
        assert F_calc.grad is not None and torch.isfinite(F_calc.grad).all()
        assert not beta.requires_grad


@pytest.mark.integration
class TestSigmaAOnRealData:
    @pytest.fixture(scope="class")
    def refinement(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "1DAW.pdb"
        mtz = mtz_dir / "1DAW.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("1DAW fixture not present")
        from torchref import LBFGSRefinement

        ref = LBFGSRefinement(
            data_file=str(mtz), pdb=str(pdb), target_mode="ml", verbose=0
        )
        ref.scaler.initialize()
        ref.scaler.refine_lbfgs()
        return ref

    def test_beta_owned_by_target(self, refinement):
        from torchref.base.targets.xray_ml_sigmaa import SigmaAEstimator
        from torchref.scaling import Scaler

        # The scaler owns scaling only — no beta machinery.
        assert isinstance(refinement.scaler, Scaler)
        assert not hasattr(refinement.scaler, "get_beta")
        # beta lives in the target's own SigmaAEstimator.
        assert isinstance(refinement.xray_target_work._sigma_a, SigmaAEstimator)
        # both targets still share the one scaler (scaling layer).
        assert (
            refinement.xray_target_work._scaler
            is refinement.scaler
            is refinement.xray_target_test._scaler
        )

    def test_beta_physical_and_falls_with_resolution(self, refinement):
        from torchref.base.reciprocal import get_scattering_vectors
        from torchref.base.targets.xray_ml_sigmaa import (
            epsilon_from_hkl,
            estimate_beta,
        )

        t = refinement.xray_target_work
        assert torch.isfinite(t.forward())
        beta, eps = t._sigma_a._cache
        v = refinement.reflection_data.masks().to(torch.bool)
        assert (beta[v] > 0).all() and torch.isfinite(beta[v]).all()

        # The falling-with-resolution trend is an estimator-math property,
        # checked in float64 so it is device-independent. Move to CPU before
        # casting -- MPS has no float64, so the double() must happen off-device.
        data = refinement.reflection_data
        with torch.no_grad():
            fc = torch.abs(t._scaled_F_calc_full()).cpu().double().reshape(-1)
            fo = data.get_corrected_data()[0].cpu().double().reshape(-1)
            epsd = epsilon_from_hkl(
                data.hkl, getattr(data, "spacegroup", None)
            ).cpu().double()
            s = get_scattering_vectors(data.hkl, data.cell)
            dss = (torch.norm(s, dim=1) ** 2).cpu().double()
            _b, bbin, _ = estimate_beta(
                fo, fc, data.centric.cpu(), epsd, dss, data.free.mask.cpu()
            )
        assert (bbin > 0).all() and torch.isfinite(bbin).all()
        # beta is an absolute model-error variance in F^2 units (~(1-sigma_A^2)*
        # Sigma_N); Sigma_N ~= <F^2> decays steeply with resolution, so absolute
        # beta falls: the low-resolution half exceeds the high-resolution half.
        h = bbin.numel() // 2
        assert bbin[:h].mean() > bbin[h:].mean()

    def test_maintenance_resets_cache(self, refinement):
        t = refinement.xray_target_work
        t.forward()  # populates the estimator cache
        assert t._sigma_a._cache is not None
        t.maintenance()
        assert t._sigma_a._cache is None

    def test_gradient_reaches_model(self, refinement):
        refinement.xray_target_work.forward().backward()
        xyz = refinement.xray_target_work._model.xyz.refinable_params
        assert xyz.grad is not None and torch.isfinite(xyz.grad).all()
