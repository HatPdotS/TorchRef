"""Tests for the 'ml' target (Read MLF with ML model-error variance beta).

- factory dispatch / thin target
- loss math: exact reduction to the unit-variance ML target at beta=eps=1
- gradient isolation (beta is a detached constant)
- end-to-end on 1DAW: beta owned by the target's SigmaAEstimator, maintenance
  resets the cache, gradients reach the model.
"""

import math

import pytest
import torch

from torchref.base.targets.xray_likelihoods import complex_var_from_beta, rice_math


def _unit_variance_mlf_closed_form(F_obs, F_calc, sigma, centric_flags, mask):
    """The unit-variance MLF, written out here on purpose as an INDEPENDENT oracle.

    Deliberately NOT imported from the library. This test used to compare the beta-Rice
    against the eager unit-variance Rice in ``base/targets/xray_ml.py``; the 2026-08 loss
    consolidation made both the same :func:`rice_math` primitive, so importing either would
    make the comparison production-against-itself -- circular, and the test vacuous while
    still passing. Six lines of arithmetic here is a strictly stronger oracle. (That module
    was deleted as a duplicate Rice.)

        acentric: -log(2 F_o/eb) + F_o^2/eb + F_c^2/eb - log I0(2 F_o F_c/eb)
        centric : -0.5 log(2/(pi eb)) + (F_o^2 + F_c^2)/(2 eb) - log cosh(F_o F_c/eb)

    with ``eb = clamp(sigma**2, min=1e-6)`` -- the floor the deleted module applied, which is
    NOT the primitive's own ``VAR_FLOOR`` (1e-10) and fires on genuinely weak data.
    """
    Fc = torch.abs(F_calc)
    eb = torch.clamp(sigma**2, min=1e-6)
    if centric_flags is None:
        centric_flags = torch.zeros_like(F_obs, dtype=torch.bool)

    arg = torch.clamp(2 * Fc * F_obs / eb, max=1e6)
    acen = (
        -torch.log(2 * F_obs / eb + 1e-12)
        + F_obs**2 / eb
        + Fc**2 / eb
        - (torch.log(torch.special.i0e(arg) + 1e-12) + arg)
    )
    z = torch.clamp(-2 * Fc * F_obs / eb, min=-80.0, max=80.0)
    cen = (
        -0.5 * torch.log(2 / (math.pi * eb) + 1e-12)
        + F_obs**2 / (2 * eb)
        + Fc**2 / (2 * eb)
        - (Fc * F_obs) / eb
        - torch.log((1 + torch.exp(z)) / 2 + 1e-12)
    )
    loss = torch.where(centric_flags, cen, acen)
    loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))
    return (loss * mask).sum()


@pytest.mark.unit
class TestMLXrayTarget:
    def test_init_thin(self):
        from torchref.refinement.targets import MLXrayTarget

        target = MLXrayTarget()
        assert target._model is None and target._data is None and target._scaler is None
        # σ_A (beta) is owned by the target via a SigmaAEstimator; the scaler
        # owns scaling only.
        from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator

        assert isinstance(target._sigma_a, SigmaAEstimator)

    def test_factory_dispatch(self):
        from torchref.refinement.targets import (
            MLXrayTarget,
            create_xray_target,
        )

        assert isinstance(
            create_xray_target(mode="ml"), MLXrayTarget
        )

    def test_the_retired_ml_sigmaa_alias_is_gone(self):
        """``ml_sigmaa`` was removed, and must not be silently reachable.

        It was a deprecated alias for ``ml``, unreachable from the CLI in any case
        (``choices`` is built from canonical names only, so argparse rejected it before
        ``by_name`` was ever consulted) and exercised by exactly this test. Removed with
        ``gaussian`` in the 2026-08 cleanup. The alias *machinery* is still there for the
        next rename and is tested separately, against a purpose-built table rather than a
        live dead name -- see ``test_alias_machinery_still_warns``.
        """
        from torchref.refinement.targets import create_xray_target
        from torchref.refinement.targets.xray._specs import XRAY_TARGETS

        assert "ml_sigmaa" not in XRAY_TARGETS.names
        assert all(not spec.aliases for spec in XRAY_TARGETS.specs), (
            "a production alias came back; keep the machinery tested via a synthetic table"
        )
        with pytest.raises(ValueError, match="Unknown X-ray target mode"):
            create_xray_target(mode="ml_sigmaa")

    def test_alias_machinery_still_warns(self):
        """The deprecation path itself, exercised without a live deprecated mode.

        No production row carries an alias any more, so this builds a throwaway table. This
        codebase renames modes regularly (``sigma_a`` -> ``ml_sigmaa`` -> ``ml``), so the
        path is kept -- but kept *tested*, or it would rot before the next rename needs it.
        """
        from torchref.refinement.targets.xray import MLXrayTarget
        from torchref.refinement.targets.xray._specs import (
            XrayTargetSpec,
            XrayTargetTable,
        )

        table = XrayTargetTable(
            specs=(
                XrayTargetSpec(
                    name="ml", target_cls=MLXrayTarget, doc="x", aliases=("legacy_name",)
                ),
            )
        )
        with pytest.warns(DeprecationWarning, match="legacy_name"):
            assert table.by_name("legacy_name").name == "ml"

    def test_factory_unknown_mode_raises(self):
        from torchref.refinement.targets import create_xray_target

        with pytest.raises(ValueError):
            create_xray_target(mode="not_a_mode")

    def test_factory_default_is_ml(self):
        """'ml' (the σ_A Read MLF target) is the promoted default."""
        from torchref.refinement.targets import (
            MLXrayTarget,
            create_xray_target,
        )

        assert type(create_xray_target()) is MLXrayTarget

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
        new = rice_math(F_obs, F_calc, complex_var_from_beta(one, one), centric, mask=mask)
        old = _unit_variance_mlf_closed_form(F_obs, F_calc, one, centric, mask)
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
        loss = rice_math(
            F_obs, F_calc, complex_var_from_beta(beta, eps), centric, mask=mask
        )
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
        from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator
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
        from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl, estimate_beta

        t = refinement.xray_target_work
        assert torch.isfinite(t.forward())
        _c = t._sigma_a._cache
        beta, eps = _c.beta, _c.epsilon
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
            sh = estimate_beta(
                fo, fc, data.centric.cpu(), epsd, dss, data.free.mask.cpu()
            )
        bbin = sh.beta
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


@pytest.mark.unit
def test_epsilon_from_hkl_returns_on_hkl_device():
    """``epsilon_from_hkl`` must answer on ``hkl``'s device, not the spacegroup's.

    ``SpaceGroup.apply_to_hkl`` moves its input onto the symmetry matrices'
    device, so a spacegroup built elsewhere than the reflection data would
    otherwise either raise inside the ``Hs == h0`` comparison or hand back a
    tensor the caller cannot multiply against its per-reflection data.
    """
    from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl
    from torchref.symmetry import SpaceGroup

    hkl = torch.tensor([[1, 0, 0], [0, 2, 0], [1, 1, 1]], dtype=torch.int32)
    sg = SpaceGroup("P 21 21 21", device="cpu")

    eps = epsilon_from_hkl(hkl, sg)
    assert eps.device == hkl.device
    assert eps.shape == (3,)
    assert (eps >= 1).all()

    # No spacegroup: the early-return path must honour the same contract.
    assert epsilon_from_hkl(hkl, None).device == hkl.device


@pytest.mark.mps
def test_epsilon_from_hkl_cross_device():
    """hkl on the accelerator, spacegroup on CPU: still answers on hkl's device."""
    from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl
    from torchref.symmetry import SpaceGroup

    hkl = torch.tensor(
        [[1, 0, 0], [0, 2, 0], [1, 1, 1]], dtype=torch.int32, device="mps"
    )
    sg = SpaceGroup("P 21 21 21", device="cpu")
    eps = epsilon_from_hkl(hkl, sg)
    assert eps.device.type == "mps"
    assert (eps.cpu() >= 1).all()
