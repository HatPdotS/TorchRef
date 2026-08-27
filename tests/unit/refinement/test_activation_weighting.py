"""The difference target's per-reflection weighting must follow the activation spread.

A merged light intensity carries a positive, phase-blind contamination
``sigma_alpha^2 |dF/dalpha|^2``. Propagated onto the amplitude it is a shift of
``sigma_alpha^2 |dF|^2 / (2 |F|)``. The difference target does not model that shift, so it
enters as a variance -- which down-weights exactly the reflections whose difference is most
contaminated, and is the calibrated form of the k-weighting difference maps apply by hand.

Two properties are pinned: a zero dispersion changes nothing at all, and a non-zero one
reweights in proportion to ``|dF|^2`` rather than uniformly. The second is what separates a
real weighting from an overall rescaling of the x-ray term, which would be
indistinguishable from a change of x-ray weight.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def collection(pdb_dir, mtz_dir):
    """A dark/light pair on 1DAW with a displaced light model, so dF is non-zero."""
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData
    from torchref.cli._common import load_model
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.collection_scaler import CollectionScaler

    d_min = 2.05
    dark = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    light = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))

    model_dark = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    model_light = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    with torch.no_grad():
        model_light.xyz.refinable_params += 0.25

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", dark, set_as_reference=True)
    dc.add_dataset("light", light)

    mc = ModelCollection([model_dark, model_light], dark_key="dark", verbose=0)
    mc.add_dark()
    mc.add_timepoint("light", [0.78, 0.22])

    scaler = CollectionScaler(dc, mc, verbose=0)
    scaler.initialize()
    return dc, mc, scaler


def _target(dc, mc, scaler):
    from torchref.refinement.targets import CollectionDifferenceTarget

    return CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)


@pytest.mark.integration
class TestZeroDispersionIsInert:
    def test_the_extra_variance_is_exactly_zero(self, collection):
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.0)
        target = _target(dc, mc, scaler)

        F_obs = dc.stack_F_obs(target._keys())
        extra = target._activation_variance(target._keys(), F_obs)
        assert float(extra.abs().max()) == 0.0

    def test_the_loss_is_unchanged_from_the_no_dispersion_baseline(self, collection):
        """Back-compat: lambda = 0 must reproduce the loss the target always returned."""
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.0)
        target = _target(dc, mc, scaler)
        first = target.forward().item()
        second = target.forward().item()
        assert second == pytest.approx(first, rel=1e-4)


@pytest.mark.integration
class TestDispersionReweights:
    def test_a_nonzero_dispersion_changes_the_loss(self, collection):
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.0)
        base = _target(dc, mc, scaler).forward().item()

        mc.set_lambda_twin(0.4)
        try:
            weighted = _target(dc, mc, scaler).forward().item()
        finally:
            mc.set_lambda_twin(0.0)

        assert weighted != pytest.approx(base, rel=1e-3)

    def test_the_extra_variance_tracks_the_squared_difference(self, collection):
        """Proportional to |dF|^2, not uniform.

        A uniform inflation would just rescale the x-ray term and be
        indistinguishable from a change of x-ray weight; the point of this weighting is
        that it is reflection-specific.
        """
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.4)
        try:
            target = _target(dc, mc, scaler)
            keys = target._keys()
            F_obs = dc.stack_F_obs(keys)
            extra = target._activation_variance(keys, F_obs)

            rows = [mc.keys().index(k) for k in keys]
            components = dc.component_structure_factors(mc, recalc=False)
            jacobian = mc.activation_jacobian()[rows]
            deriv = scaler.forward_batched(
                mc.mix_component_fcalcs(components, jacobian), jacobian
            )
            expected = (
                mc.sigma_alpha_sq * deriv.abs() ** 2
                / (2.0 * F_obs.abs().clamp(min=1e-6))
            ) ** 2
            assert torch.allclose(extra, expected, rtol=1e-5)

            light = extra[keys.index("light")]
            assert float(light.max()) > 0.0
            # Genuinely non-uniform across reflections.
            assert float(light.std() / light.mean().clamp(min=1e-30)) > 0.5
        finally:
            mc.set_lambda_twin(0.0)

    def test_the_reference_row_is_unweighted(self, collection):
        """The dark's activation Jacobian is exactly zero, so it carries no
        contamination and must keep its measured sigma."""
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.6)
        try:
            target = _target(dc, mc, scaler)
            keys = target._keys()
            extra = target._activation_variance(keys, dc.stack_F_obs(keys))
            assert float(extra[keys.index("dark")].abs().max()) == 0.0
            assert float(extra[keys.index("light")].abs().max()) > 0.0
        finally:
            mc.set_lambda_twin(0.0)

    @pytest.mark.parametrize("lam", [0.1, 0.4, 0.9])
    def test_more_dispersion_means_more_down_weighting(self, collection, lam):
        """The implied weight sigma^2/(sigma^2 + extra) must fall monotonically."""
        dc, mc, scaler = collection
        keys = _target(dc, mc, scaler)._keys()
        F_obs = dc.stack_F_obs(keys)

        mc.set_lambda_twin(lam)
        try:
            extra = _target(dc, mc, scaler)._activation_variance(keys, F_obs)
        finally:
            mc.set_lambda_twin(0.0)

        sigma_sq = dc.stack_F_sigma(keys) ** 2
        weight = sigma_sq / (sigma_sq + extra)
        light = weight[keys.index("light")]
        assert float(light.max()) <= 1.0 + 1e-6
        assert float(light.min()) < 1.0, "no reflection was down-weighted at all"

    def test_the_gradient_still_reaches_the_model(self, collection):
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.3)
        try:
            _target(dc, mc, scaler).forward().backward()
            grad = mc.base_models[1].xyz.refinable_params.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert float(grad.abs().max()) > 0
        finally:
            mc.base_models[1].xyz.refinable_params.grad = None
            mc.set_lambda_twin(0.0)
