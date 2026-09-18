"""Two-moment target limits, gradients, invalid observations and weight calibration."""

import pytest
import torch

pytestmark = pytest.mark.integration


def _target(dc, mc, scaler, **kw):
    from torchref.refinement.targets import CollectionTwoMomentIntensityTarget

    return CollectionTwoMomentIntensityTarget(dc, mc, scaler=scaler, verbose=0, **kw)


class TestCoherentLimit:
    def test_lambda_zero_reduces_to_the_squared_mean(self, difference_collection):
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(0.0)
        target = _target(dc, mc, scaler)

        model = target.intensity_model(recalc=True)

        rows = target._row_indices(target._keys())
        weights = mc.fractions_matrix()[rows]
        components = dc.component_structure_factors(mc, recalc=False)
        mean = scaler.forward_batched(
            mc.mix_component_fcalcs(components, weights), weights
        )
        assert torch.equal(model, mean.abs() ** 2)

    def test_lambda_zero_skips_nonfinite_derivatives(
        self, difference_collection, monkeypatch
    ):
        """Zero dispersion must not evaluate a potentially non-finite derivative."""
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(0.0)
        weights = mc.fractions_matrix()
        monkeypatch.setattr(mc, "fractions_matrix", lambda: weights)

        def forbidden():
            raise AssertionError("coherent prediction evaluated its variance branch")

        monkeypatch.setattr(mc, "activation_jacobian", forbidden)
        assert torch.isfinite(_target(dc, mc, scaler).forward())

    def test_full_dispersion_matches_incoherent_intensity(self, difference_collection):
        """Fully dark or fully lit crystals mix in intensity, including solvent."""
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(1.0)
        target = _target(dc, mc, scaler)
        actual = target.intensity_model(recalc=True)
        components = dc.component_structure_factors(mc, recalc=False)
        basis = torch.eye(
            mc.n_base_models, device=components.device, dtype=components.real.dtype
        )
        pure = scaler.forward_batched(components, basis)
        expected = mc.fractions_matrix() @ pure.abs().square()
        # Complex mixture sums lose relative precision near solvent cancellation.
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)

    def test_a_nonzero_lambda_changes_the_prediction(self, difference_collection):
        """Anti-vacuity: the variance branch must actually do something."""
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(0.0)
        coherent = _target(dc, mc, scaler).intensity_model(recalc=True)

        mc.set_lambda_twin(0.5)
        dispersed = _target(dc, mc, scaler).intensity_model(recalc=True)

        assert not torch.allclose(coherent, dispersed)
        # Strictly positive: |dF|^2 has no sign.
        assert bool((dispersed >= coherent - 1e-6).all())


class TestForwardModelStructure:
    def test_the_variance_term_is_sigma_sq_times_the_scaled_jacobian(
        self, difference_collection
    ):
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(0.4)
        target = _target(dc, mc, scaler)
        total = target.intensity_model(recalc=True)

        rows = target._row_indices(target._keys())
        components = dc.component_structure_factors(mc, recalc=False)
        weights = mc.fractions_matrix()[rows]
        jacobian = mc.activation_jacobian()[rows]

        mean = scaler.forward_batched(
            mc.mix_component_fcalcs(components, weights), weights
        )
        deriv = scaler.forward_batched(
            mc.mix_component_fcalcs(components, jacobian), jacobian
        )
        expected = mean.abs() ** 2 + mc.sigma_alpha_sq * deriv.abs() ** 2
        assert torch.allclose(total, expected, rtol=1e-6)

    def test_the_reference_row_carries_no_variance(self, difference_collection):
        """The dark's Jacobian row is exactly zero, so its prediction is coherent
        regardless of the dispersion -- a dark dataset holds no activation information.
        """
        dc, mc, scaler = difference_collection
        keys = _target(dc, mc, scaler)._keys()
        assert keys[0] == "dark"

        mc.set_lambda_twin(0.0)
        coherent = _target(dc, mc, scaler).intensity_model(recalc=True)[0]
        mc.set_lambda_twin(0.9)
        dispersed = _target(dc, mc, scaler).intensity_model(recalc=True)[0]

        assert torch.allclose(coherent, dispersed, rtol=1e-6)

    def test_shape_follows_the_fitted_keys(self, difference_collection):
        dc, mc, scaler = difference_collection
        target = _target(dc, mc, scaler)
        model = target.intensity_model(recalc=True)
        assert model.shape == (len(target._keys()), len(dc.hkl))


class TestLossAndReporting:
    def test_forward_is_finite_and_positive(self, difference_collection):
        dc, mc, scaler = difference_collection
        loss = _target(dc, mc, scaler).forward()
        assert torch.isfinite(loss)
        assert loss.numel() == 1

    def test_gradient_reaches_the_light_model(self, difference_collection):
        dc, mc, scaler = difference_collection
        target = _target(dc, mc, scaler)
        target.forward().backward()
        grad = mc.base_models[1].xyz.refinable_params.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs().max()) > 0

    def test_gradient_reaches_the_dispersion_when_refinable(
        self, difference_collection
    ):
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(0.3, refinable=True)
        _target(dc, mc, scaler).forward().backward()
        grad = mc._lambda_logit.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs()) > 0

    def test_rfactor_uses_the_two_moment_amplitude(self, difference_collection):
        dc, mc, scaler = difference_collection
        target = _target(dc, mc, scaler)

        rf = target.get_rfactor()
        assert set(rf) == {"per_dataset", "rwork_pct", "rfree_pct"}
        assert set(rf["per_dataset"]) == set(target._keys())
        for key, (rwork, rfree) in rf["per_dataset"].items():
            assert 0.0 < rwork < 2.0, f"{key}: {rwork}"
            assert 0.0 < rfree < 2.0, f"{key}: {rfree}"

    def test_stats_report_the_activation_moments(self, difference_collection):
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(0.25)
        stats = _target(dc, mc, scaler).stats()
        for key in (
            "alpha_mean",
            "lambda_twin",
            "sigma_alpha_sq",
            "alpha_sd",
            "dI_frac",
            "rwork",
            "rfree",
            "loss",
        ):
            assert key in stats, f"missing stat: {key}"
        assert stats["alpha_mean"].value == pytest.approx(0.22, abs=1e-4)
        assert stats["lambda_twin"].value == pytest.approx(0.25, abs=1e-4)
        assert stats["dI_frac"].value > 0.0

    def test_di_frac_is_zero_in_the_coherent_limit(self, difference_collection):
        """The stat that distinguishes "refined to zero" from "never refined"."""
        dc, mc, scaler = difference_collection
        mc.set_lambda_twin(0.0)
        assert _target(dc, mc, scaler).stats()["dI_frac"].value == 0.0

    @pytest.mark.parametrize("use_set", ["work", "free"])
    def test_subset_selection_is_honoured(self, difference_collection, use_set):
        dc, mc, scaler = difference_collection
        target = _target(dc, mc, scaler, use_set=use_set)
        assert target.use_set == use_set
        expected = sum(
            (dc[k].work if use_set == "work" else dc[k].free).n for k in target._keys()
        )
        assert target._n_reflections() == expected


class TestIntensityRequirement:
    def test_construction_fails_without_intensities(self, difference_models):
        """Reject absent intensity columns before evaluating the loss."""
        from torchref.refinement.targets import CollectionTwoMomentIntensityTarget

        dc, mc = difference_models
        for data in dc.values():
            data.I = data.I_sigma = None
        with pytest.raises(ValueError, match="I/SIGI"):
            CollectionTwoMomentIntensityTarget(dc, mc, verbose=0)


class TestNonFiniteObservations:
    """Real reflection files carry non-finite intensities, and they must not reach the
    gradient."""

    def test_a_nan_observation_does_not_poison_the_gradient(
        self, difference_collection
    ):
        dc, mc, scaler = difference_collection
        data = dc["light"]
        with torch.no_grad():
            data.I[5] = float("nan")
            data.I[11] = float("inf")

        target = _target(dc, mc, scaler)
        loss = target.forward()
        assert torch.isfinite(loss), "loss went non-finite"

        loss.backward()
        grad = mc.base_models[1].xyz.refinable_params.grad
        assert grad is not None
        assert torch.isfinite(grad).all(), (
            "non-finite observations reached the gradient; every optimizer step "
            "would be rejected and the model would not move"
        )

    def test_a_nan_sigma_does_not_poison_the_gradient(self, difference_collection):
        dc, mc, scaler = difference_collection
        data = dc["light"]
        with torch.no_grad():
            data.I_sigma[7] = float("nan")

        target = _target(dc, mc, scaler)
        loss = target.forward()
        loss.backward()
        grad = mc.base_models[1].xyz.refinable_params.grad
        assert torch.isfinite(loss) and torch.isfinite(grad).all()

    def test_the_bad_reflections_are_excluded_not_absorbed(self, difference_collection):
        """They must drop out of the sum, not contribute a large finite penalty --
        otherwise the loss depends on how many reflections the file happened to reject.
        """
        dc, mc, scaler = difference_collection
        data = dc["light"]
        target = _target(dc, mc, scaler)
        baseline = target.forward().item()
        with torch.no_grad():
            data.I[3] = float("nan")
        with_nan = _target(dc, mc, scaler).forward().item()

        # One reflection out of tens of thousands: the loss should drop slightly, not
        # jump by a penalty term.
        assert with_nan <= baseline
        assert abs(with_nan - baseline) / baseline < 1e-2


class TestWeightCalibration:
    """Intensities are squared amplitudes, so this target's gradient is orders of
    magnitude away from the amplitude target beside it. Left uncalibrated it swamps the
    geometry restraints and buys R-free by moving the model further than the data
    supports."""

    def test_calibration_equalises_the_gradient_norms(self, difference_collection):
        dc, mc, scaler = difference_collection
        from torchref.refinement.targets import CollectionDifferenceTarget

        params = [p for p in mc.base_models[1].parameters() if p.requires_grad]
        diff = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)
        target = _target(dc, mc, scaler)

        target.calibrate_base_weight(diff, params)

        def gnorm(t):
            g = torch.autograd.grad(t.forward(), params, allow_unused=True)
            return sum(float((x**2).sum()) for x in g if x is not None) ** 0.5

        assert gnorm(target) == pytest.approx(gnorm(diff), rel=0.05)

    def test_the_ratio_argument_scales_the_result(self, difference_collection):
        dc, mc, scaler = difference_collection
        from torchref.refinement.targets import CollectionDifferenceTarget

        params = [p for p in mc.base_models[1].parameters() if p.requires_grad]
        diff = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)

        a = _target(dc, mc, scaler)
        b = _target(dc, mc, scaler)
        wa = a.calibrate_base_weight(diff, params, ratio=1.0)
        wb = b.calibrate_base_weight(diff, params, ratio=0.25)
        assert wb == pytest.approx(0.25 * wa, rel=1e-3)

    def test_base_weight_scales_the_loss_on_the_work_set(self, difference_collection):
        dc, mc, scaler = difference_collection
        one = _target(dc, mc, scaler, base_weight=1.0).forward().item()
        three = _target(dc, mc, scaler, base_weight=3.0).forward().item()
        assert three == pytest.approx(3.0 * one, rel=1e-5)

    def test_the_free_set_value_is_left_unweighted(self, difference_collection):
        """The free-set number is a diagnostic and has to stay comparable across
        weightings."""
        dc, mc, scaler = difference_collection
        one = _target(dc, mc, scaler, use_set="free", base_weight=1.0).forward().item()
        five = _target(dc, mc, scaler, use_set="free", base_weight=5.0).forward().item()
        assert five == pytest.approx(one, rel=1e-6)

    def test_calibration_needs_refinable_parameters(self, difference_collection):
        dc, mc, scaler = difference_collection
        from torchref.refinement.targets import CollectionDifferenceTarget

        diff = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)
        with pytest.raises(ValueError, match="No refinable parameters"):
            _target(dc, mc, scaler).calibrate_base_weight(diff, [])
