"""The two-moment intensity target: the identity, the coherent limit, and the plumbing.

The central claim is an *identity*, not an approximation. For any finite set of
per-crystal activations, with the branching conserved,

    mean_c |F_D + a_c dF|^2  ==  |F_D + abar dF|^2  +  var(a) |dF|^2

with ``abar`` and ``var`` the **population** moments (1/M divisor). The first test builds
the left-hand side by an explicit loop over crystals -- no two-moment expression anywhere
on the generator side -- so it tests the physics rather than restating the implementation.

The trap it pins: a ``1/(M-1)`` divisor makes the identity fail by O(1/M). At M=64 that is
1.6% -- small enough to slip past a loose tolerance and far larger than the effect the
target exists to measure.
"""

import pytest
import torch


def _sample_moments(alpha: torch.Tensor):
    """Population mean and variance (1/M divisor, not 1/(M-1))."""
    return alpha.mean(), alpha.var(unbiased=False)


def _brute_force_mean_intensity(F_D, dF, alpha):
    """mean_c |F_D + a_c dF|^2, by explicit loop. No two-moment expression."""
    total = torch.zeros(F_D.shape, dtype=F_D.real.dtype)
    for a in alpha:
        total = total + (F_D + a * dF).abs() ** 2
    return total / len(alpha)


@pytest.mark.unit
class TestTheMomentIdentity:
    @pytest.mark.parametrize("m", [2, 7, 64])
    @pytest.mark.parametrize(
        "dtype,tol", [(torch.float64, 1e-13), (torch.float32, 1e-5)]
    )
    def test_identity_holds_for_any_finite_activation_set(self, m, dtype, tol):
        gen = torch.Generator().manual_seed(11)
        n = 32
        cdtype = torch.complex128 if dtype is torch.float64 else torch.complex64

        F_D = torch.randn(n, generator=gen, dtype=dtype).to(cdtype) + 1j * torch.randn(
            n, generator=gen, dtype=dtype
        ).to(cdtype)
        dF = torch.randn(n, generator=gen, dtype=dtype).to(cdtype) + 1j * torch.randn(
            n, generator=gen, dtype=dtype
        ).to(cdtype)
        alpha = torch.rand(m, generator=gen, dtype=dtype)

        brute = _brute_force_mean_intensity(F_D, dF, alpha)
        abar, var = _sample_moments(alpha)
        two_moment = (F_D + abar * dF).abs() ** 2 + var * dF.abs() ** 2

        rel = ((brute - two_moment).abs() / brute.abs().clamp(min=1e-30)).max()
        assert rel < tol, f"identity failed at {rel:.2e} (M={m}, {dtype})"

    def test_the_unbiased_variance_divisor_breaks_it(self):
        """Anti-vacuity for the divisor: the wrong one fails, and by how much."""
        gen = torch.Generator().manual_seed(3)
        n, m = 16, 64
        F_D = torch.randn(n, generator=gen, dtype=torch.float64).to(torch.complex128)
        dF = torch.randn(n, generator=gen, dtype=torch.float64).to(torch.complex128)
        alpha = torch.rand(m, generator=gen, dtype=torch.float64)

        brute = _brute_force_mean_intensity(F_D, dF, alpha)
        abar = alpha.mean()
        wrong = (F_D + abar * dF).abs() ** 2 + alpha.var(unbiased=True) * dF.abs() ** 2

        rel = ((brute - wrong).abs() / brute.abs()).max()
        assert rel > 1e-3, (
            "the unbiased divisor produced the same answer, so this test cannot "
            "detect the wrong one"
        )

    @pytest.mark.parametrize("m", [3, 16])
    def test_identity_holds_with_a_degenerate_zero_variance_set(self, m):
        """Constant activation: the variance term must vanish exactly."""
        n = 8
        gen = torch.Generator().manual_seed(5)
        F_D = torch.randn(n, generator=gen, dtype=torch.float64).to(torch.complex128)
        dF = torch.randn(n, generator=gen, dtype=torch.float64).to(torch.complex128)
        alpha = torch.full((m,), 0.31, dtype=torch.float64)

        brute = _brute_force_mean_intensity(F_D, dF, alpha)
        abar, var = _sample_moments(alpha)
        assert float(var) == pytest.approx(0.0, abs=1e-30)
        assert torch.allclose(brute, (F_D + abar * dF).abs() ** 2, rtol=1e-13)

    def test_bernoulli_activation_gives_the_incoherent_sum(self):
        """The lambda = 1 limit: fully-lit or fully-dark crystals add in intensity."""
        n = 64
        gen = torch.Generator().manual_seed(7)
        F_D = torch.randn(n, generator=gen, dtype=torch.float64).to(torch.complex128)
        F_L = torch.randn(n, generator=gen, dtype=torch.float64).to(torch.complex128)
        dF = F_L - F_D

        w = 0.25
        m = 400
        alpha = torch.zeros(m, dtype=torch.float64)
        alpha[: int(w * m)] = 1.0

        brute = _brute_force_mean_intensity(F_D, dF, alpha)
        incoherent = (1 - w) * F_D.abs() ** 2 + w * F_L.abs() ** 2
        assert torch.allclose(brute, incoherent, rtol=1e-12)

        # ...and the two-moment form reproduces it, with lambda exactly 1.
        abar, var = _sample_moments(alpha)
        assert float(var) == pytest.approx(abar * (1 - abar), rel=1e-12)
        two_moment = (F_D + abar * dF).abs() ** 2 + var * dF.abs() ** 2
        assert torch.allclose(brute, two_moment, rtol=1e-12)


# =====================================================================
# Integration against the real collection stack
# =====================================================================


@pytest.fixture(scope="module")
def collection(pdb_dir, mtz_dir):
    """A dark/light collection on 1DAW, which is the only fixture with I/SIGI."""
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
    if dark.I is None:
        pytest.skip("1DAW loaded without intensities")

    model_dark = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    model_light = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    with torch.no_grad():
        model_light.xyz.refinable_params += 0.2

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", dark, set_as_reference=True)
    dc.add_dataset("light", light)

    mc = ModelCollection([model_dark, model_light], dark_key="dark", verbose=0)
    mc.add_dark()
    mc.add_timepoint("light", [0.78, 0.22])

    scaler = CollectionScaler(dc, mc, verbose=0)
    scaler.initialize()
    return dc, mc, scaler


def _target(dc, mc, scaler, **kw):
    from torchref.refinement.targets import CollectionTwoMomentIntensityTarget

    return CollectionTwoMomentIntensityTarget(dc, mc, scaler=scaler, verbose=0, **kw)


@pytest.mark.integration
class TestCoherentLimit:
    def test_lambda_zero_reduces_to_the_squared_mean(self, collection):
        dc, mc, scaler = collection
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

    def test_lambda_zero_survives_a_poisoned_derivative(self, collection):
        """The coherent limit must skip the variance branch, not multiply it by zero.

        A non-finite entry times exactly zero is NaN, which would poison the whole
        gradient; this is what makes the short-circuit load-bearing rather than an
        optimisation.
        """
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.0)
        target = _target(dc, mc, scaler)
        assert not target._variance_is_live(mc.sigma_alpha_sq)
        assert torch.isfinite(target.forward())

    def test_a_nonzero_lambda_changes_the_prediction(self, collection):
        """Anti-vacuity: the variance branch must actually do something."""
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.0)
        coherent = _target(dc, mc, scaler).intensity_model(recalc=True)

        mc.set_lambda_twin(0.5)
        try:
            dispersed = _target(dc, mc, scaler).intensity_model(recalc=True)
        finally:
            mc.set_lambda_twin(0.0)

        assert not torch.allclose(coherent, dispersed)
        # Strictly positive: |dF|^2 has no sign.
        assert bool((dispersed >= coherent - 1e-6).all())


@pytest.mark.integration
class TestForwardModelStructure:
    def test_the_variance_term_is_sigma_sq_times_the_scaled_jacobian(self, collection):
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.4)
        try:
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
        finally:
            mc.set_lambda_twin(0.0)

    def test_the_reference_row_carries_no_variance(self, collection):
        """The dark's Jacobian row is exactly zero, so its prediction is coherent
        regardless of the dispersion -- a dark dataset holds no activation information."""
        dc, mc, scaler = collection
        keys = _target(dc, mc, scaler)._keys()
        assert keys[0] == "dark"

        mc.set_lambda_twin(0.0)
        coherent = _target(dc, mc, scaler).intensity_model(recalc=True)[0]
        mc.set_lambda_twin(0.9)
        try:
            dispersed = _target(dc, mc, scaler).intensity_model(recalc=True)[0]
        finally:
            mc.set_lambda_twin(0.0)

        assert torch.allclose(coherent, dispersed, rtol=1e-6)

    def test_shape_follows_the_fitted_keys(self, collection):
        dc, mc, scaler = collection
        target = _target(dc, mc, scaler)
        model = target.intensity_model(recalc=True)
        assert model.shape == (len(target._keys()), len(dc.hkl))


@pytest.mark.integration
class TestLossAndReporting:
    def test_forward_is_finite_and_positive(self, collection):
        dc, mc, scaler = collection
        loss = _target(dc, mc, scaler).forward()
        assert torch.isfinite(loss)
        assert loss.numel() == 1

    def test_gradient_reaches_the_light_model(self, collection):
        dc, mc, scaler = collection
        target = _target(dc, mc, scaler)
        target.forward().backward()
        grad = mc.base_models[1].xyz.refinable_params.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs().max()) > 0

    def test_gradient_reaches_the_dispersion_when_refinable(self, collection):
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.3, refinable=True)
        try:
            _target(dc, mc, scaler).forward().backward()
            grad = mc._lambda_logit.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert float(grad.abs()) > 0
        finally:
            mc._lambda_logit.grad = None
            mc.set_lambda_twin(0.0)

    def test_rfactor_uses_the_two_moment_amplitude(self, collection):
        dc, mc, scaler = collection
        target = _target(dc, mc, scaler)

        rf = target.get_rfactor()
        assert set(rf) == {"per_dataset", "rwork_pct", "rfree_pct"}
        assert set(rf["per_dataset"]) == set(target._keys())
        for key, (rwork, rfree) in rf["per_dataset"].items():
            assert 0.0 < rwork < 2.0, f"{key}: {rwork}"
            assert 0.0 < rfree < 2.0, f"{key}: {rfree}"

    def test_stats_report_the_activation_moments(self, collection):
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.25)
        try:
            stats = _target(dc, mc, scaler).stats()
            for key in ("alpha_mean", "lambda_twin", "sigma_alpha_sq", "alpha_sd",
                        "dI_frac", "rwork", "rfree", "loss"):
                assert key in stats, f"missing stat: {key}"
            assert stats["alpha_mean"].value == pytest.approx(0.22, abs=1e-4)
            assert stats["lambda_twin"].value == pytest.approx(0.25, abs=1e-4)
            assert stats["dI_frac"].value > 0.0
        finally:
            mc.set_lambda_twin(0.0)

    def test_di_frac_is_zero_in_the_coherent_limit(self, collection):
        """The stat that distinguishes "refined to zero" from "never refined"."""
        dc, mc, scaler = collection
        mc.set_lambda_twin(0.0)
        assert _target(dc, mc, scaler).stats()["dI_frac"].value == 0.0

    @pytest.mark.parametrize("use_set", ["work", "free"])
    def test_subset_selection_is_honoured(self, collection, use_set):
        dc, mc, scaler = collection
        target = _target(dc, mc, scaler, use_set=use_set)
        assert target.use_set == use_set
        expected = sum(
            (dc[k].work if use_set == "work" else dc[k].free).n
            for k in target._keys()
        )
        assert target._n_reflections() == expected


@pytest.mark.integration
class TestIntensityRequirement:
    def test_construction_fails_without_intensities(self, pdb_dir, mtz_dir):
        """Fails at construction, not inside the first loss evaluation: LossState
        probes forward at registration and that traceback is far harder to read."""
        mtz = mtz_dir / "3GR5.mtz"
        pdb = pdb_dir / "3GR5.pdb"
        if not (mtz.exists() and pdb.exists()):
            pytest.skip("3GR5 fixture not present")

        from torchref import ReflectionData
        from torchref.cli._common import load_model
        from torchref.io.datasets.collection import DatasetCollection
        from torchref.model.model_collection import ModelCollection
        from torchref.refinement.targets import CollectionTwoMomentIntensityTarget

        data = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
        if data.I is not None:
            pytest.skip("3GR5 unexpectedly carries intensities")

        model = load_model(str(pdb), max_res=2.05, device="cpu", verbose=0)
        dc = DatasetCollection(verbose=0, device="cpu")
        dc.add_dataset("dark", data, set_as_reference=True)
        mc = ModelCollection([model], dark_key="dark", verbose=0)
        mc.add_dark()

        with pytest.raises(ValueError, match="I/SIGI"):
            CollectionTwoMomentIntensityTarget(dc, mc, verbose=0)


@pytest.mark.integration
class TestNonFiniteObservations:
    """Real reflection files carry non-finite intensities, and they must not reach the
    gradient.

    Masking the *loss* is not enough. ``torch.where`` picks the finite branch for the
    value while still backpropagating through the branch it discarded, so one NaN
    observation turns every parameter gradient into NaN and every optimizer step is
    rejected -- a refinement that silently does nothing rather than one that fails.
    """

    def test_a_nan_observation_does_not_poison_the_gradient(self, collection):
        dc, mc, scaler = collection
        data = dc["light"]
        saved = data.I.clone()
        try:
            with torch.no_grad():
                data.I[5] = float("nan")
                data.I[11] = float("inf")
            data._corrected_I_fp = None

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
        finally:
            with torch.no_grad():
                data.I.copy_(saved)
            data._corrected_I_fp = None
            mc.base_models[1].xyz.refinable_params.grad = None

    def test_a_nan_sigma_does_not_poison_the_gradient(self, collection):
        dc, mc, scaler = collection
        data = dc["light"]
        saved = data.I_sigma.clone()
        try:
            with torch.no_grad():
                data.I_sigma[7] = float("nan")
            data._corrected_I_fp = None

            target = _target(dc, mc, scaler)
            loss = target.forward()
            loss.backward()
            grad = mc.base_models[1].xyz.refinable_params.grad
            assert torch.isfinite(loss) and torch.isfinite(grad).all()
        finally:
            with torch.no_grad():
                data.I_sigma.copy_(saved)
            data._corrected_I_fp = None
            mc.base_models[1].xyz.refinable_params.grad = None

    def test_the_bad_reflections_are_excluded_not_absorbed(self, collection):
        """They must drop out of the sum, not contribute a large finite penalty --
        otherwise the loss depends on how many reflections the file happened to reject.
        """
        dc, mc, scaler = collection
        data = dc["light"]
        saved = data.I.clone()
        target = _target(dc, mc, scaler)
        try:
            baseline = target.forward().item()
            with torch.no_grad():
                data.I[3] = float("nan")
            data._corrected_I_fp = None
            with_nan = _target(dc, mc, scaler).forward().item()
        finally:
            with torch.no_grad():
                data.I.copy_(saved)
            data._corrected_I_fp = None

        # One reflection out of tens of thousands: the loss should drop slightly, not
        # jump by a penalty term.
        assert with_nan <= baseline
        assert abs(with_nan - baseline) / baseline < 1e-2


@pytest.mark.integration
class TestWeightCalibration:
    """Intensities are squared amplitudes, so this target's gradient is orders of
    magnitude away from the amplitude target beside it. Left uncalibrated it swamps the
    geometry restraints and buys R-free by moving the model further than the data
    supports.
    """

    def test_the_uncalibrated_mismatch_is_large(self, collection):
        """Anti-vacuity: if the two targets already pushed equally, calibration would
        be pointless."""
        dc, mc, scaler = collection
        from torchref.refinement.targets import CollectionDifferenceTarget

        params = [p for p in mc.base_models[1].parameters() if p.requires_grad]
        diff = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)
        target = _target(dc, mc, scaler)

        def gnorm(t):
            g = torch.autograd.grad(t.forward(), params, allow_unused=True)
            return sum(float((x**2).sum()) for x in g if x is not None) ** 0.5

        ratio = gnorm(target) / gnorm(diff)
        assert ratio > 10 or ratio < 0.1, (
            f"gradient ratio is {ratio:.3g}; the two targets are already matched and "
            f"this fixture cannot show why calibration is needed"
        )

    def test_calibration_equalises_the_gradient_norms(self, collection):
        dc, mc, scaler = collection
        from torchref.refinement.targets import CollectionDifferenceTarget

        params = [p for p in mc.base_models[1].parameters() if p.requires_grad]
        diff = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)
        target = _target(dc, mc, scaler)

        target.calibrate_base_weight(diff, params)

        def gnorm(t):
            g = torch.autograd.grad(t.forward(), params, allow_unused=True)
            return sum(float((x**2).sum()) for x in g if x is not None) ** 0.5

        assert gnorm(target) == pytest.approx(gnorm(diff), rel=0.05)

    def test_the_ratio_argument_scales_the_result(self, collection):
        dc, mc, scaler = collection
        from torchref.refinement.targets import CollectionDifferenceTarget

        params = [p for p in mc.base_models[1].parameters() if p.requires_grad]
        diff = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)

        a = _target(dc, mc, scaler)
        b = _target(dc, mc, scaler)
        wa = a.calibrate_base_weight(diff, params, ratio=1.0)
        wb = b.calibrate_base_weight(diff, params, ratio=0.25)
        assert wb == pytest.approx(0.25 * wa, rel=1e-3)

    def test_base_weight_scales_the_loss_on_the_work_set(self, collection):
        dc, mc, scaler = collection
        one = _target(dc, mc, scaler, base_weight=1.0).forward().item()
        three = _target(dc, mc, scaler, base_weight=3.0).forward().item()
        assert three == pytest.approx(3.0 * one, rel=1e-5)

    def test_the_free_set_value_is_left_unweighted(self, collection):
        """The free-set number is a diagnostic and has to stay comparable across
        weightings."""
        dc, mc, scaler = collection
        one = _target(dc, mc, scaler, use_set="free", base_weight=1.0).forward().item()
        five = _target(dc, mc, scaler, use_set="free", base_weight=5.0).forward().item()
        assert five == pytest.approx(one, rel=1e-6)

    def test_calibration_needs_refinable_parameters(self, collection):
        dc, mc, scaler = collection
        from torchref.refinement.targets import CollectionDifferenceTarget

        diff = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)
        with pytest.raises(ValueError, match="No refinable parameters"):
            _target(dc, mc, scaler).calibrate_base_weight(diff, [])
