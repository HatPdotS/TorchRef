"""Properties of the bounded sigma_A solve and its stability shrinkage.

The solve deliberately changes ``ml``'s ``beta``, so it is verified against *properties*
rather than golden values. Each test below corresponds to a defect in the implementation
it replaced -- a bracket + regula-falsi root-find on an unbounded ``t in (0, inf)`` with
three sign-triggered gates (``OMEGA <= 0``, ``wiAB <= 3e-7``, "never bracketed"), a
cancellation-prone ``wi = A*B - C**2``, a 3-point arithmetic average of ``topt``, and alpha
recovered on a different surface from ``beta``.
"""

import math

import pytest
import torch

from torchref.refinement.model_error_estimation.sigma_a import ALPHA_FLOOR, RATIO_MAX, SigmaAEstimator, _shrink_to_curve, _sigma_a_sampling_var, estimate_beta

#: Tolerance for comparing ``beta``/``sigma_A`` across dtypes or devices.
#:
#: The solve is a discrete argmin over a geometric grid whose final stage steps ``beta``
#: by 0.38%. Near a flat optimum, adjacent candidates can differ by less than the float32
#: noise floor of the shell sum, so two dtypes -- or two devices -- can land on
#: neighbouring grid points. The resulting disagreement is bounded by that one step.
#:
#: 1e-2 sits between the two scales that matter: ~2.6x above the measured worst case
#: (3.8e-3 over 60 seed/sigma_A combinations) and ~14x below ``beta``'s own sampling sd
#: of ~14%. Tightening it chases precision the estimator does not have; loosening it past
#: ~1e-1 would stop catching a real regression.
GRID_STEP_RTOL = 1e-2


def synth(n, sigma_a, seed=11, dtype=torch.float64, sig_frac=0.0, dss_level=None):
    """Acentric Wilson data with a KNOWN Luzzati sigma_A.

    ``Fc ~ CN(0, Sigma_P)`` and ``F_true = sigma_A*Fc + CN(0, (1-sigma_A**2)*Sigma_N)``
    with ``Sigma_P == Sigma_N``, so ``alpha == sigma_A`` and
    ``E[|F_true|**2] == Sigma_N`` exactly.

    Getting this wrong is easy and quiet: a generator that sets ``beta = (1-s**2)*Sp/s**2``
    and calls ``s`` the answer actually has ``sigma_A**2 = s**4/(s**4 + 1 - s**2)``, i.e.
    0.094 when ``s = 0.3``. Any "recovery" test built on that measures nothing.
    """
    g = torch.Generator().manual_seed(seed)
    SN = torch.full((n,), 100.0, dtype=torch.float64)
    cn = lambda var: (  # noqa: E731
        torch.randn(n, generator=g, dtype=torch.float64) * torch.sqrt(var / 2)
        + 1j * torch.randn(n, generator=g, dtype=torch.float64) * torch.sqrt(var / 2)
    )
    fcz = cn(SN)
    ftz = sigma_a * fcz + cn((1 - sigma_a**2) * SN)
    fc, ft = fcz.abs(), ftz.abs()
    sig = sig_frac * ft.mean() * torch.ones(n, dtype=torch.float64)
    fo = (
        (ft + torch.randn(n, generator=g, dtype=torch.float64) * sig).abs()
        if sig_frac > 0
        else ft
    )
    dss = (
        torch.linspace(0.02, 0.35, n, dtype=torch.float64)
        if dss_level is None
        else torch.full((n,), float(dss_level), dtype=torch.float64)
    )
    to = lambda t: t.to(dtype)  # noqa: E731
    return dict(
        F_obs=to(fo), F_calc=to(fc),
        centric=torch.zeros(n, dtype=torch.bool),
        epsilon=to(torch.ones(n, dtype=torch.float64)),
        d_star_sq=to(dss),
        free_mask=torch.ones(n, dtype=torch.bool),
        sigma_obs=to(sig),
    )


@pytest.mark.unit
class TestBoundsAndIdentity:
    @pytest.mark.parametrize(
        "dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-5)]
    )
    def test_second_moment_identity_is_exact(self, dtype, tol):
        """``alpha**2 Sigma_P + beta_model + S2 == B``.

        The old estimator could not pass this: it fitted alpha on the raw moment ``B``
        and subtracted ``S2`` from beta afterwards, so the shipped pair implied
        ``sigma_A**2 = (B - beta_raw)/(B - S2)``, which exceeds 1 whenever
        ``beta_raw < S2``. Deriving both from one sigma_A makes the identity algebraic.
        """
        sh = estimate_beta(**synth(3000, 0.85, dtype=dtype, sig_frac=0.05))
        resid = (
            sh.alpha**2 * sh.Sigma_P + sh.beta_model + sh.S2 - sh.B
        ).abs() / sh.B
        assert float(resid.max()) < tol, f"identity off by {float(resid.max()):.2e}"

    def test_bounds(self):
        sh = estimate_beta(**synth(3000, 0.85, sig_frac=0.05))
        assert (sh.sigma_a >= 0).all() and (sh.sigma_a <= 1).all()
        assert (sh.beta_model > 0).all()
        assert (sh.beta >= sh.S2).all(), "total variance below the measurement variance"
        assert (sh.beta <= sh.B * (1 + 1e-9)).all(), "variance above the observed power"
        bound = (sh.Sigma_N / sh.Sigma_P).clamp(max=RATIO_MAX).sqrt()
        assert (sh.alpha <= bound * (1 + 1e-9)).all()
        assert (sh.alpha >= ALPHA_FLOOR).all()

    def test_beta_minus_beta_model_is_exactly_S2(self):
        """The sigma_Fo-free variance and the total differ by exactly the measurement
        term -- not by a clamped, non-invertible subtraction as before."""
        sh = estimate_beta(**synth(3000, 0.85, sig_frac=0.08))
        assert float(sh.S2.min()) > 0.0
        assert torch.allclose(sh.beta - sh.beta_model, sh.S2, atol=1e-12)

    def test_no_sigma_obs_means_S2_zero_and_the_two_coincide(self):
        """``sigma_obs=None`` is not a separate code path -- it is ``S2 == 0``."""
        a = synth(3000, 0.85)
        a["sigma_obs"] = None
        sh = estimate_beta(**a)
        assert float(sh.S2.abs().max()) == 0.0
        assert torch.equal(sh.beta, sh.beta_model)


@pytest.mark.unit
class TestRecovery:
    @pytest.mark.parametrize("sig_frac", [0.0, 0.15])
    def test_estimate_is_not_biased_high(self, sig_frac):
        """The joint (alpha, beta) fit must be near-unbiased across a resolution trend.

        This pins the reason the objective is what it is. Scoring candidates with the mean
        pinned at ``|F_c|`` -- i.e. at ``alpha == 1``, which is what this solve used to do
        before deriving ``alpha`` from the moment identity afterwards -- fits the wrong
        likelihood, and the error is systematic rather than noise:

            objective            bias (no noise)   bias (15% sigma_obs)
            alpha == 1 (old)         +0.0352            +0.0282
            joint (alpha, beta)      -0.0053            -0.0157

        A single-shell recovery test cannot see this: the bias is a *mean over shells*,
        and each shell's own sampling sd is larger than the bias. Hence a trend, many
        shells, and a bound on the mean rather than on any one shell.

        The bound is 0.02, comfortably below the +0.035 the wrong objective produces and
        comfortably above what this one achieves, so it fails loudly if the mean coupling
        is ever dropped from the objective again.
        """
        truth = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
        parts = [
            synth(3000, sa, seed=700 + i, sig_frac=sig_frac, dss_level=0.03 * (i + 1))
            for i, sa in enumerate(truth)
        ]
        cat = lambda k: torch.cat([p[k] for p in parts])  # noqa: E731
        sh = estimate_beta(
            F_obs=cat("F_obs"), F_calc=cat("F_calc"), centric=cat("centric"),
            epsilon=cat("epsilon"), d_star_sq=cat("d_star_sq"),
            free_mask=cat("free_mask"), sigma_obs=cat("sigma_obs"),
            per_bin=3000, shrink=False,
        )
        assert sh.sigma_a.numel() == len(truth)
        bias = float((sh.sigma_a.double() - torch.tensor(truth, dtype=torch.float64)).mean())
        assert abs(bias) < 0.02, f"mean sigma_A bias {bias:+.4f} over {len(truth)} shells"

    @pytest.mark.parametrize("sigma_a", [0.50, 0.80, 0.90, 0.99])
    def test_recovers_known_sigma_a(self, sigma_a):
        """Within the analytic sampling sd, which is the floor set by shell size.

        ``shrink=False`` isolates the solve. Note sigma_A is biased UPWARD as it
        approaches 0 (+0.16 at sigma_A=0.2, +0.06 at 0.5, negligible >= 0.8): standard
        ML boundary bias where the parameter is barely identified -- the analytic sd at
        sigma_A=0.2 is ~2000, i.e. unidentified. It errs toward too LITTLE model error,
        so it is asserted loosely here and recorded rather than hidden.
        """
        sh = estimate_beta(**synth(4200, sigma_a), shrink=False)
        sd = (1 - sh.sigma_a**2) / (sh.sigma_a * torch.sqrt(2 * sh.counts))
        # mean over shells, so the tolerance is the sd of that mean
        err = abs(float(sh.sigma_a.mean()) - sigma_a)
        assert err < 4 * float(sd.mean()) / math.sqrt(sh.counts.numel()) + 0.02, (
            f"sigma_A={sigma_a}: recovered {float(sh.sigma_a.mean()):.4f}"
        )

    def test_counts_not_per_bin_is_the_right_denominator(self):
        """``estimate_beta`` re-bins by equal COUNT, so a shell holds far fewer
        reflections than the requested ``per_bin``. Comparing a sampling variance at the
        requested size rather than at ``shells.counts`` already produced one false
        conclusion in this codebase."""
        sh = estimate_beta(**synth(1000, 0.85), per_bin=140)
        assert int(sh.counts.sum()) == 1000
        assert float(sh.counts.max()) <= 1000 / sh.counts.numel() + 1


@pytest.mark.unit
class TestDeterminism:
    def test_repeated_calls_bit_identical(self):
        a = synth(3000, 0.85, sig_frac=0.05, dtype=torch.float32)
        out = [estimate_beta(**a).beta for _ in range(8)]
        assert all(torch.equal(x, out[0]) for x in out)

    def test_float32_tracks_float64(self):
        """The f32 entry point must agree with f64 to within the grid's own resolution.

        Bounded by ``GRID_STEP_RTOL``, not by float32 epsilon. The solve is a discrete
        argmin over a geometric grid, and near a flat optimum two adjacent candidates can
        be separated by less than the float32 noise floor of the shell sum -- so f32 and
        f64 can land on neighbouring grid points. That costs one final-stage step, 0.38%
        in ``beta``, against a sampling sd of ~14% for the same quantity: 1/38 of a sigma.
        Measured worst case over 60 seed/sigma_A combinations is 3.8e-3.

        This is NOT the failure this test was written for. That one had a recorded max
        f32-vs-f64 difference of 13.1 -- 1310%, ~90x the sampling noise -- from an
        objective of magnitude ~4e4 deciding on differences of ~5e-3, and it made shells
        snap to the 1.0 floor at random. A tolerance of 1e-2 still catches that by three
        orders of magnitude. Do not tighten this to chase bit-agreement between dtypes:
        it is not a property the estimator has, or needs.
        """
        b32 = estimate_beta(**synth(3000, 0.85, sig_frac=0.05, dtype=torch.float32)).beta
        b64 = estimate_beta(**synth(3000, 0.85, sig_frac=0.05, dtype=torch.float64)).beta
        rel = ((b32.double() - b64).abs() / b64).max()
        assert float(rel) < GRID_STEP_RTOL, f"f32 vs f64 rel diff {float(rel):.2e}"

    def test_tie_breaking_is_not_argmin(self):
        """A flat objective must still return a defined answer, and the same one every
        time -- ``argmin`` tie-breaking is not documented as stable across backends, so
        the solve uses a strict-``<`` ``where`` tournament instead."""
        n = 400
        a = dict(
            F_obs=torch.ones(n, dtype=torch.float64),
            F_calc=torch.zeros(n, dtype=torch.float64),
            centric=torch.zeros(n, dtype=torch.bool),
            epsilon=torch.ones(n, dtype=torch.float64),
            d_star_sq=torch.linspace(0.02, 0.3, n, dtype=torch.float64),
            free_mask=torch.ones(n, dtype=torch.bool),
        )
        out = [estimate_beta(**a).sigma_a for _ in range(4)]
        assert all(torch.equal(x, out[0]) for x in out)
        assert torch.isfinite(out[0]).all()


@pytest.mark.unit
class TestFailureModes:
    def test_degenerate_free_set(self):
        a = synth(200, 0.85)
        a["free_mask"] = torch.zeros(200, dtype=torch.bool)
        sh = estimate_beta(**a)
        assert sh.degenerate
        assert float(sh.sigma_a[0]) == 0.0
        assert float(sh.alpha[0]) == pytest.approx(ALPHA_FLOOR)
        assert float(sh.beta[0]) > 0 and math.isfinite(float(sh.beta[0]))

    def test_zero_model_amplitude_gives_no_information_not_a_crash(self):
        """All ``F_calc == 0``: sigma_A must go to its floor and alpha must NOT be 0.

        ``sigma_A = 0`` is conservative for the variance (``beta = B``) but destructive
        for the mean -- ``alpha*|F_calc| == 0`` deletes a whole shell's model
        contribution. Observed on a real structure (6OHI) with the old solve.
        """
        a = synth(1000, 0.85)
        a["F_calc"] = torch.zeros_like(a["F_calc"])
        sh = estimate_beta(**a)
        assert (sh.alpha >= ALPHA_FLOOR).all(), "alpha must never reach 0"
        assert torch.isfinite(sh.beta).all() and (sh.beta > 0).all()

    def test_non_finite_and_non_positive_sigma_are_filtered_and_counted(self):
        a = synth(1000, 0.85, sig_frac=0.05)
        a["F_obs"] = a["F_obs"].clone()
        a["F_obs"][:5] = float("nan")
        a["F_obs"][5:8] = float("inf")
        a["sigma_obs"] = a["sigma_obs"].clone()
        a["sigma_obs"][10:14] = -1.0
        sh = estimate_beta(**a)
        assert sh.diagnostics["n_dropped"] == 12
        assert sh.diagnostics["n_free"] == 988
        assert torch.isfinite(sh.beta).all() and torch.isfinite(sh.alpha).all()

    def test_data_at_or_below_its_own_noise_clamps_S2_and_counts_it(self):
        a = synth(1000, 0.85)
        a["sigma_obs"] = a["F_obs"] * 3.0  # sigma^2 >> <Fo^2>
        sh = estimate_beta(**a)
        assert sh.diagnostics["n_s2_clamped"] > 0
        assert (sh.Sigma_N > 0).all()
        assert (sh.beta <= sh.B * (1 + 1e-9)).all()

    def test_single_shell(self):
        sh = estimate_beta(**synth(120, 0.85), per_bin=1000, min_bins=1)
        assert sh.sigma_a.numel() == 1
        assert float(sh.shrink_w[0]) == 0.0, "nothing to borrow from"


@pytest.mark.unit
class TestStabilityShrinkage:
    def _multi_shell(self, sigma_as, per, seed0=100, sig_frac=0.0):
        parts = [
            synth(per, sa, seed=seed0 + i, sig_frac=sig_frac, dss_level=0.05 * (i + 1))
            for i, sa in enumerate(sigma_as)
        ]
        cat = lambda k: torch.cat([p[k] for p in parts])  # noqa: E731
        return dict(
            F_obs=cat("F_obs"), F_calc=cat("F_calc"), centric=cat("centric"),
            epsilon=cat("epsilon"), d_star_sq=cat("d_star_sq"),
            free_mask=cat("free_mask"), sigma_obs=cat("sigma_obs"),
        )

    def test_the_curve_is_recovered_and_left_alone_when_the_shells_lie_on_it(self):
        """The exact-fit limit: shells ON ``exp(-(a + b*d*^2))`` must give back ``(a, b)``,
        ``tau**2 == 0``, ``w == 1`` -- and, because the target IS the data, move nothing.

        ``w == 1`` reads as "shrink fully" and is the *correct* answer here: there is no
        scatter about the curve beyond noise, so the curve carries all the information.
        Confusing ``w == 1`` with "destroyed the estimate" is the trap this pins.
        """
        x = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.25], dtype=torch.float64)
        sa = torch.exp(-(0.05 + 1.5 * x))
        var = torch.full((5,), 1e-4, dtype=torch.float64)
        out, w, tau_sq, a, b = _shrink_to_curve(sa, var, x)
        assert a == pytest.approx(0.05, abs=1e-9)
        assert b == pytest.approx(1.50, abs=1e-9)
        assert float(tau_sq) == 0.0
        assert float(w.min()) == 1.0
        # Not `== 0.0`: with w == 1 the output IS the curve, and the curve is re-evaluated
        # as exp(-(a_hat + b_hat*x)) from the fitted coefficients rather than being handed
        # back as `sa`. A fit-then-evaluate round trip through a transcendental cannot be
        # bit-exact. Held to float32 epsilon, so the bound stays meaningful whatever
        # precision the estimator is configured for.
        assert float((out - sa).abs().max()) < torch.finfo(torch.float32).eps

    def test_a_badly_determined_shell_is_shrunk_more_than_a_precise_one(self):
        """The mechanism the whole change rests on: ``w_i = var_i/(var_i + tau**2)``.

        Five shells on a common trend, one of them measured 40x worse. Its weight must be
        markedly higher than its neighbours' -- that, and not average R_free, is why this
        was landed: it engages on data-poor shells and stands aside elsewhere (measured
        -0.00060 R_free, p=0.022, on the benchmark's data-poor quartile; a tie overall).
        """
        sa = torch.tensor([0.90, 0.80, 0.70, 0.60, 0.50], dtype=torch.float64)
        dss = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.25], dtype=torch.float64)
        var = torch.tensor([1e-4, 1e-4, 4e-3, 1e-4, 1e-4], dtype=torch.float64)
        _out, w, _tau, _a, _b = _shrink_to_curve(sa, var, dss)
        assert float(w[2]) > float(w[0]) + 0.25, f"w={w.tolist()}"
        # and the precise shells are not dragged along with it
        assert float(w[0]) == pytest.approx(float(w[1]), rel=1e-9)

    def test_pure_noise_is_pooled(self):
        """Constant truth, tiny shells: the apparent spread is all sampling noise, so
        ``Q < k-2``, ``tau == 0``, ``w == 1`` and the spread collapses onto the curve."""
        sh = estimate_beta(**self._multi_shell([0.80] * 6, 60), per_bin=60, min_bins=6)
        assert sh.tau == 0.0
        assert float(sh.shrink_w.min()) > 0.2
        assert float(sh.sigma_a.std()) < 0.2 * float(sh.sigma_a_raw.std())
        # a flat truth must not acquire a slope
        assert abs(sh.curve_b) < 0.1, f"curve_b={sh.curve_b}"

    def test_shrinkage_is_one_shot_from_the_raw_estimates(self):
        """The successor to "the weights must not drift with the pass count".

        The neighbour shrinkage iterated, so re-estimating ``tau**2`` between passes was a
        live feedback loop (shrink -> smaller spread -> smaller ``tau**2`` -> larger ``w``
        -> shrink more), which drove ``w -> 1`` for every structure including precise ones.
        The curve target is fixed, so the fit is one-shot and that loop is structurally
        impossible -- which is why the pass count is gone rather than defaulted.

        What is pinned here is the remaining invariant: the shell values the estimator
        returns are exactly one application of ``_shrink_to_curve`` to the RAW estimates,
        so no second pass is hiding in the call path.
        """
        args = self._multi_shell([0.95, 0.90, 0.85, 0.75, 0.60], 4000)
        sh = estimate_beta(**args, per_bin=4000)
        direct, w, tau_sq, a, b = _shrink_to_curve(
            sh.sigma_a_raw.double(), _sigma_a_sampling_var(
                sh.sigma_a_raw.double(), sh.counts.double().clamp(min=1.0)
            ), sh.bin_dss.double(),
        )
        # `sigma_a` additionally carries the alpha backstop, which cannot fire here.
        assert sh.diagnostics["n_alpha_floored"] == 0
        assert torch.allclose(sh.sigma_a.double(), direct, atol=1e-12)
        assert torch.allclose(sh.shrink_w.double(), w, atol=1e-12)
        assert sh.tau == pytest.approx(float(tau_sq.sqrt()), rel=1e-12)
        assert (sh.curve_a, sh.curve_b) == pytest.approx((a, b), rel=1e-12)
        # repeated calls are bit-identical (no RNG, no order-dependent reduction)
        again = _shrink_to_curve(
            sh.sigma_a_raw.double(),
            _sigma_a_sampling_var(sh.sigma_a_raw.double(),
                                  sh.counts.double().clamp(min=1.0)),
            sh.bin_dss.double(),
        )[0]
        assert torch.equal(direct, again)

    def test_global_tau_drags_the_worst_fitting_shell_of_a_curved_trend(self):
        """A KNOWN LIMITATION, pinned so a future fix shows up as an intended change.

        ``tau**2`` is one global number, so where the 2-parameter form cannot follow the
        truth's curvature the systematic end-of-range residual is averaged in with the
        well-fitting middle, and the worst shell is shrunk despite being measured
        precisely. On this synthetic case (a real trend that accelerates, 4000
        reflections/shell) the top-resolution shell reaches ``w ~ 0.36`` and is pulled
        ~0.05 above its truth, while the raw solve was accurate to ~0.009.

        The real-data justification for accepting this is the 765-structure benchmark,
        where it is a tie overall (R_free p=0.083) and a win on the data-poor quartile.
        The fix is a per-shell residual scale, which is not estimable from one observation
        per shell -- but the trajectory data shows the residual is reproducible within a
        structure, so a previous cycle could supply it. Not done here.
        """
        truth = torch.tensor([0.95, 0.90, 0.85, 0.75, 0.60], dtype=torch.float64)
        sh = estimate_beta(**self._multi_shell(truth.tolist(), 4000), per_bin=4000)
        err_raw = float((sh.sigma_a_raw.double() - truth).abs().max())
        err_shrunk = float((sh.sigma_a.double() - truth).abs().max())
        assert err_raw < 0.015, f"the solve itself should be accurate here: {err_raw}"
        # The documented degradation. Bounded on BOTH sides: below, so the limitation
        # cannot quietly disappear unnoticed; above, so it cannot quietly get worse.
        assert 0.02 < err_shrunk < 0.08, f"max|shrunk-truth|={err_shrunk}"
        assert 0.2 < float(sh.shrink_w.max()) < 0.5, f"w={sh.shrink_w.tolist()}"
        # Most of the trend still survives -- this is a drag, not a flattening.
        span_raw = float(sh.sigma_a_raw.max() - sh.sigma_a_raw.min())
        span_shr = float(sh.sigma_a.max() - sh.sigma_a.min())
        assert span_shr > 0.8 * span_raw, f"{span_shr:.4f} vs {span_raw:.4f}"
        # and the well-measured low-resolution end is untouched
        assert float(sh.shrink_w[0]) < 0.02

    def test_the_fitted_slope_is_clamped_monotone(self):
        """``b >= 0``: sigma_A must not be fitted as INCREASING with resolution, which
        noisy per-shell estimates ask for on ~2 structures in 10. ``a`` is refitted after
        the clamp, so the flat curve still passes through the weighted centroid rather
        than through zero."""
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        rising = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float64)
        var = torch.full((5,), 1e-4, dtype=torch.float64)
        out, w, _tau, a, b = _shrink_to_curve(rising, var, x)
        assert b == 0.0
        flat = math.exp(-a)
        assert float(rising.min()) < flat < float(rising.max())
        assert torch.isfinite(out).all()

    def test_shrinkage_does_not_smear_betas_resolution_falloff(self):
        """Only sigma_A is pooled; ``Sigma_N`` stays per shell. That is what makes this
        safe where smoothing ``beta`` was measured harmful -- ``beta`` keeps falling with
        resolution exactly as ``Sigma_N`` does."""
        sh = estimate_beta(**self._multi_shell([0.80] * 6, 60), per_bin=60, min_bins=6)
        # Sigma_N is constant by construction here, so use a falling-Sigma_N case
        sh2 = estimate_beta(**synth(3000, 0.85, sig_frac=0.03))
        assert float(sh2.beta[0]) > float(sh2.beta[-1])
        assert (sh.beta > 0).all()

    def test_solve_is_lipschitz_where_the_old_gates_were(self):
        """Sweep the model-error level continuously across the region where the old
        ``wiAB <= 3e-7`` and ``OMEGA <= 0`` gates fired, and assert beta moves smoothly.
        The old code fails this by construction -- a gate is a discontinuity -- and a
        discontinuity on a float32 quantity is what made beta differ across GPU
        processes."""
        prev = None
        jumps = []
        for sa in torch.linspace(0.90, 0.9995, 24):
            sh = estimate_beta(**synth(2000, float(sa), seed=5), shrink=False)
            cur = float(sh.beta.mean())
            if prev is not None:
                jumps.append(abs(cur - prev) / max(prev, 1e-12))
            prev = cur
        # the sweep spans ~200x in beta over 24 steps, so ~25% per step is expected;
        # a gate would show as a single step far above the rest
        assert max(jumps) < 4 * sorted(jumps)[len(jumps) // 2], (
            f"discontinuity: max step {max(jumps):.3f} vs median "
            f"{sorted(jumps)[len(jumps)//2]:.3f}"
        )

    @pytest.mark.parametrize("sa", [0.99, 0.95, 0.70, 0.30])
    @pytest.mark.parametrize("n", [40.0, 150.0, 2000.0])
    def test_sampling_var_is_exactly_1p5x_the_ml_term(self, sa, n):
        """``var_SN`` is algebraically ``var_ml / 2``, for every sigma_A and n:

            var_SN/var_ml = [1/(4 sA^2 n)] / [1/(2 sA^2 n)] = 1/2

        Pinned because the docstring once claimed the second term "dominates when the
        model is good". It never dominates -- that statement is true of **alpha's**
        relative sd (where Sigma_N enters at 0.5/sqrt(n), which does not vanish as
        sigma_A -> 1), not of sigma_A's variance, which is what the shrinkage consumes.
        """
        dt = torch.float64  # else the comparison is limited by float32, not the algebra
        v = float(
            _sigma_a_sampling_var(
                torch.tensor([sa], dtype=dt), torch.tensor([n], dtype=dt)
            )
        )
        ml_only = ((1 - sa**2) / (sa * (2 * n) ** 0.5)) ** 2
        assert v == pytest.approx(1.5 * ml_only, rel=1e-12)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_shrink_is_noop_below_four_shells(self, k):
        """Two fitted parameters need two residual degrees of freedom. Note this is
        STRICTER than the neighbour shrinkage it replaced (``k < 3``): a 3-shell fit now
        gets no shrinkage where it previously got some."""
        sa = torch.linspace(0.8, 0.5, k, dtype=torch.float64)
        v = torch.full((k,), 0.01, dtype=torch.float64)
        dss = torch.linspace(0.1, 0.4, k, dtype=torch.float64)
        out, w, tau_sq, a, b = _shrink_to_curve(sa, v, dss)
        assert torch.equal(out, sa)
        assert float(w.max()) == 0.0 and float(tau_sq) == 0.0
        assert math.isnan(a) and math.isnan(b)

    def test_noop_when_every_shell_shares_one_d_star_sq(self):
        """The slope is unidentifiable, so no curve exists to shrink toward.

        The guard has to be RELATIVE: ``det = S*Sxx - Sx**2`` is a difference of two
        ``~(S*x)**2`` terms, so on this input it lands at the float64 cancellation floor
        (~1e-10 here, not ~1e-30). An absolute threshold never fires and the fit proceeds
        on pure rounding noise.
        """
        sa = torch.tensor([0.8, 0.7, 0.6, 0.5], dtype=torch.float64)
        v = torch.full((4,), 1e-3, dtype=torch.float64)
        dss = torch.full((4,), 0.2, dtype=torch.float64)
        out, w, tau_sq, a, b = _shrink_to_curve(sa, v, dss)
        assert torch.equal(out, sa)
        assert float(w.max()) == 0.0 and float(tau_sq) == 0.0
        assert math.isnan(a) and math.isnan(b)

    @pytest.mark.parametrize(
        "sa_list",
        [
            [0.8, 0.8, 0.8, 0.8, 0.8],        # zero spread
            [0.0, 0.7, 0.6, 0.5, 0.4],        # a collapsed shell
            [1.0, 0.7, 0.6, 0.5, 0.4],        # a saturated shell
            [1e-30, 1.0, 1e-30, 1.0, 1e-30],  # alternating at both clamps
        ],
    )
    def test_no_nan_at_degenerate_geometry(self, sa_list):
        """The log/exp round-trip and the ``1/var`` weights are the two places a degenerate
        shell can produce NaN. Both are clamped; this pins that they stay clamped."""
        sa = torch.tensor(sa_list, dtype=torch.float64)
        v = torch.full((5,), 1e-3, dtype=torch.float64)
        dss = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        out, w, tau_sq, a, b = _shrink_to_curve(sa, v, dss)
        assert torch.isfinite(out).all() and torch.isfinite(w).all()
        assert torch.isfinite(tau_sq)
        assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.unit
class TestPerReflectionDerivation:
    def test_identity_holds_per_reflection_not_only_per_shell(self):
        """sigma_A / log Sigma_N / log Sigma_P are interpolated and everything is derived,
        rather than interpolating ``beta`` -- which could give a per-reflection value
        consistent with no ``sigma_A <= 1`` at all."""
        a = synth(3000, 0.85, sig_frac=0.05)
        est = SigmaAEstimator().get(
            a["F_obs"], a["F_calc"], a["centric"], a["epsilon"],
            a["d_star_sq"], a["free_mask"], sigma_obs=a["sigma_obs"],
        )
        assert est.beta.shape == a["F_obs"].shape
        assert (est.sigma_a >= 0).all() and (est.sigma_a <= 1).all()
        assert (est.beta_model > 0).all() and (est.beta >= est.beta_model).all()
        assert (est.alpha >= ALPHA_FLOOR).all() and torch.isfinite(est.alpha).all()

    def test_cache_and_reset(self):
        a = synth(1000, 0.85)
        e = SigmaAEstimator()
        args = (a["F_obs"], a["F_calc"], a["centric"], a["epsilon"],
                a["d_star_sq"], a["free_mask"])
        first = e.get(*args)
        assert e.get(*args) is first, "second call must hit the cache"
        e.reset()
        assert e.get(*args) is not first
