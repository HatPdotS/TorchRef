"""Properties of the shell-free difference-power fit.

Pinned on seeded synthetic differences with a known power law: the power and the
dark-amplitude exponent are recovered from one dataset, and the sigma scale too when it
is asked for, including when the reported sigmas are uniformly inflated; by default the
reported sigmas are the noise and a fixed scale is kept; the power is recovered when the
sigmas follow the dark amplitude within a resolution, and pure noise reads as noise
there with a fitted scale that stays at one; sigmas that overstate the noise toward high
resolution leave the signal fraction of the signal-bearing shells exact and put no
signal in the noise-dominated ones; a fixed exponent stays fixed; a model difference's
resolution-dependent coupling and the power it leaves unexplained are recovered
together; the bounded Wiener weight never falls below its floor, so no reflection or
resolution range is removed even when the data hold no signal, and an infinite SNR gets
full weight; the centric factor is fitted, and recovered, only when both centric and
acentric reflections are present; a fitted sigma scale stays within its bounds when the
differences hold no noise; the fit stays defined when the residual holds no power; the
estimator caches one fit until reset and applies its configured exponent and sigma
scale; the fit runs under ``torch.no_grad()`` and on every available device.
"""

import pytest
import torch

from torchref.refinement.model_error_estimation.difference_power import (
    DEFAULT_SNR_FLOOR,
    SIGMA_SCALE_BOUNDS,
    DifferencePowerConfig,
    DifferencePowerEstimator,
    bounded_wiener_weight,
    fit_difference_power,
)

#: Median absolute log error of the recovered power. The fit has seven parameters
#: against 40 000 reflections; 0.15 is several times the scatter observed across seeds.
LOG_POWER_ATOL = 0.15
#: Tolerance on the exponent and on the relative sigma scale.
GAMMA_ATOL = 0.1
K_RTOL = 0.05


def synth(n=40000, sigma_inflation=1.0, signal=1.0, seed=0, device="cpu"):
    """Differences with power ``4 exp(-8 d*^2) (F / <F>)``; reported sigmas vary
    many-fold within a resolution, so the sigma scale is identifiable."""
    g = torch.Generator().manual_seed(seed)
    dss = torch.rand(n, generator=g) / 1.6**2
    f = torch.exp(torch.randn(n, generator=g) * 0.6) * 100 * torch.exp(-10 * dss)
    s_true = signal * 4.0 * torch.exp(-8.0 * dss) * (f / f.mean())
    sig = 0.5 + 6 * dss / dss.max() * torch.exp(torch.randn(n, generator=g) * 0.4)
    d = torch.randn(n, generator=g) * s_true.sqrt() + torch.randn(n, generator=g) * sig
    out = dict(delta=d, sigma=sig * sigma_inflation, dss=dss, f=f, s_true=s_true)
    return {k: v.to(device) for k, v in out.items()}


def synth_sigma_follows_f(
    n=40000, signal=1.0, sign=1.0, overstate_high_res=False, seed=0
):
    """Differences with the power of :func:`synth`, sigmas that follow the dark amplitude
    within a resolution (rank correlation ~0.7 in magnitude, of the given sign), and the
    true noise ``r(x) sigma``: ``r = 1``, or with ``overstate_high_res`` falling from one
    at half the ``sin(theta)/lambda`` range to 0.3 at its edge, as posterior amplitude
    sigmas overstate the scatter of a difference where the prior dominates."""
    g = torch.Generator().manual_seed(seed)
    dss = torch.rand(n, generator=g) / 1.6**2
    f_mean = 100 * torch.exp(-10 * dss)
    f = torch.exp(torch.randn(n, generator=g) * 0.6) * f_mean
    s_true = signal * 4.0 * torch.exp(-8.0 * dss) * (f / f.mean())
    rel = (f / f_mean) ** (0.7 * sign)
    sig = (
        (0.5 + 6 * dss / dss.max())
        * rel
        * torch.exp(0.15 * torch.randn(n, generator=g))
    )
    x = dss.sqrt() / dss.sqrt().max()
    r = torch.ones_like(x)
    if overstate_high_res:
        r = torch.where(x < 0.5, r, 1.0 - 1.4 * (x - 0.5))
    noise = r * sig
    d = (
        torch.randn(n, generator=g) * s_true.sqrt()
        + torch.randn(n, generator=g) * noise
    )
    return {
        "delta": d,
        "sigma": sig,
        "dss": dss,
        "f": f,
        "s_true": s_true,
        "noise": noise,
        "x": x,
    }


@pytest.mark.unit
@pytest.mark.parametrize("inflation", [1.0, 1.5])
def test_recovers_power_exponent_and_sigma_scale(inflation):
    s = synth(sigma_inflation=inflation)
    fit = fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"], sigma_scale=None
    )
    assert fit.converged and fit.sigma_scale_fitted
    assert fit.gamma == pytest.approx(1.0, abs=GAMMA_ATOL)
    assert fit.sigma_scale == pytest.approx(1.0 / inflation, rel=K_RTOL)
    power = fit.signal_power(s["dss"], f_dark=s["f"])
    err = (power / s["s_true"]).log().abs().median()
    assert float(err) < LOG_POWER_ATOL
    assert torch.isfinite(fit.stderr[: len(fit.coeffs)]).all()


@pytest.mark.unit
def test_default_takes_the_reported_sigmas_and_a_fixed_scale_is_kept():
    s = synth()
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    assert fit.sigma_scale == 1.0
    assert not fit.sigma_scale_fitted and not fit.sigma_scale_at_bound
    assert torch.isnan(fit.stderr[len(fit.coeffs) + 1])
    # Sigmas reported 1.5x too small, with the scale that corrects them given.
    small = synth(sigma_inflation=1.0 / 1.5)
    fit = fit_difference_power(
        small["delta"], small["sigma"], small["dss"], f_dark=small["f"], sigma_scale=1.5
    )
    assert fit.sigma_scale == 1.5 and not fit.sigma_scale_fitted
    power = fit.signal_power(small["dss"], f_dark=small["f"])
    assert float((power / small["s_true"]).log().abs().median()) < LOG_POWER_ATOL
    for bad in (0.0, -1.0, 2 * SIGMA_SCALE_BOUNDS[1]):
        with pytest.raises(ValueError):
            fit_difference_power(s["delta"], s["sigma"], s["dss"], sigma_scale=bad)


@pytest.mark.unit
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_recovers_power_when_sigmas_follow_the_dark_amplitude(sign):
    s = synth_sigma_follows_f(sign=sign)
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    assert fit.gamma == pytest.approx(1.0, abs=GAMMA_ATOL)
    power = fit.signal_power(s["dss"], f_dark=s["f"])
    assert float((power / s["s_true"]).log().abs().median()) < 0.2
    free = fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"], sigma_scale=None
    )
    assert free.sigma_scale == pytest.approx(1.0, rel=K_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_noise_reads_as_noise_when_sigmas_follow_the_dark_amplitude(sign):
    s = synth_sigma_follows_f(signal=0.0, sign=sign)
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    snr = fit.snr(s["sigma"], d_star_sq=s["dss"], f_dark=s["f"])
    assert float(snr.median()) < 0.01
    assert float((snr > 0.5).float().mean()) < 0.01
    # A fitted scale has every reflection's variance to calibrate against: it stays at
    # one instead of handing the noise to the power.
    free = fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"], sigma_scale=None
    )
    assert free.sigma_scale == pytest.approx(1.0, rel=K_RTOL)
    assert not free.sigma_scale_at_bound


@pytest.mark.unit
def test_sigmas_overstated_toward_high_resolution_do_not_read_as_signal():
    s = synth_sigma_follows_f(overstate_high_res=True)
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    power = fit.signal_power(s["dss"], f_dark=s["f"])
    noise2 = s["sigma"] ** 2
    inner = s["x"] < 0.5
    fitted = float(power[inner].sum() / (power[inner] + noise2[inner]).sum())
    true = float(
        s["s_true"][inner].sum() / (s["s_true"][inner] + s["noise"][inner] ** 2).sum()
    )
    assert fitted == pytest.approx(true, abs=0.03)
    # Where the sigmas overstate the noise the SNR errs low, never high.
    outer = ~inner
    true_snr = s["s_true"][outer] / s["noise"][outer] ** 2
    assert float((power[outer] / noise2[outer]).mean()) <= float(true_snr.mean())
    none = synth_sigma_follows_f(signal=0.0, overstate_high_res=True)
    fit = fit_difference_power(
        none["delta"], none["sigma"], none["dss"], f_dark=none["f"]
    )
    snr = fit.snr(none["sigma"], d_star_sq=none["dss"], f_dark=none["f"])
    assert float(snr.median()) < 0.01


@pytest.mark.unit
def test_fixed_gamma_is_kept():
    s = synth()
    fit = fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"], gamma=0.0
    )
    assert fit.gamma == 0.0
    fit = fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"], gamma=2.0
    )
    assert fit.gamma == 2.0


@pytest.mark.unit
def test_weight_never_removes_a_reflection_without_signal():
    s = synth(signal=0.0)
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    snr = fit.snr(s["sigma"], d_star_sq=s["dss"], f_dark=s["f"])
    w = bounded_wiener_weight(snr, 0.5)
    assert float(w.min()) >= 1.0 / 3.0 - 1e-6
    assert float(w.max()) < 1.0
    w = bounded_wiener_weight(snr)
    assert float(w.min()) >= DEFAULT_SNR_FLOOR / (1.0 + DEFAULT_SNR_FLOOR) - 1e-6
    assert float(w.min()) > 0.0
    assert float(bounded_wiener_weight(torch.zeros(1), 0.0)) == 0.0
    # A zero reported sigma gives an infinite SNR: full weight, not a NaN.
    assert float(bounded_wiener_weight(torch.tensor([float("inf")]), 0.5)) == 1.0
    with pytest.raises(ValueError):
        bounded_wiener_weight(snr, -0.1)


@pytest.mark.unit
def test_centric_factor_is_fitted_only_when_both_classes_are_present():
    # Every reflection of a centrosymmetric group is centric: the factor would be
    # collinear with the constant term, so it is left at one and the fit matches the
    # one without flags. Several seeds, because a singular Hessian is seed-dependent.
    for seed in range(4):
        s = synth(n=5000, seed=seed)
        all_centric = torch.ones_like(s["delta"], dtype=torch.bool)
        fit = fit_difference_power(
            s["delta"], s["sigma"], s["dss"], f_dark=s["f"], centric=all_centric
        )
        plain = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
        assert fit.centric_factor == 1.0
        assert torch.allclose(fit.coeffs, plain.coeffs)
    # A doubled centric power is recovered when both classes are present.
    s = synth()
    centric = torch.rand(len(s["delta"]), generator=torch.Generator().manual_seed(7))
    centric = centric < 0.3
    g = torch.Generator().manual_seed(8)
    extra = torch.randn(len(s["delta"]), generator=g) * s["s_true"].sqrt()
    delta = torch.where(centric, s["delta"] + extra, s["delta"])
    fit = fit_difference_power(
        delta, s["sigma"], s["dss"], f_dark=s["f"], centric=centric
    )
    assert fit.centric_factor == pytest.approx(2.0, rel=0.15)


@pytest.mark.unit
def test_runs_on_device(any_device):
    s = synth(n=5000, device=any_device)
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    power = fit.signal_power(s["dss"], f_dark=s["f"])
    assert power.device.type == any_device.type
    assert torch.isfinite(power).all() and bool((power > 0).all())


@pytest.mark.unit
def test_fits_under_no_grad():
    s = synth(n=5000)
    with torch.no_grad():
        fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    assert fit.converged


@pytest.mark.unit
def test_sigma_scale_stays_bounded_when_the_differences_hold_no_noise():
    # Identical datasets: every difference is zero, so the fit would drive k to zero.
    s = synth(n=5000)
    zeros = torch.zeros_like(s["delta"]) + 1e-3 * torch.randn(
        len(s["delta"]), generator=torch.Generator().manual_seed(1)
    )
    fit = fit_difference_power(
        zeros, s["sigma"], s["dss"], f_dark=s["f"], sigma_scale=None
    )
    assert fit.sigma_scale == pytest.approx(SIGMA_SCALE_BOUNDS[0], rel=1e-3)
    assert fit.sigma_scale_at_bound
    snr = fit.snr(s["sigma"], d_star_sq=s["dss"], f_dark=s["f"])
    assert bool(torch.isfinite(snr).all())
    assert not fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"], sigma_scale=None
    ).sigma_scale_at_bound


@pytest.mark.unit
def test_recovers_the_model_coupling_and_the_unexplained_power():
    g = torch.Generator().manual_seed(4)
    s = synth()
    n = len(s["delta"])
    stol = s["dss"].sqrt() / 2.0
    alpha_true = 0.8 - 0.3 * stol / stol.max()
    # The model explains part of the difference; the rest is the planted s_true.
    delta_calc = torch.randn(n, generator=g) * 3.0
    delta = s["delta"] + alpha_true * delta_calc
    fit = fit_difference_power(
        delta, s["sigma"], s["dss"], f_dark=s["f"], delta_calc=delta_calc
    )
    assert fit.converged
    alpha = fit.alpha_at(s["dss"])
    assert float((alpha - alpha_true).abs().max()) < 0.05
    beta = fit.signal_power(s["dss"], f_dark=s["f"])
    assert float((beta / s["s_true"]).log().abs().median()) < LOG_POWER_ATOL
    no_model = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    assert torch.equal(no_model.alpha_at(s["dss"]), torch.ones_like(s["dss"]))


@pytest.mark.unit
def test_estimator_caches_until_reset_and_applies_its_config():
    s = synth(n=5000)
    est = DifferencePowerEstimator(DifferencePowerConfig(gamma=0.0, sigma_scale=1.2))
    assert est.fit is None
    first = est.get(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    assert first.gamma == 0.0 and est.fit is first
    assert first.sigma_scale == 1.2 and not first.sigma_scale_fitted
    # Cached: different arguments are ignored until reset.
    assert est.get(2 * s["delta"], s["sigma"], s["dss"], f_dark=s["f"]) is first
    est.reset()
    assert est.fit is None
    assert est.get(s["delta"], s["sigma"], s["dss"], f_dark=s["f"]) is not first
    with pytest.raises(ValueError):
        DifferencePowerConfig(gamma=10.0)
    with pytest.raises(ValueError):
        DifferencePowerConfig(sigma_scale=0.0)
    assert DifferencePowerConfig().sigma_scale == 1.0
    assert DifferencePowerConfig(sigma_scale=None).sigma_scale is None


@pytest.mark.unit
def test_fit_is_defined_when_the_residual_holds_no_power():
    # All differences zero, and differences exactly explained by the model: the
    # residual power is zero, which the estimator's target path meets on identical data.
    s = synth(n=5000)
    cases = {
        "zero": (torch.zeros_like(s["delta"]), None),
        "explained": (s["delta"], s["delta"].clone()),
    }
    for name, (delta, calc) in cases.items():
        est = DifferencePowerEstimator()
        fit = est.get(delta, s["sigma"], s["dss"], f_dark=s["f"], delta_calc=calc)
        snr = fit.snr(s["sigma"], d_star_sq=s["dss"], f_dark=s["f"])
        assert bool(torch.isfinite(snr).all()), name
        assert float(snr.max()) < 1e-2, name
        if calc is not None:
            assert float((fit.alpha_at(s["dss"]) - 1.0).abs().max()) < 1e-3
