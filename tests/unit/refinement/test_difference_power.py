"""Properties of the shell-free difference-power fit.

Pinned on seeded synthetic differences with a known power law: the power, the
dark-amplitude exponent and the sigma scale are recovered from one dataset, including
when the reported sigmas are uniformly inflated; a fixed exponent stays fixed; a model
difference's resolution-dependent coupling and the power it leaves unexplained are
recovered together; the bounded Wiener weight never falls below its floor, so no
reflection or resolution range is removed even when the data hold no signal; the sigma
scale stays within its bounds when the differences hold no noise; the estimator caches
one fit until reset and applies its configured exponent; the fit runs under
``torch.no_grad()`` and on every available device.
"""

import pytest
import torch

from torchref.refinement.model_error_estimation.difference_power import (
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


@pytest.mark.unit
@pytest.mark.parametrize("inflation", [1.0, 1.5])
def test_recovers_power_exponent_and_sigma_scale(inflation):
    s = synth(sigma_inflation=inflation)
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    assert fit.converged
    assert fit.gamma == pytest.approx(1.0, abs=GAMMA_ATOL)
    assert fit.sigma_scale == pytest.approx(1.0 / inflation, rel=K_RTOL)
    power = fit.signal_power(s["dss"], f_dark=s["f"])
    err = (power / s["s_true"]).log().abs().median()
    assert float(err) < LOG_POWER_ATOL
    assert torch.isfinite(fit.stderr[: len(fit.coeffs)]).all()


@pytest.mark.unit
def test_fixed_gamma_is_kept():
    s = synth()
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"], gamma=0.0)
    assert fit.gamma == 0.0
    fit = fit_difference_power(s["delta"], s["sigma"], s["dss"], f_dark=s["f"], gamma=2.0)
    assert fit.gamma == 2.0


@pytest.mark.unit
def test_weight_never_removes_a_reflection_without_signal():
    s = synth(signal=0.0)
    fit = fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"], fit_sigma_scale=False
    )
    snr = fit.snr(s["sigma"], d_star_sq=s["dss"], f_dark=s["f"])
    w = bounded_wiener_weight(snr, 0.5)
    assert float(w.min()) >= 1.0 / 3.0 - 1e-6
    assert float(w.max()) < 1.0
    assert float(bounded_wiener_weight(torch.zeros(1), 0.0)) == 0.0
    with pytest.raises(ValueError):
        bounded_wiener_weight(snr, -0.1)


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
    fit = fit_difference_power(zeros, s["sigma"], s["dss"], f_dark=s["f"])
    assert fit.sigma_scale == pytest.approx(SIGMA_SCALE_BOUNDS[0], rel=1e-3)
    assert fit.sigma_scale_at_bound
    snr = fit.snr(s["sigma"], d_star_sq=s["dss"], f_dark=s["f"])
    assert bool(torch.isfinite(snr).all())
    assert not fit_difference_power(
        s["delta"], s["sigma"], s["dss"], f_dark=s["f"]
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
    est = DifferencePowerEstimator(DifferencePowerConfig(gamma=0.0))
    assert est.fit is None
    first = est.get(s["delta"], s["sigma"], s["dss"], f_dark=s["f"])
    assert first.gamma == 0.0 and est.fit is first
    # Cached: different arguments are ignored until reset.
    assert est.get(2 * s["delta"], s["sigma"], s["dss"], f_dark=s["f"]) is first
    est.reset()
    assert est.fit is None
    assert est.get(s["delta"], s["sigma"], s["dss"], f_dark=s["f"]) is not first
    with pytest.raises(ValueError):
        DifferencePowerConfig(gamma=10.0)
