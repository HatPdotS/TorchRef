"""Properties of the sigma_D difference-power estimator.

Pinned on seeded synthetic differences with a KNOWN power law: the per-shell power is
recovered from a single dataset through ``mean(dF**2) - mean(sigma**2)``, the dark-
amplitude exponent is recovered and can be fixed, the moment identity with a difference
model holds exactly, clamps are counted and an all-noise input is flagged rather than
weighted, degenerate inputs stay finite, per-reflection weights lie in ``[0, 1)`` with the
shell mean of the power preserved, the fit is deterministic and device-independent, and
the cached estimator resets on demand.
"""

import pytest
import torch

from torchref.refinement.model_error_estimation._shells import interp_in_dss
from torchref.refinement.model_error_estimation.sigma_d import (
    GAMMA_DEFAULT,
    SigmaDConfig,
    SigmaDEstimator,
    estimate_sigma_d,
    sigma_d_per_reflection,
)

#: Tolerance on the recovered shell power relative to the truth. A shell of 140
#: reflections estimates ``B`` with a relative sd of ``sqrt(2/140) = 12%``; the line
#: shrinkage pools shells, and the decile means below average ~14 shells, so 10% is
#: ~3 sd of what remains.
POWER_RTOL = 0.10
#: Tolerance on the fitted exponent. Its standard error at 30 000 reflections is ~0.04;
#: 0.15 is well above that and well below the difference between the pure shell model
#: (0) and the default (1).
GAMMA_ATOL = 0.15


def synth_diff(
    n=30000,
    gamma=1.0,
    sig_frac=1.0,
    seed=7,
    dtype=torch.float32,
    device="cpu",
    with_model=False,
    alpha_true=0.8,
):
    """Signed differences with power ``Sigma_N(d*^2) * (F / <F>)**gamma``.

    ``F`` is Wilson-like (the modulus of a complex normal) so the amplitude classes are
    populated realistically; ``Sigma_N`` falls with resolution. The measurement sigma is
    ``sig_frac`` times the rms true difference, constant across reflections so the
    inverse-variance and sigma_D weights differ only through ``S``.
    """
    g = torch.Generator().manual_seed(seed)
    dss = torch.linspace(0.02, 0.35, n, dtype=torch.float64)
    f = (
        torch.randn(n, generator=g, dtype=torch.float64) ** 2
        + torch.randn(n, generator=g, dtype=torch.float64) ** 2
    ).sqrt() * 10.0
    sigma_n = 0.05 * torch.exp(-3.0 * dss)
    s_true = sigma_n * (f / f.mean()) ** gamma
    d_true = torch.randn(n, generator=g, dtype=torch.float64) * s_true.sqrt()
    sig = torch.full(
        (n,), float(sig_frac) * float(s_true.mean().sqrt()), dtype=torch.float64
    )
    d_obs = d_true + torch.randn(n, generator=g, dtype=torch.float64) * sig
    out = {
        "delta_obs": d_obs,
        "sigma_diff": sig,
        "d_star_sq": dss,
        "f_dark": f,
        "s_true": s_true,
        "fit_mask": torch.ones(n, dtype=torch.bool),
    }
    if with_model:
        beta_true = 0.3 * s_true
        out["delta_calc"] = (
            d_true + torch.randn(n, generator=g, dtype=torch.float64) * beta_true.sqrt()
        ) / alpha_true
    return {
        k: (
            v.to(device=device, dtype=dtype)
            if v.dtype.is_floating_point
            else v.to(device)
        )
        for k, v in out.items()
    }


def _decile_means(values, dss, n_dec=10):
    order = torch.argsort(dss)
    chunks = torch.chunk(values[order], n_dec)
    return torch.stack([c.mean() for c in chunks])


@pytest.mark.unit
def test_recovers_shell_power_from_one_dataset(any_device):
    d = synth_diff(device=any_device)
    sh = estimate_sigma_d(
        d["delta_obs"],
        d["sigma_diff"],
        None,
        d["d_star_sq"],
        d["f_dark"],
        d["fit_mask"],
    )
    est = sigma_d_per_reflection(sh, d["d_star_sq"], None, d["f_dark"], d["sigma_diff"])
    assert not sh.degenerate and not sh.all_zero
    assert (sh.Sigma_N > 0).all()
    got = _decile_means(est.S, d["d_star_sq"])
    want = _decile_means(d["s_true"], d["d_star_sq"])
    assert torch.allclose(got, want, rtol=POWER_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize("gamma", [1.0, 0.5])
def test_recovers_the_amplitude_exponent(gamma):
    d = synth_diff(gamma=gamma, dtype=torch.float64)
    sh = estimate_sigma_d(
        d["delta_obs"],
        d["sigma_diff"],
        None,
        d["d_star_sq"],
        d["f_dark"],
        d["fit_mask"],
    )
    assert sh.gamma_fitted and sh.diagnostics["gamma_reason"] == "fitted"
    assert abs(sh.gamma - gamma) < GAMMA_ATOL
    assert sh.diagnostics["gamma_se"] < GAMMA_ATOL


@pytest.mark.unit
def test_fixed_exponent_is_honoured():
    d = synth_diff(n=5000)
    sh = estimate_sigma_d(
        d["delta_obs"],
        d["sigma_diff"],
        None,
        d["d_star_sq"],
        d["f_dark"],
        d["fit_mask"],
        gamma=0.7,
    )
    assert sh.gamma == 0.7 and not sh.gamma_fitted
    assert sh.diagnostics["gamma_reason"] == "fixed"
    with pytest.raises(ValueError):
        SigmaDConfig(gamma=3.0)


@pytest.mark.unit
def test_without_dark_amplitude_the_power_is_flat_within_a_shell():
    d = synth_diff(n=5000)
    sh = estimate_sigma_d(
        d["delta_obs"], d["sigma_diff"], None, d["d_star_sq"], None, d["fit_mask"]
    )
    assert sh.gamma == GAMMA_DEFAULT and not sh.gamma_fitted
    assert sh.diagnostics["gamma_reason"] == "no_f_dark"
    est = sigma_d_per_reflection(sh, d["d_star_sq"], None, None, d["sigma_diff"])
    # Reflections at the same resolution share the power regardless of amplitude.
    order = torch.argsort(d["d_star_sq"])
    close = est.S[order][:200]
    assert float(close.max() / close.min()) < 1.05


@pytest.mark.unit
@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 1e-4), (torch.float64, 1e-10)])
def test_moment_identity_with_a_difference_model(dtype, rtol):
    d = synth_diff(dtype=dtype, with_model=True)
    sh = estimate_sigma_d(
        d["delta_obs"],
        d["sigma_diff"],
        None,
        d["d_star_sq"],
        d["f_dark"],
        d["fit_mask"],
        delta_calc=d["delta_calc"],
        shrink=False,
    )
    assert sh.has_model
    assert sh.diagnostics["n_s2_clamped"] == 0
    # Sampling noise can push alpha**2 Sigma_P above Sigma_N in a few shells; the clamp
    # there is counted, and the identity is exact everywhere it did not fire.
    unclamped = sh.alpha**2 * sh.Sigma_P <= sh.Sigma_N
    assert (
        int(unclamped.sum())
        == sh.diagnostics["n_shell"] - sh.diagnostics["n_beta_clamped"]
    )
    assert float(unclamped.float().mean()) > 0.8
    lhs = sh.alpha**2 * sh.Sigma_P + sh.beta_model + sh.S2
    assert torch.allclose(lhs[unclamped], sh.B[unclamped], rtol=rtol)
    # alpha is the Gaussian coupling S / (S + beta_true) / alpha_true-scaled slope; it must
    # be positive and below one for this generator.
    assert (sh.alpha > 0).all() and (sh.alpha < 1).all()


@pytest.mark.unit
def test_clamps_are_counted_and_all_noise_is_flagged():
    d = synth_diff(n=5000, sig_frac=5.0)
    sh = estimate_sigma_d(
        d["delta_obs"],
        d["sigma_diff"],
        None,
        d["d_star_sq"],
        d["f_dark"],
        d["fit_mask"],
    )
    assert sh.diagnostics["n_s2_clamped"] > 0
    noise = synth_diff(n=5000, sig_frac=50.0)
    sh2 = estimate_sigma_d(
        noise["delta_obs"],
        noise["sigma_diff"] * 1.2,
        None,
        noise["d_star_sq"],
        noise["f_dark"],
        noise["fit_mask"],
        shrink=False,
    )
    assert sh2.all_zero
    est = sigma_d_per_reflection(
        sh2, noise["d_star_sq"], None, noise["f_dark"], noise["sigma_diff"]
    )
    assert torch.equal(est.w, torch.zeros_like(est.w))


@pytest.mark.unit
def test_pure_noise_with_calibrated_sigma_gets_no_power():
    """Nothing tells the estimator whether a difference exists: on pure noise with
    calibrated sigmas the shrinkage must not manufacture power from the positive half
    of the noise in ``B - S2``, and the weights collapse onto inverse variance."""
    g = torch.Generator().manual_seed(3)
    n = 30000
    dss = torch.linspace(0.02, 0.35, n, dtype=torch.float64)
    f = torch.rand(n, generator=g, dtype=torch.float64) * 20.0 + 1.0
    sig = 0.2 + 0.8 * dss
    d_obs = torch.randn(n, generator=g, dtype=torch.float64) * sig
    mask = torch.ones(n, dtype=torch.bool)
    sh = estimate_sigma_d(d_obs, sig, None, dss, f, mask)
    assert not sh.degenerate
    # Shell power is below a few per cent of the noise power in every shell.
    assert (sh.Sigma_N <= 0.05 * sh.S2).all()
    est = sigma_d_per_reflection(sh, dss, None, f, sig)
    ivw = 1.0 / sig**2
    ivw = ivw / ivw.mean()
    w_sd = est.w / est.w.mean().clamp(min=1e-30)
    if not sh.all_zero:
        assert torch.corrcoef(torch.stack([w_sd, ivw]))[0, 1] > 0.97


@pytest.mark.unit
def test_degenerate_input_stays_finite():
    d = synth_diff(n=100)
    mask = torch.zeros(100, dtype=torch.bool)
    mask[0] = True
    sh = estimate_sigma_d(
        d["delta_obs"], d["sigma_diff"], None, d["d_star_sq"], d["f_dark"], mask
    )
    assert sh.degenerate and not sh.all_zero
    est = sigma_d_per_reflection(sh, d["d_star_sq"], None, d["f_dark"], d["sigma_diff"])
    assert torch.isfinite(est.S).all() and torch.isfinite(est.w).all()
    assert est.S.shape == (100,)


@pytest.mark.unit
def test_per_reflection_weights_and_shell_mean(any_device):
    d = synth_diff(device=any_device)
    eps = torch.where(
        torch.arange(d["delta_obs"].numel(), device=any_device) % 7 == 0, 2.0, 1.0
    ).to(d["delta_obs"].dtype)
    sh = estimate_sigma_d(
        d["delta_obs"], d["sigma_diff"], eps, d["d_star_sq"], d["f_dark"], d["fit_mask"]
    )
    est = sigma_d_per_reflection(sh, d["d_star_sq"], eps, d["f_dark"], d["sigma_diff"])
    assert (est.w >= 0).all() and (est.w < 1).all()
    assert est.S.device == d["delta_obs"].device
    # The multiplier has shell mean one, so S / epsilon averages to Sigma_N over a shell.
    counts = sh.counts.to(torch.long)  # dtype-ok: split sizes; PyTorch requires int64
    order = torch.argsort(d["d_star_sq"])
    per_shell = torch.stack(
        [c.mean() for c in torch.split((est.S / eps)[order], counts.tolist())]
    )
    assert torch.allclose(per_shell, sh.Sigma_N, rtol=0.15)
    # A missing dark amplitude means a multiplier of one.
    f_missing = d["f_dark"].clone()
    f_missing[:50] = float("nan")
    est2 = sigma_d_per_reflection(sh, d["d_star_sq"], eps, f_missing, d["sigma_diff"])
    log_sn = interp_in_dss(d["d_star_sq"][:50], sh.bin_dss, torch.log(sh.Sigma_N))
    assert torch.allclose(est2.S[:50], eps[:50] * torch.exp(log_sn), rtol=1e-4)


@pytest.mark.unit
def test_deterministic_and_device_independent(any_device):
    d_cpu = synth_diff()
    a = estimate_sigma_d(
        d_cpu["delta_obs"],
        d_cpu["sigma_diff"],
        None,
        d_cpu["d_star_sq"],
        d_cpu["f_dark"],
        d_cpu["fit_mask"],
    )
    b = estimate_sigma_d(
        d_cpu["delta_obs"],
        d_cpu["sigma_diff"],
        None,
        d_cpu["d_star_sq"],
        d_cpu["f_dark"],
        d_cpu["fit_mask"],
    )
    assert torch.equal(a.Sigma_N, b.Sigma_N) and a.gamma == b.gamma
    d_dev = synth_diff(device=any_device)
    c = estimate_sigma_d(
        d_dev["delta_obs"],
        d_dev["sigma_diff"],
        None,
        d_dev["d_star_sq"],
        d_dev["f_dark"],
        d_dev["fit_mask"],
    )
    assert torch.allclose(c.Sigma_N.cpu(), a.Sigma_N, rtol=1e-4)
    assert abs(c.gamma - a.gamma) < 1e-3


@pytest.mark.unit
def test_estimator_caches_until_reset_and_remaps():
    d = synth_diff(n=5000)
    est = SigmaDEstimator(SigmaDConfig(gamma=1.0))
    first = est.get(
        d["delta_obs"],
        d["sigma_diff"],
        None,
        d["d_star_sq"],
        d["f_dark"],
        d["fit_mask"],
    )
    assert (
        est.get(
            d["delta_obs"],
            d["sigma_diff"],
            None,
            d["d_star_sq"],
            d["f_dark"],
            d["fit_mask"],
        )
        is first
    )
    est.reset()
    assert est._cache is None
    target = d["d_star_sq"][:1000]
    remapped = est.get(
        d["delta_obs"],
        d["sigma_diff"],
        None,
        d["d_star_sq"],
        d["f_dark"],
        d["fit_mask"],
        target_dss=target,
        out_f_dark=d["f_dark"][:1000],
        out_sigma_diff=d["sigma_diff"][:1000],
    )
    assert remapped.S.shape == (1000,) and est.shells is not None
