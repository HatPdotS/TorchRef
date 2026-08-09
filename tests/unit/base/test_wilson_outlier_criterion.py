"""Contracts for the two-tailed, model-free Wilson outlier criterion.

Complements ``test_wilson_outliers.py``, which pins the French-Wilson statistic
itself. Each piece here has its own way of being quietly wrong: ``log Phi`` by
underflowing on an accelerator, the predictive tail by disagreeing with the
density it claims to integrate, ``Sigma`` by being dragged up by the very
outliers it is meant to expose, and the threshold by not scaling with the number
of simultaneous tests.
"""

import math

import pytest
import torch

from torchref.base.reciprocal.basis import get_scattering_vectors
from torchref.base.wilson_outliers import (
    _normal_quantile,
    anisotropic_correction,
    fit_anisotropic_scale,
    log_normal_cdf,
    robust_mean_intensity,
    wilson_log_upper_tail,
    wilson_outlier_mask,
)

CELL = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])


# =============================================================================
# log Phi
# =============================================================================


@pytest.mark.unit
def test_log_normal_cdf_matches_the_reference_across_the_tail():
    x = torch.linspace(-40.0, 10.0, 5001, dtype=torch.float64)

    error = (log_normal_cdf(x) - torch.special.log_ndtr(x)).abs()

    assert float(error.max()) < 1e-6


@pytest.mark.unit
def test_log_normal_cdf_survives_float32_in_the_far_tail():
    """No -inf where the probability is small but perfectly representable."""
    x = torch.tensor([-40.0, -20.0, -10.0, -5.0], dtype=torch.float32)

    got = log_normal_cdf(x)

    assert bool(torch.isfinite(got).all())
    assert float(got[0]) == pytest.approx(-804.608, abs=1e-2)


@pytest.mark.unit
def test_log_normal_cdf_agrees_across_devices(any_device):
    """MPS erfc returns exactly zero past argument 4, so a device split here
    would be silent."""
    x = torch.tensor([-30.0, -8.0, -5.0, -2.0, 0.0, 5.0])

    reference = log_normal_cdf(x)
    got = log_normal_cdf(x.to(any_device)).cpu()

    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_normal_quantile_inverts_the_cdf():
    for p in (0.5, 1e-3, 1e-7, 1e-12):
        x = _normal_quantile(p)
        got = float(log_normal_cdf(torch.tensor(x, dtype=torch.float64)))
        assert got == pytest.approx(math.log(p), abs=1e-6)


# =============================================================================
# The predictive tail
# =============================================================================


def _numerical_upper_tail(I, sigma, Sigma, centric=False):
    """``P(I' > I)`` by direct quadrature over the Wilson prior."""
    import numpy as np
    from scipy.special import erfc

    scale = 2.0 * Sigma if centric else Sigma
    J = np.linspace(0.0, 80.0 * Sigma, 800001)
    prior = np.exp(-J / scale) / scale
    survival = 0.5 * erfc((I - J) / (sigma * math.sqrt(2.0)))
    return float(np.trapezoid(prior * survival, J))


@pytest.mark.unit
@pytest.mark.parametrize("I", [-50.0, 0.0, 100.0, 500.0, 2000.0])
def test_upper_tail_matches_quadrature_of_the_convolution(I):
    Sigma, sigma = 200.0, 30.0

    log_p = float(
        wilson_log_upper_tail(
            torch.tensor(I, dtype=torch.float64),
            torch.tensor(sigma, dtype=torch.float64),
            torch.tensor(Sigma, dtype=torch.float64),
        )
    )

    assert math.exp(log_p) == pytest.approx(
        _numerical_upper_tail(I, sigma, Sigma), rel=1e-3
    )


@pytest.mark.unit
def test_upper_tail_is_monotone_and_bounded():
    I = torch.linspace(-500.0, 5000.0, 400, dtype=torch.float64)
    sigma = torch.full_like(I, 30.0)
    Sigma = torch.full_like(I, 200.0)

    log_p = wilson_log_upper_tail(I, sigma, Sigma)

    assert bool((log_p <= 1e-9).all()), "a log probability cannot exceed zero"
    assert bool((log_p.diff() <= 1e-9).all()), "P(I' > I) must decrease with I"


@pytest.mark.unit
def test_centric_reflections_have_the_heavier_tail():
    """Twice the variance per degree of freedom, so the same intensity is less
    surprising and must not be flagged on the acentric threshold."""
    I = torch.tensor([1500.0], dtype=torch.float64)
    sigma = torch.tensor([30.0], dtype=torch.float64)
    Sigma = torch.tensor([200.0], dtype=torch.float64)

    acentric = wilson_log_upper_tail(I, sigma, Sigma, is_centric=False)
    centric = wilson_log_upper_tail(I, sigma, Sigma, is_centric=True)

    assert float(centric) > float(acentric)


# =============================================================================
# Sigma
# =============================================================================


def _wilson_intensities(n=20000, Sigma=500.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    I = -Sigma * torch.log(torch.rand(n, generator=generator).clamp(min=1e-12))
    return I, torch.linspace(8.0, 1.5, n)


@pytest.mark.unit
def test_robust_sigma_recovers_a_known_value():
    I, d = _wilson_intensities()
    assign = torch.ones_like(I, dtype=torch.bool)

    Sigma = robust_mean_intensity(I, d, assign)

    assert float(Sigma.median()) == pytest.approx(500.0, rel=0.05)


@pytest.mark.unit
def test_robust_sigma_ignores_the_outliers_a_mean_absorbs():
    """A shell's mean is raised by its own contamination; its median is not,
    which is the difference between exposing an outlier and hiding it."""
    I, d = _wilson_intensities()
    contaminated = I.clone()
    contaminated[::100] *= 100.0  # 1% of reflections, two orders of magnitude up
    assign = torch.ones_like(I, dtype=torch.bool)

    median_based = float(robust_mean_intensity(contaminated, d, assign).median())

    assert median_based == pytest.approx(500.0, rel=0.05)
    assert float(contaminated.mean()) > 1.5 * 500.0


@pytest.mark.unit
def test_held_out_reflections_still_receive_a_sigma():
    """Holding a reflection out of the estimate must not stop it being tested --
    that is how a flagged outlier silently un-flags itself on the next pass."""
    I, d = _wilson_intensities()
    assign = torch.ones_like(I, dtype=torch.bool)
    estimate = assign.clone()
    estimate[:50] = False

    Sigma = robust_mean_intensity(I, d, assign, estimate)

    assert bool(torch.isfinite(Sigma[:50]).all())
    assert bool((Sigma[:50] > 0).all())


@pytest.mark.unit
def test_sigma_is_absent_where_no_shell_could_be_formed():
    I, d = _wilson_intensities(n=100)
    assign = torch.ones_like(I, dtype=torch.bool)

    Sigma = robust_mean_intensity(I, d, assign, per_shell=250)

    assert bool(torch.isnan(Sigma).all())


# =============================================================================
# Anisotropy
# =============================================================================


def _anisotropic_dataset(U_true, half_width=22, Sigma=500.0, seed=1):
    axis = torch.arange(-half_width, half_width + 1)
    hkl = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    hkl = hkl.reshape(-1, 3)
    hkl = hkl[hkl.abs().sum(dim=1) > 0].to(torch.float32)

    s = get_scattering_vectors(hkl, CELL)
    d = 1.0 / torch.linalg.vector_norm(s, dim=1).clamp(min=1e-6)

    truth = anisotropic_correction(s, U_true)
    generator = torch.Generator().manual_seed(seed)
    w = -torch.log(torch.rand(len(hkl), generator=generator).clamp(min=1e-12))
    return hkl, s, d, Sigma * truth * w, truth


@pytest.mark.unit
def test_anisotropic_fit_recovers_a_planted_correction():
    # Magnitudes chosen to span roughly a factor of five between the strong and
    # weak directions, which is what real anisotropy looks like. A percent-level
    # U is not recoverable against Exp(1) scatter and testing one proves nothing.
    U_true = torch.tensor([0.25, -0.15, -0.10, 0.0, 0.0, 0.0])
    _, s, _, I, truth = _anisotropic_dataset(U_true)
    sigma = 0.02 * I + 1.0
    Sigma_iso = torch.full_like(I, float(I.median() / math.log(2.0)))
    fittable = torch.ones_like(I, dtype=torch.bool)

    recovered = anisotropic_correction(
        s, fit_anisotropic_scale(I, sigma, Sigma_iso, s, fittable)
    )

    # The correction is what matters, not U itself: any isotropic part of U is
    # absorbed by the shell estimate and is not identifiable from it.
    correlation = torch.corrcoef(torch.stack([recovered, truth]))[0, 1]
    assert float(correlation) > 0.99


@pytest.mark.unit
def test_anisotropic_correction_has_unit_mean():
    """Otherwise it competes with the shell medians for the absolute scale."""
    U = torch.tensor([0.30, -0.10, 0.15, 0.04, 0.0, -0.02])
    _, s, _, _, _ = _anisotropic_dataset(U)

    assert float(anisotropic_correction(s, U).mean()) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.unit
def test_anisotropic_fit_declines_rather_than_guessing_on_thin_data():
    I = torch.full((20,), 100.0)
    s = torch.zeros(20, 3)
    Sigma = torch.full((20,), 100.0)

    U = fit_anisotropic_scale(I, 0.1 * I, Sigma, s, torch.ones(20, dtype=torch.bool))

    assert bool((U == 0).all())


# =============================================================================
# The mask
# =============================================================================


def _clean_dataset(n_per_axis=18, Sigma=500.0, seed=3):
    axis = torch.arange(-n_per_axis, n_per_axis + 1)
    hkl = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    hkl = hkl.reshape(-1, 3)
    hkl = hkl[hkl.abs().sum(dim=1) > 0].to(torch.float32)

    s = get_scattering_vectors(hkl, CELL)
    d = 1.0 / torch.linalg.vector_norm(s, dim=1).clamp(min=1e-6)

    generator = torch.Generator().manual_seed(seed)
    w = -torch.log(torch.rand(len(hkl), generator=generator).clamp(min=1e-12))
    I = Sigma * w
    return hkl, d, I, 0.02 * I + 5.0


@pytest.mark.unit
def test_clean_wilson_data_survives_the_test():
    """Data drawn from the distribution the criterion assumes must not be
    rejected by it: the threshold is family-wise over the whole dataset."""
    hkl, d, I, sigma = _clean_dataset()

    keep, info = wilson_outlier_mask(I, sigma, hkl, d, CELL, d_max=100.0)

    assert info["n_tested"] > 10000
    assert int((~keep).sum()) <= 2


@pytest.mark.unit
def test_planted_zingers_are_caught():
    hkl, d, I, sigma = _clean_dataset()
    corrupted = I.clone()
    generator = torch.Generator().manual_seed(7)
    planted = torch.randperm(len(I), generator=generator)[:100]
    corrupted[planted] *= 50.0

    keep, _ = wilson_outlier_mask(corrupted, sigma, hkl, d, CELL, d_max=100.0)

    flagged = set(torch.nonzero(~keep, as_tuple=True)[0].tolist())
    caught = len(flagged & set(planted.tolist()))
    assert caught > 60, f"caught only {caught}/100"
    # Precision matters as much as recall: this rejects real measurements.
    assert len(flagged - set(planted.tolist())) < 10


@pytest.mark.unit
def test_threshold_tightens_with_dataset_size():
    """A fixed per-reflection p-value rejects a fixed fraction of good data; the
    threshold has to know how many simultaneous tests it is making."""
    hkl, d, I, sigma = _clean_dataset()

    _, small = wilson_outlier_mask(
        I[:20000], sigma[:20000], hkl[:20000], d[:20000], CELL, d_max=100.0
    )
    _, large = wilson_outlier_mask(I, sigma, hkl, d, CELL, d_max=100.0)

    ratio = large["n_tested"] / small["n_tested"]
    assert small["log_p_threshold"] - large["log_p_threshold"] == pytest.approx(
        math.log(ratio), abs=1e-6
    )


@pytest.mark.unit
def test_unusable_rows_are_kept_rather_than_called_outliers():
    hkl, d, I, sigma = _clean_dataset()
    usable = torch.ones_like(I, dtype=torch.bool)
    usable[:100] = False

    keep, info = wilson_outlier_mask(
        I, sigma, hkl, d, CELL, usable=usable, d_max=100.0
    )

    assert bool(keep[:100].all())
    assert info["n_tested"] <= len(I) - 100


@pytest.mark.unit
def test_low_resolution_reflections_are_left_untested():
    """Bulk solvent dominates there, so departures from Wilson statistics have
    nothing to do with being an outlier."""
    hkl, d, I, sigma = _clean_dataset()
    corrupted = I.clone()
    low_resolution = torch.nonzero(d > 6.0, as_tuple=True)[0]
    assert len(low_resolution) > 0
    corrupted[low_resolution] *= 1000.0

    keep, _ = wilson_outlier_mask(corrupted, sigma, hkl, d, CELL, d_max=4.0)

    assert bool(keep[low_resolution].all())


@pytest.mark.unit
def test_epsilon_enhanced_reflections_are_not_flagged_for_being_enhanced():
    """A reflection on a symmetry axis is genuinely stronger; normalising by
    epsilon is what stops that reading as an outlier."""
    hkl, d, I, sigma = _clean_dataset()
    epsilon = torch.ones_like(I)
    on_axis = torch.nonzero((hkl[:, 0] == 0) & (hkl[:, 1] == 0), as_tuple=True)[0]
    assert len(on_axis) > 0
    epsilon[on_axis] = 4.0
    enhanced = I.clone()
    enhanced[on_axis] *= 4.0

    with_epsilon, _ = wilson_outlier_mask(
        enhanced, sigma, hkl, d, CELL, epsilon=epsilon, d_max=100.0
    )
    without_epsilon, _ = wilson_outlier_mask(
        enhanced, sigma, hkl, d, CELL, d_max=100.0
    )

    n_with = int((~with_epsilon[on_axis]).sum())
    n_without = int((~without_epsilon[on_axis]).sum())
    assert n_with <= n_without
    assert n_with == 0
