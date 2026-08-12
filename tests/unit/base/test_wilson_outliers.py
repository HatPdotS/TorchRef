"""Wilson-probability outlier rejection.

``h = I/sigma - sigma/Sigma`` is the standardized argument of the Wilson prior
convolved with Gaussian measurement error, so French-Wilson's ``h >= h_min``
guard is a tail-probability cut on the observation. These tests lock the
statistic to its historical inline form, pin the behaviour that matters
(negative intensities explainable as noise are *kept*), and document the one
thing the amplitude-only path cannot do.
"""

import pytest
import torch

from torchref.base.french_wilson import (
    french_wilson,
    french_wilson_h,
    french_wilson_valid_mask,
    intensities_from_amplitudes,
)


@pytest.mark.unit
def test_h_matches_original_inline_formulas():
    """Regression lock against the expressions french_wilson_* used inline."""
    I = torch.tensor([100.0, 5.0, -15.0, 0.5])
    sigma_I = torch.tensor([10.0, 8.0, 7.0, 3.0])
    Sigma = torch.tensor([80.0, 80.0, 80.0, 40.0])

    acentric = french_wilson_h(I, sigma_I, Sigma, is_centric=False)
    torch.testing.assert_close(acentric, (I / sigma_I) - (sigma_I / Sigma))

    centric = french_wilson_h(I, sigma_I, Sigma, is_centric=True)
    torch.testing.assert_close(centric, (I / sigma_I) - (sigma_I / (2.0 * Sigma)))

    # None must behave exactly like all-acentric.
    torch.testing.assert_close(french_wilson_h(I, sigma_I, Sigma), acentric)


@pytest.mark.unit
def test_h_centric_penalty_is_half_the_acentric_one():
    I = torch.tensor([10.0, -4.0])
    sigma_I = torch.tensor([6.0, 6.0])
    Sigma = torch.tensor([50.0, 50.0])

    acentric = french_wilson_h(I, sigma_I, Sigma, is_centric=False)
    centric = french_wilson_h(I, sigma_I, Sigma, is_centric=True)

    # Both subtract sigma/Sigma from I/sigma; the centric one subtracts half.
    torch.testing.assert_close(centric - acentric, 0.5 * sigma_I / Sigma)

    # A per-reflection mask must agree with the homogeneous calls elementwise.
    mixed = french_wilson_h(
        I, sigma_I, Sigma, is_centric=torch.tensor([True, False])
    )
    torch.testing.assert_close(mixed, torch.stack([centric[0], acentric[1]]))


@pytest.mark.unit
def test_negative_intensities_within_noise_are_kept():
    """The whole point: a negative I explainable as noise is not an outlier."""
    sigma_I = torch.full((4,), 10.0)
    Sigma = torch.full((4,), 500.0)
    #                 mildly negative, well within noise ... then absurd
    I = torch.tensor([-5.0, -15.0, -25.0, -5000.0])

    keep = french_wilson_valid_mask(I, sigma_I, Sigma, h_min=-4.0)

    # The first three are ordinary weak measurements that happen to come out
    # negative; only the last is too negative for any Wilson reflection.
    assert keep.tolist() == [True, True, True, False]


@pytest.mark.unit
def test_inflated_sigma_drives_h_down():
    """The -sigma/Sigma term is what folds sigma and the shell mean together."""
    I = torch.tensor([0.0, 0.0])
    Sigma = torch.tensor([100.0, 100.0])
    sigma_I = torch.tensor([10.0, 1000.0])  # second sigma is nonsense

    h = french_wilson_h(I, sigma_I, Sigma)
    assert h[0] > h[1]
    keep = french_wilson_valid_mask(I, sigma_I, Sigma, h_min=-4.0)
    assert keep.tolist() == [True, False]


@pytest.mark.unit
def test_valid_mask_rejects_rather_than_propagating_non_finite():
    I = torch.tensor([10.0, 10.0, float("nan")])
    sigma_I = torch.tensor([1.0, 0.0, 1.0])  # zero sigma -> h is not finite
    Sigma = torch.tensor([50.0, 50.0, 50.0])

    keep = french_wilson_valid_mask(I, sigma_I, Sigma, h_min=-4.0)
    assert keep.tolist() == [True, False, False]


@pytest.mark.unit
def test_intensities_from_amplitudes_never_divides_by_zero():
    F = torch.tensor([10.0, 0.0, 10.0, -1.0, float("nan")])
    sigma_F = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])

    I, sigma_I = intensities_from_amplitudes(F, sigma_F)

    assert torch.isfinite(I).all()
    assert torch.isfinite(sigma_I).all()
    # Only the first row is a usable measurement.
    torch.testing.assert_close(I[0], torch.tensor(100.0))
    torch.testing.assert_close(sigma_I[0], torch.tensor(20.0))
    assert (sigma_I[1:] == 0).all()


@pytest.mark.unit
def test_amplitude_round_trip_is_lossy_in_the_documented_direction():
    """F is a positive posterior mean, so negative-intensity evidence is gone.

    This is not a defect to fix -- it is why the amplitude-only path is
    structurally weaker than the intensity path, and it must stay documented.
    """
    I = torch.tensor([-40.0, 5.0, 500.0])
    sigma_I = torch.tensor([10.0, 10.0, 10.0])
    Sigma = torch.tensor([200.0, 200.0, 200.0])

    h_true = french_wilson_h(I, sigma_I, Sigma)
    F, sigma_F, _ = french_wilson(I, sigma_I, Sigma)

    # French-Wilson output is strictly positive whatever the input intensity.
    assert (F > 0).all()

    I_rt, sigma_I_rt = intensities_from_amplitudes(F, sigma_F)
    assert (I_rt >= 0).all()

    h_rt = french_wilson_h(I_rt, sigma_I_rt, Sigma)
    # The strongly negative reflection round-trips to a non-negative h: the
    # reconstruction cannot see that it was ever negative.
    assert h_true[0] < 0 < h_rt[0]
