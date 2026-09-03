"""The weighting half of the split: properties, not a preferred answer.

Every assertion here is something that has to be true whatever the caps end up
being. The numbers themselves are screened, not asserted -- pinning a cap in a
unit test would make the screen unfalsifiable.
"""

import pytest
import torch

from torchref.scaling.weighting import (
    DEFAULT_SNR_CAP, empirical_sigma_a, information_weight,
    inverse_variance_weight, normalise_weight, snr_from_amplitude,
)

pytestmark = pytest.mark.unit


def test_information_weight_saturates_at_one():
    """Past the cap, better measurement buys nothing: the model is the limit."""
    snr = torch.tensor([0.0, 1.0, 5.0, 50.0, 1e4], dtype=torch.float64)
    w = information_weight(snr, cap=5.0)
    assert float(w[0]) == 0.0
    assert float(w[2]) == pytest.approx(0.5), "the cap is where it reaches a half"
    assert float(w[-1]) == pytest.approx(1.0, abs=1e-6)
    assert bool((w[1:] > w[:-1]).all()), "must be monotone in signal-to-noise"


def test_information_weight_is_the_variance_ratio_in_disguise():
    """``w = 1/(1 + sigma_meas^2/sigma_model^2)`` -- not a sigmoid picked by eye.

    With the cap standing for the signal-to-noise at which the two errors are
    equal, the weight must equal that expression exactly.
    """
    snr = torch.linspace(0.1, 40.0, 200, dtype=torch.float64)
    cap = 5.0
    expected = 1.0 / (1.0 + (cap / snr) ** 2)
    assert torch.allclose(information_weight(snr, cap=cap), expected)


def test_below_the_cap_the_weight_is_quadratic_in_snr():
    """Where measurement error dominates, information goes as snr^2."""
    snr = torch.tensor([0.01, 0.02, 0.04], dtype=torch.float64)
    w = information_weight(snr, cap=DEFAULT_SNR_CAP)
    assert float(w[1] / w[0]) == pytest.approx(4.0, rel=1e-3)
    assert float(w[2] / w[1]) == pytest.approx(4.0, rel=1e-3)


def test_snr_uses_the_intensity_convention():
    """``I/sigma_I`` with ``I = F^2`` is ``F/(2 sigma_F)``, half the amplitude's.

    Only a factor of two, and only a rescaling of the cap -- but quoting a cap
    against the wrong signal-to-noise silently doubles it.
    """
    F = torch.tensor([10.0, 100.0], dtype=torch.float64)
    sig = torch.tensor([1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(snr_from_amplitude(F, sig),
                          torch.tensor([5.0, 50.0], dtype=torch.float64))


def test_the_weight_rises_as_either_error_falls():
    """Inverse variance: better data or a better model both mean more weight."""
    snr = torch.tensor([1.0, 1.0, 10.0], dtype=torch.float64)
    sa = torch.tensor([0.1, 0.9, 0.1], dtype=torch.float64)
    w = inverse_variance_weight(snr, sa, cap=1e9)
    assert float(w[1]) > float(w[0]), "a more reliable model must weigh more"
    assert float(w[2]) > float(w[0]), "a better measurement must weigh more"


def test_measurement_error_bounds_the_low_resolution_weight():
    """``sigma_A -> 1`` sends the model variance to zero; ``1/snr^2`` is what
    stops the weight diverging, and it must, because that regime is where the
    strongest reflections live and one of them could otherwise carry the run."""
    sa = torch.tensor([0.999999, 0.999999], dtype=torch.float64)
    snr = torch.tensor([5.0, 50.0], dtype=torch.float64)
    w = inverse_variance_weight(snr, sa, cap=1e9)
    assert bool(torch.isfinite(w).all())
    # snr^2 is the ceiling, approached from below: the residual model variance
    # is small but not zero, and it bites harder the better the measurement.
    assert float(w[0]) <= 25.0 and float(w[0]) == pytest.approx(25.0, rel=1e-2)
    assert float(w[1]) <= 2500.0 and float(w[1]) == pytest.approx(2500.0, rel=1e-2)


def test_the_weight_keeps_its_dependence_on_model_error():
    """The factorised form lost this, which is why it was abandoned.

    A product of a ``snr`` term and ``sigma_A/(eps - sigma_A^2)`` came out
    identical for a 0.5 A and a 1.0 A coordinate error, because the singularity
    set the shape and the model error dropped out. The coupled form must not.
    """
    import math
    s = torch.linspace(0.02, 0.5, 12, dtype=torch.float64)
    snr = torch.full_like(s, 8.0)
    curves = []
    for dv in (0.5, 1.0):
        sa = torch.exp(-(2.0 / 3.0) * (math.pi ** 2) * s * s * dv * dv)
        curves.append(normalise_weight(
            inverse_variance_weight(snr, sa, cap=1e9)))
    spread = float((curves[0] / curves[1] - 1).abs().max())
    assert spread > 0.1, (
        f"the weight barely moved ({spread:.3f}) between a 0.5 A and a 1.0 A "
        f"model error; it has stopped carrying model information"
    )


def test_epsilon_enters_the_variance_not_the_signal():
    """``V = eps - sigma_A^2``, so higher multiplicity means more variance and
    therefore less weight, at fixed model reliability and measurement error."""
    sa = torch.full((4,), 0.5, dtype=torch.float64)
    snr = torch.full((4,), 10.0, dtype=torch.float64)
    plain = inverse_variance_weight(snr, sa, cap=1e9)
    axial = inverse_variance_weight(
        snr, sa, eps=torch.full((4,), 2.0, dtype=torch.float64), cap=1e9)
    assert bool((axial < plain).all())


def test_normalise_weight_sets_the_mean():
    w = torch.rand(1000, dtype=torch.float64) * 7.0 + 0.5
    assert float(normalise_weight(w).mean()) == pytest.approx(1.0)


def test_a_constant_weight_survives_normalisation_as_ones():
    w = torch.full((100,), 3.7, dtype=torch.float64)
    assert torch.allclose(normalise_weight(w), torch.ones_like(w))


def test_empirical_sigma_a_ignores_the_absolute_scale():
    """The data's scale is arbitrary and the model's is electrons; neither is information.

    Before this held, the ratio's level set the answer: a flat sigma_A of 0.2 on
    one structure and 0.33 on another, with no resolution dependence at all.
    """
    s = torch.linspace(0.07, 0.25, 40, dtype=torch.float64)
    obs = torch.exp(-30.0 * s * s)
    calc = torch.exp(-30.0 * s * s)
    base = empirical_sigma_a(obs, calc)
    torch.testing.assert_close(empirical_sigma_a(1000.0 * obs, calc), base)
    torch.testing.assert_close(empirical_sigma_a(obs, 1e-3 * calc), base)


def test_empirical_sigma_a_reads_the_shape():
    """Identical shapes: full trust everywhere. A low-resolution deficit: less trust there."""
    s = torch.linspace(0.07, 0.25, 40, dtype=torch.float64)
    calc = torch.exp(-30.0 * s * s)
    same = empirical_sigma_a(7.0 * calc, calc)
    assert float(same.min()) > 0.999
    # The model predicts more scattering at low resolution than the data have
    # -- the bulk solvent it lacks -- and matches at high resolution.
    deficit = torch.where(s < 0.12, torch.full_like(s, 0.4), torch.ones_like(s))
    sa = empirical_sigma_a(calc * deficit, calc)
    assert float(sa[s < 0.12].max()) < float(sa[s > 0.15].min())
    assert bool((sa <= 1.0).all()) and bool((sa > 0.0).all())
