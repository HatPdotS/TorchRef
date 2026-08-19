"""
Tests for `torchref.experimental.alignment.ml_rotation.compute_sigma_a_luzzati`.

Closed-form Luzzati σA(s) = exp(−2π²·s²·ΔVRMS²) — used to pre-weight the
FRF input field so the bare correlation is already an LLG proxy
(Phaser-style; see phaser_vs_torchref_rotation.md §6).
"""

import math

import torch

from torchref.experimental.alignment.ml_rotation import compute_sigma_a_luzzati


def test_dc_value_is_one():
    """σA(0) = 1.0 for any ΔVRMS — full agreement at the origin."""
    s = torch.tensor([0.0], dtype=torch.float64)
    for vrms in [0.1, 1.0, 2.5, 10.0]:
        v = compute_sigma_a_luzzati(s, vrms).item()
        assert abs(v - 1.0) < 1e-12, f"σA(0; ΔVRMS={vrms}) = {v}, want 1.0"


def test_monotone_decrease_in_s():
    """σA strictly decreases with |s| at fixed ΔVRMS."""
    s = torch.linspace(0.0, 0.5, 50, dtype=torch.float64)
    sa = compute_sigma_a_luzzati(s, delta_vrms_A=1.0)
    diffs = sa[1:] - sa[:-1]
    assert (diffs <= 0).all(), "σA(s) must be monotone non-increasing in |s|"
    assert sa[0] > sa[-1], "σA must actually drop, not just be flat"


def test_known_value_at_quarter_inverse_angstrom():
    """Regression: σA(s=0.25, ΔVRMS=1.0) = exp(−π²/8) ≈ 0.291."""
    s = torch.tensor([0.25], dtype=torch.float64)
    got = compute_sigma_a_luzzati(s, delta_vrms_A=1.0).item()
    expected = math.exp(-math.pi ** 2 / 8.0)  # = 0.2910...
    assert abs(got - expected) < 1e-12, f"got {got}, want {expected}"


def test_vector_broadcast():
    """Vector input returns vector output, same shape + dtype + device."""
    s = torch.linspace(0.05, 0.4, 16, dtype=torch.float32)
    sa = compute_sigma_a_luzzati(s, 1.5)
    assert sa.shape == s.shape
    assert sa.dtype == s.dtype


def test_increasing_delta_vrms_steepens_falloff():
    """Larger ΔVRMS ⇒ faster decay at fixed s — sanity for the parameter knob."""
    s = torch.tensor([0.2], dtype=torch.float64)
    a = compute_sigma_a_luzzati(s, 0.5).item()
    b = compute_sigma_a_luzzati(s, 1.0).item()
    c = compute_sigma_a_luzzati(s, 2.0).item()
    assert a > b > c, f"expected a > b > c; got {a}, {b}, {c}"
