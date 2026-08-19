"""Unit tests for the three Phaser model-prep helpers added to
`torchref.experimental.alignment.frf.preprocessing`:

- ``bulk_solvent_factor`` (Babinet, Phaser solTerm.h:9)
- ``oeffner_vrms`` (Oeffner empirical, rms_estimate.cc:37)
- ``fit_relative_wilson_b`` (Wilson-B regression, EnsemblePDB.cc:793-851)
"""
from __future__ import annotations

import math

import pytest
import torch

from torchref.experimental.alignment.frf.preprocessing import (
    bulk_solvent_factor,
    fit_relative_wilson_b,
    oeffner_vrms,
)


# -----------------------------------------------------------------------------
# bulk_solvent_factor
# -----------------------------------------------------------------------------


def test_bulk_solvent_factor_low_res_suppressed():
    """At s→0 the Babinet term → 1 − fsol ≈ 0.05 (default fsol=0.95)."""
    s = torch.tensor([1e-6, 1e-5])  # essentially zero
    val = bulk_solvent_factor(s, fsol=0.95, bsol=300.0)
    assert torch.allclose(val, torch.tensor([0.05, 0.05]), atol=1e-3)


def test_bulk_solvent_factor_high_res_unaffected():
    """At large s the exponential vanishes → term → 1."""
    s = torch.tensor([1.0, 2.0])  # very high res
    val = bulk_solvent_factor(s, fsol=0.95, bsol=300.0)
    assert torch.allclose(val, torch.ones_like(s), atol=1e-6)


def test_bulk_solvent_factor_sigA_min_clamp():
    """Clamp to sigA_min when Babinet would go below."""
    s = torch.tensor([1e-10])
    val = bulk_solvent_factor(s, fsol=0.999, bsol=300.0, sigA_min=0.05)
    # 1 - 0.999·1 = 0.001 < sigA_min=0.05 → clamped to 0.05
    assert val.item() == pytest.approx(0.05, abs=1e-6)


def test_bulk_solvent_factor_monotonic_in_s():
    """Strictly non-decreasing in |s|."""
    s = torch.linspace(0.001, 1.0, 100)
    val = bulk_solvent_factor(s)
    diffs = val[1:] - val[:-1]
    assert (diffs >= -1e-10).all()


# -----------------------------------------------------------------------------
# oeffner_vrms
# -----------------------------------------------------------------------------


def test_oeffner_vrms_small_perfect_model():
    """N_res = 125 (lower clamp), identity = 1 (perfect):
    vrms = 0.0569 · (173 + 125)^(1/3) · 1 ≈ 0.379 Å.
    """
    val = oeffner_vrms(125, identity=1.0)
    expected = 0.0569 * (173 + 125) ** (1.0 / 3.0)
    assert val == pytest.approx(expected, abs=1e-6)
    assert 0.35 < val < 0.45


def test_oeffner_vrms_large_perfect_model():
    """N_res = 1500 (upper clamp): vrms = 0.0569 · 1673^(1/3) ≈ 0.675 Å."""
    val = oeffner_vrms(1500, identity=1.0)
    expected = 0.0569 * (173 + 1500) ** (1.0 / 3.0)
    assert val == pytest.approx(expected, abs=1e-6)
    assert 0.6 < val < 0.7


def test_oeffner_vrms_clamps_inputs():
    """N_res < 125 clamps to 125; N_res > 1500 clamps to 1500."""
    low = oeffner_vrms(50, identity=1.0)
    expected_125 = oeffner_vrms(125, identity=1.0)
    assert low == pytest.approx(expected_125)
    high = oeffner_vrms(5000, identity=1.0)
    expected_1500 = oeffner_vrms(1500, identity=1.0)
    assert high == pytest.approx(expected_1500)


def test_oeffner_vrms_identity_dependence():
    """Lower identity → larger vrms (exp(C·(1-ident)) grows)."""
    v_perfect = oeffner_vrms(500, identity=1.0)
    v_30pct = oeffner_vrms(500, identity=0.3)
    assert v_30pct > v_perfect
    # exp(1.52 · 0.7) ≈ 2.9× larger
    assert v_30pct / v_perfect == pytest.approx(math.exp(1.52 * 0.7), rel=1e-6)


# -----------------------------------------------------------------------------
# fit_relative_wilson_b
# -----------------------------------------------------------------------------


def test_fit_relative_wilson_b_recovers_synthetic_B():
    """Synthetic: F_calc = F_obs · exp(-B_true · s² / 4) should recover B_true.

    log(<F_obs²>/<F_calc²>) = log(exp(B·s²/2)) = B/2 · s². Slope = B/2, and
    `WilsonB = -2·slope` per Phaser EnsemblePDB.cc:850 means recovery is
    NEGATIVE: the fit returns -B_true (because we put the B on F_calc, the
    regression sees Σ_N/Σ_P = exp(+B·s²/2) → positive slope → returns -B).

    To test recovery of a POSITIVE B (model more disordered than data), we
    multiply F_obs by exp(-B·s²/4) instead → Σ_N/Σ_P = exp(-B·s²/2)
    → negative slope → returns positive B.
    """
    torch.manual_seed(0)
    N = 5000
    s = torch.linspace(0.05, 0.5, N, dtype=torch.float64)  # 2 - 20 Å range
    F_obs_base = (1.0 + 0.05 * torch.randn(N, dtype=torch.float64)).abs() + 0.1
    F_calc = F_obs_base.clone()
    # Apply B = +10 to F_obs (obs more disordered): Σ_N/Σ_P = exp(-10·s²/2)
    B_true = 10.0
    F_obs = F_obs_base * torch.exp(-B_true * s * s / 4.0)
    val = fit_relative_wilson_b(F_obs, F_calc, s, n_shells=20)
    assert val == pytest.approx(B_true, abs=1.5)


def test_fit_relative_wilson_b_zero_when_matched():
    """F_obs ≈ F_calc → B ≈ 0."""
    torch.manual_seed(0)
    N = 5000
    s = torch.linspace(0.05, 0.5, N, dtype=torch.float64)
    F = (1.0 + 0.05 * torch.randn(N, dtype=torch.float64)).abs() + 0.1
    val = fit_relative_wilson_b(F, F.clone(), s, n_shells=20)
    assert abs(val) < 1.0  # Should be near zero modulo discretisation


def test_fit_relative_wilson_b_clamp():
    """Clamp guard: extreme synthetic case clamps to ±clamp_b."""
    N = 5000
    s = torch.linspace(0.05, 0.5, N, dtype=torch.float64)
    F_obs = torch.ones(N, dtype=torch.float64) * 1e-10  # vanishing obs
    F_calc = torch.ones(N, dtype=torch.float64)
    val = fit_relative_wilson_b(F_obs, F_calc, s, n_shells=20, clamp_b=50.0)
    assert -50.0 <= val <= 50.0


def test_fit_relative_wilson_b_handles_sparse_data():
    """Too few shells contribute (all at low res) → returns 0 (no fit)."""
    s = torch.linspace(0.001, 0.005, 20, dtype=torch.float64)  # all > 200 Å
    F = torch.ones(20, dtype=torch.float64)
    val = fit_relative_wilson_b(F, F.clone(), s, n_shells=10)
    assert val == 0.0
