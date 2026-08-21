"""Tier 1 invariant tests for frf_separate.

Mathematical truths that must hold regardless of input data. Failure
here means a sign / aliasing / normalisation bug in the FFT or the
adaptive sample list.
"""
from __future__ import annotations

import math

import pytest
import torch

from torchref.experimental.alignment.frf.sitelist_ang import (
    adjust_gridding,
    build_adaptive_sample_list,
    build_dense_map_per_beta,
    evaluate_rotation_function,
)
from torchref.experimental.alignment.frf.wigner_d import wigner_contraction_per_beta
from torchref.experimental.alignment.wigner import small_d_packed


def _make_xi(L: int, seed: int = 42) -> torch.Tensor:
    """ξ_{lmn} built as an outer product of two complex SH coefficient sets
    that each derive from a real-valued function on S² (so each has the
    symmetry ``c_{l,-m} = (-1)^m · conj(c_{l, m})``). The resulting ξ has
    the symmetry required for a real-valued RF without us needing to
    derive the index relationship by hand.
    """
    torch.manual_seed(seed)
    # Two random complex SH coefficient vectors with the real-function symmetry.
    def _real_sh_coeffs(L):
        c = torch.zeros((L, 2 * L - 1), dtype=torch.complex128)
        for l in range(2, L, 2):
            # m = 0 entries must be real.
            c[l, L - 1] = torch.randn(1, dtype=torch.float64).item()
            for m in range(1, l + 1):
                v = torch.randn(2, dtype=torch.float64)
                z = complex(v[0].item(), v[1].item())
                c[l, L - 1 + m] = z
                c[l, L - 1 - m] = ((-1) ** m) * complex(z.real, -z.imag)
        return c

    c_obs = _real_sh_coeffs(L)
    c_calc = _real_sh_coeffs(L)
    # ξ_{l, m, n} = c_obs_{l, n} · conj(c_calc_{l, m})
    xi = torch.einsum("ln,lm->lmn", c_obs, torch.conj(c_calc))
    return xi


def test_adjust_gridding_basic():
    assert adjust_gridding(180) == 180         # 180 = 4·45 = 4·9·5 (5-smooth)
    assert adjust_gridding(7) == 8             # rounds up to next 5-smooth
    assert adjust_gridding(243) == 243         # 3^5
    assert adjust_gridding(1) == 1


def test_sample_count_matches_so3_measure():
    """Phaser's adaptive sample count is ≈ (720·360·n_β)/(π·Δ²)."""
    Δ = 5.0
    _, _, _, _, beta_grid = build_adaptive_sample_list(Δ)
    n_beta = beta_grid.shape[0]
    # Manually count by re-building (cheap).
    alphas, _, _, _, _ = build_adaptive_sample_list(Δ)
    n_samples = alphas.shape[0]
    expected = (720.0 * 360.0 * n_beta) / (math.pi * Δ * Δ)
    # 10 % tolerance because of dedup + integer rounding.
    assert 0.85 * expected <= n_samples <= 1.15 * expected, (
        f"got {n_samples}, expected ~{expected:.0f}"
    )


def test_polar_caps_are_one_dimensional():
    """At β=0 only α + γ matters → samples lie on the diagonal α = γ."""
    alphas, betas, gammas, beta_starts, beta_grid = build_adaptive_sample_list(5.0)
    i0, i1 = int(beta_starts[0].item()), int(beta_starts[1].item())
    # β=0 slice: α should equal γ for every sample.
    assert torch.allclose(alphas[i0:i1], gammas[i0:i1], atol=1e-9)


def test_identity_rotation_value():
    """At (α=0, β=0, γ=0): RF = Σ_{l, m} ξ_{l, m, m}.

    The first sample (β=0, p=0) has (α, γ) = (0, 0), so the interpolated
    map value at that sample equals RF(0, 0, 0) / N² (the implicit
    inverse-FFT normalisation).
    """
    L = 8
    xi = _make_xi(L)
    Δ = 10.0
    arf = evaluate_rotation_function(xi, grid_sampling_deg=Δ)

    # Expected RF(0,0,0): Σ_l Σ_m ξ_{l, m, m} since d^l_{m,n}(0) = δ_{m,n}.
    expected = 0.0
    for l in range(2, L, 2):
        for m in range(-l, l + 1):
            expected += xi[l, L - 1 + m, L - 1 + m].real.item()

    # The map at (α=γ=0) corresponds to sample index 0 (β=0, p=0).
    # fft2 (no normalisation) gives RF directly.
    measured = arf.values[0].item()
    assert abs(measured - expected) / max(abs(expected), 1e-9) < 1e-5, (
        f"measured={measured}, expected={expected}"
    )


def test_real_output_from_hermitian_xi():
    """RF must be real (imaginary residue at numerical-noise level only)."""
    L = 8
    xi = _make_xi(L)
    Δ = 10.0
    bmax = int(math.ceil(180.0 / Δ))
    N = adjust_gridding(2 * max(bmax, 2 * L - 1), max_prime=5)
    _, _, _, _, beta_grid = build_adaptive_sample_list(Δ)
    M = build_dense_map_per_beta(xi, beta_grid, N)
    # Imaginary part divided by typical magnitude should be < 1e-10.
    typ = M.real.abs().mean()
    imag_residue = M.imag.abs().mean()
    assert (imag_residue / max(typ.item(), 1e-30)) < 1e-9, (
        f"imag/real = {imag_residue.item()/typ.item():.2e}"
    )


def test_wigner_contraction_symmetry():
    """At β = π/2 the small-d satisfies d^l_{m,n}(π/2) = (-1)^{l+m} d^l_{m,-n}(π/2)."""
    L = 8
    betas = torch.tensor([math.pi / 2], dtype=torch.float64)
    d = small_d_packed(L, betas)[0]   # (L, 2L-1, 2L-1)
    for l in range(2, L, 2):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                lhs = d[l, L - 1 + m, L - 1 + n].item()
                rhs = ((-1) ** (l + m)) * d[l, L - 1 + m, L - 1 - n].item()
                assert abs(lhs - rhs) < 1e-10, (
                    f"l={l} m={m} n={n}: {lhs} vs {rhs}"
                )


def test_beta_reflection_identity():
    """d^l_{m,n}(π − β) = (-1)^{l+m} d^l_{m,-n}(β) (small-d β-reflection)."""
    L = 6
    beta = 0.37
    betas = torch.tensor([beta, math.pi - beta], dtype=torch.float64)
    d = small_d_packed(L, betas)
    for l in range(2, L, 2):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                lhs = d[1, l, L - 1 + m, L - 1 + n].item()                   # d(π-β)
                rhs = ((-1) ** (l + m)) * d[0, l, L - 1 + m, L - 1 - n].item()  # (-1)^(l+m) d(β)|n→-n
                assert abs(lhs - rhs) < 1e-10


def test_zero_xi_gives_zero_rf():
    """Trivial sanity: empty input → empty output."""
    L = 6
    xi = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.complex128)
    arf = evaluate_rotation_function(xi, grid_sampling_deg=10.0)
    assert arf.values.abs().max().item() < 1e-15
