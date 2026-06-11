"""
Unit tests for torchref.experimental.alignment.sh: spherical harmonic primitives.

Conventions verified:
- Y_{l,m} are fully orthonormal physics SH with Condon-Shortley phase.
- Matches scipy.special.sph_harm.
- For a centrosymmetric (Friedel-symmetric) input, sh_expand_ball produces
  exactly-zero odd-l coefficients when enforce_friedel=True.
"""
import math

import numpy as np
import pytest
import torch

scipy_special = pytest.importorskip("scipy.special")

from torchref.experimental.alignment.sh import (
    _bar_legendre_recurrence,
    evaluate_ylm,
    sh_expand_ball,
    equal_count_shell_edges,
    assign_shells,
)


def _scipy_ylm(l, m, theta, phi):
    """Reference: scipy spherical harmonics with C-S phase, physics convention.

    scipy >= 1.15 replaced ``sph_harm(m, l, phi, theta)`` with
    ``sph_harm_y(n, m, theta, phi)``; prefer the new API and fall back to the
    old one for older scipy installs.
    """
    if hasattr(scipy_special, "sph_harm_y"):
        return scipy_special.sph_harm_y(l, m, theta, phi)
    return scipy_special.sph_harm(m, l, phi, theta)


@pytest.mark.parametrize("L", [4, 8, 16])
def test_ylm_matches_scipy(L):
    """Our Y_lm should match scipy.special.sph_harm at random points."""
    torch.manual_seed(0)
    n = 20
    theta = torch.rand(n, dtype=torch.float64) * math.pi
    phi = (torch.rand(n, dtype=torch.float64) - 0.5) * 2 * math.pi

    Y = evaluate_ylm(theta, phi, L)  # (n, L, 2L-1)

    for l in range(L):
        for m in range(-l, l + 1):
            ref = _scipy_ylm(l, m, theta.numpy(), phi.numpy())
            got = Y[:, l, L - 1 + m].numpy()
            np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-10,
                                       err_msg=f"mismatch at l={l}, m={m}")


def test_ylm_orthonormality_on_grid():
    """Y_lm should be ~orthonormal when integrated on a fine spherical grid."""
    L = 6
    # Gauss-Legendre in cos(theta), uniform in phi
    n_theta = 2 * L + 4
    n_phi = 4 * L + 4
    # GL nodes in [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    theta_np = np.arccos(x_gl)
    phi_np = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    weights = (np.repeat(w_gl, n_phi)) * (2 * math.pi / n_phi)  # (n_theta*n_phi,)
    theta = torch.tensor(np.repeat(theta_np, n_phi), dtype=torch.float64)
    phi = torch.tensor(np.tile(phi_np, n_theta), dtype=torch.float64)

    Y = evaluate_ylm(theta, phi, L)  # (n_pts, L, 2L-1)
    Yflat = Y.reshape(-1, L * (2 * L - 1))
    w = torch.tensor(weights, dtype=torch.float64)

    # G[a,b] = sum_pts w * Y*_a * Y_b
    G = torch.einsum("p,pa,pb->ab", w.to(Yflat.dtype), Yflat.conj(), Yflat)
    G_np = G.numpy()

    # Only diagonals for valid (l,m) entries (m | <= l) should be 1; off-diagonal ~0.
    expected = np.zeros_like(G_np, dtype=np.complex128)
    for l in range(L):
        for m in range(-l, l + 1):
            idx = l * (2 * L - 1) + (L - 1 + m)
            expected[idx, idx] = 1.0
    # Zero out the entries we don't care about (l < |m|, where Y is zero anyway)
    valid_mask = np.zeros(L * (2 * L - 1), dtype=bool)
    for l in range(L):
        for m in range(-l, l + 1):
            valid_mask[l * (2 * L - 1) + (L - 1 + m)] = True

    G_valid = G_np[np.ix_(valid_mask, valid_mask)]
    exp_valid = expected[np.ix_(valid_mask, valid_mask)]
    np.testing.assert_allclose(G_valid, exp_valid, atol=1e-9, rtol=1e-9)


def test_bar_legendre_pole_values():
    """At the north pole (cos θ = 1), bar_P_l^m = 0 for m > 0 and bar_P_l^0 ≠ 0."""
    L = 5
    theta = torch.tensor([0.0], dtype=torch.float64)
    bar_P = _bar_legendre_recurrence(torch.cos(theta), torch.sin(theta), L)  # (1, L, L)
    # m > 0 must be zero (sin θ = 0 kills sectorals; vertical recurrence then ~ 0)
    for l in range(L):
        for m in range(1, l + 1):
            assert abs(bar_P[0, l, m].item()) < 1e-14, f"l={l}, m={m}: {bar_P[0,l,m].item()}"
    # m = 0: bar_P_l^0(1) = sqrt((2l+1)/(4π))   (Legendre polynomial at 1 is 1)
    for l in range(L):
        expected = math.sqrt((2 * l + 1) / (4 * math.pi))
        np.testing.assert_allclose(bar_P[0, l, 0].item(), expected, atol=1e-12)


def test_friedel_enforces_even_l():
    """sh_expand_ball with enforce_friedel=True must give exactly-zero odd-l coefficients."""
    torch.manual_seed(1)
    L = 8
    P = 4
    n_pts = 1000
    s_vectors = torch.randn(n_pts, 3, dtype=torch.float64) * 0.5
    # |s| roughly in [0, 1]; make sure non-zero
    s_vectors = s_vectors / (1 + s_vectors.norm(dim=-1, keepdim=True) * 0.1)
    s_mags = s_vectors.norm(dim=-1)
    edges, _ = equal_count_shell_edges(s_mags, P)
    shell_idx = assign_shells(s_mags, edges)
    values = torch.rand(n_pts, dtype=torch.float64)

    f_plm = sh_expand_ball(s_vectors, values, shell_idx, P, L, enforce_friedel=True)

    # Odd-l rows must be exactly zero (after explicit zero of FP drift).
    for l in range(1, L, 2):
        assert f_plm[:, l, :].abs().max().item() == 0.0, f"l={l} not zero"


def test_friedel_without_enforce_has_odd_l_for_nonsymmetric_input():
    """Without Friedel enforcement, a non-centrosymmetric scatter produces nonzero odd-l."""
    torch.manual_seed(2)
    L = 6
    P = 1
    # Place all mass at the north pole — extremely non-centrosymmetric
    s_vectors = torch.tensor([[0.0, 0.0, 1.0]] * 5, dtype=torch.float64)
    values = torch.ones(5, dtype=torch.float64)
    shell_idx = torch.zeros(5, dtype=torch.int64)

    f_no_friedel = sh_expand_ball(s_vectors, values, shell_idx, P, L,
                                  enforce_friedel=False)
    # odd-l (l=1) entries should be non-trivial
    odd_norm = f_no_friedel[:, 1, :].abs().max().item()
    assert odd_norm > 1e-3, "expected nonzero odd-l without Friedel enforcement"


def test_shell_assignment_round_trip():
    """assign_shells gives indices that round-trip through equal_count_shell_edges."""
    torch.manual_seed(3)
    s_mags = torch.rand(2000, dtype=torch.float64) * 2.0
    P = 16
    edges, centers = equal_count_shell_edges(s_mags, P)
    idx = assign_shells(s_mags, edges)
    # all should be in [0, P-1]
    assert idx.min().item() >= 0
    assert idx.max().item() == P - 1
    # roughly equal counts (within 2x because of tie-breaking at edges)
    counts = torch.bincount(idx, minlength=P)
    assert counts.min().item() >= s_mags.numel() / (4 * P)


def test_sh_expand_zero_when_l_too_large_for_no_points():
    """Empty shell should give all-zero coefficients."""
    L = 4
    P = 3
    n_pts = 100
    torch.manual_seed(4)
    s_vectors = torch.randn(n_pts, 3, dtype=torch.float64)
    values = torch.ones(n_pts, dtype=torch.float64)
    # Force shell_idx == 1 for all points (shell 0 and 2 empty)
    shell_idx = torch.ones(n_pts, dtype=torch.int64)
    f = sh_expand_ball(s_vectors, values, shell_idx, P, L, enforce_friedel=False)
    assert f[0].abs().max() == 0
    assert f[2].abs().max() == 0
    assert f[1].abs().max() > 0
