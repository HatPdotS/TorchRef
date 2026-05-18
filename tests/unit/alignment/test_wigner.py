"""
Unit tests for torchref.alignment.wigner.

Conventions verified:
- D^l_{m,n}(α,β,γ) = e^{-imα} d^l_{m,n}(β) e^{-inγ}                (Edmonds)
- d^l_{m,n}(0) = δ_{m,n};   d^l_{m,n}(π) = (-1)^{l+m} δ_{m,-n}.
- Unitarity:   D^l D^l† = I  for any (α,β,γ).
- Composition (sanity):   the inverse FFT path agrees with the pointwise path.
"""
import math

import numpy as np
import pytest
import torch

from torchref.alignment.wigner import (
    small_d_block,
    small_d_packed,
    wigner_D_pointwise,
    evaluate_rotation_function_grid,
    evaluate_rotation_function_pointwise,
)


@pytest.mark.parametrize("l", [0, 1, 2, 3, 5, 8])
def test_small_d_identity_at_zero(l):
    """d^l_{m,n}(0) = δ_{m,n}."""
    beta = torch.tensor([0.0], dtype=torch.float64)
    d = small_d_block(l, beta)[0]  # (2l+1, 2l+1)
    expected = torch.eye(2 * l + 1, dtype=torch.float64)
    np.testing.assert_allclose(d.numpy(), expected.numpy(), atol=1e-12)


@pytest.mark.parametrize("l", [0, 1, 2, 3, 5, 8])
def test_small_d_at_pi(l):
    """d^l_{m,n}(π) = (-1)^{l+m} δ_{m,-n}."""
    beta = torch.tensor([math.pi], dtype=torch.float64)
    d = small_d_block(l, beta)[0]  # (2l+1, 2l+1)
    size = 2 * l + 1
    expected = torch.zeros(size, size, dtype=torch.float64)
    for m_idx in range(size):
        m = m_idx - l
        expected[m_idx, -m_idx - 1 + size] = (-1.0) ** (l + m)  # n = -m → index size-1 - m_idx
    np.testing.assert_allclose(d.numpy(), expected.numpy(), atol=1e-10)


@pytest.mark.parametrize("l", [1, 2, 4, 6])
def test_small_d_unitary(l):
    """d^l(β) is real-orthogonal: d^T d = I."""
    beta = torch.tensor([0.3, 1.1, 2.5], dtype=torch.float64)
    d = small_d_block(l, beta)  # (3, 2l+1, 2l+1)
    for k in range(3):
        dk = d[k]
        prod = dk.T @ dk
        np.testing.assert_allclose(prod.numpy(), np.eye(2 * l + 1), atol=1e-10,
                                   err_msg=f"l={l} β={beta[k]:.3f}: d^T d ≠ I")


@pytest.mark.parametrize("L", [3, 5])
def test_wigner_D_unitary(L):
    """D^l(R) is unitary for any (α,β,γ)."""
    torch.manual_seed(0)
    n = 4
    alpha = torch.rand(n, dtype=torch.float64) * 2 * math.pi
    beta = torch.rand(n, dtype=torch.float64) * math.pi
    gamma = torch.rand(n, dtype=torch.float64) * 2 * math.pi
    D = wigner_D_pointwise(alpha, beta, gamma, L)  # (n, L, 2L-1, 2L-1)

    for k in range(n):
        for l in range(L):
            sl = slice(L - 1 - l, L - 1 + l + 1)
            Dl = D[k, l, sl, sl]
            prod = Dl @ Dl.conj().transpose(-1, -2)
            eye = torch.eye(2 * l + 1, dtype=Dl.dtype)
            np.testing.assert_allclose(prod.numpy(), eye.numpy(), atol=1e-10,
                                       err_msg=f"l={l} not unitary at k={k}")


def test_wigner_D_diagonal_for_pure_z_rotation():
    """For β=γ=0, D^l_{m,n}(α,0,0) = δ_{m,n} e^{-imα}."""
    L = 4
    alpha = torch.tensor([0.5], dtype=torch.float64)
    zero = torch.zeros_like(alpha)
    D = wigner_D_pointwise(alpha, zero, zero, L)[0]  # (L, 2L-1, 2L-1)
    for l in range(L):
        sl = slice(L - 1 - l, L - 1 + l + 1)
        Dl = D[l, sl, sl].numpy()
        # diagonal entries
        for idx in range(2 * l + 1):
            m = idx - l
            expected = np.exp(-1j * m * 0.5)
            np.testing.assert_allclose(Dl[idx, idx], expected, atol=1e-12)
        # off-diagonal must vanish
        offdiag = Dl - np.diag(np.diag(Dl))
        assert np.abs(offdiag).max() < 1e-12


def test_pointwise_matches_grid():
    """evaluate_rotation_function_pointwise and *_grid agree at grid points."""
    L = 4
    torch.manual_seed(1)
    # Random xi coefficients (only valid (l, |m|<=l, |n|<=l) entries non-zero).
    xi = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.complex128)
    for l in range(L):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                xi[l, L - 1 + m, L - 1 + n] = (torch.randn(1).item()
                                               + 1j * torch.randn(1).item())

    C_grid, alphas, betas, gammas = evaluate_rotation_function_grid(
        xi, L, n_alpha=2 * L, n_beta=2 * L, n_gamma=2 * L
    )
    # Pick a few grid points and verify pointwise matches.
    rng = np.random.default_rng(0)
    for _ in range(5):
        ka = int(rng.integers(0, 2 * L))
        kb = int(rng.integers(0, 2 * L))
        kg = int(rng.integers(0, 2 * L))
        C_from_grid = C_grid[kg, kb, ka]
        a = alphas[ka:ka + 1]
        b = betas[kb:kb + 1]
        g = gammas[kg:kg + 1]
        C_pointwise = evaluate_rotation_function_pointwise(xi, a, b, g, L)[0]
        np.testing.assert_allclose(C_from_grid.item(), C_pointwise.item(),
                                   atol=1e-10, rtol=1e-8)


def test_pointwise_real_for_hermitian_xi():
    """If ξ_{l,m,n} satisfies the conjugacy relation expected of a real cross-correlation,
    then C(R) is real-valued."""
    L = 3
    torch.manual_seed(2)
    # Build xi from a real-field convention:
    #   ξ_{l,m,n} = conj(f_{l,m}) g_{l,n}, where f and g are SH coefficients of real fields
    #   so f_{l,-m} = (-1)^m conj(f_{l,m}).
    def random_real_field_coeffs(L):
        f = torch.zeros((L, 2 * L - 1), dtype=torch.complex128)
        for l in range(L):
            for m in range(0, l + 1):
                r = torch.randn(1).item() + 1j * torch.randn(1).item()
                if m == 0:
                    r = complex(r.real, 0.0)
                f[l, L - 1 + m] = r
                if m > 0:
                    f[l, L - 1 - m] = ((-1) ** m) * np.conj(r)
        return f

    f = random_real_field_coeffs(L)
    g = random_real_field_coeffs(L)
    xi = torch.zeros((L, 2 * L - 1, 2 * L - 1), dtype=torch.complex128)
    for l in range(L):
        for mi in range(2 * L - 1):
            for ni in range(2 * L - 1):
                xi[l, mi, ni] = f[l, mi].conj() * g[l, ni]

    C_grid, _, _, _ = evaluate_rotation_function_grid(xi, L)
    imag = C_grid.imag.abs().max().item()
    real = C_grid.real.abs().max().item()
    assert imag < 1e-10 * max(real, 1.0), f"C imag={imag} too large (real max {real})"
