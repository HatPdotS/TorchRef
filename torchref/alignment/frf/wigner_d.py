"""Wigner small-d matrices and Wigner-D pointwise evaluation.

Phaser source: ``phaser/lib/wigner.h`` (the C++ template
``djmn_recursive_table`` used in ``FastRot.cc:41`` per-l, per-β).

Phaser uses the Sakurai recurrence convention; we use the equivalent
Edmonds (4.1.23) direct-sum formula, already validated against Phaser's
output by the convention tests in
``tests/unit/alignment/test_wigner.py``. To avoid duplicating maths,
this module re-exports the existing implementation from
``torchref.alignment.wigner`` (which is the same convention) and adds
Phaser-specific helpers on top.
"""
from __future__ import annotations

from typing import Tuple

import torch

# Re-export the existing validated implementations.
from ..wigner import (
    _build_half_angle_pow_tables,
    small_d_block,
    small_d_packed,
    small_d_table,
    wigner_D_pointwise,
)

__all__ = [
    "small_d_block",
    "small_d_packed",
    "small_d_stable",
    "small_d_table",
    "wigner_D_pointwise",
    "wigner_contraction_per_beta",
]


def small_d_stable(L: int, betas: torch.Tensor) -> torch.Tensor:
    """Numerically stable Wigner small-d table via J_y diagonalization.

    ``d^l(β) = exp(-iβ J_y^{(l)})``. J_y is a tiny real tridiagonal generator,
    so ``eigh(i·J_y)`` gives integer eigenvalues μ ∈ [-l, l] and a basis V with
        ``d^l_{m n}(β) = Σ_μ V_{m μ} e^{-iβ μ} V*_{n μ}``  (real).
    This is exactly the π/2 / SOFT Fourier-over-μ decomposition (V are the π/2
    matrices up to phase), and it has NO catastrophic cancellation — unlike the
    Edmonds direct-sum ``small_d_packed`` which explodes to |d|~1e11 at l≥50.

    Returns ``(n_beta, L, 2L-1, 2L-1)`` with ``d[k, l, m+L-1, n+L-1] = d^l_{m,n}(β_k)``,
    matching ``small_d_packed`` exactly (same convention, verified at l≤40).
    """
    betas = betas.to(torch.float64)
    n_beta = betas.shape[0]
    dim = 2 * L - 1
    device = betas.device
    out = torch.zeros((n_beta, L, dim, dim), dtype=torch.float64, device=device)
    out[:, 0, L - 1, L - 1] = 1.0  # l=0
    for l in range(1, L):
        sz = 2 * l + 1
        p = torch.arange(sz - 1, dtype=torch.float64, device=device)
        sup = 0.5 * torch.sqrt((2 * l - p) * (p + 1.0))          # J_y off-diagonal magnitudes
        A = torch.diag(sup, 1) - torch.diag(sup, -1)            # A = -i J_y, real antisymmetric
        H = 1j * A.to(torch.complex128)                          # Hermitian
        w, V = torch.linalg.eigh(H)                              # w≈[-l..l], V complex
        # d_l(β) = Re( V · diag(e^{-iβ w}) · V^H ), batched over β
        phase = torch.exp(-1j * betas.unsqueeze(1) * w.unsqueeze(0))   # (n_beta, sz)
        d_l = torch.einsum("ma,ka,na->kmn", V, phase, V.conj()).real   # (n_beta, sz, sz)
        lo, hi = L - 1 - l, L - 1 + l + 1
        out[:, l, lo:hi, lo:hi] = d_l
    return out


def wigner_contraction_per_beta(
    xi_lmn: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """Compute ``S_{m1, m2}(β) = Σ_l ξ_{l, m1, m2} · d^l_{m1, m2}(β)``.

    Phaser source: ``SiteListAng::DoRfftStuff`` (FastRot.cc:39-59) — the
    inner ``for (l_index, m1_index, m2_index)`` triple-loop. We do it
    in one tensor contraction instead of a Python loop over l.

    Parameters
    ----------
    xi_lmn : torch.Tensor (complex), shape (L, 2L-1, 2L-1)
        SH-Bessel coefficients with l ∈ [0, L), |m|, |n| ≤ L-1,
        zero-padded outside |m| > l or |n| > l. (Already n-summed over
        the Bessel radial index by the caller.)
    betas : torch.Tensor (real), shape (n_beta,)
        β values to evaluate at, in radians.

    Returns
    -------
    S : torch.Tensor (complex), shape (n_beta, 2L-1, 2L-1)
        ``S[k, m1+L-1, m2+L-1] = Σ_l ξ_{l, m1, m2} · d^l_{m1, m2}(β_k)``.
    """
    if xi_lmn.ndim != 3:
        raise ValueError(f"xi_lmn must be 3-D, got shape {tuple(xi_lmn.shape)}")
    L = xi_lmn.shape[0]
    dim = 2 * L - 1
    device = xi_lmn.device
    betas = betas.to(torch.float64)
    n_beta = betas.shape[0]
    xi = xi_lmn.to(torch.complex128)

    # Fused per-l loop: compute each d^l(β) block via J_y eigendecomposition
    # (small_d_stable's method, stable to any l) and contract it into S
    # immediately. Never materialises the full (n_beta, L, 2L-1, 2L-1) table
    # (~19 GB at L=100) nor a 4-D einsum intermediate — peak memory is one
    # (n_beta, 2l+1, 2l+1) block (~0.4 GB at l=99).
    S = torch.zeros((n_beta, dim, dim), dtype=torch.complex128, device=device)
    c = L - 1
    S[:, c, c] += xi[0, c, c]                                  # l=0: d^0 = 1
    for l in range(1, L):
        sz = 2 * l + 1
        p = torch.arange(sz - 1, dtype=torch.float64, device=device)
        sup = 0.5 * torch.sqrt((2 * l - p) * (p + 1.0))
        A = torch.diag(sup, 1) - torch.diag(sup, -1)           # A = -i J_y
        w, V = torch.linalg.eigh(1j * A.to(torch.complex128))  # w∈[-l..l]
        phase = torch.exp(-1j * betas.unsqueeze(1) * w.unsqueeze(0))   # (n_beta, sz)
        VP = V.unsqueeze(0) * phase.unsqueeze(1)               # (n_beta, sz, sz) = (k,m,a)
        d_l = (VP @ V.conj().transpose(-1, -2)).real           # (n_beta, sz, sz)
        lo, hi = c - l, c + l + 1
        S[:, lo:hi, lo:hi] += xi[l, lo:hi, lo:hi].unsqueeze(0) * d_l.to(torch.complex128)
    return S
