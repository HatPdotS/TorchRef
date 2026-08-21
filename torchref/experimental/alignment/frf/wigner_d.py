"""Wigner small-d matrices and Wigner-D pointwise evaluation.

Phaser source: ``phaser/lib/wigner.h`` (the C++ template
``djmn_recursive_table`` used in ``FastRot.cc:41`` per-l, per-β).

Phaser uses the Sakurai recurrence convention; the equivalent Edmonds
(4.1.23) convention is used throughout this package and is pinned against
Phaser's output by ``tests/unit/alignment/test_wigner.py``.
``wigner_contraction_per_beta`` builds the small-d table it needs from the
``J_y`` eigendecomposition inline, which stays bounded to any ``l``.
"""
from __future__ import annotations

import torch

__all__ = ["wigner_contraction_per_beta"]

#: Memo for the per-l J_y eigendecomposition, keyed on (L, device-str).
#: It depends only on the bandwidth, so repeat calls at the same L reuse it.
_WIGNER_EIG_CACHE: dict = {}


def _wigner_eig_table(L: int, device: torch.device):
    """Return [(w_l, V_l)] for l ∈ [1, L) — the J_y eigendecomposition per l.

    ``d^l(β) = Re(V_l · diag(e^{-iβ w_l}) · V_l^H)``. ``w_l ≈ [-l..l]`` and
    ``V_l`` are independent of β and the data, so they are memoised.
    """
    key = (int(L), str(device))
    cached = _WIGNER_EIG_CACHE.get(key)
    if cached is not None:
        return cached
    table = []
    for l in range(1, L):
        sz = 2 * l + 1
        p = torch.arange(sz - 1, dtype=torch.float64, device=device)
        sup = 0.5 * torch.sqrt((2 * l - p) * (p + 1.0))
        A = torch.diag(sup, 1) - torch.diag(sup, -1)           # A = -i J_y
        w, V = torch.linalg.eigh(1j * A.to(torch.complex128))  # w∈[-l..l]
        table.append((w, V))
    _WIGNER_EIG_CACHE[key] = table
    return table


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
    eig_table = _wigner_eig_table(L, device)                   # cached (w_l, V_l)
    for l in range(1, L):
        w, V = eig_table[l - 1]                                # data-independent
        phase = torch.exp(-1j * betas.unsqueeze(1) * w.unsqueeze(0))   # (n_beta, sz)
        VP = V.unsqueeze(0) * phase.unsqueeze(1)               # (n_beta, sz, sz) = (k,m,a)
        d_l = (VP @ V.conj().transpose(-1, -2)).real           # (n_beta, sz, sz)
        lo, hi = c - l, c + l + 1
        S[:, lo:hi, lo:hi] += xi[l, lo:hi, lo:hi].unsqueeze(0) * d_l.to(torch.complex128)
    return S
