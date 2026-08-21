"""
Pure-PyTorch Wigner small-d and Wigner-D evaluation for the alignment module.

Conventions (locked, asserted by tests/unit/alignment/test_wigner.py):

    D^l_{m,n}(α, β, γ) = e^{-i m α} · d^l_{m,n}(β) · e^{-i n γ}                    (Edmonds)

with the Euler angles paired to the rotation matrix used by
`torchref.experimental.alignment.transform.rotation_matrix_from_euler` — i.e. ZYZ.

Small-d uses the direct sum formula (Edmonds 4.1.23) with log-factorials so the
recurrence never forms `(2l)!` explicitly:

    d^l_{m,n}(β) = Σ_k (-1)^k · √[(l+m)!(l-m)!(l+n)!(l-n)!]
                            / [(l+m-k)! · k! · (l-n-k)! · (k+n-m)!]
                  · cos(β/2)^(2l+m-n-2k) · sin(β/2)^(2k+n-m)

k runs over the integers that keep every factorial non-negative:
    max(0, m-n) ≤ k ≤ min(l+m, l-n).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


def _log_factorial(n: torch.Tensor) -> torch.Tensor:
    """log(n!) for non-negative integer tensor."""
    return torch.lgamma(n.to(torch.float64) + 1.0)


def _build_half_angle_pow_tables(
    beta: torch.Tensor, max_exp: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute `cos(β/2)^j` and `sin(β/2)^j` for j ∈ [0, max_exp].

    Why: `torch.pow(tensor, tensor)` takes the slow `exp(log(x) * y)` path
    even for integer exponents. The Wigner-d sum gathers cos/sin to powers
    in {0, 1, …, 2l} for every (m, n, k) cell; replacing `cos_h ** k` with
    a `cumprod`-built table + fancy index is ~10× faster on CPU for L≥16
    and dominates the `small_d_packed` cost when sharing the table across
    the L iterations.

    Returns tensors of shape `(*beta.shape, max_exp + 1)`, float64.
    """
    beta64 = beta.to(torch.float64)
    half = 0.5 * beta64
    cos_h = torch.cos(half).unsqueeze(-1)  # (*beta, 1)
    sin_h = torch.sin(half).unsqueeze(-1)
    ones = torch.ones_like(cos_h)
    if max_exp < 1:
        return ones, ones
    base_cos = cos_h.expand(*beta.shape, max_exp)
    base_sin = sin_h.expand(*beta.shape, max_exp)
    cos_seq = torch.cat([ones, base_cos], dim=-1)
    sin_seq = torch.cat([ones, base_sin], dim=-1)
    return torch.cumprod(cos_seq, dim=-1), torch.cumprod(sin_seq, dim=-1)


def small_d_block(
    l: int,
    beta: torch.Tensor,
    cos_pow_table: Optional[torch.Tensor] = None,
    sin_pow_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Evaluate d^l_{m,n}(β) for fixed l, all m,n ∈ [-l, l], batched over β.

    Parameters
    ----------
    l : int
        Wigner degree.
    beta : torch.Tensor (real)
        Euler β angle(s), arbitrary shape. Values in [0, π].
    cos_pow_table, sin_pow_table : torch.Tensor, optional
        Precomputed tables with `cos_pow_table[..., j] = cos(β/2)^j` and
        likewise for sin, shape `(*beta.shape, max_exp+1)` with
        `max_exp >= 2*l`. If omitted, built locally. Pass them when looping
        over l with shared β (see `small_d_packed`) to avoid repeated
        `pow`-via-`exp(log·y)` evaluations.

    Returns
    -------
    d : torch.Tensor (real, float64 internally, cast to beta.dtype on return)
        Shape (..., 2l+1, 2l+1). `d[..., m+l, n+l] = d^l_{m,n}(β)`.
    """
    if l == 0:
        out = torch.ones((*beta.shape, 1, 1), dtype=beta.dtype, device=beta.device)
        return out

    device = beta.device
    out_dtype = beta.dtype

    # Build (or reuse) the pow tables. Local build is cheap (~max_exp small
    # ops) so we only skip it when the caller hands us one.
    if cos_pow_table is None or sin_pow_table is None:
        cos_pow_table, sin_pow_table = _build_half_angle_pow_tables(beta, 2 * l)
    max_exp = cos_pow_table.shape[-1] - 1
    assert max_exp >= 2 * l, (
        f"pow table max_exp={max_exp} insufficient for degree l={l}"
    )

    # Precompute log factorials for arguments in [0, 2l].
    n_table = torch.arange(0, 2 * l + 1, device=device)
    log_fac = _log_factorial(n_table)  # (2l+1,) float64

    size = 2 * l + 1
    # Build (m, n) index grids: m_idx = m + l, n_idx = n + l, m,n ∈ [-l, l].
    m_grid = torch.arange(-l, l + 1, dtype=torch.int64, device=device)  # (size,)
    n_grid = m_grid.clone()
    M = m_grid.view(size, 1).expand(size, size)  # (size, size)
    N = n_grid.view(1, size).expand(size, size)

    # k range for each (m, n) pair.
    k_lo = torch.clamp(M - N, min=0)                       # (size, size)
    k_hi = torch.minimum(torch.full_like(M, l) + M, torch.full_like(M, l) - N)
    # Universal k range across all (m,n): k in [0, 2l].
    K = torch.arange(0, 2 * l + 1, dtype=torch.int64, device=device)
    # Build the validity mask for each (m, n, k):
    K_mn = K.view(1, 1, -1)
    mask = (K_mn >= k_lo.unsqueeze(-1)) & (K_mn <= k_hi.unsqueeze(-1))  # (size, size, 2l+1)

    # Coefficient log( (l+m)!(l-m)!(l+n)!(l-n)! / [(l+m-k)! k! (l-n-k)! (k+n-m)!] )^(1/2)
    # Common numerator (depends on m, n only)
    L_t = torch.full_like(M, l)
    log_num = 0.5 * (
        log_fac[L_t + M] + log_fac[L_t - M] + log_fac[L_t + N] + log_fac[L_t - N]
    )  # (size, size)

    # Denominator term per (m, n, k) — guard out-of-range indices with mask.
    # Use clamp into [0, 2l] so the index is always valid; result will be masked off.
    def _safe_lf(idx):
        return log_fac[idx.clamp(min=0, max=2 * l)]

    idx_a = (L_t + M).unsqueeze(-1) - K_mn        # (l+m-k)
    idx_b = K_mn.expand(size, size, -1)            # k
    idx_c = (L_t - N).unsqueeze(-1) - K_mn        # (l-n-k)
    idx_d = K_mn + (N - M).unsqueeze(-1)            # (k+n-m)

    log_den = _safe_lf(idx_a) + _safe_lf(idx_b) + _safe_lf(idx_c) + _safe_lf(idx_d)
    log_coef = log_num.unsqueeze(-1) - log_den    # (size, size, 2l+1)

    coef = torch.exp(log_coef)
    sign = torch.where((K_mn % 2 == 0), torch.ones_like(coef), -torch.ones_like(coef))
    # Zero out invalid k entries
    coef = torch.where(mask, sign * coef, torch.zeros_like(coef))

    # Per-k exponents. In the *valid* (mask=True) region these lie in [0, 2l];
    # invalid entries can fall outside, so we clamp to [0, max_exp] and rely
    # on coef=0 to nuke their contribution.
    exp_cos = 2 * l + M.unsqueeze(-1) - N.unsqueeze(-1) - 2 * K_mn  # (size, size, 2l+1)
    exp_sin = 2 * K_mn + N.unsqueeze(-1) - M.unsqueeze(-1)
    exp_cos_safe = exp_cos.clamp(min=0, max=max_exp)
    exp_sin_safe = exp_sin.clamp(min=0, max=max_exp)

    # Gather cos(β/2)^exp_cos and sin(β/2)^exp_sin from the precomputed
    # tables via index_select. Faster + cleaner dispatch than fancy
    # indexing (`table[..., idx]`) on CPU. Flatten the (size, size, 2l+1)
    # index into 1D, gather along the last axis, then reshape back.
    flat_idx_cos = exp_cos_safe.reshape(-1)
    flat_idx_sin = exp_sin_safe.reshape(-1)
    cos_pow = cos_pow_table.index_select(-1, flat_idx_cos).reshape(
        *cos_pow_table.shape[:-1], *exp_cos_safe.shape
    )
    sin_pow = sin_pow_table.index_select(-1, flat_idx_sin).reshape(
        *sin_pow_table.shape[:-1], *exp_sin_safe.shape
    )

    out64 = (coef * cos_pow * sin_pow).sum(dim=-1)       # (*beta, size, size)

    return out64.to(out_dtype)


def small_d_table(L: int, beta: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Compute d^l_{m,n}(β) for all l ∈ [0, L), batched over β.

    Returned as a list of tensors of shape (..., 2l+1, 2l+1) — variable in
    final two dims because the small-d matrix for degree l has size 2l+1.

    Use `small_d_packed` if you want a single dense (L, 2L-1, 2L-1) tensor with
    zero-padding for the off-diagonal entries beyond |m|, |n| > l.
    """
    return tuple(small_d_block(l, beta) for l in range(L))


def small_d_packed(L: int, beta: torch.Tensor) -> torch.Tensor:
    """
    Compute the small-d matrices for all l ∈ [0, L), packed into a single
    dense tensor of shape (..., L, 2L-1, 2L-1) with zero padding for entries
    where |m| > l or |n| > l.

    `d_packed[..., l, L-1+m, L-1+n] = d^l_{m,n}(β)` if |m|, |n| ≤ l, else 0.

    Internally builds the cos/sin half-angle pow tables once (shared across
    all L iterations) so each `small_d_block` call gathers from a table
    instead of running a tensor-exponent `pow`. A fully vectorised-over-l
    implementation would need a (n_beta, L, 2L-1, 2L-1, 2L-1) intermediate
    that at L=32, n_beta=64 is multi-GB — the shared pow table buys most
    of the speedup while staying memory-bounded.
    """
    if L <= 0:
        raise ValueError(f"L must be >= 1, got {L}")
    out = torch.zeros((*beta.shape, L, 2 * L - 1, 2 * L - 1),
                      dtype=beta.dtype, device=beta.device)
    max_exp = max(2 * (L - 1), 0)
    cos_pow_table, sin_pow_table = _build_half_angle_pow_tables(beta, max_exp)
    for l in range(L):
        d_l = small_d_block(
            l, beta,
            cos_pow_table=cos_pow_table,
            sin_pow_table=sin_pow_table,
        )  # (..., 2l+1, 2l+1)
        out[..., l, L - 1 - l : L - 1 + l + 1, L - 1 - l : L - 1 + l + 1] = d_l
    return out


def wigner_D_pointwise(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """
    Evaluate `D^l_{m,n}(α, β, γ) = e^{-imα} d^l_{m,n}(β) e^{-inγ}` for all
    l, m, n with l < L, |m|, |n| ≤ L-1, batched over the (α, β, γ) triples.

    Returns
    -------
    D : torch.Tensor (complex), shape (..., L, 2L-1, 2L-1)
    """
    assert alpha.shape == beta.shape == gamma.shape

    real_dtype = beta.dtype
    if real_dtype == torch.float64:
        complex_dtype = torch.complex128
    elif real_dtype == torch.float32:
        complex_dtype = torch.complex64
    else:
        raise TypeError(f"Unsupported dtype {real_dtype}")

    device = beta.device
    d = small_d_packed(L, beta)  # (..., L, 2L-1, 2L-1) real
    m_vals = torch.arange(-(L - 1), L, dtype=real_dtype, device=device)
    n_vals = m_vals.clone()

    # phase_m[..., m_idx] = e^{-i m α}, phase_n[..., n_idx] = e^{-i n γ}
    ma = alpha.unsqueeze(-1) * m_vals
    ng = gamma.unsqueeze(-1) * n_vals
    phase_m = torch.complex(torch.cos(-ma), torch.sin(-ma))   # (..., 2L-1)
    phase_n = torch.complex(torch.cos(-ng), torch.sin(-ng))   # (..., 2L-1)

    # D[..., l, m_idx, n_idx] = d[..., l, m_idx, n_idx] · phase_m[..., m_idx] · phase_n[..., n_idx]
    # Broadcast shapes: d is (..., L, 2L-1, 2L-1); need phase_m as (..., 1, 2L-1, 1)
    # and phase_n as (..., 1, 1, 2L-1).
    D = d.to(complex_dtype) * phase_m[..., None, :, None] * phase_n[..., None, None, :]
    return D


def evaluate_rotation_function_grid(
    xi_lmn: torch.Tensor,
    L: int,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    n_gamma: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate `C(α, β, γ) = Σ_l Σ_{m,n} ξ_{l,m,n} D^l_{m,n}(α, β, γ)` on a
    uniform Euler grid, via per-β contraction + 2D IFFT in (α, γ).

    The mathematical identity used:
        C(α, β, γ) = Σ_{m,n} M_{m,n}(β) · e^{-i m α} · e^{-i n γ}
        with  M_{m,n}(β) := Σ_l ξ_{l,m,n} · d^l_{m,n}(β).
    For each β, M_{m,n}(β) is a (2L-1)×(2L-1) matrix; the (α, γ) dependence is
    a 2-D Fourier series, evaluated on a regular (n_α, n_γ) grid via IFFT.

    Parameters
    ----------
    xi_lmn : torch.Tensor (complex)
        Wigner coefficients, shape (L, 2L-1, 2L-1). Layout:
        `xi_lmn[l, L-1+m, L-1+n] = ξ_{l,m,n}` for |m|, |n| ≤ l, else expected zero.
    L : int
        SH / Wigner bandlimit.
    n_alpha, n_beta, n_gamma : int, optional
        Grid sizes in α, β, γ. Defaults: n_alpha = n_gamma = 2L (oversampled
        FFT grid), n_beta = 2L (midpoint quadrature in β).

    Returns
    -------
    C : torch.Tensor (complex)
        Shape (n_gamma, n_beta, n_alpha). Real-valued in exact arithmetic; the
        imaginary part is returned for diagnostics. Layout: C[k_γ, k_β, k_α].
    alpha_grid, beta_grid, gamma_grid : torch.Tensor (real)
        1-D grids in radians.
    """
    if n_alpha is None:
        n_alpha = 2 * L
    if n_gamma is None:
        n_gamma = 2 * L
    if n_beta is None:
        n_beta = 2 * L

    device = xi_lmn.device
    real_dtype = torch.float64 if xi_lmn.dtype == torch.complex128 else torch.float32
    complex_dtype = xi_lmn.dtype

    # Grids
    alpha_grid = (2.0 * torch.pi / n_alpha) * torch.arange(n_alpha, dtype=real_dtype, device=device)
    gamma_grid = (2.0 * torch.pi / n_gamma) * torch.arange(n_gamma, dtype=real_dtype, device=device)
    # β: midpoint rule on (0, π).
    beta_grid = (torch.pi * (torch.arange(n_beta, dtype=real_dtype, device=device) + 0.5)
                 / n_beta)

    # Build M_{m,n}(β_k) = Σ_l ξ_{l,m,n} d^l_{m,n}(β_k) for ALL β_k at once.
    # `small_d_packed` accepts a batched β tensor and returns
    # (n_beta, L, 2L-1, 2L-1); collapsing the previous `for kb in range(n_beta):`
    # loop into a single call eliminates ~64× of Python+torch dispatch
    # overhead (the dominant cost in this stage on CPU).
    d_all = small_d_packed(L, beta_grid)              # (n_beta, L, 2L-1, 2L-1)
    d_all_c = d_all.to(complex_dtype)
    M_all = (xi_lmn.unsqueeze(0) * d_all_c).sum(dim=1)   # (n_beta, 2L-1, 2L-1)

    # M_{m,n}(β) gives Fourier coefficients in (-m·α, -n·γ):
    #   C(α, γ | β) = Σ_{m,n} M_{m,n} e^{-i m α} e^{-i n γ}
    # Build a zero-padded (n_beta, n_alpha, n_gamma) coefficient grid by
    # placing each M_{m,n}(β) entry at index (m mod n_alpha, n mod n_gamma);
    # torch.fft.fft2 then yields C(α_k, γ_j | β) with the correct sign
    # (`fft` uses exp(-2π i k n / N) which matches e^{-i m α}).
    Mhat = torch.zeros(
        (n_beta, n_alpha, n_gamma), dtype=complex_dtype, device=device,
    )
    m_idx = torch.arange(-(L - 1), L, device=device) % n_alpha   # (2L-1,)
    n_idx = torch.arange(-(L - 1), L, device=device) % n_gamma   # (2L-1,)
    # Vectorised scatter: M_all[:, m+L-1, n+L-1] → Mhat[:, m_idx, n_idx].
    Mhat[:, m_idx.unsqueeze(-1), n_idx.unsqueeze(0)] = M_all

    # Batched 2-D FFT over (α, γ).
    slice_C = torch.fft.fft2(Mhat, dim=(-2, -1))      # (n_beta, n_alpha, n_gamma)
    # Re-order to (γ, β, α) layout per our convention C[k_γ, k_β, k_α].
    C = slice_C.permute(2, 0, 1).contiguous()         # (n_gamma, n_beta, n_alpha)

    return C, alpha_grid, beta_grid, gamma_grid


@dataclass
class AdaptiveRotationFunction:
    """
    Phaser-style ragged rotation-function grid.

    On SO(3) the natural area element is `sin(β) dα dβ dγ`. A uniform Euler
    cube oversamples the polar caps; this structure stores per-β slices of
    variable `(qmax_k, pmax_k)` shape with sampling density matching
    `pmax(β) = 720/Δ · cos(β/2)` and `qmax(β) = 360/Δ · sin(β/2)` (Phaser
    FastRot.cc:92-96).

    Attributes
    ----------
    betas : torch.Tensor
        Shape `(n_β,)`, midpoint quadrature on `(0, π)`.
    slices : list[torch.Tensor]
        Length `n_β`. Slice `k` has shape `(qmax_k, pmax_k)` complex, indexed
        as `slices[k][k_γ, k_α]` (matches dense convention `C[k_γ, k_β, k_α]`).
    alpha_grids, gamma_grids : list[torch.Tensor]
        Length `n_β`. Per-slice α and γ grid coordinates in radians.
    grid_sampling_deg : float
        Phaser's `grid_sampling` argument — target angular resolution in degrees.
    """

    betas: torch.Tensor
    slices: List[torch.Tensor]
    alpha_grids: List[torch.Tensor]
    gamma_grids: List[torch.Tensor]
    grid_sampling_deg: float

    def total_samples(self) -> int:
        return sum(s.numel() for s in self.slices)


def evaluate_rotation_function_grid_adaptive(
    xi_lmn: torch.Tensor,
    L: int,
    grid_sampling_deg: float = 3.0,
    n_beta: Optional[int] = None,
) -> AdaptiveRotationFunction:
    """
    Evaluate `C(α, β, γ)` on a Phaser-faithful adaptive Euler grid.

    Per β, the (α, γ) sampling density follows

        pmax(β) = max(1, round(720 / grid_sampling_deg · cos(β/2)))
        qmax(β) = max(1, round(360 / grid_sampling_deg · sin(β/2)))

    Total sample count ≈ `(720 · 360) / grid_sampling_deg²` — independent of L
    and free of the polar duplication that a uniform `(2L)³` grid produces.

    Computes `M_{m,n}(β_k) = Σ_l ξ_{l,m,n} d^l_{m,n}(β_k)` via the existing
    batched `small_d_packed`, then for each β does a zero-padded scatter of
    `M` into a `(pmax_k, qmax_k)` array and runs `torch.fft.fft2` on that
    slice. The scatter is intentional: aliasing past the per-slice Nyquist
    is the physically correct behaviour — those frequencies cannot be
    resolved at that β.
    """
    if n_beta is None:
        n_beta = 2 * L

    device = xi_lmn.device
    if xi_lmn.dtype == torch.complex128:
        real_dtype = torch.float64
    elif xi_lmn.dtype == torch.complex64:
        real_dtype = torch.float32
    else:
        raise TypeError(f"xi_lmn must be complex, got {xi_lmn.dtype}")
    complex_dtype = xi_lmn.dtype

    # β: midpoint rule on (0, π).
    beta_grid = (torch.pi * (torch.arange(n_beta, dtype=real_dtype, device=device) + 0.5)
                 / n_beta)

    # Batched M_{m,n}(β_k) = Σ_l ξ_{l,m,n} d^l_{m,n}(β_k).
    d_all = small_d_packed(L, beta_grid).to(complex_dtype)        # (n_β, L, 2L-1, 2L-1)
    M_all = (xi_lmn.unsqueeze(0) * d_all).sum(dim=1)              # (n_β, 2L-1, 2L-1)

    # Per-β IFFT2 with adaptive shape.
    half_beta = beta_grid * 0.5
    cos_half = torch.cos(half_beta)
    sin_half = torch.sin(half_beta)
    pmax_all = torch.clamp(
        (720.0 / grid_sampling_deg * cos_half).round().to(torch.long), min=1,
    )
    qmax_all = torch.clamp(
        (360.0 / grid_sampling_deg * sin_half).round().to(torch.long), min=1,
    )

    m_vals = torch.arange(-(L - 1), L, dtype=torch.long, device=device)   # (2L-1,)
    n_vals = m_vals.clone()

    slices: List[torch.Tensor] = []
    alpha_grids: List[torch.Tensor] = []
    gamma_grids: List[torch.Tensor] = []

    for k in range(n_beta):
        pmax_k = int(pmax_all[k].item())
        qmax_k = int(qmax_all[k].item())

        # Scatter M_{m,n} into the (pmax_k, qmax_k) Fourier-coefficient grid.
        m_idx = m_vals % pmax_k                                  # (2L-1,)
        n_idx = n_vals % qmax_k                                  # (2L-1,)
        m_grid = m_idx.unsqueeze(-1).expand(2 * L - 1, 2 * L - 1)
        n_grid = n_idx.unsqueeze(0).expand(2 * L - 1, 2 * L - 1)
        flat_idx = m_grid * qmax_k + n_grid                      # (2L-1, 2L-1)
        Mhat = torch.zeros((pmax_k, qmax_k), dtype=complex_dtype, device=device)
        Mhat.view(-1).index_add_(0, flat_idx.reshape(-1), M_all[k].reshape(-1))

        # `torch.fft.fft2` uses exp(-2π i k n / N) so positive (m, n) frequencies
        # at index (m, n) reconstruct e^{-i m α} e^{-i n γ} on the uniform grid.
        C_slice = torch.fft.fft2(Mhat, dim=(-2, -1))              # (pmax_k, qmax_k)

        alpha_grid_k = (2.0 * torch.pi / pmax_k) * torch.arange(
            pmax_k, dtype=real_dtype, device=device,
        )
        gamma_grid_k = (2.0 * torch.pi / qmax_k) * torch.arange(
            qmax_k, dtype=real_dtype, device=device,
        )

        # Transpose to (γ, α) layout to match the dense `C[k_γ, k_β, k_α]` convention.
        slices.append(C_slice.transpose(0, 1).contiguous())
        alpha_grids.append(alpha_grid_k)
        gamma_grids.append(gamma_grid_k)

    return AdaptiveRotationFunction(
        betas=beta_grid,
        slices=slices,
        alpha_grids=alpha_grids,
        gamma_grids=gamma_grids,
        grid_sampling_deg=grid_sampling_deg,
    )


def evaluate_rotation_function_pointwise(
    xi_lmn: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """
    Evaluate the rotation function at arbitrary Euler triples. Slow but exact
    and differentiable — used for sub-voxel peak refinement and convention tests.

    Parameters
    ----------
    xi_lmn : torch.Tensor (complex), shape (L, 2L-1, 2L-1)
    alpha, beta, gamma : torch.Tensor (real), same shape (..., )

    Returns
    -------
    C : torch.Tensor (complex), shape (...,). In exact arithmetic real for real
    input fields, but kept complex so callers can inspect drift.
    """
    D = wigner_D_pointwise(alpha, beta, gamma, L)  # (..., L, 2L-1, 2L-1)
    # xi_lmn has shape (L, 2L-1, 2L-1); broadcasting aligns on the trailing dims.
    C = (xi_lmn * D).sum(dim=(-3, -2, -1))
    return C
