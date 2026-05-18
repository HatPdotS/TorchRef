"""
Pure-PyTorch spherical harmonic expansion for the alignment module.

Conventions (locked, asserted by tests/unit/alignment/test_sh.py):

    Y_{l,m}(θ, φ) = (-1)^m · √[(2l+1)/(4π) · (l-m)!/(l+m)!] · P_l^m(cos θ) · e^{imφ}    (m ≥ 0)
    Y_{l,-m}(θ, φ) = (-1)^m · conj(Y_{l,m}(θ, φ))                                       (m > 0)

i.e. fully orthonormal physics convention with Condon-Shortley phase included.
Matches scipy.special.sph_harm and the convention used in Edmonds, Sakurai, etc.

Numerical core is a stable forward recurrence on the fully-normalized associated
Legendre `bar_P_l^m(cosθ) = √[(2l+1)/(4π) · (l-m)!/(l+m)!] · P_l^m(cosθ)` so we
never form (2l)! explicitly.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _bar_legendre_recurrence(
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """
    Compute fully-normalized associated Legendre `bar_P_l^m(cos θ)` for
    all l in [0, L), m in [0, l].

    Definition:
        bar_P_l^m(x) = √[(2l+1)/(4π) · (l-m)!/(l+m)!] · P_l^m(x)
    where P_l^m is the *unsigned* associated Legendre (no Condon-Shortley phase).

    Returns
    -------
    bar_P : torch.Tensor, real
        Shape (..., L, L). `bar_P[..., l, m]` is bar_P_l^m for m <= l, else 0.
    """
    batch_shape = cos_theta.shape
    dtype = cos_theta.dtype
    device = cos_theta.device

    bar_P = torch.zeros((*batch_shape, L, L), dtype=dtype, device=device)

    # Seed: bar_P_0^0 = 1 / sqrt(4π)
    inv_sqrt_4pi = 1.0 / math.sqrt(4.0 * math.pi)
    bar_P[..., 0, 0] = inv_sqrt_4pi

    # Sectoral recurrence on the diagonal m == l:
    #   bar_P_m^m = sqrt((2m+1)/(2m)) · sinθ · bar_P_{m-1}^{m-1}
    for m in range(1, L):
        factor = math.sqrt((2.0 * m + 1.0) / (2.0 * m))
        bar_P[..., m, m] = factor * sin_theta * bar_P[..., m - 1, m - 1]

    # Vertical recurrence (l > m, fixed m):
    #   bar_P_l^m = a_l^m · cosθ · bar_P_{l-1}^m - b_l^m · bar_P_{l-2}^m
    # with a_l^m = sqrt((2l-1)(2l+1)/((l-m)(l+m)))
    #      b_l^m = sqrt((2l+1)(l+m-1)(l-m-1)/((l-m)(l+m)(2l-3)))   [0 when l = m+1]
    for m in range(0, L - 1):
        # l = m+1 step (b term vanishes because l-m-1 = 0)
        l = m + 1
        a = math.sqrt((2.0 * l - 1.0) * (2.0 * l + 1.0) / ((l - m) * (l + m)))
        bar_P[..., l, m] = a * cos_theta * bar_P[..., l - 1, m]
        # l from m+2 to L-1
        for l in range(m + 2, L):
            a = math.sqrt((2.0 * l - 1.0) * (2.0 * l + 1.0) / ((l - m) * (l + m)))
            b = math.sqrt(
                (2.0 * l + 1.0) * (l + m - 1.0) * (l - m - 1.0)
                / ((l - m) * (l + m) * (2.0 * l - 3.0))
            )
            bar_P[..., l, m] = a * cos_theta * bar_P[..., l - 1, m] - b * bar_P[..., l - 2, m]

    return bar_P


def evaluate_ylm(
    theta: torch.Tensor,
    phi: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """
    Evaluate Y_{l,m}(θ, φ) for all (l, m) with l ∈ [0, L), m ∈ [-(L-1), L-1].

    Parameters
    ----------
    theta : torch.Tensor
        Polar angle, shape (...,), values in [0, π].
    phi : torch.Tensor
        Azimuthal angle, shape (...,), values in [0, 2π).
    L : int
        Maximum SH degree (exclusive: l_max = L - 1).

    Returns
    -------
    Y : torch.Tensor, complex
        Shape (..., L, 2L-1).  Y[..., l, L-1+m] = Y_{l,m}(θ, φ) for |m| ≤ l,
        zero otherwise. dtype is complex128 if input is float64, else complex64.
    """
    assert theta.shape == phi.shape, "theta and phi must have the same shape"

    real_dtype = theta.dtype
    if real_dtype == torch.float64:
        complex_dtype = torch.complex128
    elif real_dtype == torch.float32:
        complex_dtype = torch.complex64
    else:
        raise TypeError(f"Unsupported real dtype: {real_dtype}")

    device = theta.device
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta).clamp(min=0.0)  # numerical floor at the poles

    bar_P = _bar_legendre_recurrence(cos_theta, sin_theta, L)  # (..., L, L)

    # Y_{l,m}(θ,φ) = (-1)^m · bar_P_l^m(cosθ) · e^{i m φ}   for m ≥ 0
    # Y_{l,-m}    = (-1)^m · conj(Y_{l,m})                  for m > 0
    Y = torch.zeros((*theta.shape, L, 2 * L - 1), dtype=complex_dtype, device=device)

    # Precompute e^{i m φ} for m = 0..L-1
    # (use stacking to keep things vectorized)
    m_vals = torch.arange(L, dtype=real_dtype, device=device)
    m_phi = phi.unsqueeze(-1) * m_vals  # (..., L)
    expo = torch.complex(torch.cos(m_phi), torch.sin(m_phi))  # (..., L), e^{i m φ}

    # Fill m ≥ 0 columns
    for m in range(L):
        sign = (-1.0) ** m
        # bar_P[..., :, m] is (..., L); only entries l >= m are non-zero (others left at 0)
        Y[..., :, L - 1 + m] = sign * bar_P[..., :, m].to(complex_dtype) * expo[..., m].unsqueeze(-1)

    # Fill m < 0 columns by hermitian symmetry: Y_{l,-m} = (-1)^m · conj(Y_{l,m})
    for m in range(1, L):
        sign = (-1.0) ** m
        Y[..., :, L - 1 - m] = sign * torch.conj(Y[..., :, L - 1 + m])

    return Y


def angular_density_weights(
    s_vectors: torch.Tensor,
    k_neighbors: int = 12,
) -> torch.Tensor:
    """
    Per-sample weights that compensate for non-uniform angular sampling on the
    unit sphere. Returns w_i ∝ (1 / local_density)^... so the weighted sum
    `Σ_i w_i · v_i · Y*_lm(ŝ_i)` is an unbiased Monte-Carlo estimate of the
    SH integral on the sphere.

    Heuristic: w_i ~ (k-th NN great-circle distance)². Normalised so that
    Σ w_i = N.

    Pure-torch O(N · k) memory; suitable up to N ~ 30k. For larger N use a
    chunked KNN, but typical resolution-cut datasets fit easily.

    Parameters
    ----------
    s_vectors : torch.Tensor, shape (N, 3)
    k_neighbors : int, default 12
        Number of nearest angular neighbours to estimate local density.

    Returns
    -------
    w : torch.Tensor, shape (N,)
    """
    device = s_vectors.device
    dtype = s_vectors.dtype
    N = s_vectors.shape[0]
    norm = s_vectors.norm(dim=-1).clamp(min=1e-30)
    s_hat = s_vectors / norm.unsqueeze(-1)              # (N, 3)
    # cos(angle) between every pair via dot product
    # Memory: (N, N). For N ~ 30k, ~3 GB at fp32 — chunk if too big.
    if N <= 8000:
        dots = (s_hat @ s_hat.transpose(0, 1)).clamp(-1.0, 1.0)
        ang = torch.acos(dots)                          # (N, N)
        # k-th NN distance (excluding self at column-diagonal). topk smallest.
        # Set diagonal large so it doesn't show up as nearest.
        ang.fill_diagonal_(float("inf"))
        kth_dist, _ = torch.topk(ang, k_neighbors, dim=-1, largest=False)  # (N, k)
        d_local = kth_dist[:, -1]                       # k-th NN distance
    else:
        # Chunked: still O(N²) compute but bounded memory.
        chunk = 1024
        d_local = torch.empty(N, dtype=dtype, device=device)
        for i0 in range(0, N, chunk):
            i1 = min(i0 + chunk, N)
            dots = (s_hat[i0:i1] @ s_hat.transpose(0, 1)).clamp(-1.0, 1.0)
            ang = torch.acos(dots)
            for j, gi in enumerate(range(i0, i1)):
                ang[j, gi] = float("inf")
            kth_dist, _ = torch.topk(ang, k_neighbors, dim=-1, largest=False)
            d_local[i0:i1] = kth_dist[:, -1]

    w = d_local ** 2                                    # ~ local Voronoi area
    w = w * (N / w.sum().clamp(min=1e-30))              # normalise to Σw = N
    return w


def sh_expand_ball(
    s_vectors: torch.Tensor,
    values: torch.Tensor,
    shell_idx: torch.Tensor,
    P: int,
    L: int,
    enforce_friedel: bool = True,
    chunk_size: int = 2048,
    angular_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Analytical spherical-harmonic expansion of a scattered-point real field
    on a set of radial shells.

        f_{p,l,m} = Σ_{i ∈ shell p} values_i · conj(Y_{l,m}(θ_i, φ_i))

    When `enforce_friedel=True` the input is augmented with the antipodal copy
    `(-s_i, values_i)`. Y_{l,m}(-ŝ) = (-1)^l Y_{l,m}(ŝ), so the sum then has
        f_{p,l,m} = (1 + (-1)^l) · Σ_i v_i · Y*_{l,m}(ŝ_i)
    i.e. odd-l rows are exactly zero by construction (and even-l rows get a
    factor of 2 which we keep — this absorbs into the cross-correlation
    normalisation when the same convention is applied to both operands).

    Parameters
    ----------
    s_vectors : torch.Tensor
        Reciprocal-lattice vectors, shape (N, 3). Direction only is used;
        magnitudes do not enter (shell assignment is done by the caller).
    values : torch.Tensor
        Real-valued samples (e.g. |E(h)|), shape (N,).
    shell_idx : torch.Tensor (int64)
        Shell index in [0, P) for each reflection, shape (N,).
    P : int
        Number of radial shells.
    L : int
        SH bandlimit (l in [0, L)).
    enforce_friedel : bool, default True
        Augment input with (-s, value) pairs and zero odd-l coefficients.
    chunk_size : int
        Points per chunk for memory control during Y_lm evaluation.

    Returns
    -------
    f_plm : torch.Tensor, complex
        Shape (P, L, 2L-1). `f_plm[p, l, L-1+m] = f_{p,l,m}`.
    """
    assert s_vectors.dim() == 2 and s_vectors.shape[-1] == 3
    assert values.dim() == 1 and values.shape[0] == s_vectors.shape[0]
    assert shell_idx.dim() == 1 and shell_idx.shape[0] == s_vectors.shape[0]

    real_dtype = s_vectors.dtype
    if real_dtype == torch.float64:
        complex_dtype = torch.complex128
    elif real_dtype == torch.float32:
        complex_dtype = torch.complex64
    else:
        raise TypeError(f"Unsupported dtype {real_dtype}")
    device = s_vectors.device

    if enforce_friedel:
        s_vectors = torch.cat([s_vectors, -s_vectors], dim=0)
        values = torch.cat([values, values], dim=0)
        shell_idx = torch.cat([shell_idx, shell_idx], dim=0)
        if angular_weights is not None:
            angular_weights = torch.cat([angular_weights, angular_weights], dim=0)

    if angular_weights is not None:
        values = values * angular_weights.to(values.dtype)

    # Direction (θ, φ).  At |s|=0 the direction is undefined; the caller should
    # have excluded F(000), but we guard anyway.
    norm = s_vectors.norm(dim=-1).clamp(min=1e-30)
    s_hat = s_vectors / norm.unsqueeze(-1)
    cos_theta = s_hat[..., 2].clamp(min=-1.0, max=1.0)
    theta = torch.acos(cos_theta)
    phi = torch.atan2(s_hat[..., 1], s_hat[..., 0])

    f_plm = torch.zeros((P, L, 2 * L - 1), dtype=complex_dtype, device=device)

    N = s_vectors.shape[0]
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        Y = evaluate_ylm(theta[start:stop], phi[start:stop], L)  # (n, L, 2L-1)
        # contribution to f_{p,l,m} is value_i * conj(Y_{l,m}(s_i))
        contrib = values[start:stop].to(complex_dtype).view(-1, 1, 1) * torch.conj(Y)
        # scatter-add into shells
        f_plm.index_add_(0, shell_idx[start:stop], contrib)

    if enforce_friedel:
        # Zero odd-l rows explicitly (they should already be ~0; this kills FP drift).
        l_vals = torch.arange(L, device=device)
        odd_mask = (l_vals % 2 == 1)
        f_plm[:, odd_mask, :] = 0.0

    return f_plm


def equal_count_shell_edges(
    s_magnitudes: torch.Tensor,
    P: int,
    s_min: Optional[float] = None,
    s_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute equal-count radial shell edges for a list of |s| magnitudes.

    Each of the P shells receives (approximately) the same number of reflections.
    Reflections outside [s_min, s_max] (if given) are excluded.

    Returns
    -------
    edges : torch.Tensor, real, shape (P+1,)
        Shell boundaries, increasing.
    centers : torch.Tensor, real, shape (P,)
        Mid-points of each shell.
    """
    s = s_magnitudes
    if s_min is not None:
        s = s[s >= s_min]
    if s_max is not None:
        s = s[s <= s_max]
    s_sorted, _ = torch.sort(s)
    N = s_sorted.numel()
    # quantile-based partition
    idx = torch.linspace(0, N - 1, P + 1, dtype=torch.float64, device=s.device).round().long()
    edges = s_sorted[idx]
    # nudge endpoints so the data is fully covered (avoid floating-point miss)
    if s_min is not None:
        edges[0] = min(edges[0].item(), s_min)
    else:
        edges[0] = edges[0] - 1e-6
    if s_max is not None:
        edges[-1] = max(edges[-1].item(), s_max)
    else:
        edges[-1] = edges[-1] + 1e-6
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def fit_overall_anisotropy(
    F_obs: torch.Tensor,
    s_vectors: torch.Tensor,
    shell_idx: torch.Tensor,
    P: int,
    min_count: int = 20,
) -> torch.Tensor:
    """
    Fit the overall anisotropy tensor U from F_obs alone (no model needed).

    The Popov-Bourenkov anisotropy correction models the observed structure-
    factor amplitudes as a per-shell isotropic Wilson piece modulated by an
    overall anisotropic Debye-Waller term:

        |F_obs(h)|² ≈ <|F_iso|²>(s) · exp(−2π²·s·U·s)

    Taking logs and fitting the linear regression
        ln |F_obs|² − ln<|F_iso|²>(s) = −2π² s·U·s
    over all reflections gives the 6-parameter U directly (linear in U). We
    parametrise as U_xx, U_yy, U_zz, U_xy, U_xz, U_yz and ignore reflections
    in shells with fewer than `min_count` entries (poor shell mean estimate).

    The returned U is the correction to *apply* in the form
        F_obs_corrected(h) = F_obs(h) · exp(+π²·s·U·s)
    so that the resulting amplitudes have the same per-shell mean square
    regardless of direction.

    Parameters
    ----------
    F_obs : (N,) real
    s_vectors : (N, 3) real (1/Å)
    shell_idx : (N,) int64 — assigns each reflection to a shell in [0, P)
    P : int, number of shells

    Returns
    -------
    U : (3, 3) symmetric real tensor (Å²)
    """
    device = F_obs.device
    dtype = F_obs.dtype
    valid = shell_idx >= 0
    F = F_obs[valid]
    s = s_vectors[valid].to(dtype)
    idx = shell_idx[valid]
    count = torch.zeros(P, dtype=torch.int64, device=device)
    count.index_add_(0, idx, torch.ones_like(idx))
    F2 = F * F
    sum_F2 = torch.zeros(P, dtype=dtype, device=device)
    sum_F2.index_add_(0, idx, F2)
    mean_F2 = sum_F2 / count.clamp(min=1).to(dtype)
    # Mask shells with too few reflections
    good = count >= min_count
    if good.sum() == 0:
        return torch.zeros((3, 3), dtype=dtype, device=device)
    keep = good[idx]
    F2k = F2[keep].clamp(min=1e-30)
    sk = s[keep]
    mean_F2_k = mean_F2[idx[keep]].clamp(min=1e-30)
    # y = ln|F|² - ln<|F|²> = -2π² · sUs
    y = (torch.log(F2k) - torch.log(mean_F2_k)).to(torch.float64)
    sk = sk.to(torch.float64)
    # Design matrix X for u = (Uxx, Uyy, Uzz, Uxy, Uxz, Uyz):
    # s·U·s = Uxx sx² + Uyy sy² + Uzz sz² + 2 Uxy sx sy + 2 Uxz sx sz + 2 Uyz sy sz
    X = torch.stack([
        sk[:, 0] ** 2, sk[:, 1] ** 2, sk[:, 2] ** 2,
        2.0 * sk[:, 0] * sk[:, 1],
        2.0 * sk[:, 0] * sk[:, 2],
        2.0 * sk[:, 1] * sk[:, 2],
    ], dim=-1)
    A = -2.0 * (torch.pi ** 2) * X                           # y ≈ A · u
    # Least-squares solve A u = y
    u_vec, _, _, _ = torch.linalg.lstsq(A, y.unsqueeze(-1))
    u_vec = u_vec.squeeze(-1)
    Uxx, Uyy, Uzz, Uxy, Uxz, Uyz = u_vec.tolist()
    U = torch.tensor(
        [[Uxx, Uxy, Uxz], [Uxy, Uyy, Uyz], [Uxz, Uyz, Uzz]],
        dtype=dtype, device=device,
    )
    return U


def apply_overall_anisotropy(
    F: torch.Tensor,
    s_vectors: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the inverse of the anisotropy tensor to amplitudes:
        F_corrected(h) = F(h) · exp(+π²·s·U·s)
    (Inverse direction = +π² to undo the Debye-Waller effect.)
    """
    device = F.device
    dtype = F.dtype
    s = s_vectors.to(device).to(dtype)
    U_t = U.to(device).to(dtype)
    s_dot_U = s @ U_t                                       # (N, 3)
    arg = (torch.pi ** 2) * (s_dot_U * s).sum(dim=-1)
    return F * torch.exp(arg.clamp(min=-10.0, max=10.0))


def compute_patterson_shell_variance(
    patt: torch.Tensor,
    shell_idx: torch.Tensor,
    P: int,
    min_count: int = 8,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Empirical per-shell variance of a Patterson coefficient (typically E²−1).

    For each shell p, returns Var_p = <patt²>_p - <patt>_p².

    Shells with fewer than `min_count` reflections inherit the variance of the
    nearest shell that does meet the count threshold (search outward, prefer
    higher-resolution / larger-index shells first). The result is clamped at
    `eps` so the inverse-sqrt weight `1/√Var_p` stays bounded.

    Parameters
    ----------
    patt : torch.Tensor, shape (N,)
        Per-reflection Patterson coefficient (e.g. E² − 1).
    shell_idx : torch.Tensor, shape (N,), int64
        Shell assignment in [0, P). Reflections with index -1 are ignored.
    P : int
        Number of shells.
    min_count : int, default 8
        Shells with fewer reflections than this borrow from a neighbour.
    eps : float, default 1e-3
        Lower bound on returned variance.

    Returns
    -------
    var : torch.Tensor, shape (P,), same dtype as `patt`
    """
    device = patt.device
    dtype = patt.dtype
    valid = shell_idx >= 0
    patt_v = patt[valid]
    idx_v = shell_idx[valid]
    count = torch.zeros(P, dtype=torch.int64, device=device)
    count.index_add_(0, idx_v, torch.ones_like(idx_v))
    sum1 = torch.zeros(P, dtype=dtype, device=device)
    sum2 = torch.zeros(P, dtype=dtype, device=device)
    sum1.index_add_(0, idx_v, patt_v)
    sum2.index_add_(0, idx_v, patt_v * patt_v)
    safe_count = count.clamp(min=1).to(dtype)
    mean = sum1 / safe_count
    var = sum2 / safe_count - mean * mean
    var = var.clamp(min=eps)

    counts_l = count.tolist()
    var_l = var.tolist()
    good = [i for i, c in enumerate(counts_l) if c >= min_count]
    if not good:
        return torch.full_like(var, max(eps, 1.0))
    for i in range(P):
        if counts_l[i] >= min_count:
            continue
        nearest = min(good, key=lambda g, ii=i: (abs(g - ii), -g))
        var_l[i] = var_l[nearest]
    return torch.tensor(var_l, dtype=dtype, device=device).clamp(min=eps)


def assign_shells(
    s_magnitudes: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """
    Assign each reflection to a shell index in [0, P).

    Reflections strictly outside the edges get index -1 (caller may filter).
    """
    # bucketize returns indices in [0, P+1]; we want shell indices in [0, P).
    # Reflections with s == edges[0] go to bucket 0 (use right=True trick).
    idx = torch.bucketize(s_magnitudes.contiguous(), edges.contiguous(), right=True) - 1
    P = edges.shape[0] - 1
    invalid = (idx < 0) | (idx >= P)
    idx = idx.clamp(min=0, max=P - 1)
    idx = torch.where(invalid, torch.full_like(idx, -1), idx)
    return idx
