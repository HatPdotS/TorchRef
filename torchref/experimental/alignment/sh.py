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
import os
import time
from typing import Optional, Tuple

import torch

_PROFILE = bool(os.environ.get("FRF_PROFILE"))
_YLM_PROF = {"recurrence": 0.0, "assembly": 0.0}


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

    # Precompute the recurrence coefficients as (L, L) tables, indexed [l, m]:
    #   a_l^m = sqrt((2l-1)(2l+1)/((l-m)(l+m)))
    #   b_l^m = sqrt((2l+1)(l+m-1)(l-m-1)/((l-m)(l+m)(2l-3)))   [0 when l = m+1]
    # b vanishes at l = m+1 (factor l-m-1 = 0), so no special-casing is needed.
    # Computed in float64 then cast to `dtype` (matches the original, which used
    # float64 python scalars multiplied into the working-dtype tensors).
    ll = torch.arange(L, dtype=torch.float64, device=device).view(L, 1)
    mm = torch.arange(L, dtype=torch.float64, device=device).view(1, L)
    valid = (ll > mm)  # l > m (vertical recurrence region)
    denom = (ll - mm) * (ll + mm)
    denom_safe = torch.where(valid, denom, torch.ones_like(denom))
    a_coef = torch.sqrt((2.0 * ll - 1.0) * (2.0 * ll + 1.0) / denom_safe)
    b_num = (2.0 * ll + 1.0) * (ll + mm - 1.0) * (ll - mm - 1.0)
    b_den = denom_safe * (2.0 * ll - 3.0)
    b_coef = torch.sqrt(torch.clamp(b_num / torch.where(b_den == 0, torch.ones_like(b_den), b_den), min=0.0))
    a_coef = torch.where(valid, a_coef, torch.zeros_like(a_coef)).to(dtype)
    b_coef = torch.where(valid, b_coef, torch.zeros_like(b_coef)).to(dtype)
    # Sectoral diagonal factor sqrt((2m+1)/(2m)) for m = l.
    m_arange = torch.arange(L, dtype=torch.float64, device=device)
    sect = torch.sqrt((2.0 * m_arange + 1.0) / (2.0 * m_arange).clamp(min=1.0)).to(dtype)

    cos_e = cos_theta.unsqueeze(-1)  # (..., 1)
    # Single loop over l; at each l update all m ∈ [0, l] at once. The vertical
    # recurrence (m < l) and the sectoral diagonal (m = l) both read only level
    # l-1 (and l-2), already computed — so this is the same recurrence as the
    # original double loop, just reordered to vectorise over m.
    for l in range(1, L):
        prev1 = bar_P[..., l - 1, :l]                       # (..., l)
        if l >= 2:
            prev2 = bar_P[..., l - 2, :l]                   # (..., l)
        else:
            prev2 = torch.zeros_like(prev1)
        bar_P[..., l, :l] = (
            a_coef[l, :l] * cos_e * prev1 - b_coef[l, :l] * prev2
        )
        # Sectoral m == l.
        bar_P[..., l, l] = sect[l] * sin_theta * bar_P[..., l - 1, l - 1]

    return bar_P


def evaluate_ylm(
    theta: torch.Tensor,
    phi: torch.Tensor,
    L: int,
    l_indices: Optional[torch.Tensor] = None,
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
    l_indices : torch.Tensor, optional
        If given (1-D long tensor of l values), assemble and return Y only for
        those degrees, shape ``(..., len(l_indices), 2L-1)`` with row ``i``
        holding ``Y_{l_indices[i], m}``. The Legendre recurrence still runs over
        the full degree range (it is a recurrence), but the costly complex Y
        assembly is restricted to the requested rows. Used by the FRF expansion,
        which only needs even degrees (the odd-l and l=0 rows are zeroed by
        Patterson centrosymmetry) — halving the dominant assembly cost.

    Returns
    -------
    Y : torch.Tensor, complex
        Shape (..., L, 2L-1) (or (..., len(l_indices), 2L-1) if l_indices given).
        ``Y[..., l, L-1+m] = Y_{l,m}(θ, φ)`` for |m| ≤ l, zero otherwise. dtype is
        complex128 if input is float64, else complex64.
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

    if _PROFILE:
        t0 = time.perf_counter()
    bar_P = _bar_legendre_recurrence(cos_theta, sin_theta, L)  # (..., L, L)
    if l_indices is not None:
        bar_P = bar_P[..., l_indices, :]               # (..., n_sel, L) — even rows only
    n_rows = bar_P.shape[-2]
    if _PROFILE:
        _YLM_PROF["recurrence"] += time.perf_counter() - t0
        t0 = time.perf_counter()

    # Y_{l,m}(θ,φ) = (-1)^m · bar_P_l^m(cosθ) · e^{i m φ}   for m ≥ 0
    # Y_{l,-m}    = (-1)^m · conj(Y_{l,m})                  for m > 0
    Y = torch.zeros((*theta.shape, n_rows, 2 * L - 1), dtype=complex_dtype, device=device)

    # Precompute e^{i m φ} for m = 0..L-1 and the Condon-Shortley signs (-1)^m.
    m_vals = torch.arange(L, dtype=real_dtype, device=device)
    m_phi = phi.unsqueeze(-1) * m_vals  # (..., L)
    expo = torch.complex(torch.cos(m_phi), torch.sin(m_phi))  # (..., L), e^{i m φ}
    signs = ((-1.0) ** m_vals).to(complex_dtype)  # (L,)

    # Fill m ≥ 0 columns (L-1 .. 2L-2): Y_{l,m} = (-1)^m bar_P_l^m e^{imφ},
    # vectorised over (l, m). phase = (-1)^m e^{imφ} broadcasts over l.
    phase_pos = (signs * expo).unsqueeze(-2)            # (..., 1, L)
    Y_pos = bar_P.to(complex_dtype) * phase_pos         # (..., n_rows, L)
    Y[..., :, L - 1:] = Y_pos

    # Fill m < 0 columns by hermitian symmetry: Y_{l,-m} = (-1)^m conj(Y_{l,m}).
    # For m = 1..L-1 these land in columns L-2 .. 0, i.e. the reversed prefix.
    neg = signs[1:] * torch.conj(Y_pos[..., :, 1:])     # (..., L, L-1), m = 1..L-1
    Y[..., :, : L - 1] = torch.flip(neg, dims=(-1,))

    if _PROFILE:
        _YLM_PROF["assembly"] += time.perf_counter() - t0
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


def get_axis_order(sym_mats: torch.Tensor, axis: int) -> int:
    """
    Order of the highest-multiplicity proper rotation about a principal axis.

    `sym_mats` is the spacegroup rotation matrices (n_ops, 3, 3). `axis` is
    0/1/2 for x/y/z. Returns the largest n such that some R in `sym_mats` is a
    rotation by 2π/n around that axis. For non-rotational operations (or
    rotations not aligned with the axis), the spacegroup element is skipped.
    Returns 1 if no proper rotation around the axis exists.

    Used by the Phaser-style m-symmetry filter on the spherical-harmonic
    coefficients: the Patterson is invariant under the spacegroup rotations,
    so m-values that violate the highest-order axis symmetry are pure noise.
    """
    a = torch.zeros(3, dtype=torch.float64, device=sym_mats.device)
    a[axis] = 1.0
    max_order = 1
    n_ops = sym_mats.shape[0]
    for k in range(n_ops):
        R = sym_mats[k].to(torch.float64)
        # Axis must be invariant under R (proper or improper rotation about it).
        if (R @ a - a).norm().item() > 1e-3:
            continue
        # Trace of a rotation by angle θ about the preserved axis is 1+2cosθ.
        tr = R.diagonal().sum().item()
        cos_a = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        # Identity (angle ~0) → order 1.
        if cos_a >= 1.0 - 1e-6:
            continue
        angle = math.acos(cos_a)
        n = round(2 * math.pi / angle)
        if n > max_order:
            max_order = n
    return max_order


def get_high_order_axis(sym_mats: torch.Tensor) -> Tuple[int, int]:
    """
    Return (axis, zsymm) where `axis` ∈ {0, 1, 2} (x/y/z) maximises
    `get_axis_order`, with z preferred on ties (matches Phaser's
    `highOrderAxis()` in SpaceGroup.cc).
    """
    orders = [get_axis_order(sym_mats, a) for a in (0, 1, 2)]
    # Phaser: axis=3 (z); axis=2 if orderY > orderZ; axis=1 if orderX > both.
    if orders[0] > orders[1] and orders[0] > orders[2]:
        axis = 0
    elif orders[1] > orders[2]:
        axis = 1
    else:
        axis = 2
    return axis, orders[axis]


def sh_expand_ball(
    s_vectors: torch.Tensor,
    values: torch.Tensor,
    shell_idx: torch.Tensor,
    P: int,
    L: int,
    enforce_friedel: bool = True,
    chunk_size: int = 2048,
    angular_weights: Optional[torch.Tensor] = None,
    zsymm: int = 1,
    skip_odd_l: bool = False,
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

    if enforce_friedel or skip_odd_l:
        # Zero odd-l rows explicitly (they should already be ~0; this kills FP drift
        # and is the only required step when skip_odd_l is set without Friedel).
        l_vals = torch.arange(L, device=device)
        odd_mask = (l_vals % 2 == 1)
        f_plm[:, odd_mask, :] = 0.0

    # F1: m-symmetry filter (Phaser DataMR.cc:1019 / 1117). The Patterson is
    # invariant under the spacegroup rotation operators, so SH coefficients
    # whose m-index violates the highest-order rotation axis are pure noise.
    # Zero them out post-expansion. With `zsymm=1` (no filter) this is a no-op.
    if zsymm > 1:
        m_vals = torch.arange(-(L - 1), L, device=device)            # (2L-1,)
        m_invalid = (m_vals.abs() % zsymm) != 0
        f_plm[:, :, m_invalid] = 0.0

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
    centric: torch.Tensor,
    P: int,
    min_count: int = 20,
    n_iter: int = 12,
) -> torch.Tensor:
    """
    Fit the overall anisotropy tensor U from F_obs alone (no model needed).

    The Popov-Bourenkov correction models the observed intensities as a
    per-shell isotropic Wilson piece modulated by an overall anisotropic
    Debye-Waller term::

        E[ I(h) / <I>_shell ] = c * exp(-2 pi^2 s.U.s)

    That expectation is exact in **intensity** space, which is where this fits
    it: a free constant ``c`` absorbs the overall scale, the weights come from
    ``Var(I/<I>)`` -- 1 for acentric reflections and 2 for centric ones -- and
    non-positive or non-finite amplitudes are dropped. Gauss-Newton from
    ``U = 0``.

    Fitting the same relation in log space instead is what the earlier version
    did, and it is biased: ``E[ln(I/<I>)]`` is ``-gamma = -0.577`` for acentric
    and ``-gamma - ln 2`` for centric reflections, not zero. Without a constant
    term that offset can only be absorbed by the quadratic form, so U comes back
    with a large spurious component -- and because centric reflections lie on
    the zones perpendicular to the symmetry axes, the bias is
    direction-dependent rather than a harmless overall scale.

    The returned U is the correction to *apply* in the form::

        F_obs_corrected(h) = F_obs(h) * exp(+pi^2 s.U.s)

    so the corrected amplitudes have the same mean square in every direction.
    Project it onto the point group with :func:`symmetrize_anisotropy` before
    applying it: an unconstrained six-component fit can return a tensor the
    lattice forbids.

    Parameters
    ----------
    F_obs : (N,) real
    s_vectors : (N, 3) real, reciprocal-space Cartesian (1/Angstrom)
    shell_idx : (N,) int64 -- shell of each reflection, in [0, P); negative
        entries are excluded
    centric : (N,) bool
    P : int, number of shells
    min_count : int, optional
        Shells with fewer reflections than this are dropped, since their mean
        intensity is too noisy to normalise against.
    n_iter : int, optional
        Gauss-Newton iterations.

    Returns
    -------
    U : (3, 3) symmetric real tensor (Angstrom squared). Zero if too few
        reflections survive to constrain seven parameters.
    """
    valid = shell_idx >= 0
    F = F_obs[valid].to(torch.float64)
    s = s_vectors[valid].to(torch.float64)
    idx = shell_idx[valid]
    cen = centric[valid].bool()

    ok = torch.isfinite(F) & (F > 0)
    F, s, idx, cen = F[ok], s[ok], idx[ok], cen[ok]

    I = F * F
    count = torch.zeros(P, dtype=torch.int64, device=F.device)
    total = torch.zeros(P, dtype=torch.float64, device=F.device)
    count.index_add_(0, idx, torch.ones_like(idx))
    total.index_add_(0, idx, I)
    mean_I = (total / count.clamp(min=1).to(torch.float64)).clamp(min=1e-30)

    keep = (count >= min_count)[idx]
    if int(keep.sum()) < 50:
        return torch.zeros((3, 3), dtype=F_obs.dtype, device=F_obs.device)
    ratio = I[keep] / mean_I[idx[keep]]
    sk, cenk = s[keep], cen[keep]

    x, y, z = sk[:, 0], sk[:, 1], sk[:, 2]
    # s.U.s = Uxx sx^2 + Uyy sy^2 + Uzz sz^2
    #         + 2 Uxy sx sy + 2 Uxz sx sz + 2 Uyz sy sz
    quad = torch.stack([x * x, y * y, z * z,
                        2 * x * y, 2 * x * z, 2 * y * z], dim=1)
    # Column 0 is the free constant ln(c); the rest carry -2 pi^2 s.U.s.
    A = torch.cat([torch.ones_like(x).unsqueeze(1),
                   -2.0 * (torch.pi ** 2) * quad], dim=1)
    w = torch.where(cenk, torch.full_like(ratio, 0.5), torch.ones_like(ratio))

    theta = torch.zeros(7, dtype=torch.float64, device=F.device)
    for _ in range(n_iter):
        model = torch.exp((A @ theta).clamp(min=-20.0, max=20.0))
        J = model.unsqueeze(1) * A
        Jw = J * w.unsqueeze(1)
        H = J.transpose(0, 1) @ Jw
        grad = Jw.transpose(0, 1) @ (ratio - model)
        H = H + torch.eye(7, dtype=H.dtype, device=H.device) * 1e-12 * float(
            torch.diagonal(H).abs().max().clamp(min=1e-30))
        theta = theta + torch.linalg.solve(H, grad)

    u = theta[1:]
    return torch.tensor(
        [[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]],
        dtype=F_obs.dtype, device=F_obs.device,
    )


def hkl_symops_to_cartesian(
    sg_mats: torch.Tensor,
    rec_basis: torch.Tensor,
) -> torch.Tensor:
    """
    Convert spacegroup symmetry operators that act on integer Miller indices
    into the equivalent rotation operators acting on Cartesian reciprocal-
    space vectors (`s = h @ rec_basis`, column-vector form: `s = M @ h` with
    `M = rec_basis^T`).

    For column vectors: `s' = M · S · M⁻¹ · s` so `P_cart = M · S · M⁻¹`.

    For orthogonal cells (orthorhombic+) M is diagonal and `P_cart == S`
    exactly. For non-orthogonal cells (monoclinic, hex/trig with γ=120°,
    triclinic), the Cartesian form differs and matters for any operation
    that mixes the axes (e.g. averaging tensors over the point group).

    Parameters
    ----------
    sg_mats : torch.Tensor, shape (n_ops, 3, 3)
        Integer Miller-index symops (data.spacegroup.matrices).
    rec_basis : torch.Tensor, shape (3, 3)
        Reciprocal basis matrix such that `s = h @ rec_basis` (row-vector
        convention used throughout torchref).

    Returns
    -------
    sym_mats_cart : torch.Tensor, shape (n_ops, 3, 3), real
    """
    dtype = torch.float64
    M = rec_basis.to(dtype).transpose(-1, -2)            # (3, 3)
    M_inv = torch.linalg.inv(M)
    S = sg_mats.to(dtype)                                 # (n_ops, 3, 3)
    # S^T, not S: reciprocal space transforms as h' = h.S, so the operator
    # acting on Cartesian s as a column vector is (B^-1 S B)^T = M S^T M^-1
    # with M = B^T. Using S here returns matrices that are not rotations at all
    # in a non-orthogonal basis -- measured orthogonality error 5.33 for
    # P 3_1 2 1 and P 6_5 2 2, versus 2e-7 with the transpose. The two agree
    # whenever the symmetry matrices are orthogonal, i.e. everywhere except
    # trigonal/hexagonal, which is why this survived.
    return torch.einsum("ij,klj,lm->kim", M, S, M_inv)


def symmetrize_anisotropy(
    U: torch.Tensor,
    sym_mats_cart: torch.Tensor,
) -> torch.Tensor:
    """
    Project a symmetric 3×3 tensor `U` onto the point-group-invariant
    subspace of the spacegroup by averaging over its Cartesian rotation
    operators:

        U_sym = (1/n) Σ_k  P_k · U · P_k^T

    This mirrors Phaser's `site_symmetry.average_u_star()` and the
    `RefineANO.cc:116-142` constraint construction. After symmetrisation the
    tensor automatically satisfies the crystal's point-group symmetry:

        - cubic:        U_sym = (trace U / 3) · I              (1 DOF)
        - tetragonal:   U_sym = diag(λ, λ, μ)                  (2 DOF)
        - orthorhombic: U_sym = diag(λ, μ, ν)                  (3 DOF)
        - monoclinic:   diagonal + one off-diagonal             (4 DOF)
        - triclinic:    unchanged                               (6 DOF)

    This is the structural fix that prevents an unconstrained 6-component
    regression from producing physically impossible anisotropy in
    high-symmetry cells (e.g. fitting a 70 Å² eigenvalue on a cubic dataset
    where every eigenvalue must be equal by symmetry).

    Parameters
    ----------
    U : torch.Tensor, shape (3, 3), symmetric
    sym_mats_cart : torch.Tensor, shape (n_ops, 3, 3)
        Output of `hkl_symops_to_cartesian` — Cartesian-space rotation
        operators of the spacegroup.

    Returns
    -------
    U_sym : torch.Tensor, shape (3, 3), symmetric, same dtype/device as U
    """
    dtype = U.dtype
    device = U.device
    sym = sym_mats_cart.to(device=device, dtype=dtype)
    # Vectorised: U_avg = mean_k  P_k · U · P_k^T
    U_avg = torch.einsum("kij,jl,knl->in", sym, U, sym) / sym.shape[0]
    # Enforce symmetry (numerical hygiene; pure rotation averaging preserves
    # symmetry exactly, but FP drift can leave 1e-15 asymmetry).
    return 0.5 * (U_avg + U_avg.transpose(-1, -2))


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
