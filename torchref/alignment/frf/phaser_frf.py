"""
Phaser-faithful Fast Rotation Function (FRF) — a clean, parallel implementation.

Translates Phaser's `DataMR.cc` (observed-side SH expansion) + `Ensemble.cc`
(model-side σA Eterm) + `FastRot.cc` (Wigner-D contraction + 2-D FFT) into
PyTorch. Designed to be benchmarked directly against torchref's existing
`ball_rotation_search` in
`tests/integration/alignment/benchmark_phaser_frf.py`.

The single fundamental difference from `ball_rotation_search` is the **radial
basis**: this module expands the per-reflection Patterson onto **spherical
Bessel functions** rather than shell-step indicators. The Bessel basis is the
natural orthonormal radial basis on the unit ball (Crowther 1972, Navaza FAST);
shell-step is its coarsest piecewise-constant approximation. Phaser's
`DataMR.cc:1102–1134` accumulates

    clmn[l, m, n]  = Σ_h  Y*_{l,m}(ŝ_h) · I(h) · √u · j_u(2π·a·|s_h|) / (2π·a·|s_h|)

with u = l + 2n − 1 and n ∈ [1, nmax(l)] where nmax(l) = (lmax − l)/2 + 1.

Other Phaser-faithful choices we make:
- Even l only (Patterson is centrosymmetric); odd-l rows zeroed.
- m-symmetry filter on observed side only (`DataMR.cc:863-870, 1117`):
  obs SH coefficients with `m % ZSYMM != 0` are dropped; calc side keeps all m.
- σA Eterm on calc side (`Ensemble.cc:36-46`):
  `Eterm(s) = exp(-2π² · s² · ΔVRMS²)`. Per-reflection (not per-shell).
- LERF1-style observed intensity: `I_obs(h) = cweight · (E²−1)` with
  `cweight = 1` for centric, `2` for acentric.
- Euler convention: this module computes in **Edmonds ZYZ**
  `R = R_z(α) R_y(β) R_z(γ)` — same as the rest of torchref. Phaser's
  internal convention is `R = R_z(γ) R_y(β) R_z(α)` (swapped α↔γ) but only
  the rotation matrix matters for our rank-of-truth metric, so we never
  swap explicitly — the orbit_rank metric in the benchmark builds rotation
  matrices via `rotation_matrix_from_edmonds_euler` and the answer is
  parameterization-independent.

What we deliberately DO NOT yet implement (out of scope per plan):
- French-Wilson per-reflection DFAC. Phaser uses `(E²-V)/V² · DFAC²` with
  per-reflection Luzzati DFAC ∈ [0.05, 10]. Here we use DFAC = 1.
- Adaptive (α, γ) grid sampling (Phaser's `pmax = 720·cos(β/2)/grid_sampling`).
  We use uniform 2L grid.
- Axis permutation: Phaser detects the high-order axis and may permute the
  coordinate frame. We use the z-preferred selection from
  `get_high_order_axis`, applied without permutation. For groups where the
  high-order axis is already z (most cases including P432 with body-diagonal
  3-folds along z), this is identical to Phaser.

References (paths under
`/das/work/p17/p17490/Peter/Library/torchref/reverse_engineering/phenix/phenix-1.20-4459/modules/phaser/codebase/phaser/src/`):
- `DataMR.cc:863-1148`        observed-side SH + intensity + ZSYMM filter
- `Ensemble.cc:36-46`         model-side Eterm
- `FastRot.cc:30-217`         Wigner contraction + FFT + Z-score
- `runMR_FRF.cc:546-587`      peak picking + Z-score normalisation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .ball_search import (
    RotationPeak,
    find_rotation_peaks_adaptive,
    refine_peaks_subvoxel_adaptive,
)
from ..sh import (
    assign_shells,
    equal_count_shell_edges,
    evaluate_ylm,
    get_high_order_axis,
)
from ..wigner import (
    AdaptiveRotationFunction,
    evaluate_rotation_function_grid_adaptive,
)


# =============================================================================
# Spherical Bessel j_u(x) via Miller's downward recurrence
# =============================================================================


def spherical_bessel_table(
    x: torch.Tensor,
    u_max: int,
    n_extra: int = 25,
) -> torch.Tensor:
    """
    Tabulate spherical Bessel `j_u(x)` for u ∈ [0, u_max], batched over x.

    Uses Miller's downward recurrence (the standard stable choice for
    `j_n(x)` with `n > x`):

        j_{u-1}(x) = (2u + 1) / x · j_u(x)  −  j_{u+1}(x)

    Seed: choose `n_start = u_max + n_extra`, set
    `j_{n_start+1} = 0`, `j_{n_start} = 1` (unnormalised), recur down to
    j_0, then renormalise using the exact `j_0(x) = sin(x) / x`. Float64
    internally for accuracy at moderate `u/x` ratios; cast back to input
    dtype.

    Returns
    -------
    j_table : torch.Tensor
        Shape `(*x.shape, u_max + 1)`, dtype = `x.dtype`.
    """
    real_dtype = x.dtype
    device = x.device
    x64 = x.to(torch.float64)
    safe_x = x64.clamp(min=1e-30)
    inv_x = 1.0 / safe_x

    n_start = max(u_max + n_extra, u_max + 2)
    j_high = torch.zeros_like(x64)            # j_{n_start + 1}
    j_mid = torch.ones_like(x64)              # j_{n_start} (arbitrary scale)
    j_table = torch.zeros(
        (u_max + 1, *x64.shape), dtype=torch.float64, device=device,
    )

    # Recur from n = n_start down to 1, generating j_{n-1} at each step.
    for n in range(n_start, 0, -1):
        j_low = (2.0 * n + 1.0) * inv_x * j_mid - j_high
        if n - 1 <= u_max:
            j_table[n - 1] = j_low
        j_high = j_mid
        j_mid = j_low

    # Normalise against the exact j_0(x) = sin(x)/x. Handle x ≈ 0 (where
    # j_0(0) = 1, j_u(0) = 0 for u ≥ 1) so the renormalisation factor is
    # well-defined.
    true_j0 = torch.sin(x64) * inv_x
    true_j0 = torch.where(x64 < 1e-30, torch.ones_like(x64), true_j0)
    computed_j0 = j_table[0]
    safe_j0 = torch.where(
        computed_j0.abs() < 1e-30, torch.ones_like(computed_j0), computed_j0,
    )
    scale = true_j0 / safe_j0
    j_table = j_table * scale.unsqueeze(0)

    # Re-arrange `(u_max+1, *x.shape) → (*x.shape, u_max+1)`.
    perm = list(range(1, j_table.dim())) + [0]
    j_table = j_table.permute(*perm).contiguous()
    return j_table.to(real_dtype)


# =============================================================================
# Phaser-style Bessel-radial SH expansion
# =============================================================================


@dataclass
class BesselSHCoefficients:
    """Phaser-style clmn coefficients.

    `c_nlm[n, l, m+L-1]` is the (n-th radial Bessel) × (l, m) coefficient.
    """
    c_nlm: torch.Tensor          # complex, shape (N_radial, L, 2L-1)
    L: int                       # bandwidth (l ∈ [0, L), even-l only filled)
    N_radial: int                # number of Bessel radial terms = (lmax-2)/2 + 1
    bessel_h_scale: float        # `h = bessel_h_scale · |s|` (Phaser uses lmax · d_min)
    zsymm: int                   # m-symmetry filter applied (1 = none)


def bessel_sh_expand(
    s_vectors: torch.Tensor,
    intensity: torch.Tensor,
    *,
    L: int,
    bessel_h_scale: float,
    zsymm: int = 1,
    enforce_friedel: bool = True,
    chunk_size: int = 2048,
) -> BesselSHCoefficients:
    """
    Compute the Phaser-style spherical-Bessel radial × spherical-harmonic
    angular expansion of a scattered-point intensity field on the unit ball.

        c_nlm[n, l, m] = Σ_h  Y*_{l,m}(ŝ_h) · I_h · √u · j_u(h_h) / h_h

    where `h_h = bessel_h_scale · |s_h|` and u = l + 2n + 1 (n ∈ [0, N_radial)).
    Phaser (`DataMR.cc:1107`) uses `h = lmax · |s| · HIRES` where HIRES is the
    high-resolution limit `d_min` in Å. With that scaling, h_max ≈ lmax (so the
    highest-u Bessel functions are sampled near their first peak rather than
    deep in their decaying tail). Pass `bessel_h_scale = (L - 1) * d_min` to
    match Phaser exactly.

    For each l, only n with `2n + l + 1 ≤ lmax + 1` (i.e. `n ≤ (lmax − l)/2`)
    are filled — others stay 0 (Phaser's truncation:
    `nmax(l) = (lmax - l + 2)/2`).

    Even l only (the Patterson is centrosymmetric: Y_{l,m}(−ŝ) = (−1)^l Y_{l,m},
    so odd-l rows cancel exactly when Friedel mates are summed).

    When `zsymm > 1`, SH coefficients with `|m| % zsymm ≠ 0` are zeroed
    post-expansion (Phaser `DataMR.cc:863-870, 1117`). This is the
    m-symmetry filter; applied **only** on the observed side by the caller,
    NOT on the calc side (which is a rotated P1 ensemble and not spacegroup-
    invariant).

    Parameters
    ----------
    s_vectors : (N, 3) real
        Reciprocal-lattice vectors in 1/Å.
    intensity : (N,) real
        Per-reflection intensity (LERF1 obs intensity or σA-weighted calc).
    L : int
        Wigner/SH bandwidth. lmax = L - 1, restricted to even.
    bessel_h_scale : float
        The pre-multiplier on |s| inside the Bessel argument: `h = bessel_h_scale · |s|`.
        Phaser uses `lmax · d_min` (where d_min is HIRES in Å). This puts h_max ≈ lmax
        so the highest-u terms are sampled near their first peak.
    zsymm : int, default 1
        m-symmetry filter (zero coefficients with |m| not divisible by zsymm).
        zsymm=1 is no filter.
    enforce_friedel : bool, default True
        Augment the sum with (-s, I) pairs to make odd-l rows exactly zero
        (otherwise we rely on the explicit zero-out at the end).
    chunk_size : int
        Reflections per Y_lm chunk for memory control.

    Returns
    -------
    BesselSHCoefficients with `c_nlm[n, l, m+L-1]` populated.
    """
    assert s_vectors.dim() == 2 and s_vectors.shape[-1] == 3
    assert intensity.dim() == 1 and intensity.shape[0] == s_vectors.shape[0]

    real_dtype = s_vectors.dtype
    if real_dtype == torch.float64:
        complex_dtype = torch.complex128
    elif real_dtype == torch.float32:
        complex_dtype = torch.complex64
    else:
        raise TypeError(f"Unsupported real dtype: {real_dtype}")
    device = s_vectors.device

    if enforce_friedel:
        s_vectors = torch.cat([s_vectors, -s_vectors], dim=0)
        intensity = torch.cat([intensity, intensity], dim=0)

    s_mag = s_vectors.norm(dim=-1).clamp(min=1e-30)
    s_hat = s_vectors / s_mag.unsqueeze(-1)
    cos_theta = s_hat[..., 2].clamp(min=-1.0, max=1.0)
    theta = torch.acos(cos_theta)
    phi = torch.atan2(s_hat[..., 1], s_hat[..., 0])

    lmax = L - 1
    lmax_even = lmax if (lmax % 2 == 0) else (lmax - 1)
    # Phaser indexing: n ∈ [1, nmax(l)] with nmax(l) = (lmax - l + 2)/2.
    # Our 0-indexed n: n ∈ [0, N_radial - 1] with N_radial = nmax(l=2).
    if lmax_even < 2:
        raise ValueError(
            f"L={L} too small; need lmax_even >= 2 (so L >= 3)."
        )
    N_radial = (lmax_even - 2) // 2 + 1
    # Highest Bessel order needed: for l = 2 and n = N_radial − 1 (1-indexed
    # n_max), u = l + 2n + 1 = 2 + 2(N_radial − 1) + 1 = lmax_even + 1.
    u_max = lmax_even + 1

    # Bessel j_u(bessel_h_scale · |s|) for u ∈ [0, u_max], per reflection.
    x = bessel_h_scale * s_mag                                      # (M,)
    j_all = spherical_bessel_table(x, u_max)                        # (M, u_max+1)

    # bessel_factor[h, l, n] = √(2u+1) · j_u(x_h) / x_h, with u = l + 2n + 1.
    # Matches Phaser DataMR.cc:993 + 1109:
    #   sqrt_table[i] = sqrt(2*i+1); besselx[u] = sqrt_table[u] · sphbessel(u, h) / h
    # (Crowther/Navaza unit-ball radial-basis weight; the previous mimic used
    # √u, an unweighted variant that biased the radial power spectrum.)
    M = s_vectors.shape[0]
    bessel = torch.zeros((M, L, N_radial), dtype=real_dtype, device=device)
    safe_x = x.clamp(min=1e-30)
    for l in range(2, lmax_even + 1, 2):
        n_l = (lmax_even - l) // 2 + 1
        for n in range(n_l):
            u = l + 2 * n + 1
            bessel[:, l, n] = math.sqrt(float(2 * u + 1)) * j_all[:, u] / safe_x

    # Accumulate the SH expansion in chunks.
    # c_nlm[n, l, m] = Σ_h conj(Y_lm(ŝ_h)) · I_h · bessel[h, l, n]
    c_nlm = torch.zeros(
        (N_radial, L, 2 * L - 1), dtype=complex_dtype, device=device,
    )
    intensity_c = intensity.to(complex_dtype)

    for start in range(0, M, chunk_size):
        stop = min(start + chunk_size, M)
        Y = evaluate_ylm(theta[start:stop], phi[start:stop], L)      # (n, L, 2L-1)
        # weight × conj(Y) per reflection.
        Y_w = torch.conj(Y) * intensity_c[start:stop].view(-1, 1, 1)  # (n, L, 2L-1)
        b_slice = bessel[start:stop].to(complex_dtype)               # (n, L, N_radial)
        # einsum: sum over h, contract bessel(h,l,n) × Y_w(h,l,m).
        contrib = torch.einsum("hln,hlm->nlm", b_slice, Y_w)
        c_nlm += contrib

    # Zero odd-l rows (and l = 0, which Phaser also doesn't use).
    l_vals = torch.arange(L, device=device)
    odd_or_zero_mask = (l_vals % 2 == 1) | (l_vals == 0)
    c_nlm[:, odd_or_zero_mask, :] = 0.0

    # m-symmetry filter (applied to observed side only; caller passes
    # zsymm = 1 for calc side).
    if zsymm > 1:
        m_vals = torch.arange(-(L - 1), L, device=device)             # (2L-1,)
        m_invalid = (m_vals.abs() % zsymm) != 0
        c_nlm[:, :, m_invalid] = 0.0

    return BesselSHCoefficients(
        c_nlm=c_nlm, L=L, N_radial=N_radial,
        bessel_h_scale=float(bessel_h_scale), zsymm=int(zsymm),
    )


# =============================================================================
# Phaser-style data preparation: Wilson normalisation + σA Eterm
# =============================================================================


def _wilson_normalise(
    F: torch.Tensor,
    s_mag: torch.Tensor,
    n_shells: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-shell Wilson normalisation of amplitudes:

        E_h = F_h / sqrt(<F²>_p)    where p = shell containing h.

    Phaser's `Feff[r] / SIGMAN.sqrt_epsnSN[r]` (`DataMR.cc:925`) does
    roughly this (modulo French-Wilson and explicit ε-factor handling
    which we skip; the input F is already anisotropy-corrected by the
    caller in practice).

    Returns (E_h, sqrt_mean_F2_per_h).
    """
    edges, _ = equal_count_shell_edges(s_mag, n_shells)
    shell_idx = assign_shells(s_mag, edges)
    valid = shell_idx >= 0
    F_dtype = F.dtype
    F2 = F * F
    count = torch.zeros(n_shells, dtype=torch.int64, device=F.device)
    sumF2 = torch.zeros(n_shells, dtype=F_dtype, device=F.device)
    F2_v = F2[valid]
    idx_v = shell_idx[valid]
    count.index_add_(0, idx_v, torch.ones_like(idx_v))
    sumF2.index_add_(0, idx_v, F2_v)
    mean_F2 = sumF2 / count.clamp(min=1).to(F_dtype)
    mean_F2 = mean_F2.clamp(min=1e-12)
    sqrt_mean = mean_F2.sqrt()
    per_h = torch.ones_like(F)
    per_h[valid] = sqrt_mean[idx_v]
    E = F / per_h
    return E, per_h


# -----------------------------------------------------------------------------
# French-Wilson posterior expected values  (math_FrenchWilson.cc)
# -----------------------------------------------------------------------------


def _expectE_FW_acen(eosq, sigesq):
    """
    Acentric posterior expected E from normalised observed intensity (eosq)
    and its standard deviation (sigesq). Translates verbatim from Phaser's
    `lib/math_FrenchWilson.cc:expectEFWacen` (lines 8-44). Vectorised NumPy.

    `eosq = Iobs / <I>`, `sigesq = σIobs / <I>`.
    """
    import numpy as np
    from scipy.special import erfc, pbdv
    CROSS1, CROSS2 = -12.5, 18.0
    SQRT2 = np.sqrt(2.0)
    x = (eosq - sigesq ** 2) / sigesq
    xsqr = x * x
    ee = np.empty_like(eosq)
    # Large negative argument: asymptotic
    m_neg = x < CROSS1
    if m_neg.any():
        xs = xsqr[m_neg]
        num = (-916620705. + xs *
               (91891800. + xs *
                (-11531520. + xs *
                 (1935360. + xs *
                  (-491520. + xs * 262144.)))))
        den = (-495452160. + xs *
               (55050240. + xs *
                (-7864320. + xs *
                 (1572864. + xs *
                  (-524288. + xs * 524288.)))))
        ee[m_neg] = np.sqrt(-np.pi * sigesq[m_neg] / x[m_neg]) * num / den
    # Large positive argument: asymptotic
    m_pos = x > CROSS2
    if m_pos.any():
        xs = xsqr[m_pos]
        num = (-45045. + 32. * xs *
               (-315. + 8. * xs *
                (-15. - 16. * xs + 128. * xs * xs)))
        ee[m_pos] = (np.sqrt(sigesq[m_pos]) * num /
                     (32768. * x[m_pos] ** 7.5))
    # Moderate arguments
    m_mid = ~(m_neg | m_pos)
    if m_mid.any():
        xm = x[m_mid]
        pcd, _ = pbdv(-1.5, -xm)
        ee[m_mid] = (np.sqrt(sigesq[m_mid] / 2.0) * np.exp(-xm * xm / 4.0) *
                     pcd / erfc(-xm / SQRT2))
    return ee


def _expectEsq_FW_acen(eosq, sigesq):
    """Acentric posterior <E²>. From `expectEsqFWacen` (lines 46-78)."""
    import numpy as np
    from scipy.special import erfc
    CROSS1, CROSS2 = -8.9, 5.7
    SQRT2_BY_PI = np.sqrt(2.0 / np.pi)
    SQRT2 = np.sqrt(2.0)
    eesq_base = eosq - sigesq ** 2          # baseline value
    x = eesq_base / (SQRT2 * sigesq)
    xsqr = x * x
    eesq = eesq_base.copy()
    m_neg = x < CROSS1
    if m_neg.any():
        xs = xsqr[m_neg]
        num = (-135135. + xs * (20790. + xs * (-3780. + xs *
                (840. + xs * (-240. + xs * (96. - xs * 64.))))))
        den = (-135135. + xs * (20790. + xs * (-3780. + xs *
                (840. + xs * (-240. + xs * (96. + xs *
                 (-64. + xs * 128.)))))))
        eesq[m_neg] = eesq_base[m_neg] * num / den
    m_mid = (x >= CROSS1) & (x <= CROSS2)
    if m_mid.any():
        xm = x[m_mid]
        eesq[m_mid] = (eesq_base[m_mid] +
                       SQRT2_BY_PI * sigesq[m_mid] /
                       (np.exp(xm * xm) * erfc(-xm)))
    # x > CROSS2: eesq stays at eesq_base (default per source)
    return eesq


def _expectE_FW_cen(eosq, sigesq):
    """Centric posterior <E>. From `expectEFWcen` (lines 80-113)."""
    import numpy as np
    from scipy.special import pbdv
    CROSS1, CROSS2 = -17.5, 17.5
    SQRTPI = np.sqrt(np.pi)
    x = sigesq / 2.0 - eosq / sigesq
    xsqr = x * x
    pcdratio = np.empty_like(x)
    m_neg = x < CROSS1
    if m_neg.any():
        xn, xs = x[m_neg], xsqr[m_neg]
        pcdratio[m_neg] = ((1024. * SQRTPI * (-xn) ** 6.5) /
                           (3465. + xs *
                            (840. + xs *
                             (384. + xs * 1024.))))
    m_pos = x > CROSS2
    if m_pos.any():
        xp, xs = x[m_pos], xsqr[m_pos]
        num = (3440640. + xs *
               (-491520. + xs *
                (98304. + xs *
                 (-32768. + xs * 32768.))))
        den = (675675. + xs *
               (-110880. + xs *
                (26880. + xs *
                 (-12288. + xs * 32768.))))
        pcdratio[m_pos] = num / (den * np.sqrt(xp))
    m_mid = ~(m_neg | m_pos)
    if m_mid.any():
        xm = x[m_mid]
        d_neg1, _ = pbdv(-1.0, xm)
        d_neghalf, _ = pbdv(-0.5, xm)
        pcdratio[m_mid] = d_neg1 / d_neghalf
    return np.sqrt(sigesq / np.pi) * pcdratio


def _expectEsq_FW_cen(eosq, sigesq):
    """Centric posterior <E²>. From `expectEsqFWcen` (lines 115-152)."""
    import numpy as np
    from scipy.special import pbdv
    CROSS1, CROSS2 = -17.5, 17.5
    x = sigesq / 2.0 - eosq / sigesq
    xsqr = x * x
    pcdratio = np.empty_like(x)
    m_neg = x < CROSS1
    if m_neg.any():
        xn, xs = x[m_neg], xsqr[m_neg]
        num = (45045. + xs *
               (10080. + xs *
                (3840. + xs *
                 (4096. - xs * 32768.))))
        den = xn * (55440. + xs *
                    (13440. + xs *
                     (6144. + xs * 16384.)))
        pcdratio[m_neg] = num / den
    m_pos = x > CROSS2
    if m_pos.any():
        xp, xs = x[m_pos], xsqr[m_pos]
        num = (11486475. + xs *
               (-1441440. + xs *
                (241920. + xs *
                 (-61440. + xs * 32768.))))
        den = xp * (675675. + xs *
                    (-110880. + xs *
                     (26880. + xs *
                      (-12288. + xs * 32768.))))
        pcdratio[m_pos] = num / den
    m_mid = ~(m_neg | m_pos)
    if m_mid.any():
        xm = x[m_mid]
        d_neg15, _ = pbdv(-1.5, xm)
        d_neghalf, _ = pbdv(-0.5, xm)
        pcdratio[m_mid] = d_neg15 / d_neghalf
    return sigesq * pcdratio / 2.0


def _french_wilson_posterior(eosq, sigesq, centric_mask):
    """Wrap centric/acentric branches.

    Phaser `expectEFW` / `expectEsqFW` (lines 154-178): if sigesq <= 0 the
    measurement is treated as exact and (eEFW, eEsqFW) = (sqrt(eosq), eosq).
    """
    import numpy as np
    eEFW = np.empty_like(eosq)
    eEsqFW = np.empty_like(eosq)
    zero_sig = sigesq <= 0.0
    if zero_sig.any():
        eEFW[zero_sig] = np.sqrt(np.maximum(eosq[zero_sig], 0.0))
        eEsqFW[zero_sig] = np.maximum(eosq[zero_sig], 0.0)
    valid = ~zero_sig
    if valid.any():
        cen = centric_mask & valid
        acen = (~centric_mask) & valid
        if cen.any():
            eEFW[cen] = _expectE_FW_cen(eosq[cen], sigesq[cen])
            eEsqFW[cen] = _expectEsq_FW_cen(eosq[cen], sigesq[cen])
        if acen.any():
            eEFW[acen] = _expectE_FW_acen(eosq[acen], sigesq[acen])
            eEsqFW[acen] = _expectEsq_FW_acen(eosq[acen], sigesq[acen])
    return eEFW, eEsqFW


# -----------------------------------------------------------------------------
# DFAC via Halley iteration  (math_RiceLLG.cc:getDfactor)
# -----------------------------------------------------------------------------


def _i0e_full(x):
    """Phaser's `eBesselI0(x) = I0(x)·exp(-|x|)`. Symmetric in x.

    See `math_eBesselI0.cc` — Phaser's Rice-moment formulas use the
    exp-scaled Bessel (the un-scaled `I0` cancels analytically with the
    Gaussian envelope of the Rice distribution).
    """
    import numpy as np
    from scipy.special import i0e
    return i0e(np.abs(x))


def _i1e_full(x):
    """Phaser's `eBesselI1(x) = I1(x)·exp(-|x|)`. Antisymmetric in x."""
    import numpy as np
    from scipy.special import i1e
    return np.sign(x) * i1e(np.abs(x))


def _effSigaRoot_acen(ee, eesq, sa):
    """`effSigaRootAcen` (math_RiceLLG.cc:12-34)."""
    import numpy as np
    sigbsqr = 1.0 - sa * sa
    x = 0.5 * (eesq - sigbsqr) / sigbsqr
    return (np.sqrt(np.pi * sigbsqr) / (2.0 * sigbsqr) *
            (eesq * _i0e_full(x) + (eesq - sigbsqr) * _i1e_full(x)) - ee)


def _deffSigaRoot_acen(eesq, sa):
    """`deffSigaRootAcen_by_dsa` (lines 36-52)."""
    import numpy as np
    sigbsqr = 1.0 - sa * sa
    x = 0.5 * (eesq - sigbsqr) / sigbsqr
    return np.sqrt(np.pi / sigbsqr) * (sa / 2.0) * _i1e_full(x)


def _d2effSigaRoot_acen(eesq, sa):
    """`d2effSigaRootAcen_by_dsa2` (lines 54-81)."""
    import numpy as np
    sigasqr = sa * sa
    sigapow4 = sigasqr * sigasqr
    sigbsqr = 1.0 - sigasqr
    xnum = eesq - sigbsqr
    x = 0.5 * xnum / sigbsqr
    out = np.empty_like(eesq)
    big = xnum > 1e-10
    if big.any():
        I0 = _i0e_full(x[big])
        I1 = _i1e_full(x[big])
        out[big] = (np.sqrt(np.pi / sigbsqr[big]) / (2.0 * sigbsqr[big] ** 2) *
                    (eesq[big] * sigasqr[big] * I0 +
                     (eesq[big] - 1.0 - (-2.0 + eesq[big] * (2.0 + eesq[big])) * sigasqr[big] +
                      (eesq[big] - 1.0) * sigapow4[big]) * I1 / xnum[big]))
    small = ~big
    if small.any():
        samin = np.sqrt(np.maximum(1.0 - eesq[small], 0.0))
        out[small] = (np.sqrt(np.pi) * samin *
                      ((3.0 + samin * samin) * sa[small] -
                       2.0 * (samin + samin ** 3)) /
                      (4.0 * eesq[small] ** 2.5))
    return out


def _effSigaRoot_cen(ee, eesq, sa):
    """`effSigaRootCen` (lines 83-105)."""
    import numpy as np
    from scipy.special import erf
    sigbsqr = 1.0 - sa * sa
    x = 0.5 * (eesq - sigbsqr) / sigbsqr
    x_safe = np.maximum(x, 0.0)             # erf(sqrt(x)) ill-defined for x<0
    return (np.exp(-x) * np.sqrt(2.0 * sigbsqr / np.pi) +
            np.sqrt(np.maximum(eesq - sigbsqr, 0.0)) * erf(np.sqrt(x_safe)) - ee)


def _deffSigaRoot_cen(eesq, sa):
    """`deffSigaRootCen_by_dsa` (lines 107-130)."""
    import numpy as np
    from scipy.special import erf
    sigbsqr = 1.0 - sa * sa
    xnum = eesq - sigbsqr
    x = 0.5 * xnum / sigbsqr
    out = np.empty_like(eesq)
    big = np.abs(xnum) > 1e-10
    if big.any():
        x_safe = np.maximum(x[big], 0.0)
        out[big] = (sa[big] * erf(np.sqrt(x_safe)) /
                    np.sqrt(np.maximum(xnum[big], 1e-30)) -
                    np.exp(-x[big]) * np.sqrt(2.0 * sigbsqr[big] / np.pi) *
                    sa[big] / sigbsqr[big])
    small = ~big
    if small.any():
        out[small] = (xnum[small] * np.sqrt(2.0 / np.pi) * sa[small] /
                      (3.0 * sigbsqr[small] ** 1.5))
    return out


def _d2effSigaRoot_cen(eesq, sa):
    """`d2effSigaRootCen_by_dsa2` (lines 132-159)."""
    import numpy as np
    from scipy.special import erf
    sigasqr = sa * sa
    sigapow4 = sigasqr * sigasqr
    sigbsqr = 1.0 - sigasqr
    xnum = eesq - sigbsqr
    x = 0.5 * xnum / sigbsqr
    sigbsqrtpi = np.sqrt(np.pi * sigbsqr)
    out = np.empty_like(eesq)
    big = np.abs(xnum) > 1e-10
    if big.any():
        x_safe = np.maximum(x[big], 0.0)
        d2num = ((eesq[big] - 1.0) * sigbsqr[big] ** 2 * sigbsqrtpi[big] *
                 erf(np.sqrt(x_safe)))
        exp_part = np.where(
            x[big] < 20.0,
            np.sqrt(np.maximum(2.0 * xnum[big], 0.0)) * np.exp(-x[big]) *
            (1.0 - eesq[big] + sigasqr[big] *
             (eesq[big] + eesq[big] ** 2 - 2.0) + sigapow4[big]),
            np.zeros_like(x[big]),
        )
        d2num = d2num + exp_part
        out[big] = d2num / (sigbsqr[big] ** 2 *
                            np.maximum(xnum[big], 1e-30) ** 1.5 *
                            sigbsqrtpi[big])
    small = ~big
    if small.any():
        out[small] = (np.sqrt(2.0 / np.pi) * sa[small] /
                      (1.5 * sigbsqr[small] ** 1.5))
    return out


def get_dfactor_vectorised(ee_np, eesq_np, centric_np):
    """
    Vectorised version of Phaser's `math_RiceLLG.cc:getDfactor` (lines 191-250).
    Halley's method with bisection fallback, run over all reflections in
    parallel. Each reflection has its own bracket [dflo, dfhi].

    Returns a (N,) numpy float64 array of DFAC values in (0, 1).
    """
    import numpy as np

    EPS1 = 1e-7
    EPS2 = 1e-10
    MAXDFAC = 1.0 - EPS1

    ee = np.asarray(ee_np, dtype=np.float64)
    eesq = np.asarray(eesq_np, dtype=np.float64)
    cen = np.asarray(centric_np, dtype=bool)
    N = ee.shape[0]

    # Case 1: no observational error (eesq - ee² ≤ 0) → DFAC = 1.0
    out = np.ones(N, dtype=np.float64)
    has_err = (eesq - ee * ee) > 0.0

    if not has_err.any():
        return out

    ee_a, eesq_a, cen_a = ee[has_err], eesq[has_err], cen[has_err]
    # Bracket. dflo = max(sqrt(1 - min(eesq, 1)) + EPS, EPS).
    dflo = np.maximum(np.sqrt(np.maximum(1.0 - np.minimum(eesq_a, 1.0), 0.0)) + EPS1, EPS1)
    dfhi = np.full_like(dflo, MAXDFAC)

    # Early return: if dflo >= MAXDFAC, just use dflo.
    early = dflo >= MAXDFAC
    if early.any():
        # No iteration for those; keep dflo
        pass

    dfmid = 0.5 * (dflo + dfhi)
    # Compute initial fmid
    fmid = np.empty_like(dfmid)
    if cen_a.any():
        fmid[cen_a] = _effSigaRoot_cen(ee_a[cen_a], eesq_a[cen_a], dfmid[cen_a])
    if (~cen_a).any():
        fmid[~cen_a] = _effSigaRoot_acen(ee_a[~cen_a], eesq_a[~cen_a], dfmid[~cen_a])

    active = ~early    # reflections still being iterated
    for _ in range(50):
        if not active.any():
            break
        # Convergence check
        conv = (dfhi - dflo) <= EPS1
        conv |= np.abs(fmid) <= EPS2
        active = active & ~conv
        if not active.any():
            break

        # slope and curve at dfmid (only for active reflections)
        slope = np.empty_like(dfmid)
        curve = np.empty_like(dfmid)
        cen_act = cen_a & active
        acen_act = (~cen_a) & active
        if cen_act.any():
            slope[cen_act] = _deffSigaRoot_cen(eesq_a[cen_act], dfmid[cen_act])
            curve[cen_act] = _d2effSigaRoot_cen(eesq_a[cen_act], dfmid[cen_act])
        if acen_act.any():
            slope[acen_act] = _deffSigaRoot_acen(eesq_a[acen_act], dfmid[acen_act])
            curve[acen_act] = _d2effSigaRoot_acen(eesq_a[acen_act], dfmid[acen_act])

        # Halley step (fall back to `fmid · slope` if curve <= 0, matching
        # the Phaser source literally — math_RiceLLG.cc:230-231).
        denom_halley = 2.0 * (slope ** 2 - fmid * curve)
        use_halley = (curve > 0.0) & (np.abs(denom_halley) > 1e-30)
        step = np.where(
            use_halley,
            2.0 * fmid * slope / np.where(use_halley, denom_halley, 1.0),
            fmid * slope,
        )
        dfnew = dfmid - step
        # If new value out of bracket, bisect instead
        in_bracket = (dfnew > dflo) & (dfnew < dfhi)
        dfmid_new = np.where(in_bracket, dfnew, 0.5 * (dflo + dfhi))

        # Apply only on active reflections; converged ones keep dfmid.
        dfmid = np.where(active, dfmid_new, dfmid)

        # Re-evaluate fmid only on active reflections.
        if cen_act.any():
            fmid[cen_act] = _effSigaRoot_cen(ee_a[cen_act], eesq_a[cen_act],
                                              dfmid[cen_act])
        if acen_act.any():
            fmid[acen_act] = _effSigaRoot_acen(ee_a[acen_act], eesq_a[acen_act],
                                                  dfmid[acen_act])

        # Update bracket based on fmid sign.
        below = (fmid < 0.0) & active
        above = (fmid >= 0.0) & active
        dflo = np.where(below, dfmid, dflo)
        dfhi = np.where(above, dfmid, dfhi)

    out[has_err] = dfmid
    return np.clip(out, EPS1, MAXDFAC)


def french_wilson_preprocess(
    F: torch.Tensor,
    sig_F: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    *,
    n_wilson_shells: int = 20,
) -> dict:
    """
    Phaser-style preprocessing from raw (F, σF, centric) to (eEobs, DFAC).

    Implements the chain documented at the top of the module:
      1. equal-count Wilson shells over `s_mag`
      2. per-shell `<F²>_p` (Phaser's `SIGMAN.BINS`)
      3. per-reflection normalised intensity `eosq = F² / <F²>` and σ
         `sigesq = σI / <I> ≈ 2·F·σF / <F²>`
      4. French-Wilson posterior `eEFW, eEsqFW` (`math_FrenchWilson.cc`)
      5. DFAC via Halley iteration on Rice moments (`math_RiceLLG.cc`)
      6. `eEobs = sqrt(eEsqFW + (DFAC²−1)/DFAC²)`, clamped to ≤10
         (Phaser `Dfactor.cc:87-93`).

    Returns a dict with torch tensors back on the input device:
      eEobs: (N,) effective normalised amplitude
      DFAC : (N,) per-reflection D-factor ∈ [1e-7, 1−1e-7]
      sqrt_mean_F2: (N,) per-reflection √<F²>_p (caller can multiply
                    back to recover absolute-scale Feff if needed)
    """
    import numpy as np

    device = F.device
    F_np = F.detach().to("cpu").to(torch.float64).numpy()
    sigF_np = sig_F.detach().to("cpu").to(torch.float64).numpy()
    s_np = s_mag.detach().to("cpu").to(torch.float64).numpy()
    cen_np = centric.detach().to("cpu").bool().numpy()

    # 1+2. Per-shell <F²> by equal-count binning over |s|.
    sorted_idx = np.argsort(s_np)
    edges_idx = np.linspace(0, len(s_np) - 1, n_wilson_shells + 1).round().astype(np.int64)
    s_edges = s_np[sorted_idx][edges_idx]
    # Nudge endpoints
    s_edges[0] -= 1e-6
    s_edges[-1] += 1e-6
    shell_idx = np.clip(
        np.searchsorted(s_edges, s_np, side="right") - 1, 0, n_wilson_shells - 1,
    )
    F2 = F_np * F_np
    mean_F2 = np.zeros(n_wilson_shells, dtype=np.float64)
    counts = np.zeros(n_wilson_shells, dtype=np.int64)
    np.add.at(mean_F2, shell_idx, F2)
    np.add.at(counts, shell_idx, 1)
    mean_F2 = mean_F2 / np.maximum(counts, 1)
    mean_F2 = np.maximum(mean_F2, 1e-12)
    mean_I_per_h = mean_F2[shell_idx]              # <I>_p mapped to each h
    sqrt_mean_F2 = np.sqrt(mean_I_per_h)

    # 3. Normalised intensity + its sigma.
    eosq = F2 / mean_I_per_h
    sigesq = 2.0 * F_np * sigF_np / mean_I_per_h
    # Guard: sigesq must be positive for FW. If σF=0 we let zero_sig path
    # in `_french_wilson_posterior` handle it.
    sigesq = np.maximum(sigesq, 0.0)

    # 4. French-Wilson posterior.
    eEFW, eEsqFW = _french_wilson_posterior(eosq, sigesq, cen_np)
    # Guard non-physical posteriors (numerical accidents): make sure
    # eEsqFW ≥ eEFW² (the second moment must dominate the first squared).
    bad = eEsqFW < eEFW * eEFW
    if bad.any():
        eEsqFW[bad] = eEFW[bad] ** 2 + 1e-12

    # 5. DFAC via Halley iteration.
    DFAC = get_dfactor_vectorised(eEFW, eEsqFW, cen_np)

    # 6. eEobs = sqrt(eEsqFW + (DFAC²-1)/DFAC²), clamp to ≤ 10; recompute
    # DFAC if clamped (mirrors `Dfactor.cc:87-93`).
    dfsqr = DFAC * DFAC
    eEobs_sqr = eEsqFW + (dfsqr - 1.0) / np.maximum(dfsqr, 1e-30)
    eEobs_sqr = np.maximum(eEobs_sqr, 0.0)
    eEobs = np.sqrt(eEobs_sqr)
    # Clamp at 10 and recompute DFAC where clamped (rare).
    clamp_mask = (eEobs > 10.0) & (eEsqFW > 1.0)
    if clamp_mask.any():
        # eEobs = 10 → solve for DFAC: 100 = eEsqFW + (dfsqr-1)/dfsqr
        #   → (100 - eEsqFW) = (dfsqr - 1)/dfsqr
        #   → dfsqr (100 - eEsqFW) = dfsqr - 1
        #   → dfsqr (100 - eEsqFW - 1) = -1
        #   → dfsqr (eEsqFW - 99) = 1
        eEobs[clamp_mask] = 10.0
        DFAC[clamp_mask] = 1.0 / np.sqrt(np.maximum(eEsqFW[clamp_mask] - 99.0, 1e-30))
        DFAC[clamp_mask] = np.clip(DFAC[clamp_mask], 1e-7, 1.0 - 1e-7)

    return {
        "eEobs": torch.from_numpy(eEobs).to(device=device, dtype=F.dtype),
        "DFAC": torch.from_numpy(DFAC).to(device=device, dtype=F.dtype),
        "sqrt_mean_F2": torch.from_numpy(sqrt_mean_F2).to(device=device, dtype=F.dtype),
    }


def eterm_sigma_a(s_mag: torch.Tensor, delta_vrms_A: float) -> torch.Tensor:
    """
    Phaser's σA Eterm, literal port of `Ensemble.cc:42`:

        Eterm(s) = exp(-(2π²/3) · s² · ΔVRMS_var)

    where `ΔVRMS_var` is the *coordinate variance* in Å². We accept the RMS
    coordinate error `delta_vrms_A` (Å) per the standard σA convention and
    square it internally: `ΔVRMS_var = delta_vrms_A²`.

    Per-reflection (`s_mag` may be any shape). Returns the same shape.

    NB: an earlier version of this mimic used `exp(-2π² · s² · ΔVRMS²)` —
    that is *3×* too aggressive in the exponent and over-attenuated calc at
    high resolution. See `phaser_frf_known_bugs.md` for the audit.
    """
    s2 = s_mag * s_mag
    return torch.exp(-(2.0 / 3.0) * (math.pi ** 2) * s2 * (delta_vrms_A ** 2))


# =============================================================================
# Top-level FRF entry point — mirrors `ball_rotation_search` API
# =============================================================================


def phaser_rotation_search(
    s_obs: torch.Tensor,
    F_obs: torch.Tensor,
    centric_obs: torch.Tensor,
    s_calc: torch.Tensor,
    F_calc: torch.Tensor,
    sym_mats: torch.Tensor,
    *,
    L: int = 24,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    delta_vrms_A: float = 1.0,
    n_wilson_shells: int = 20,
    n_peaks: int = 500,
    refine_subvoxel: bool = True,
    n_refine: int = 50,
    sigma_threshold: float = -5.0,
    bessel_h_scale: Optional[float] = None,
    use_lerf1_intensity: bool = True,
    use_m_symmetry_filter: bool = True,
    sig_F_obs: Optional[torch.Tensor] = None,
    use_french_wilson: bool = False,
    use_shell_variance_weights: bool = False,
    n_var_shells: int = 20,
    grid_sampling_deg: float = 3.0,
) -> Tuple[AdaptiveRotationFunction, List[RotationPeak]]:
    """
    Phaser-faithful Fast Rotation Function.

    Parameters
    ----------
    s_obs : (N_o, 3) real
        Observed-side reciprocal-lattice vectors in 1/Å.
    F_obs : (N_o,) real
        Observed amplitudes (already anisotropy-corrected by caller, if
        applicable).
    centric_obs : (N_o,) bool
        Centric flag for each observed reflection.
    s_calc, F_calc : (N_c, 3), (N_c,) real
        Model reciprocal vectors and amplitudes.
    sym_mats : (n_ops, 3, 3) real
        Spacegroup rotation matrices — used to detect the high-order axis
        for the m-symmetry filter (Phaser `highOrderAxis()`).
    L : int, default 24
        Wigner/SH bandwidth. lmax = L − 1 (rounded down to even).
    d_min, d_max : float, optional
        Resolution window. Reflections outside `[1/d_max, 1/d_min]` (in |s|)
        are discarded on both sides before normalisation.
    delta_vrms_A : float, default 1.0
        ΔVRMS in Å for the Luzzati σA Eterm applied to the calc side.
    n_wilson_shells : int, default 20
        Number of equal-count shells used in per-shell Wilson normalisation.
    n_peaks : int, default 500
        Maximum number of rotation peaks returned.
    refine_subvoxel : bool, default True
        Apply quadratic sub-voxel refinement to the top `n_refine` peaks.
    n_refine : int, default 50
        Number of peaks to refine sub-voxel.
    sigma_threshold : float, default -5.0
        Minimum Z-score for a peak to be returned. Negative ≈ "keep everything".
    bessel_h_scale : float, optional
        Pre-multiplier on |s| in the Bessel argument: `h = bessel_h_scale · |s|`.
        Default: `(L - 1) · d_min` (Phaser's `lmax · HIRES`). Picking this
        scale puts the highest-u Bessel functions near their first peak at
        the maximum |s| in the data, so the radial basis covers the full
        resolution range without wasted bandwidth.
    use_lerf1_intensity : bool, default True
        If True, observed intensity is `cweight · (E_obs² − 1)`; if False,
        plain `E_obs² − 1` (cweight = 1 everywhere).
    use_m_symmetry_filter : bool, default True
        If True, detect ZSYMM from `sym_mats` and apply the m-symmetry
        filter to the observed-side SH coefficients.
    sig_F_obs : (N_o,) real, optional
        Standard deviation of `F_obs`. Required if `use_french_wilson=True`;
        ignored otherwise.
    use_french_wilson : bool, default False
        If True, the observed-side intensity is built via the full Phaser
        preprocessing chain: per-shell Wilson normalisation → French-Wilson
        posterior → per-reflection Luzzati DFAC → effective Eobs. Observed
        intensity becomes `cweight · (eEobs² − 1) · DFAC²` instead of the
        simpler `cweight · (E² − 1)`. Requires `sig_F_obs`.
    use_shell_variance_weights : bool, default False
        If True, compute per-shell empirical variance of `intensity_obs`
        and weight each reflection by `1/√Var_p` before the SH expansion.
        Analog of torchref's `auto_variance_weights=True`. Downweights
        shells whose obs Patterson coefficient is dominated by intermolecular
        noise — particularly important for cubic/tetragonal groups where the
        spacegroup-invariant SH subspace has few coefficients and per-shell
        SNR matters disproportionately.
    n_var_shells : int, default 20
        Number of shells for the variance estimate. Same scheme as
        `n_wilson_shells`.
    grid_sampling_deg : float, default 3.0
        Target Euler-angle resolution in degrees. Drives the per-β α/γ sample
        counts via Phaser's `pmax(β) = 720/Δ · cos(β/2)`,
        `qmax(β) = 360/Δ · sin(β/2)` (FastRot.cc:92-96). Total adaptive
        sample count ≈ `(720 · 360) / grid_sampling_deg²`.

    Returns
    -------
    adaptive_rf : AdaptiveRotationFunction
        Ragged Euler-grid evaluation of `C(α, β, γ)` with per-β
        variable-shape `(qmax_k, pmax_k)` slices, alpha/gamma grids, and the
        β midpoint quadrature points.
    peaks : list of RotationPeak (sorted by descending score).
    """
    device = s_obs.device
    real_dtype = s_obs.dtype

    # 1. Resolution mask on both sides.
    def _resmask(s_vec, F, centric=None):
        smag = s_vec.norm(dim=-1)
        lo = 1.0 / d_max if d_max is not None else 0.0
        hi = 1.0 / d_min if d_min is not None else float("inf")
        keep = (smag >= lo) & (smag <= hi)
        s_vec = s_vec[keep]
        F = F[keep]
        if centric is not None:
            centric = centric[keep]
            return s_vec, F, centric, smag[keep]
        return s_vec, F, smag[keep]

    # Apply resolution mask to obs side; also mask sig_F if provided.
    if use_french_wilson:
        if sig_F_obs is None:
            raise ValueError("use_french_wilson=True requires sig_F_obs.")
        smag_obs_pre = s_obs.norm(dim=-1)
        lo = 1.0 / d_max if d_max is not None else 0.0
        hi = 1.0 / d_min if d_min is not None else float("inf")
        keep_obs = (smag_obs_pre >= lo) & (smag_obs_pre <= hi)
        s_obs = s_obs[keep_obs]
        F_obs = F_obs[keep_obs]
        sig_F_obs = sig_F_obs[keep_obs]
        centric_obs = centric_obs[keep_obs]
        smag_obs = smag_obs_pre[keep_obs]
        s_calc, F_calc, smag_calc = _resmask(s_calc, F_calc)
    else:
        s_obs, F_obs, centric_obs, smag_obs = _resmask(s_obs, F_obs, centric_obs)
        s_calc, F_calc, smag_calc = _resmask(s_calc, F_calc)

    if s_obs.shape[0] < n_wilson_shells * 5:
        raise ValueError(
            f"Too few obs reflections ({s_obs.shape[0]}) for "
            f"{n_wilson_shells} Wilson shells in [{d_min}, {d_max}] Å."
        )

    # 2. Bessel argument scale. Default = lmax · d_min (Phaser DataMR.cc:1107).
    if bessel_h_scale is None:
        if d_min is None:
            raise ValueError("bessel_h_scale must be set when d_min is None")
        lmax = L - 1
        lmax_even = lmax if lmax % 2 == 0 else lmax - 1
        bessel_h_scale = float(lmax_even) * float(d_min)

    # 3. Wilson-normalise both sides (+ FW + DFAC on obs if requested).
    if use_french_wilson:
        fw = french_wilson_preprocess(
            F_obs, sig_F_obs, smag_obs, centric_obs,
            n_wilson_shells=n_wilson_shells,
        )
        eEobs = fw["eEobs"]
        DFAC = fw["DFAC"]
    else:
        E_obs, _ = _wilson_normalise(F_obs, smag_obs, n_wilson_shells)
        eEobs = E_obs
        DFAC = torch.ones_like(E_obs)
    E_calc, _ = _wilson_normalise(F_calc, smag_calc, n_wilson_shells)

    # 4. Observed intensity (LERF1-style): cweight · (eEobs² − 1) · DFAC².
    if use_lerf1_intensity:
        cweight = torch.where(
            centric_obs.bool(),
            torch.ones_like(eEobs),
            2.0 * torch.ones_like(eEobs),
        )
    else:
        cweight = torch.ones_like(eEobs)
    intensity_obs = cweight * (eEobs * eEobs - 1.0) * (DFAC * DFAC)

    # 4b. Optional per-shell empirical-variance weighting (Phaser does this
    # via per-shell BINS + best(r) — torchref's `auto_variance_weights=True`
    # is the cleanest analog). Downweight shells whose intensity_obs has high
    # empirical variance, normalising so the mean weight is ~1.
    if use_shell_variance_weights:
        from ..sh import compute_patterson_shell_variance
        edges_var, _ = equal_count_shell_edges(smag_obs, n_var_shells)
        shell_idx_obs = assign_shells(smag_obs, edges_var)
        valid_obs = shell_idx_obs >= 0
        var_p = compute_patterson_shell_variance(
            intensity_obs[valid_obs].to(torch.float64),
            shell_idx_obs[valid_obs],
            P=n_var_shells,
        )
        inv_sqrt_var = 1.0 / var_p.sqrt().clamp(min=1e-30)
        # Normalise mean weight = 1 so the absolute scale of intensity_obs
        # doesn't shift (only the SHAPE across shells matters for the
        # cross-correlation rotation function).
        inv_sqrt_var = (inv_sqrt_var *
                        (n_var_shells / inv_sqrt_var.sum().clamp(min=1e-30)))
        weights_per_h = torch.ones_like(intensity_obs)
        weights_per_h[valid_obs] = inv_sqrt_var[shell_idx_obs[valid_obs]].to(
            intensity_obs.dtype,
        )
        intensity_obs = intensity_obs * weights_per_h

    # 5. Model intensity with σA Eterm² weighting (per-reflection).
    eterm = eterm_sigma_a(smag_calc, delta_vrms_A)
    intensity_calc = (eterm ** 2) * (E_calc * E_calc - 1.0)

    # 6. Detect ZSYMM (high-order rotation axis from sym_mats).
    zsymm = 1
    if use_m_symmetry_filter and sym_mats is not None:
        _, zsymm = get_high_order_axis(
            sym_mats.to(torch.float64).cpu(),
        )
        zsymm = int(zsymm)

    # 7. Bessel-radial × SH expansion of both sides.
    c_obs = bessel_sh_expand(
        s_obs, intensity_obs.to(real_dtype),
        L=L, bessel_h_scale=bessel_h_scale,
        zsymm=zsymm, enforce_friedel=True,
    )
    c_calc = bessel_sh_expand(
        s_calc, intensity_calc.to(real_dtype),
        L=L, bessel_h_scale=bessel_h_scale,
        zsymm=1, enforce_friedel=True,        # NO m-filter on calc side
    )

    # 8. Cross-correlation in Wigner basis: contract on the radial-Bessel axis.
    # Convention chosen to match torchref's existing `ball_search.py` (line 182):
    #     xi[l, m, n] = Σ_r c_obs[r, l, n] · conj(c_calc[r, l, m])
    # i.e. obs-side SH index labels the OUTPUT "n" axis (→ γ Fourier),
    # calc-side SH index labels the OUTPUT "m" axis (→ α Fourier). With this
    # ordering, the peak Euler triple satisfies `s_calc = R · s_obs` (column
    # vector), matching the test-driver scenario where F_calc was generated by
    # applying R to the model coordinates.
    xi = torch.einsum(
        "rln,rlm->lmn",
        c_obs.c_nlm,
        torch.conj(c_calc.c_nlm),
    )

    # 9. Evaluate on the Phaser-faithful adaptive Euler grid.
    arf = evaluate_rotation_function_grid_adaptive(
        xi, L, grid_sampling_deg=grid_sampling_deg, n_beta=2 * L,
    )

    # 10. Peak picking + sub-voxel refinement on the ragged grid.
    peaks = find_rotation_peaks_adaptive(
        arf, n_peaks=n_peaks, sigma_threshold=sigma_threshold,
    )
    if refine_subvoxel and peaks:
        head = peaks[: min(n_refine, len(peaks))]
        head = refine_peaks_subvoxel_adaptive(head, arf)
        peaks = head + peaks[len(head):]
        peaks.sort(key=lambda r: r.score, reverse=True)

    return arf, peaks
