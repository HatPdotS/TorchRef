"""Bessel-radial × spherical-harmonic expansion of obs and calc Pattersons.

Mirrors ``DataMR::dataMR_FRF`` (DataMR.cc) and the helper sums in
``Ensemble.cc``. We import the validated ``bessel_sh_expand`` from
``torchref.alignment.phaser_frf`` and add the obs/calc cross-correlation
contraction that's the input to ``SiteListAng::get_FRF``.

Citations:
  * Bessel-radial × SH expansion: DataMR.cc:993, 1107-1117
  * sqrt(2u+1) · j_u(h)/h radial weight: DataMR.cc:993
  * Even-l only (Patterson centrosymmetry): implicit in DataMR.cc
  * m-symmetry filter: DataMR.cc:863-870, 1117
"""
from __future__ import annotations

import math

import torch

from .phaser_frf import spherical_bessel_table
from ..sh import evaluate_ylm

from .types import BesselSHCoefficients

__all__ = [
    "bessel_sh_expand",
    "cross_correlate_xi",
]


def bessel_sh_expand(
    s_vectors: torch.Tensor,
    intensity: torch.Tensor,
    *,
    L: int,
    bessel_h_scale: float,
    zsymm: int = 1,
    enforce_friedel: bool = True,
    chunk_size: int = -1,
) -> BesselSHCoefficients:
    """Phaser-style ``c_nlm = Σ_h Y*_lm(ŝ) · I · sqrt(2u+1) · j_u(h)/h``.

    Memory-bounded chunked reimplementation of
    ``torchref.alignment.phaser_frf.bessel_sh_expand`` (identical math,
    verified element-wise by ``tests/unit/frf_separate``). The legacy
    version materialises the full ``(M, L, N_radial)`` Bessel table and
    ``(M, u_max+1)`` j-table for *all* reflections at once — at L≈100 with
    a symmetry-unrolled obs set (≳10⁶ reflections) that is tens of GB and
    OOMs. Here the j-table, Bessel weights and Y_lm are all computed
    *inside* the reflection-chunk loop, so peak memory is set by one chunk.

    Citations:
      * radial × SH expansion, sqrt(2u+1)·j_u(h)/h weight: DataMR.cc:993, 1107
      * even-l only (Patterson centrosymmetry) + m-filter: DataMR.cc:863-870, 1117

    Parameters
    ----------
    chunk_size : int
        Reflections per chunk. ``-1`` (default) auto-sizes so the per-chunk
        Y_lm block ``(chunk, L, 2L-1)`` stays near ~256 MB.
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

    lmax = L - 1
    lmax_even = lmax if (lmax % 2 == 0) else (lmax - 1)
    if lmax_even < 2:
        raise ValueError(f"L={L} too small; need lmax_even >= 2 (so L >= 3).")
    N_radial = (lmax_even - 2) // 2 + 1
    u_max = lmax_even + 1

    # (l, n) -> u = l + 2n + 1 and the sqrt(2u+1) weight, precomputed once.
    even_ls = list(range(2, lmax_even + 1, 2))
    ln_u = []          # (l, n, u, sqrt(2u+1))
    for l in even_ls:
        n_l = (lmax_even - l) // 2 + 1
        for n in range(n_l):
            u = l + 2 * n + 1
            ln_u.append((l, n, u, math.sqrt(float(2 * u + 1))))

    M = s_vectors.shape[0]
    if chunk_size <= 0:
        # target ~256 MB for the complex (chunk, L, 2L-1) Y block
        chunk_size = int(max(256, min(8192, 16_000_000 // max(1, L * (2 * L - 1)))))

    c_nlm = torch.zeros(
        (N_radial, L, 2 * L - 1), dtype=complex_dtype, device=device,
    )

    for start in range(0, M, chunk_size):
        stop = min(start + chunk_size, M)
        s_c = s_vectors[start:stop]
        i_c = intensity[start:stop]
        s_mag = s_c.norm(dim=-1).clamp(min=1e-30)
        s_hat = s_c / s_mag.unsqueeze(-1)
        cos_theta = s_hat[..., 2].clamp(min=-1.0, max=1.0)
        theta = torch.acos(cos_theta)
        phi = torch.atan2(s_hat[..., 1], s_hat[..., 0])

        x = (bessel_h_scale * s_mag).clamp(min=1e-30)            # (c,)
        j_all = spherical_bessel_table(x, u_max)                 # (c, u_max+1)

        bessel = torch.zeros((stop - start, L, N_radial), dtype=real_dtype, device=device)
        for (l, n, u, w) in ln_u:
            bessel[:, l, n] = w * j_all[:, u] / x

        Y = evaluate_ylm(theta, phi, L)                          # (c, L, 2L-1)
        Y_w = torch.conj(Y) * i_c.to(complex_dtype).view(-1, 1, 1)
        c_nlm += torch.einsum("hln,hlm->nlm", bessel.to(complex_dtype), Y_w)

    # Zero odd-l rows and l = 0 (Patterson centrosymmetry; Phaser drops l=0).
    l_vals = torch.arange(L, device=device)
    c_nlm[:, (l_vals % 2 == 1) | (l_vals == 0), :] = 0.0

    # m-symmetry filter (observed side only; caller passes zsymm=1 for calc).
    if zsymm > 1:
        m_vals = torch.arange(-(L - 1), L, device=device)
        c_nlm[:, :, (m_vals.abs() % zsymm) != 0] = 0.0

    return BesselSHCoefficients(
        coeffs=c_nlm,
        L=L,
        N_radial=N_radial,
        bessel_h_scale=float(bessel_h_scale),
    )


def cross_correlate_xi(
    c_obs: BesselSHCoefficients,
    c_calc: BesselSHCoefficients,
) -> torch.Tensor:
    """Contract obs/calc Bessel-SH coefficients on the radial-Bessel axis.

    Phaser source: the radial sum ``Σ_n c_obs[n,l,m] · conj(c_calc[n,l,m'])``
    happens inside ``DataMR::dataMR_FRF`` before being fed into
    ``SiteListAng::DoRfftStuff`` as the ``clmn`` tensor (FastRot.cc:39).

    Convention (matches torchref's existing ball_search.py:182):
        xi[l, m, n] = Σ_r c_obs[r, l, n] · conj(c_calc[r, l, m])
    so that the peak Euler triple satisfies ``s_calc = R · s_obs``.

    Returns
    -------
    xi : torch.Tensor (complex), shape (L, 2L-1, 2L-1)
    """
    if c_obs.L != c_calc.L:
        raise ValueError(f"L mismatch: obs={c_obs.L} calc={c_calc.L}")
    return torch.einsum(
        "rln,rlm->lmn",
        c_obs.coeffs,
        torch.conj(c_calc.coeffs),
    )
