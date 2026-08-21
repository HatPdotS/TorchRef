"""Per-β rotation function evaluation on Phaser's adaptive SO(3) sample list.

Mirrors ``SiteListAng`` from
``reverse_engineering/phenix/phenix-1.20-4459/modules/phaser/codebase/phaser/src/FastRot.cc``.

The crucial point is that **the FFT itself is NOT per-β-adaptive**.
Phaser does:

1. ``get_FRF`` (FastRot.cc:90-167) loops over a uniform β grid
   ``β_b = b · Δ`` for ``b ∈ [0, bmax)``, ``bmax = ceil(180/Δ)``.
2. For each β: ``DoRfftStuff`` (FastRot.cc:19-88) builds the per-β
   Fourier-mode amplitudes
       ``S_{m1, m2}(β) = Σ_l ξ_{l, m1, m2} · d^l_{m1, m2}(β)``
   on the full ``(2L-1) × (2L-1)`` grid (asymmetric-unit storage only —
   the Friedel mate is added by cctbx via ``conjugate_flag=true``).
3. The 2D inverse FFT runs at a **fixed shape**
   ``amax = adjust_gridding(2·max(bmax, lmax), max_prime=5)`` for every β.
   The result is a dense ``M_β(α, γ)`` map indexed in ``[0, 1)`` along
   each axis.
4. The **adaptive sample list** is built once by ``allocate_memory``
   (FastRot.cc:169-262): for each β,
       ``pmax(β) = 720/Δ · cos(β/2)``
       ``qmax(β) = 360/Δ · sin(β/2)``
   and the ``(p, q)`` lattice is mapped to ``(α, γ)`` via
       ``α = (p/pmax + q/qmax) mod 1``
       ``γ = sign · (p/pmax − q/qmax) mod 1`` (FastRot.cc:216-219)
   with the β=0 special case keeping only the ``p == p`` diagonal
   (FastRot.cc:189-207) because only ``α + γ`` is meaningful at the pole.
5. ``M_β`` is **bilinearly interpolated** at each ``(α, γ)`` sample point
   to give the RF value (FastRot.cc:146-152, ``four_point_interpolation``).

Making the FFT shape itself ``(pmax(β), qmax(β))`` would collapse to
``(N, 1)`` at small β and lose all γ Fourier information. The FFT stays
dense; adaptivity is only in the sample list and the interpolation.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch

from .types import AdaptiveRotationFunction
from .wigner_d import wigner_contraction_per_beta

__all__ = [
    "adjust_gridding",
    "build_dense_map_per_beta",
    "build_adaptive_sample_list",
    "evaluate_rotation_function",
]


def adjust_gridding(target: int, max_prime: int = 5) -> int:
    """Smallest integer ≥ target whose largest prime factor is ≤ max_prime.

    Phaser source: ``scitbx::fftpack::adjust_gridding`` (FastRot.cc:66-69).
    For ``max_prime=5`` this is the standard 5-smooth (Hamming) numbers.
    """
    if target <= 1:
        return 1
    primes = [2, 3, 5, 7, 11, 13][: min(max(max_prime // 2, 1), 6)]
    primes = [p for p in [2, 3, 5, 7, 11, 13] if p <= max_prime]
    n = int(target)
    while True:
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def build_dense_map_per_beta(
    xi_lmn: torch.Tensor,
    betas: torch.Tensor,
    fft_size: int,
) -> torch.Tensor:
    """Return the dense FFT map ``M_β(α, γ)`` for every β.

    Phaser source: ``DoRfftStuff`` (FastRot.cc:19-88), but tensor-batched
    over β and with a single 2D ``torch.fft.ifft2`` per β instead of a
    cctbx ``real_to_complex_3d`` of shape ``(1, amax, amax)`` — they
    produce equivalent dense (α, γ) grids.

    Parameters
    ----------
    xi_lmn : torch.Tensor (complex), shape (L, 2L-1, 2L-1)
        Cross-correlation coefficients with l ∈ [0, L), m, n ∈ [-(L-1), L-1].
    betas : torch.Tensor (real), shape (n_beta,)
        β values in radians.
    fft_size : int
        Fixed FFT grid size N. The map ``M_β`` will be ``(N, N)`` for every β,
        indexed as ``M[k', l'] = RF(2π k'/N, β, 2π l'/N) / N²``.

    Returns
    -------
    M : torch.Tensor (complex), shape (n_beta, fft_size, fft_size)
        The dense maps. Use bilinear interpolation in the (α, γ) plane to
        evaluate at non-grid points.
    """
    L = xi_lmn.shape[0]
    n_beta = betas.shape[0]
    device = xi_lmn.device

    # 1. Per-β Wigner-d contraction: S[k, m1+L-1, m2+L-1] = Σ_l ξ d^l_{m1,m2}(β_k).
    S = wigner_contraction_per_beta(xi_lmn, betas)   # (n_beta, 2L-1, 2L-1)

    # 2. Place S into the (fft_size, fft_size) Fourier grid by FFT-frequency
    #    mapping: m → (m mod N), n → (n mod N). For m, n ∈ [-(L-1), L-1] and
    #    N >> 2L-1, this puts negative frequencies at the high end of each axis.
    if fft_size < 2 * L - 1:
        raise ValueError(
            f"fft_size={fft_size} must be >= 2L-1={2*L-1} to avoid aliasing"
        )
    pad = torch.zeros(
        (n_beta, fft_size, fft_size), dtype=S.dtype, device=device,
    )
    m_vals = torch.arange(-(L - 1), L, device=device)
    idx = (m_vals % fft_size).to(torch.int64)
    pad[:, idx.unsqueeze(1), idx.unsqueeze(0)] = S

    # 3. Forward 2D FFT — torch convention:
    #    fft2(X)[k, l] = Σ_{m, n} X[m, n] · exp(-2πi (m·k/N + n·l/M))
    #    which gives M[k, l] = RF(α = 2π·k/N, β, γ = 2π·l/N) directly, with
    #    Edmonds D^l_{m,n} = exp(-imα) d^l_{m,n}(β) exp(-inγ). Using ifft2
    #    here (the cctbx default in Phaser's pipeline) would give RF at
    #    (-α, -γ) which then requires negating alpha_frac/gamma_frac when
    #    recording sample Euler angles — Phaser's FastRot.cc:153 does this
    #    explicit ``-360 * alpha`` flip. We do the equivalent by using fft2
    #    so the Euler labels are already in the right sign.
    M = torch.fft.fft2(pad, dim=(-2, -1))
    return M


def _build_beta_grid(grid_sampling_deg: float) -> Tuple[torch.Tensor, int]:
    """Return (β_grid in radians, bmax). β = b · Δ for b ∈ [0, bmax)."""
    bmax = int(math.ceil(180.0 / grid_sampling_deg))
    if bmax < 1:
        raise ValueError(f"grid_sampling_deg={grid_sampling_deg} too coarse")
    b = torch.arange(bmax, dtype=torch.float64)
    betas_rad = b * grid_sampling_deg * (math.pi / 180.0)
    return betas_rad, bmax


# Module-level memo for the data-independent sample list, keyed on
# (grid_sampling_deg, device-str, dtype). The list depends only on geometry, so
# repeat FRF calls at the same grid reuse it; a single cold call still pays the
# (now vectorised, CPU-built) construction once.
_SAMPLE_LIST_CACHE: dict = {}


def build_adaptive_sample_list(
    grid_sampling_deg: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-β (α, γ) sample list.

    Phaser source: ``SiteListAng::allocate_memory`` (FastRot.cc:169-262).
    Returns (in radians):
        alphas    : (N_samples,)         α value per sample
        betas     : (N_samples,)         β value per sample
        gammas    : (N_samples,)         γ value per sample
        beta_starts: (bmax + 1,) int64    slice [beta_starts[b]:beta_starts[b+1]]
                                          is the samples at β = b · Δ
        beta_grid : (bmax,)              the β values in radians

    The construction is purely geometric (independent of the data / ξ), so it is
    memoised on ``(grid_sampling_deg, device, dtype)``. It is also built on the
    CPU — the per-β tensors are tiny and the work is launch-latency-bound, so a
    single host-side build + one device transfer is far cheaper than thousands of
    small CUDA kernels. The per-key dedup is vectorised (no ``.tolist()`` /
    Python scan), so there is no host sync inside the loop.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    cache_key = (float(grid_sampling_deg), str(device), dtype)
    cached = _SAMPLE_LIST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bmax = int(math.ceil(180.0 / grid_sampling_deg))
    if bmax < 1:
        raise ValueError(f"grid_sampling_deg={grid_sampling_deg} too coarse")
    cpu = torch.device("cpu")

    alphas_list: List[torch.Tensor] = []
    gammas_list: List[torch.Tensor] = []
    betas_list: List[torch.Tensor] = []
    beta_starts: List[int] = [0]

    deg2rad = math.pi / 180.0
    for b in range(bmax):
        beta_rad = b * grid_sampling_deg * deg2rad        # plain math: no sync
        cosb = math.cos(beta_rad / 2.0)
        sinb = math.sin(beta_rad / 2.0)
        # Truncation toward zero, with NO clamp to a minimum of 1 -- Phaser
        # has none (FastRot.cc:214-215). Near beta = 180 deg, cos(beta/2) drives
        # pmax to 0 and Phaser's `for (p=0; p<pmax; p++)` body never runs, so
        # the beta section is genuinely empty. Clamping to 1 invents a section
        # Phaser does not sample.
        pmax = int(720.0 / grid_sampling_deg * cosb)
        qmax = int(360.0 / grid_sampling_deg * sinb)
        if pmax == 0 or (b > 0 and qmax == 0):
            beta_starts.append(beta_starts[-1])
            continue

        if b == 0:
            # β=0: only α = γ = p/pmax for p < pmax/2 (FastRot.cc:189-207).
            p_idx = torch.arange(pmax, device=cpu)
            p_ratio = p_idx.to(torch.float64) / pmax
            keep = p_ratio < 0.5
            p_ratio = p_ratio[keep]
            alpha_frac = p_ratio
            gamma_frac = p_ratio
        else:
            # (p, q) ∈ [0, pmax) × [0, qmax), mapped to (α, γ) via
            # FastRot.cc:216-219 — with the negative branch for γ when
            # p_ratio < q_ratio (gives γ ∈ [0, 1) without negative values).
            p_idx = torch.arange(pmax, device=cpu)
            q_idx = torch.arange(qmax, device=cpu)
            p_ratio = (p_idx.to(torch.float64) / pmax).unsqueeze(1)  # (pmax, 1)
            q_ratio = (q_idx.to(torch.float64) / qmax).unsqueeze(0)  # (1, qmax)
            alpha_frac = torch.fmod(p_ratio + q_ratio, 1.0)
            diff = p_ratio - q_ratio
            gamma_frac = torch.where(
                diff >= 0.0,
                torch.fmod(diff, 1.0),
                1.0 - torch.fmod(-diff, 1.0),
            )
            alpha_frac = alpha_frac.reshape(-1)
            gamma_frac = gamma_frac.reshape(-1)

            # Dedup: when p ≥ pmax/2, the (p, q) → (α, γ) map can collide
            # with (p − pmax/2, q') for some q' (FastRot.cc:222-241). Keep the
            # first occurrence (in original order) of each rounded (α, γ) key.
            # Vectorised first-occurrence: stable-sort the unique-group labels,
            # mark group boundaries, scatter back — same kept set & order as the
            # original dict scan, but no host sync / Python loop.
            # Hash the two rounded fracs (each in [0, 1e6]) into one int64 so we
            # can use the fast 1-D unique instead of a 2-D row lexsort.
            a_round = (alpha_frac * 1_000_000).round().to(torch.int64)
            g_round = (gamma_frac * 1_000_000).round().to(torch.int64)
            key_hash = a_round * 1_000_001 + g_round
            _, uniq_idx = torch.unique(key_hash, return_inverse=True)
            n = uniq_idx.shape[0]
            order = torch.argsort(uniq_idx, stable=True)
            sorted_u = uniq_idx[order]
            first_in_sorted = torch.ones(n, dtype=torch.bool, device=cpu)
            first_in_sorted[1:] = sorted_u[1:] != sorted_u[:-1]
            keep_mask = torch.zeros(n, dtype=torch.bool, device=cpu)
            keep_mask[order[first_in_sorted]] = True
            alpha_frac = alpha_frac[keep_mask]
            gamma_frac = gamma_frac[keep_mask]

        n_this = alpha_frac.shape[0]
        alphas_list.append((alpha_frac * (2.0 * math.pi)).to(dtype))
        gammas_list.append((gamma_frac * (2.0 * math.pi)).to(dtype))
        betas_list.append(torch.full((n_this,), beta_rad, dtype=dtype, device=cpu))
        beta_starts.append(beta_starts[-1] + n_this)

    # Concatenate on CPU, then move to the target device in one transfer each.
    alphas = torch.cat(alphas_list).to(device)
    gammas = torch.cat(gammas_list).to(device)
    betas_flat = torch.cat(betas_list).to(device)
    beta_starts_t = torch.tensor(beta_starts, dtype=torch.int64, device=device)
    b = torch.arange(bmax, dtype=torch.float64, device=cpu)
    betas_rad = (b * grid_sampling_deg * deg2rad).to(device=device, dtype=dtype)

    result = (alphas, betas_flat, gammas, beta_starts_t, betas_rad)
    _SAMPLE_LIST_CACHE[cache_key] = result
    return result


def _bilinear_interp_periodic(
    M: torch.Tensor,  # (N, N) complex
    alpha_frac: torch.Tensor,  # (n,) in [0, 1)
    gamma_frac: torch.Tensor,  # (n,) in [0, 1)
) -> torch.Tensor:
    """Periodic bilinear interpolation of M at (alpha_frac · N, gamma_frac · N).

    Mirrors ``four_point_interpolation`` (FastRot.cc:146) on a periodic
    map: both axes wrap around modulo N.
    """
    N = M.shape[-1]
    af = (alpha_frac % 1.0) * N
    gf = (gamma_frac % 1.0) * N
    a0 = torch.floor(af).to(torch.int64) % N
    g0 = torch.floor(gf).to(torch.int64) % N
    a1 = (a0 + 1) % N
    g1 = (g0 + 1) % N
    da = (af - torch.floor(af)).to(M.real.dtype)
    dg = (gf - torch.floor(gf)).to(M.real.dtype)
    da_c = da.to(M.dtype)
    dg_c = dg.to(M.dtype)
    v00 = M[a0, g0]
    v01 = M[a0, g1]
    v10 = M[a1, g0]
    v11 = M[a1, g1]
    return (
        v00 * ((1 - da_c) * (1 - dg_c))
        + v01 * ((1 - da_c) * dg_c)
        + v10 * (da_c * (1 - dg_c))
        + v11 * (da_c * dg_c)
    )


def evaluate_rotation_function(
    xi_lmn: torch.Tensor,
    grid_sampling_deg: float = 2.0,
    fft_size: int = -1,
) -> AdaptiveRotationFunction:
    """Compute the rotation function on Phaser's adaptive SO(3) sample list.

    Phaser source: composition of ``get_FRF`` + ``allocate_memory`` +
    ``four_point_interpolation`` (FastRot.cc:90-262). Returns
    real-valued samples (the rotation function is real).

    Parameters
    ----------
    xi_lmn : torch.Tensor (complex), shape (L, 2L-1, 2L-1)
    grid_sampling_deg : float
        Phaser's ``grid_sampling`` keyword. β grid is uniform at this
        spacing; (α, γ) sample density per β follows pmax/qmax.
    fft_size : int, optional
        Dense FFT shape. Default: ``adjust_gridding(2·max(bmax, 2L-1), 5)``.
    """
    if xi_lmn.ndim != 3:
        raise ValueError(f"xi_lmn must be 3-D (L, 2L-1, 2L-1), got {tuple(xi_lmn.shape)}")
    L = xi_lmn.shape[0]
    device = xi_lmn.device
    real_dtype = (
        torch.float64
        if xi_lmn.dtype in (torch.complex128, torch.float64)
        else torch.float32
    )

    bmax = int(math.ceil(180.0 / grid_sampling_deg))
    if fft_size < 0:
        fft_size = adjust_gridding(2 * max(bmax, 2 * L - 1), max_prime=5)

    # 1. Build adaptive sample list (purely geometric — independent of xi).
    alphas, betas_flat, gammas, beta_starts, beta_grid = build_adaptive_sample_list(
        grid_sampling_deg, dtype=real_dtype, device=device,
    )

    # 2. Dense FFT map per β.
    M = build_dense_map_per_beta(xi_lmn, beta_grid, fft_size)  # (n_beta, N, N)

    # 3. Bilinear interp at each sample's (α, γ).
    values = torch.zeros(alphas.shape[0], dtype=real_dtype, device=device)
    n_beta = beta_grid.shape[0]
    for b in range(n_beta):
        i0, i1 = int(beta_starts[b].item()), int(beta_starts[b + 1].item())
        if i1 <= i0:
            continue
        af = alphas[i0:i1] / (2.0 * math.pi)
        gf = gammas[i0:i1] / (2.0 * math.pi)
        v_complex = _bilinear_interp_periodic(M[b], af, gf)
        # The rotation function is real; the imaginary residue is at the
        # numerical-noise level for a Hermitian-symmetric input ξ. Drop it.
        values[i0:i1] = v_complex.real

    return AdaptiveRotationFunction(
        alphas=alphas,
        betas=betas_flat,
        gammas=gammas,
        values=values,
        beta_starts=beta_starts,
        beta_grid=beta_grid,
        grid_sampling_deg=grid_sampling_deg,
    )
