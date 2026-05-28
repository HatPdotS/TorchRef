"""
Fast FFT-based translation search for molecular replacement.

Translation t shifts phase: F(hkl, t) = F(hkl) * exp(2*pi*i * hkl.t)
Correlation: C(t) = IFFT{ conj(F_obs) * F_calc }

This module provides efficient FFT-based translation search that finds the
optimal translation to position a model after rotation has been determined.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TranslationPeak:
    """
    Translation search peak.

    Attributes
    ----------
    translation : np.ndarray
        Fractional coordinates (3,).
    score : float
        Correlation score.
    sigma : float
        Z-score above mean.
    """
    translation: np.ndarray
    score: float
    sigma: float


def fft_translation_search(
    F_obs: np.ndarray,
    F_calc: np.ndarray,
    hkl: np.ndarray,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    n_peaks: int = 10,
    cluster_radius: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[TranslationPeak]]:
    """
    FFT-based translation search (vectorized).

    The translation function is:
        TF(t) = Re{ IFFT{ conj(F_obs) * F_calc } }

    This finds translation t such that F_calc shifted by t best matches F_obs.

    Parameters
    ----------
    F_obs : np.ndarray
        Observed structure factor amplitudes (or complex), shape (N,).
    F_calc : np.ndarray
        Calculated structure factors (complex), shape (N,).
    hkl : np.ndarray
        Miller indices, shape (N, 3).
    grid_shape : tuple, optional
        (Nx, Ny, Nz) grid size for FFT. If None, auto-computed from HKL range.
    n_peaks : int
        Number of peaks to return.
    cluster_radius : float
        Minimum fractional distance between peaks for clustering.

    Returns
    -------
    correlation_map : np.ndarray
        Full translation function, shape grid_shape.
    best_translation : np.ndarray
        Best translation in fractional coordinates, shape (3,).
    peaks : list
        Top peaks as TranslationPeak objects.

    Examples
    --------
    ::

        import numpy as np
        from torchref.alignment.translation import fft_translation_search

        # Known translation test
        hkl = np.array([[1,0,0], [0,1,0], [1,1,0], [0,0,1]])
        F_obs = np.array([1.0, 1.0, 1.0, 1.0])
        F_calc = np.exp(2j * np.pi * hkl @ [0.25, 0.0, 0.0])
        _, best, peaks = fft_translation_search(F_obs, F_calc, hkl)
        print(f'Recovered: {best}')  # Should be ~[0.25, 0, 0]
    """
    # Auto grid shape from HKL range
    if grid_shape is None:
        hkl_abs = np.abs(hkl).astype(int)
        grid_shape = tuple(2 * (hkl_abs[:, i].max() + 1) for i in range(3))

    Nx, Ny, Nz = grid_shape
    product_grid = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Standard translation function: TF(t) = Re{ IFFT{ conj(F_obs) * F_calc } }
    # This finds t such that F_calc(t) = F_calc * exp(2*pi*i*hkl.t) matches F_obs
    product = np.conj(F_obs) * F_calc

    # Place at HKL positions using vectorized add.at for accumulation
    hkl_int = hkl.astype(int)
    h_idx = hkl_int[:, 0] % Nx
    k_idx = hkl_int[:, 1] % Ny
    l_idx = hkl_int[:, 2] % Nz

    np.add.at(product_grid, (h_idx, k_idx, l_idx), product)

    # IFFT gives correlation at all translations
    correlation_map = np.fft.ifftn(product_grid).real

    # Find peaks
    peaks = find_translation_peaks(correlation_map, n_peaks, cluster_radius)
    best = peaks[0].translation if peaks else np.zeros(3)

    return correlation_map, best, peaks


def find_translation_peaks(
    correlation_map: np.ndarray,
    n_peaks: int = 10,
    cluster_radius: float = 0.05,
) -> List[TranslationPeak]:
    """
    Extract and cluster peaks from translation function.

    Parameters
    ----------
    correlation_map : np.ndarray
        Translation function values, shape (Nx, Ny, Nz).
    n_peaks : int
        Maximum number of peaks to return.
    cluster_radius : float
        Minimum fractional distance between peaks (periodic).

    Returns
    -------
    peaks : list
        List of TranslationPeak objects sorted by score.
    """
    Nx, Ny, Nz = correlation_map.shape
    mean_val = correlation_map.mean()
    std_val = correlation_map.std()

    if std_val < 1e-10:
        return []

    flat = correlation_map.flatten()
    sorted_idx = np.argsort(flat)[::-1]

    peaks = []
    used = []

    for idx in sorted_idx:
        if len(peaks) >= n_peaks:
            break

        pos_3d = np.unravel_index(idx, correlation_map.shape)
        trans = np.array([pos_3d[0] / Nx, pos_3d[1] / Ny, pos_3d[2] / Nz])
        score = flat[idx]
        sigma = (score - mean_val) / std_val

        # Check clustering - skip if too close to existing peak
        is_new = True
        for prev in used:
            diff = np.abs(trans - prev)
            diff = np.minimum(diff, 1 - diff)  # Periodic boundary
            if np.linalg.norm(diff) < cluster_radius:
                is_new = False
                break

        if is_new:
            peaks.append(TranslationPeak(trans, score, sigma))
            used.append(trans)

    return peaks


def fft_translation_search_torch(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    hkl: torch.Tensor,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, List[TranslationPeak]]:
    """
    Torch wrapper for fft_translation_search.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes (or complex).
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    hkl : torch.Tensor
        Miller indices.
    **kwargs
        Additional arguments passed to fft_translation_search.

    Returns
    -------
    correlation_map : np.ndarray
        Full translation function.
    best_translation : np.ndarray
        Best translation in fractional coordinates.
    peaks : list
        Top peaks as TranslationPeak objects.
    """
    return fft_translation_search(
        F_obs.detach().cpu().numpy(),
        F_calc.detach().cpu().numpy(),
        hkl.detach().cpu().numpy(),
        **kwargs,
    )


def apply_translation_to_fcalc(
    F_calc: np.ndarray,
    hkl: np.ndarray,
    translation_frac: np.ndarray,
) -> np.ndarray:
    """
    Apply translation phase shift to calculated structure factors.

    F(hkl, t) = F(hkl) * exp(2*pi*i * hkl.t)

    Parameters
    ----------
    F_calc : np.ndarray
        Calculated structure factors (complex), shape (N,).
    hkl : np.ndarray
        Miller indices, shape (N, 3).
    translation_frac : np.ndarray
        Translation in fractional coordinates, shape (3,).

    Returns
    -------
    F_calc_shifted : np.ndarray
        Phase-shifted structure factors, shape (N,).
    """
    phase_shift = 2 * np.pi * (hkl @ translation_frac)
    return F_calc * np.exp(1j * phase_shift)


def amplitude_translation_search(
    F_obs: torch.Tensor,
    interpolator,
    R_rotation: torch.Tensor,
    hkl: torch.Tensor,
    spacegroup,
    real_cell,
    grid_steps: int = 16,
    n_peaks: int = 20,
    cluster_radius: float = 0.05,
    batch_size: int = 256,
    use_e_values: bool = True,
    n_shells: int = 20,
    precomputed_G: Optional[torch.Tensor] = None,
    precomputed_h_R: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, List[TranslationPeak]]:
    """
    Coarse-grid translation search via |F|²-correlation.

    For each candidate fractional translation `t` on a `grid_steps`³ grid in
    `[0, 1)³`, scores the model at the current rotation translated by `t`
    against the observed amplitudes by Pearson correlation of `|F_obs|²` and
    `|F_calc(h, t)|²`. The structure-factor sum uses the spacegroup symmetry
    expansion

        F_calc(h, t) = Σ_i G_i(h) · exp(2πi (h R_i) · t)
        G_i(h) = exp(2πi h · t_i) · F_p1(h R_i)

    with `F_p1(h R_i)` looked up via the supplied interpolator at the rotation
    already applied to the model. The `G_i` factors are computed once; only the
    phase exponential changes per candidate, so the scan is efficient.

    Parameters
    ----------
    F_obs : torch.Tensor, shape (N,)
        Observed amplitudes (complex inputs are coerced to |·|).
    interpolator : LattmanLoveInterpolator
        Provides `evaluate(R, hkl, real_cell, return_amplitude=False)`.
    R_rotation : torch.Tensor, shape (3, 3)
        Rotation that has been applied to the model coordinates.
    hkl : torch.Tensor, shape (N, 3)
        Integer Miller indices of the observed reflections.
    spacegroup : SpaceGroup
        Provides `matrices` and `translations`.
    real_cell : Cell
        Real crystal cell.
    grid_steps : int, default 16
        Per-axis grid resolution. Total candidates = grid_steps³.
    n_peaks : int, default 20
        Number of peaks returned (after clustering).
    cluster_radius : float, default 0.05
        Minimum fractional separation between returned peaks.
    batch_size : int, default 256
        Number of candidate translations evaluated per inner batch.
    use_e_values : bool, default True
        Normalize `|F_obs|` and `|F_calc(h, t)|` per resolution shell to unit
        Wilson variance (E-values) before correlating. This removes the
        resolution-dependent envelope mismatch between real F_obs (with bulk
        solvent + thermal falloff) and a model that doesn't model these — a
        per-shell mean subtraction in the Pearson correlation alone doesn't
        cover it because the falloff is multiplicative, not additive.
    n_shells : int, default 20
        Number of equal-count radial shells used by `use_e_values`.

    Returns
    -------
    correlation_map : np.ndarray, shape (grid_steps, grid_steps, grid_steps)
        Pearson correlation of |F_obs|² and |F_calc(t)|² at each grid point.
    best_translation : np.ndarray, shape (3,)
        Top-scoring fractional translation.
    peaks : list of TranslationPeak
        Top-`n_peaks` peaks sorted by descending correlation.
    """
    device = getattr(interpolator, "device", hkl.device)
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    F_obs_t = F_obs.detach().to(device)
    if F_obs_t.is_complex():
        F_obs_t = F_obs_t.abs()
    F_obs_t = F_obs_t.to(real_dtype)

    hkl_t = hkl.detach().to(device).to(real_dtype)        # (N, 3)

    # Precompute per-shell normalisation if requested. We bin reflections by
    # |s| into n_shells equal-count shells and normalise F → F / sqrt(<F²>_shell)
    # (Wilson E-value). The same shell norm is applied to F_calc(t) inside the
    # batch loop. This makes the Pearson correlation a Patterson-style
    # correlation of "E²−1" — robust to bulk-solvent / B-factor mismatch.
    if use_e_values:
        rec_basis_real = real_cell.reciprocal_basis_matrix.to(device).to(real_dtype)
        s_mag = (hkl_t @ rec_basis_real).norm(dim=-1)
        order = torch.argsort(s_mag)
        shell_idx = torch.zeros_like(s_mag, dtype=torch.int64)
        chunk = s_mag.numel() // max(n_shells, 1)
        for k in range(n_shells):
            a = k * chunk
            b = (k + 1) * chunk if k < n_shells - 1 else s_mag.numel()
            shell_idx[order[a:b]] = k
        shell_norm_obs = torch.zeros(n_shells, dtype=real_dtype, device=device)
        for k in range(n_shells):
            m = shell_idx == k
            if m.any():
                shell_norm_obs[k] = (F_obs_t[m] ** 2).mean().clamp(min=1e-30).sqrt()
        E_obs = F_obs_t / shell_norm_obs[shell_idx]
        F_obs2 = E_obs * E_obs
    else:
        shell_idx = None
        F_obs2 = F_obs_t * F_obs_t
    F_obs2_centered = F_obs2 - F_obs2.mean()

    # Pre-compute G_i(h) = exp(2πi h·t_i) · F_p1(h R_i)  (or reuse caller's)
    two_pi_i = 2j * torch.pi
    if precomputed_G is not None and precomputed_h_R is not None:
        G = precomputed_G.to(device).to(complex_dtype)
        h_R = precomputed_h_R.to(device).to(real_dtype)
    else:
        G, h_R = precompute_G_for_rotation(
            interpolator, R_rotation, hkl, spacegroup, real_cell, device=device,
        )

    # Crowther–Blow FFT translation function (Acta Cryst. B23 (1967) 544).
    # The grid-evaluated score
    #     num(t) = Σ_h F_obs²_centered(h) · |F_calc(h, t)|²
    # expands as
    #     num(t) = Σ_{i,j} [Σ_h F_obs²_c(h) · G_i*(h) · G_j(h)]
    #                       · exp(2πi · (h·R_j − h·R_i) · t)
    # and on a regular fractional t-grid t = (jx, jy, jz) / G this is exactly
    # an inverse DFT of the bracketed coefficients accumulated onto a 3-D
    # reciprocal grid at integer indices (h·R_j − h·R_i) mod G.
    #
    # We accumulate two such reciprocal grids in one sym-op pass:
    #   W_num : weight per h = F_obs²_centered(h)  → num(t)
    #   W_den : weight per h = 1                   → Σ_h |F_calc(h,t)|²
    # Score(t) = num(t) / Σ_h|F_calc(h,t)|²  — a per-t scale-normalised
    # Pearson proxy (Phaser's TF uses the full Pearson denominator; ours
    # uses the same scaling that the previous separable-phase code applied
    # via explicit per-t centering, achieved here without materialising
    # |F_calc(h,t)|² per t-point).
    #
    # One IFFT pair replaces G³ grid evaluations — for our defaults this is
    # ~5000× less arithmetic than the separable-phase scoring it supersedes,
    # and orders of magnitude less than the original explicit grid loop.
    S_eff, N_eff = G.shape
    h_R_int = h_R.round().to(torch.int64)                        # (S, N, 3)
    F_obs2_c_complex = F_obs2_centered.to(complex_dtype)         # (N,)
    ones_complex = torch.ones(N_eff, dtype=complex_dtype, device=device)

    W_num_flat = torch.zeros(
        grid_steps ** 3, dtype=complex_dtype, device=device,
    )
    W_den_flat = torch.zeros(
        grid_steps ** 3, dtype=complex_dtype, device=device,
    )
    G_stride_xy = grid_steps * grid_steps
    for i in range(S_eff):
        Gi_conj = G[i].conj()                                    # (N,)
        pair = Gi_conj.view(1, -1) * G                           # (S, N)
        coeff_num = F_obs2_c_complex.view(1, -1) * pair          # (S, N)
        coeff_den = ones_complex.view(1, -1) * pair              # (S, N)
        dh = (h_R_int - h_R_int[i:i + 1]) % grid_steps           # (S, N, 3)
        flat = (dh[..., 0] * G_stride_xy
                + dh[..., 1] * grid_steps + dh[..., 2])          # (S, N)
        flat_flat = flat.reshape(-1)
        W_num_flat.index_add_(0, flat_flat, coeff_num.reshape(-1))
        W_den_flat.index_add_(0, flat_flat, coeff_den.reshape(-1))

    W_num = W_num_flat.view(grid_steps, grid_steps, grid_steps)
    W_den = W_den_flat.view(grid_steps, grid_steps, grid_steps)
    # IFFT scales by 1/G³; undo so values are raw integrals.
    num_t = (torch.fft.ifftn(W_num, dim=(0, 1, 2)).real
             * (grid_steps ** 3)).to(real_dtype)
    den_t = (torch.fft.ifftn(W_den, dim=(0, 1, 2)).real
             * (grid_steps ** 3)).to(real_dtype)
    corr_map = num_t / den_t.clamp(min=1e-30)
    corr_map_np = corr_map.detach().cpu().numpy().astype(np.float32)
    peaks = find_translation_peaks(corr_map_np, n_peaks=n_peaks,
                                    cluster_radius=cluster_radius)
    best = peaks[0].translation if peaks else np.zeros(3)
    return corr_map_np, best, peaks


def llg_translation_rescore(
    F_obs: torch.Tensor,
    hkl: torch.Tensor,
    centric: torch.Tensor,
    shell_idx: torch.Tensor,
    n_shells: int,
    G: torch.Tensor,
    h_R: torch.Tensor,
    t_candidates: torch.Tensor,
    sigma_a: torch.Tensor,
    interp_var: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-translation Rice / Woolfson log-likelihood, using the symmetry-summed
    interpolator contributions ``G`` (Phaser EM_search analogue) and a fixed
    per-shell σA.

    For each candidate t:
        F_calc(h, t) = Σ_i G_i(h) · exp(2πi (h R_i) · t)
        E_calc(h, t) = |F_calc(h, t)| / sqrt(<F²>_per_shell)
        LLG(t) = Σ_shell [LL_Rice(E_obs, D·E_calc, var) − LL_Wilson(E_obs)]
        where var = (1 − D²) + interp_var.

    Phase B alignment likelihood-TF. Replaces the |F|² Pearson correlation
    in `amplitude_translation_search` as the scoring rule when the caller
    re-ranks the FFT-cheap pre-filter peaks.

    Parameters
    ----------
    F_obs : (N,) real
    hkl   : (N, 3) — unused here but kept for symmetry with the rest of the
            module (and future extension to per-h variance models).
    centric : (N,) bool
    shell_idx : (N,) int64 — same binning as used to fit sigma_a / interp_var.
    n_shells : int
    G : (S, N) complex — per-sym F_p1 contributions × per-sym translation phase
                          (output of `precompute_G_for_rotation`).
    h_R : (S, N, 3) — per-sym rotated reciprocal indices.
    t_candidates : (K, 3) fractional translations to score.
    sigma_a : (n_shells,) — fixed per-shell σA (shared across candidates).
    interp_var : (N,) optional — per-reflection variance inflation.

    Returns
    -------
    llg : (K,) torch.Tensor — log-likelihood gain per candidate.
    """
    from .distributions import rice_log_likelihood, woolfson_log_likelihood

    device = G.device
    real_dtype = torch.float64
    complex_dtype = G.dtype

    K = t_candidates.shape[0]
    S, N = G.shape

    t_cand = t_candidates.to(device).to(real_dtype)               # (K, 3)
    # Phase factor for each (k, i, n): exp(2πi · (h_R[i, n] · t[k]))
    phase_arg = torch.einsum("ind,kd->kin", h_R.to(real_dtype), t_cand)
    phase = torch.exp(2j * torch.pi * phase_arg.to(complex_dtype))  # (K, S, N)
    # F_calc(k, n) = Σ_i G[i, n] · phase[k, i, n]
    Fc_complex = (G.view(1, S, N) * phase).sum(dim=1)             # (K, N)
    F_calc = Fc_complex.abs().to(real_dtype)                       # (K, N)

    # Per-shell E normalisation of F_calc across the K-batch.
    shell_idx_l = shell_idx.to(device).long()
    cnt = torch.bincount(shell_idx_l, minlength=n_shells).to(real_dtype)
    shell_idx_k = shell_idx_l.view(1, -1).expand(K, N)
    F2 = F_calc * F_calc
    sum_per_shell = torch.zeros((K, n_shells), dtype=real_dtype, device=device)
    sum_per_shell.scatter_add_(1, shell_idx_k, F2)
    mean_per_shell = (sum_per_shell / cnt.clamp(min=1.0).unsqueeze(0)).clamp(min=1e-30)
    norm_per_refl = mean_per_shell.sqrt().gather(1, shell_idx_k)  # (K, N)
    E_calc = F_calc / norm_per_refl                                 # (K, N)

    F_obs_t = F_obs.to(device).to(real_dtype)
    sum_F_obs2 = torch.zeros(n_shells, dtype=real_dtype, device=device)
    sum_F_obs2.scatter_add_(0, shell_idx_l, F_obs_t * F_obs_t)
    mean_F_obs2 = (sum_F_obs2 / cnt.clamp(min=1.0)).clamp(min=1e-30)
    E_obs = F_obs_t / mean_F_obs2.sqrt().index_select(0, shell_idx_l)

    sigma_a_d = sigma_a.to(device).to(real_dtype)                  # (n_shells,)
    D_per_refl = sigma_a_d.index_select(0, shell_idx_l)            # (N,)
    var_d = (1.0 - D_per_refl * D_per_refl).clamp(min=1e-4)        # (N,)
    if interp_var is not None:
        var_per_refl = (var_d + interp_var.to(device).to(real_dtype)).clamp(min=1e-4)
    else:
        var_per_refl = var_d

    F_mean = D_per_refl.view(1, N) * E_calc                        # (K, N)
    var_full = var_per_refl.view(1, N).expand(K, N)
    E_obs_full = E_obs.view(1, N).expand(K, N)
    cent_full = centric.to(device).to(torch.bool).view(1, N)

    ll_acent = rice_log_likelihood(E_obs_full, F_mean, var_full)
    ll_cent = woolfson_log_likelihood(E_obs_full, F_mean, var_full)
    ll = torch.where(cent_full, ll_cent, ll_acent)                 # (K, N)

    # Wilson reference (data only): F_mean = 0, var = 1.
    var0 = torch.ones_like(E_obs)
    F_mean0 = torch.zeros_like(E_obs)
    ll_wil_acent = rice_log_likelihood(E_obs, F_mean0, var0)
    ll_wil_cent = woolfson_log_likelihood(E_obs, F_mean0, var0)
    ll_wil_per_refl = torch.where(centric.to(device).to(torch.bool),
                                   ll_wil_cent, ll_wil_acent)
    ll_wil_total = ll_wil_per_refl.sum()

    return ll.sum(dim=1) - ll_wil_total                            # (K,)


def precompute_G_for_rotation(
    interpolator,
    R_rotation: torch.Tensor,
    hkl: torch.Tensor,
    spacegroup,
    real_cell,
    device=None,
):
    """
    Pre-compute per-symmetry F_asu contributions `G_i(h)` for a fixed rotation.

    These are the only inputs that depend on `R_rotation` (and therefore on
    expensive interpolator/model-forward evaluations). Passing the result
    into `amplitude_translation_search` and `local_translation_refine` lets
    them share a single set of (n_sym) model evaluations across the coarse
    TF and the fine refinement, instead of recomputing each call.

    Returns
    -------
    G : (S, N) complex128
    h_R : (S, N, 3) float64
    """
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    if device is None:
        device = getattr(interpolator, "device", hkl.device)

    hkl_t = hkl.detach().to(device).to(real_dtype)
    sym_R = spacegroup.matrices.detach().to(device).to(real_dtype)
    sym_t = spacegroup.translations.detach().to(device).to(real_dtype)
    S = sym_R.shape[0]
    N = hkl_t.shape[0]
    R_rot = R_rotation.detach().to(device).to(real_dtype)

    # Batched: h_R[i, n, d] = Σ_e hkl[n, e] · sym_R[i, e, d]
    h_R = torch.einsum("ne,ied->ind", hkl_t, sym_R)              # (S, N, 3)
    # Per-sym-op translation phase: exp(2πi · h · t_i)
    two_pi_i = 2j * torch.pi
    phase_arg = torch.einsum("ne,ie->in", hkl_t, sym_t)          # (S, N)
    phase = torch.exp(two_pi_i * phase_arg.to(complex_dtype))    # (S, N)

    # One interpolator.evaluate over all (S × N) rotated indices: lets the
    # backend do a single grid_sample instead of S sequential ones.
    h_R_flat = h_R.reshape(-1, 3)                                # (S·N, 3)
    F_flat = interpolator.evaluate(
        R_rot, h_R_flat, real_cell, return_amplitude=False,
    )
    F_all = F_flat.reshape(S, N).to(complex_dtype)               # (S, N)

    G = F_all * phase
    return G, h_R


def local_translation_refine(
    F_obs: torch.Tensor,
    interpolator,
    R_rotation: torch.Tensor,
    hkl: torch.Tensor,
    spacegroup,
    real_cell,
    t_init: torch.Tensor,
    radius: float = 0.06,
    grid_steps: int = 13,
    n_refinement_passes: int = 2,
    batch_size: int = 1024,
    precomputed_G: Optional[torch.Tensor] = None,
    precomputed_h_R: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Fine-grid Patterson translation refinement around ``t_init``.

    For each candidate ``t`` in a `grid_steps`³ cubic grid of half-width
    ``radius`` centered on ``t_init`` (fractional), computes |F_calc(h, t)|²
    via the symmetry expansion and the analytical-scale R-factor
        R(t) = Σ ||F_obs| − k·|F_calc(t)|| / Σ |F_obs|
        k(t) = Σ |F_obs|·|F_calc(t)| / Σ |F_calc(t)|²
    against `F_obs`. Returns the (t, R) at the minimum.

    Use `n_refinement_passes > 1` to do a multi-pass zoom: each pass shrinks
    the radius by `grid_steps/2` and re-centers on the previous best. For
    `radius=0.06, grid_steps=13, n_refinement_passes=2`, the final fractional
    resolution is ~0.005 (≈0.3 Å for a 60 Å cell).

    The analytical-scale R-factor uses a single global scale; it is not the
    same number a full crystallographic Scaler would return, but its
    *minimum location* is robust because both numerator and denominator share
    the same per-shell envelope. Use a full Scaler to compute the final
    R-work after this routine selects (R, t).
    """
    device = getattr(interpolator, "device", hkl.device)
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    F_obs_t = F_obs.detach().to(device)
    if F_obs_t.is_complex():
        F_obs_t = F_obs_t.abs()
    F_obs_t = F_obs_t.to(real_dtype)
    F_obs_sum = F_obs_t.sum().clamp(min=1e-30)

    two_pi_i = 2j * torch.pi
    if precomputed_G is not None and precomputed_h_R is not None:
        G = precomputed_G.to(device).to(complex_dtype)
        h_R = precomputed_h_R.to(device).to(real_dtype)
    else:
        G, h_R = precompute_G_for_rotation(
            interpolator, R_rotation, hkl, spacegroup, real_cell, device=device,
        )

    # Adapt batch_size to keep the largest inner einsum tensor under ~250 MB
    # of complex128. The (S, B, N) phase tensor is the offender:
    # S × B × N × 16 bytes ≤ 2.5e8 → B ≤ 2.5e8 / (16 × S × N).
    S_eff, N_eff = G.shape
    safe_b = max(8, int(2.5e8 / (16.0 * max(S_eff, 1) * max(N_eff, 1))))
    batch_size = min(batch_size, safe_b)

    # Crowther–Blow FFT refinement on a fine grid around t_init.
    # Bake t_init into G as a per-h_R phase factor, then the IFFT trick from
    # `amplitude_translation_search` works on the offset grid Δt with the
    # same (num/den) Pearson-proxy scoring. The previous nested-grid Python
    # evaluation paid O(G³ · N · S) Bessel/exp/einsum per call; this pays
    # one IFFT pair on G_fft³ + an O(N · S) F_calc evaluation at the final t.
    S, N = G.shape
    t_init_t = torch.as_tensor(t_init, dtype=real_dtype, device=device)
    h_R_int = h_R.round().to(torch.int64)                                # (S, N, 3)

    # Bake t_init into G:
    phase_init = torch.exp(
        two_pi_i * torch.einsum("snd,d->sn", h_R, t_init_t).to(complex_dtype)
    )                                                                    # (S, N)
    G_shifted = G * phase_init                                           # (S, N)

    # Pick G_fft so the IFFT spacing matches the requested fine grid:
    # spacing = 2·radius / (grid_steps − 1), G_fft = round(1 / spacing).
    # Cap at 128 to bound memory (128³ complex128 ≈ 32 MB).
    desired_spacing = max(2.0 * float(radius) / max(grid_steps - 1, 1), 1e-6)
    G_fft = max(grid_steps, int(round(1.0 / desired_spacing)))
    G_fft = min(G_fft, 128)
    half_window = max(1, int(round(float(radius) * G_fft)))

    F_obs2 = (F_obs_t * F_obs_t).to(real_dtype)
    F_obs2_centered = F_obs2 - F_obs2.mean()
    F_obs2_c_complex = F_obs2_centered.to(complex_dtype)
    ones_complex = torch.ones(N, dtype=complex_dtype, device=device)

    W_num_flat = torch.zeros(G_fft ** 3, dtype=complex_dtype, device=device)
    W_den_flat = torch.zeros(G_fft ** 3, dtype=complex_dtype, device=device)
    G_stride_xy = G_fft * G_fft
    for i in range(S):
        Gi_conj = G_shifted[i].conj()
        pair = Gi_conj.view(1, -1) * G_shifted                           # (S, N)
        coeff_num = F_obs2_c_complex.view(1, -1) * pair
        coeff_den = ones_complex.view(1, -1) * pair
        dh = (h_R_int - h_R_int[i:i + 1]) % G_fft                        # (S, N, 3)
        flat = (dh[..., 0] * G_stride_xy
                + dh[..., 1] * G_fft + dh[..., 2])                       # (S, N)
        flat_flat = flat.reshape(-1)
        W_num_flat.index_add_(0, flat_flat, coeff_num.reshape(-1))
        W_den_flat.index_add_(0, flat_flat, coeff_den.reshape(-1))

    W_num = W_num_flat.view(G_fft, G_fft, G_fft)
    W_den = W_den_flat.view(G_fft, G_fft, G_fft)
    num_t = torch.fft.ifftn(W_num, dim=(0, 1, 2)).real * (G_fft ** 3)
    den_t = torch.fft.ifftn(W_den, dim=(0, 1, 2)).real * (G_fft ** 3)
    score = num_t / den_t.clamp(min=1e-30)

    # Roll so the (Δt = 0) cell sits in the centre of a (2·half_window+1)
    # window, then look for the maximum within the radius-sphere.
    score_rolled = torch.roll(
        score, shifts=(half_window, half_window, half_window), dims=(0, 1, 2),
    )
    w = 2 * half_window + 1
    score_window = score_rolled[:w, :w, :w]
    idx_flat = int(score_window.argmax().item())
    jx = idx_flat // (w * w)
    rem = idx_flat % (w * w)
    jy = rem // w
    jz = rem % w
    Delta_t = torch.tensor(
        [(jx - half_window) / G_fft,
         (jy - half_window) / G_fft,
         (jz - half_window) / G_fft],
        dtype=real_dtype, device=device,
    )
    best_t = t_init_t + Delta_t

    # Compute the analytical-scale R-factor at best_t (one t evaluation),
    # which is what the caller uses to rank rotation × translation
    # candidates. The local-refine grid search above optimised the
    # FFT-scored Pearson proxy; analytical R is monotonically related on
    # this neighbourhood so the choice of which fine-grid maximum to
    # commit to is preserved.
    phase_best = torch.exp(
        two_pi_i * torch.einsum("snd,d->sn", h_R, best_t).to(complex_dtype)
    )                                                                    # (S, N)
    F_calc_best = (G * phase_best).sum(dim=0)                            # (N,) complex
    F_c_abs = F_calc_best.abs().to(real_dtype)
    num_a = (F_obs_t * F_c_abs).sum()
    den_a = (F_c_abs ** 2).sum().clamp(min=1e-30)
    k = num_a / den_a
    best_R = float(
        ((F_obs_t - k * F_c_abs).abs().sum() / F_obs_sum).item()
    )
    # Unused: `n_refinement_passes`, `batch_size` kept in signature for
    # back-compat with callers passing them.
    _ = n_refinement_passes
    _ = batch_size

    return best_t.cpu(), best_R


def local_rotation_translation_refine(
    F_obs: torch.Tensor,
    interpolator,
    R_initial: torch.Tensor,
    t_initial: torch.Tensor,
    hkl: torch.Tensor,
    spacegroup,
    real_cell,
    centric: torch.Tensor,
    n_shells: int = 15,
    rotation_grid_steps: int = 5,
    rotation_radius_rad: float = 0.04,
    translation_grid_steps: int = 9,
    translation_radius_frac: float = 0.02,
    batch_size: int = 1024,
    verbose: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Joint (R, t) fine-grid refinement scored by the Sim MLRF log-likelihood
    gain (LLG) — fully scale-invariant.

    Scale invariance: F_obs is shell-normalised to E-values (Wilson
    statistics per resolution shell) and F_calc(R, t) is shell-normalised
    per-candidate. The LLG is then a per-shell σA fit + sum of Rice
    (acentric) / Woolfson (centric) log-likelihoods — none of which depend
    on the absolute magnitude of either F_obs or F_calc.

    Procedure:
    1. Build a `rotation_grid_steps³` cubic grid of small rotation
       perturbations in (Δα, Δβ, Δγ) around `R_initial`, parametrised as
       axis-angle rotations of magnitude up to `rotation_radius_rad`.
    2. For each R candidate:
       - Pre-compute the per-symmetry F_asu via `interpolator.evaluate(R, …)`
         (the expensive step — one model.forward per sym op per R).
       - Run an analytical inner translation grid of
         `translation_grid_steps³` candidates around `t_initial`. Within
         this inner loop, only phase factors change (cheap).
       - Pick the inner-best t (by an analytical R-factor proxy — fast).
    3. For each (R, best-inner-t) pair, evaluate the **full** Sim MLRF LLG
       (per-shell σA fit). Pick the global best by LLG.

    Returns
    -------
    R_best : torch.Tensor (3, 3)
        Refined rotation = R_initial @ R_perturb_best (column-vector form).
    t_best : torch.Tensor (3,)
        Refined fractional translation.
    llg_best : float
        Sim MLRF LLG at the returned (R_best, t_best).
    """
    from torchref.alignment.ml_rotation import llg_for_rotation_batch
    device = getattr(interpolator, "device", hkl.device)
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    two_pi_i = 2j * torch.pi

    F_obs_t = F_obs.detach().to(device)
    if F_obs_t.is_complex():
        F_obs_t = F_obs_t.abs()
    F_obs_t = F_obs_t.to(real_dtype)
    F_obs_sum = F_obs_t.sum().clamp(min=1e-30)

    hkl_t = hkl.detach().to(device).to(real_dtype)
    R_init = R_initial.detach().to(device).to(real_dtype)
    t_init = t_initial.detach().to(device).to(real_dtype)
    centric_t = centric.detach().to(device).to(torch.bool)

    # Per-shell normalisation of F_obs → E_obs (Wilson, shell-equal-count).
    rec_basis = real_cell.reciprocal_basis_matrix.to(device).to(real_dtype)
    s_mag = (hkl_t @ rec_basis).norm(dim=-1)
    order = torch.argsort(s_mag)
    shell_idx = torch.zeros_like(s_mag, dtype=torch.int64)
    chunk = max(1, s_mag.numel() // max(n_shells, 1))
    for k in range(n_shells):
        a = k * chunk
        b = (k + 1) * chunk if k < n_shells - 1 else s_mag.numel()
        shell_idx[order[a:b]] = k
    shell_sigma_obs = torch.zeros(n_shells, dtype=real_dtype, device=device)
    for k in range(n_shells):
        m = shell_idx == k
        if m.any():
            shell_sigma_obs[k] = (F_obs_t[m] ** 2).mean().clamp(min=1e-30).sqrt()
    E_obs = F_obs_t / shell_sigma_obs[shell_idx]

    # Rotation perturbation grid.
    # We parametrise (Δα, Δβ, Δγ) ∈ [-r, r]³ via the small-angle rotation
    # R_perturb ≈ I + ω_x · Lx + ω_y · Ly + ω_z · Lz, exponentiated via
    # the matrix exponential of the skew-symmetric generator. For small ω
    # (≤ ~3°) Rodrigues is well-conditioned.
    def _so3_exp(omega):
        # omega: (3,) axis-angle
        th = omega.norm()
        if th.item() < 1e-12:
            return torch.eye(3, dtype=real_dtype, device=device)
        axis = omega / th
        K = torch.tensor(
            [[0.0, -axis[2].item(), axis[1].item()],
             [axis[2].item(), 0.0, -axis[0].item()],
             [-axis[1].item(), axis[0].item(), 0.0]],
            dtype=real_dtype, device=device,
        )
        return (torch.eye(3, dtype=real_dtype, device=device)
                + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K))

    coords_r = torch.linspace(-rotation_radius_rad, rotation_radius_rad,
                                rotation_grid_steps, dtype=real_dtype, device=device)
    omega_grid = torch.stack(torch.meshgrid(coords_r, coords_r, coords_r,
                                              indexing="ij"), dim=-1).reshape(-1, 3)

    # Inner translation grid: (Δtx, Δty, Δtz) ∈ [-rt, rt]³ around t_initial.
    coords_t = torch.linspace(-translation_radius_frac, translation_radius_frac,
                                translation_grid_steps, dtype=real_dtype, device=device)
    t_offsets = torch.stack(torch.meshgrid(coords_t, coords_t, coords_t,
                                             indexing="ij"), dim=-1).reshape(-1, 3)
    t_candidates = t_init.unsqueeze(0) + t_offsets        # (T, 3)

    # For each rotation candidate, pre-compute G_i and run inner t scan.
    best_llg = -float("inf")
    best_R = R_init.clone()
    best_t = t_init.clone()

    for r_idx, omega in enumerate(omega_grid):
        R_perturb = _so3_exp(omega)
        R_cand = R_init @ R_perturb
        # Build G_i for this rotation candidate (the only expensive step).
        G, h_R = precompute_G_for_rotation(
            interpolator, R_cand, hkl, spacegroup, real_cell, device=device,
        )

        # Inner translation scan: scored by analytical R-factor for speed.
        scores = torch.empty(t_candidates.shape[0], dtype=real_dtype, device=device)
        for start in range(0, t_candidates.shape[0], batch_size):
            stop = min(start + batch_size, t_candidates.shape[0])
            t_batch = t_candidates[start:stop]
            dot = torch.einsum("ind,bd->ibn", h_R, t_batch)
            phase = torch.exp(two_pi_i * dot.to(complex_dtype))
            F_calc = torch.einsum("in,ibn->bn", G, phase)
            F_c_abs = F_calc.abs().to(real_dtype)
            num = (F_obs_t.unsqueeze(0) * F_c_abs).sum(dim=-1)
            den = (F_c_abs ** 2).sum(dim=-1).clamp(min=1e-30)
            k = num / den
            R_b = ((F_obs_t.unsqueeze(0) - k.unsqueeze(-1) * F_c_abs).abs()
                    .sum(dim=-1)) / F_obs_sum
            scores[start:stop] = R_b
        best_inner = int(scores.argmin().item())
        t_best_inner = t_candidates[best_inner]

        # Score (R_cand, t_best_inner) by full Sim MLRF LLG (scale-invariant
        # per-shell σA fit on E-values).
        dot = (h_R * t_best_inner.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (S, N)
        phase = torch.exp(two_pi_i * dot.to(complex_dtype))
        F_calc = (G * phase).sum(dim=0)                                     # (N,)
        F_calc_abs = F_calc.abs().to(real_dtype).unsqueeze(0)               # (1, N)
        llg = llg_for_rotation_batch(
            F_obs=F_obs_t, shell_idx=shell_idx, n_shells=n_shells,
            E_obs=E_obs, centric=centric_t, F_calc=F_calc_abs,
        )[0].item()

        if verbose > 1:
            print(f"  R-refine {r_idx}/{omega_grid.shape[0]}: "
                  f"|ω|={omega.norm().item():.4f} rad, R={scores[best_inner]:.4f}, "
                  f"LLG={llg:.2f}", flush=True)

        if llg > best_llg:
            best_llg = llg
            best_R = R_cand.clone()
            best_t = t_best_inner.clone()
    return best_R.cpu(), best_t.cpu(), float(best_llg)


def patterson_translation_function(
    F_obs: torch.Tensor,
    interpolator,
    R_rotation: torch.Tensor,
    hkl: torch.Tensor,
    spacegroup,
    real_cell,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    n_peaks: int = 20,
    cluster_radius: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[TranslationPeak]]:
    """
    Crowther-Blow Patterson translation function for molecular replacement.

    Computes T(t) = Σ_h |F_obs(h)|² · |F_calc(h, t)|² on a fractional grid by
    expanding |F_calc(h, t)|² over symmetry-operator pairs and inverse-FFTing
    the result. The peaks of T(t) are the translations that best place the
    rotated model against the observed amplitudes — unlike the bare
    `fft_translation_search`, this is the standard MR translation function and
    works on amplitude-only F_obs.

    For each symmetry operator (R_i, t_i) with `x_new = R_i x_old + t_i`,
    a per-symmetry asymmetric-unit structure factor is computed as
        F_asu_i(h) = interpolator.evaluate(R_rotation, h R_i, real_cell,
                                           return_amplitude=False)
                   * exp(2πi h · t_i)
    (the "h R_i" notation follows from F(h, R x + t) = exp(2πi h·t)·F(R^T h, x);
    in tensor form: `hkl @ R_i`). For each ordered pair (i, j) with i ≠ j, the
    contribution
        |F_obs(h)|² · conj(F_asu_i(h)) · F_asu_j(h)
    is scattered into a 3-D reciprocal grid at h' = h @ (R_j − R_i), then the
    inverse FFT gives the translation function. Diagonal pairs (i = j) are
    t-independent.

    Parameters
    ----------
    F_obs : torch.Tensor, shape (N,)
        Observed amplitudes. Complex inputs are coerced to |·|.
    interpolator : LattmanLoveInterpolator
        Provides `evaluate(R, hkl, real_cell, return_amplitude=False)` returning
        complex F_calc of the P1 ASU at arbitrary HKL.
    R_rotation : torch.Tensor, shape (3, 3)
        Rotation that has already been applied to the model coordinates, in
        the convention `xyz_new = R · xyz_old`. Passed through to the
        interpolator so it evaluates F of the rotated model.
    hkl : torch.Tensor, shape (N, 3)
        Integer Miller indices of the observed reflections.
    spacegroup : SpaceGroup
        Provides `matrices` (n_ops, 3, 3, integer in fractional basis) and
        `translations` (n_ops, 3, fractional).
    real_cell : Cell
        Real crystal cell, passed to interpolator.evaluate.
    grid_shape : tuple of int, optional
        Translation-function grid (Nx, Ny, Nz). Default: 4·max(|hkl|) per axis,
        which covers `h @ (R_j − R_i)^T` for any standard spacegroup.
    n_peaks : int, default 20
        Number of translation peaks returned.
    cluster_radius : float, default 0.05
        Minimum fractional separation between returned peaks.

    Returns
    -------
    correlation_map : np.ndarray, shape (Nx, Ny, Nz)
        Real-valued translation function T(t).
    best_translation : np.ndarray, shape (3,)
        Fractional coordinates of the top peak.
    peaks : list of TranslationPeak
        Top-`n_peaks` peaks sorted by descending T value.
    """
    device = getattr(interpolator, "device", hkl.device)
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    F_obs_t = F_obs.detach().to(device)
    if F_obs_t.is_complex():
        F_obs_t = F_obs_t.abs()
    F_obs_t = F_obs_t.to(real_dtype)
    F_obs2 = F_obs_t * F_obs_t                          # (N,)

    hkl_t = hkl.detach().to(device).to(real_dtype)       # (N, 3)

    sym_R = spacegroup.matrices.detach().to(device).to(real_dtype)       # (S, 3, 3)
    sym_t = spacegroup.translations.detach().to(device).to(real_dtype)   # (S, 3)
    S = sym_R.shape[0]

    R_rot = R_rotation.detach().to(device).to(real_dtype)

    # Per-symmetry F_asu_i(h) = F_model_rot(h R_i) · exp(2πi h · t_i)
    F_asu = torch.zeros((S, hkl_t.shape[0]), dtype=complex_dtype, device=device)
    two_pi_i = 2j * torch.pi
    for i in range(S):
        hkl_i = hkl_t @ sym_R[i]
        F_i = interpolator.evaluate(R_rot, hkl_i, real_cell, return_amplitude=False)
        F_i = F_i.to(complex_dtype)
        phase = torch.exp(two_pi_i * (hkl_t @ sym_t[i])).to(complex_dtype)
        F_asu[i] = F_i * phase

    # Translation grid extent: bound by max |h @ (R_j − R_i)^T|. For
    # crystallographic R_op (entries in {-1, 0, 1}, occasionally 2 for trigonal
    # subgroups), |R_j − R_i| has entries up to 2 → 2·max|h| per axis is the
    # natural extent. Use 4·max(|hkl|) for an oversampled, periodic grid.
    if grid_shape is None:
        max_h = hkl_t.abs().max(dim=0).values
        grid_shape = tuple(int(4 * (m.item() + 1)) for m in max_h)
    Nx, Ny, Nz = grid_shape

    W = torch.zeros((Nx, Ny, Nz), dtype=complex_dtype, device=device)
    W_flat = W.view(-1)
    # Cross-pair accumulation (skip i == j: t-independent, only shifts DC).
    for i in range(S):
        Fi_conj = torch.conj(F_asu[i])
        for j in range(S):
            if j == i:
                continue
            diff_R = sym_R[j] - sym_R[i]
            h_diff = (hkl_t @ diff_R).round().to(torch.int64)
            ix = h_diff[:, 0] % Nx
            iy = h_diff[:, 1] % Ny
            iz = h_diff[:, 2] % Nz
            flat_idx = ix * (Ny * Nz) + iy * Nz + iz
            weight = F_obs2 * Fi_conj * F_asu[j]
            W_flat.index_add_(0, flat_idx, weight)

    TF_complex = torch.fft.ifftn(W, dim=(0, 1, 2))
    TF = TF_complex.real

    TF_np = TF.detach().cpu().numpy().astype(np.float32)
    peaks = find_translation_peaks(TF_np, n_peaks=n_peaks, cluster_radius=cluster_radius)
    best = peaks[0].translation if peaks else np.zeros(3)
    return TF_np, best, peaks


def apply_translation_to_fcalc_torch(
    F_calc: torch.Tensor,
    hkl: torch.Tensor,
    translation_frac: torch.Tensor,
) -> torch.Tensor:
    """
    Apply translation phase shift to calculated structure factors (PyTorch).

    F(hkl, t) = F(hkl) * exp(2*pi*i * hkl.t)

    Parameters
    ----------
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    hkl : torch.Tensor
        Miller indices.
    translation_frac : torch.Tensor
        Translation in fractional coordinates.

    Returns
    -------
    F_calc_shifted : torch.Tensor
        Phase-shifted structure factors.
    """
    phase_shift = 2 * torch.pi * (hkl.to(translation_frac.dtype) @ translation_frac)
    return F_calc * torch.exp(1j * phase_shift)
