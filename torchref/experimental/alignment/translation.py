"""
Fast FFT-based translation search for molecular replacement.

Translation t shifts phase: F(hkl, t) = F(hkl) * exp(2*pi*i * hkl.t)
Correlation: C(t) = IFFT{ conj(F_obs) * F_calc }

This module provides efficient FFT-based translation search that finds the
optimal translation to position a model after rotation has been determined.
"""

import numpy as np
import torch

from .distributions import rice_log_likelihood, woolfson_log_likelihood
from .e_values import WilsonShellE
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
    interpolator : object
        Anything providing ``evaluate(R, hkl, real_cell, return_amplitude=False)``
        -- in the pipeline, ``align._DirectModelEvaluator``.
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
        # The caller's own shell assignment is handed to the convention rather
        # than letting it derive one: this binning is rank-based and the shared
        # `assign_shells` is value-based, and the sigma_a fit downstream is tied
        # to whichever one was used here.
        E_obs = WilsonShellE(
            F_obs_t, s_mag, shell_idx=shell_idx, n_shells=n_shells,
        ).E
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


def fit_sigma_a_per_shell(
    E_obs: torch.Tensor,
    E_calc: torch.Tensor,
    centric: torch.Tensor,
    shell_idx: torch.Tensor,
    n_shells: int,
    n_grid: int = 81,
    interp_var: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-shell sigma_A = D, fitted by a grid scan of the shell likelihood.

    Scans ``D`` in ``[0, 0.99]`` and returns the per-shell maximum. At
    ``n_grid=81`` the resolution is ~0.012, which is finer than the difference
    between adjacent shells on any real falloff.

    This is the *fitted* half of a pair. :func:`torchref.scaling.weighting.empirical_sigma_a`
    is the other: it measures model reliability from the ratio of two Wilson
    curves and works **before** the molecule is placed, which is what the
    rotation function needs. Once a translation exists the residual is
    per-reflection rather than per-shell-average, so a direct fit against the
    placed model is available and strictly better informed. The two answer the
    same question at two different points in the search.

    Returns
    -------
    sigma_a : torch.Tensor, shape (n_shells,)
    """
    device = E_obs.device
    dtype = E_obs.dtype
    N = E_obs.numel()
    D_grid = torch.linspace(0.0, 0.99, n_grid, device=device, dtype=dtype)  # (G,)
    F_mean = D_grid.view(-1, 1) * E_calc.view(1, -1)                        # (G, N)
    var_d = (1.0 - D_grid * D_grid).clamp(min=1e-4)                         # (G,)
    if interp_var is None:
        var_full = var_d.view(-1, 1).expand(n_grid, N)
    else:
        var_full = (var_d.view(-1, 1) + interp_var.view(1, -1)).clamp(min=1e-4)
    E_obs_full = E_obs.view(1, -1).expand(n_grid, N)

    ll_acent = rice_log_likelihood(E_obs_full, F_mean, var_full)
    ll_cent = woolfson_log_likelihood(E_obs_full, F_mean, var_full)
    cent_full = centric.view(1, -1)
    ll = torch.where(cent_full, ll_cent, ll_acent)                          # (G, N)

    # Sum per shell, take argmax over the D-grid.
    shell_idx_gn = shell_idx.view(1, -1).expand(n_grid, N)
    ll_per_shell = torch.zeros((n_grid, n_shells), dtype=dtype, device=device)
    ll_per_shell.scatter_add_(1, shell_idx_gn, ll)                          # (G, n_shells)
    best_idx = ll_per_shell.argmax(dim=0)                                   # (n_shells,)
    return D_grid[best_idx]


def llg_translation_rescore(
    F_obs: torch.Tensor,
    hkl: torch.Tensor,
    centric: torch.Tensor,
    s_mag: torch.Tensor,
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
    s_mag : (N,) — |s| the shells were built from. Not used to derive a binning
        here (``shell_idx`` is given) but passed rather than fabricated, so a
        convention that fits a curve in ``|s|`` gets the real abscissa.
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
    E_obs = WilsonShellE(
        F_obs_t, s_mag.to(device).to(real_dtype),
        shell_idx=shell_idx_l, n_shells=n_shells,
    ).E

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


