"""
Maximum-Likelihood Rotation Function (Sim MLRF) rescoring of peaks from the
fast ball-search.

Phaser paper §2.1.2: the fast rotation function is a "shortlist generator";
discrimination of the correct orientation comes from rescoring the top peaks
with a slow, full ML target. This file implements that rescoring.

Per-shell σA (= D) is fitted on-the-fly for each candidate rotation. We work in
E-value (normalized structure-factor amplitude) space, which gives the standard
Rice / Woolfson likelihood forms

    P(E_obs | σA · E_calc, 1 − σA²)        (acentric, Rice)
    P(E_obs | σA · E_calc, 1 − σA²)        (centric,  Woolfson)

The "log-likelihood gain" relative to a Wilson reference (σA = 0) is the
discriminating score Phaser reports as LLG.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .ball_search import (
    RotationPeak,
    edmonds_euler_from_rotation_matrix,
    rotation_matrix_from_edmonds_euler,
    rotation_matrix_from_edmonds_euler_batch,
)
from .distributions import rice_log_likelihood, woolfson_log_likelihood
from .lattman_love import LattmanLoveInterpolator


# =============================================================================
# Helpers
# =============================================================================


def _equal_count_shell_idx(s_mag: torch.Tensor, n_shells: int) -> torch.Tensor:
    """
    Partition `s_mag` into `n_shells` shells with (approximately) equal counts.

    Returns
    -------
    shell_idx : torch.Tensor (int64), shape (N,)
        Shell index in [0, n_shells).
    """
    n = s_mag.numel()
    order = torch.argsort(s_mag)
    chunk = max(n // n_shells, 1)
    positions = torch.arange(n, device=s_mag.device, dtype=torch.int64)
    sorted_labels = (positions // chunk).clamp(max=n_shells - 1)
    shell_idx = torch.empty(n, dtype=torch.int64, device=s_mag.device)
    shell_idx[order] = sorted_labels
    return shell_idx


def _normalize_to_e(F: torch.Tensor, shell_idx: torch.Tensor,
                    n_shells: int) -> torch.Tensor:
    """E = F / sqrt(<F²> per shell). Vectorised across shells via scatter."""
    F2 = F ** 2
    sum_per_shell = torch.zeros(n_shells, dtype=F2.dtype, device=F.device)
    sum_per_shell.scatter_add_(0, shell_idx, F2)
    count_per_shell = torch.bincount(shell_idx, minlength=n_shells).to(F2.dtype)
    mean_per_shell = (sum_per_shell / count_per_shell.clamp(min=1.0)).clamp(min=1e-30)
    norm_per_refl = mean_per_shell.sqrt().index_select(0, shell_idx)
    return F / norm_per_refl


def _shell_ll(
    E_obs: torch.Tensor,
    E_calc: torch.Tensor,
    centric: torch.Tensor,
    D: float,
) -> torch.Tensor:
    """
    Per-reflection log-likelihood at a given σA = D for one shell, in E-value space.

    Acentric: Rice with F_mean = D · E_calc, variance = 1 − D².
    Centric:  Woolfson with F_mean = D · E_calc, variance = 1 − D².
    """
    var = torch.full_like(E_obs, max(1.0 - D * D, 1e-4))
    F_mean = D * E_calc
    ll = torch.where(
        centric,
        woolfson_log_likelihood(E_obs, F_mean, var),
        rice_log_likelihood(E_obs, F_mean, var),
    )
    return ll


def _optimize_D_in_shell(
    E_obs: torch.Tensor,
    E_calc: torch.Tensor,
    centric: torch.Tensor,
    n_grid: int = 21,
    n_refine: int = 12,
) -> float:
    """
    Find the σA = D ∈ [0, 0.99] that maximizes the sum log-likelihood for this
    shell. Two-stage: coarse grid search, then golden-section refinement.

    Returns the optimal D.
    """
    # Coarse grid
    D_grid = torch.linspace(0.0, 0.99, n_grid, device=E_obs.device)
    best_D = 0.0
    best_ll = -float("inf")
    for D in D_grid.tolist():
        ll = _shell_ll(E_obs, E_calc, centric, D).sum().item()
        if ll > best_ll:
            best_ll = ll
            best_D = D
    # Golden-section refinement around best_D
    span = 1.0 / (n_grid - 1)
    lo = max(0.0, best_D - span)
    hi = min(0.99, best_D + span)
    phi = (math.sqrt(5.0) - 1) / 2.0
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1 = _shell_ll(E_obs, E_calc, centric, x1).sum().item()
    f2 = _shell_ll(E_obs, E_calc, centric, x2).sum().item()
    for _ in range(n_refine):
        if f1 > f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - phi * (hi - lo)
            f1 = _shell_ll(E_obs, E_calc, centric, x1).sum().item()
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + phi * (hi - lo)
            f2 = _shell_ll(E_obs, E_calc, centric, x2).sum().item()
    return 0.5 * (lo + hi)


# =============================================================================
# Public API
# =============================================================================


def llg_for_rotation(
    F_obs: torch.Tensor,
    s_mag: torch.Tensor,
    shell_idx: torch.Tensor,
    n_shells: int,
    E_obs: torch.Tensor,
    centric: torch.Tensor,
    F_calc: torch.Tensor,
    shell_weights: Optional[torch.Tensor] = None,
) -> float:
    """
    Total log-likelihood gain (Sim − Wilson) for a single candidate rotation.
    Thin wrapper around `llg_for_rotation_batch` for a single (1,N) input.
    """
    F_calc_batch = F_calc.unsqueeze(0) if F_calc.dim() == 1 else F_calc
    return llg_for_rotation_batch(
        F_obs=F_obs, shell_idx=shell_idx, n_shells=n_shells,
        E_obs=E_obs, centric=centric, F_calc=F_calc_batch,
        shell_weights=shell_weights,
    )[0].item()


def llg_for_rotation_batch(
    F_obs: torch.Tensor,
    shell_idx: torch.Tensor,
    n_shells: int,
    E_obs: torch.Tensor,
    centric: torch.Tensor,
    F_calc: torch.Tensor,
    n_D_grid: int = 41,
    shell_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Vectorized log-likelihood gain across a batch of candidate rotations.

    Per-shell σA fit is performed on a coarse grid of `n_D_grid` D values;
    the grid maximum is taken (no golden refinement — for shortlisting only).

    Parameters
    ----------
    F_obs : torch.Tensor, shape (N,)
    shell_idx : torch.Tensor (int64), shape (N,)
    n_shells : int
    E_obs : torch.Tensor, shape (N,)
        F_obs normalized to unit variance per shell.
    centric : torch.Tensor (bool), shape (N,)
    F_calc : torch.Tensor, shape (B, N)
        Per-rotation |F_calc| at the same HKL set.
    n_D_grid : int, default 41
        Number of σA grid points in [0, 0.99].
    shell_weights : torch.Tensor, shape (n_shells,), optional
        Per-shell weight applied to the per-shell LL gain before accumulation.
        Used to implement Phaser-style empirical variance correction
        (`w_p = 1/√Var(E_obs²-1)_p`). The weight multiplies *both* the Sim and
        Wilson LL contributions uniformly per shell, so the LL gain
        interpretation is preserved.

    Returns
    -------
    llg : torch.Tensor, shape (B,)
        Total log-likelihood gain (Sim − Wilson) per rotation candidate.
    """
    B, N = F_calc.shape
    device = F_calc.device
    dtype = F_calc.dtype

    # --- Per-shell E normalisation of F_calc, fully vectorised ---
    # Build (B, n_shells) shell-mean of F_calc² via scatter_add, then gather
    # back per reflection. Replaces the n_shells-step Python loop that
    # masked + meaned one shell at a time.
    shell_idx_b = shell_idx.view(1, N).expand(B, N)
    F_calc2 = F_calc * F_calc
    sum_per_shell_b = torch.zeros((B, n_shells), dtype=dtype, device=device)
    sum_per_shell_b.scatter_add_(1, shell_idx_b, F_calc2)
    count_per_shell = torch.bincount(shell_idx, minlength=n_shells).to(dtype)
    mean_per_shell_b = (
        sum_per_shell_b / count_per_shell.clamp(min=1.0).unsqueeze(0)
    ).clamp(min=1e-30)
    norm_per_refl_b = mean_per_shell_b.sqrt().gather(1, shell_idx_b)  # (B, N)
    E_calc = F_calc / norm_per_refl_b

    # --- Joint (D, B, N) likelihood evaluation ---
    # Memory: D · B · N · 8 B. For default args (D=41, B≤100, N≈3 k) this is
    # ~100 MB, comparable to what the per-shell loop already built per
    # iteration. For dense-R we typically pass n_D_grid=11, so cost is small.
    D_grid = torch.linspace(0.0, 0.99, n_D_grid, device=device, dtype=dtype)
    F_mean = D_grid.view(-1, 1, 1) * E_calc.unsqueeze(0)              # (D, B, N)
    var_d = (1.0 - D_grid * D_grid).clamp(min=1e-4)
    var_full = var_d.view(-1, 1, 1).expand(n_D_grid, B, N)
    E_obs_full = E_obs.view(1, 1, -1).expand(n_D_grid, B, N)

    ll_acent = rice_log_likelihood(E_obs_full, F_mean, var_full)
    ll_cent = woolfson_log_likelihood(E_obs_full, F_mean, var_full)
    cent_full = centric.view(1, 1, -1)
    ll = torch.where(cent_full, ll_cent, ll_acent)                    # (D, B, N)

    # --- Sum per shell across N, max over D, sum weighted across shells ---
    shell_idx_dbn = shell_idx.view(1, 1, -1).expand(n_D_grid, B, N)
    ll_per_shell = torch.zeros((n_D_grid, B, n_shells), dtype=dtype, device=device)
    ll_per_shell.scatter_add_(2, shell_idx_dbn, ll)
    ll_sim_per_shell, _ = ll_per_shell.max(dim=0)                     # (B, n_shells)

    # Wilson reference at D = 0 (data-only): F_mean = 0, var = 1.
    var0 = torch.ones_like(E_obs)
    F_mean0 = torch.zeros_like(E_obs)
    ll_wil_acent = rice_log_likelihood(E_obs, F_mean0, var0)
    ll_wil_cent = woolfson_log_likelihood(E_obs, F_mean0, var0)
    ll_wil_per_refl = torch.where(centric, ll_wil_cent, ll_wil_acent)
    ll_wil_per_shell = torch.zeros(n_shells, dtype=dtype, device=device)
    ll_wil_per_shell.scatter_add_(0, shell_idx, ll_wil_per_refl)

    gain_per_shell = ll_sim_per_shell - ll_wil_per_shell.unsqueeze(0) # (B, n_shells)
    if shell_weights is not None:
        gain_per_shell = gain_per_shell * shell_weights.to(dtype).view(1, -1)
    total_gain = gain_per_shell.sum(dim=-1)                           # (B,)
    return total_gain


def sim_mlrf_rescore(
    peaks: List[RotationPeak],
    F_obs: torch.Tensor,
    hkl_real: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    interpolator: LattmanLoveInterpolator,
    real_cell,
    n_shells: int = 20,
    n_refine: Optional[int] = None,
    batch_size: int = 100,
    verbose: int = 0,
    shell_weights: Optional[torch.Tensor] = None,
    auto_variance_weights: bool = True,
    n_D_grid: int = 41,
) -> List[RotationPeak]:
    """
    Rescore a list of peaks from `ball_rotation_search` by the per-shell-fitted
    Sim Maximum-Likelihood Rotation Function (LLG). Returns a new list sorted by
    descending LLG with `score = LLG` and `sigma = Z-score(LLG)`.

    Batches candidates of size `batch_size` for fast vectorized evaluation.

    Parameters
    ----------
    peaks : list of RotationPeak
    F_obs : torch.Tensor, shape (N,)
    hkl_real : torch.Tensor (int), shape (N, 3)
    s_mag : torch.Tensor, shape (N,)
    centric : torch.Tensor (bool), shape (N,)
    interpolator : LattmanLoveInterpolator
    real_cell : Cell
    n_shells : int, default 20
    n_refine : int, optional
        Number of top peaks to rescore (default: all).
    batch_size : int, default 100
        Number of candidates evaluated together in one LL.evaluate + LLG call.
    """
    if not peaks:
        return []

    if n_refine is None:
        n_refine = len(peaks)
    head = peaks[: n_refine]
    tail = peaks[n_refine:]

    shell_idx = _equal_count_shell_idx(s_mag, n_shells)
    E_obs = _normalize_to_e(F_obs, shell_idx, n_shells)

    if shell_weights is None and auto_variance_weights:
        from .sh import compute_patterson_shell_variance
        patt_obs = (E_obs.to(torch.float64) ** 2) - 1.0
        var_p = compute_patterson_shell_variance(
            patt_obs, shell_idx, P=n_shells,
        )
        w = 1.0 / var_p.sqrt()
        w = w * (n_shells / w.sum().clamp(min=1e-30))
        shell_weights = w.to(F_obs.dtype)

    # Build all rotation matrices up front. The peak's Euler triple represents
    # "the rotation applied to the model coords" (synthetic-test convention of
    # ball_rotation_search). For ML scoring, we need "the rotation to apply to
    # the current model to align it to obs" — which is R^T. We transpose here.
    # Vectorised over peaks: previously this list comprehension built M·9
    # small (3,3) tensors per dense-R pass.
    alpha_t = torch.tensor([p.alpha for p in head], dtype=torch.float64)
    beta_t = torch.tensor([p.beta for p in head], dtype=torch.float64)
    gamma_t = torch.tensor([p.gamma for p in head], dtype=torch.float64)
    R_all = rotation_matrix_from_edmonds_euler_batch(
        alpha_t, beta_t, gamma_t,
    ).transpose(-1, -2).to(torch.float32)            # (M, 3, 3)

    llg_chunks: List[torch.Tensor] = []
    M = R_all.shape[0]
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        R_batch = R_all[start:stop]  # (B, 3, 3)
        # Batched LL interpolation: returns (B, N)
        F_calc = interpolator.evaluate(
            R_batch, hkl_real, real_cell, return_amplitude=True,
        )
        F_calc = F_calc.to(F_obs.dtype)
        llg_batch = llg_for_rotation_batch(
            F_obs=F_obs, shell_idx=shell_idx, n_shells=n_shells,
            E_obs=E_obs, centric=centric, F_calc=F_calc,
            shell_weights=shell_weights, n_D_grid=n_D_grid,
        )
        llg_chunks.append(llg_batch)
        if verbose > 1:
            print(f"  ML rescore batch {start}-{stop}/{M}", flush=True)

    # Concatenate on-device, compute z-score on-device, then ONE bulk
    # transfer at the end. The previous code did `.cpu().tolist()` per
    # batch — fine on CPU but a per-batch GPU↔CPU stall on cuda.
    llgs_t = torch.cat(llg_chunks)
    mean_t = llgs_t.mean()
    std_t = llgs_t.std().clamp(min=1e-30)
    sigmas_t = (llgs_t - mean_t) / std_t

    llgs_list = llgs_t.tolist()
    sigmas_list = sigmas_t.tolist()
    rescored = [
        RotationPeak(
            alpha=p.alpha, beta=p.beta, gamma=p.gamma,
            score=llg, sigma=sigma,
        )
        for p, llg, sigma in zip(head, llgs_list, sigmas_list)
    ]
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored + tail


# =============================================================================
# Brute-force ML rotation search over a uniform SO(3) sample
# =============================================================================


def _uniform_random_rotations(n: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """
    Generate `n` uniformly-distributed random rotation matrices via QR of
    Gaussian matrices. Returns shape (n, 3, 3) with det = +1.
    """
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(n, 3, 3, generator=g, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # Make det = 1 (flip first column if needed)
    diag_sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))  # (n, 3)
    Q = Q * diag_sign.unsqueeze(-2)
    det = torch.det(Q)
    flip = det < 0
    Q[flip, :, 0] = -Q[flip, :, 0]
    return Q


def brute_ml_rotation_search(
    F_obs: torch.Tensor,
    hkl_real: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    interpolator: LattmanLoveInterpolator,
    real_cell,
    n_candidates: int = 5000,
    n_shells: int = 15,
    batch_size: int = 100,
    seed: int = 0,
    verbose: int = 0,
) -> List[RotationPeak]:
    """
    Evaluate ML LLG on `n_candidates` uniformly-random SO(3) rotations.
    Returns peaks sorted by descending LLG. Acts as a shortlist generator that
    bypasses the fast ball-search (which has known sphere-sampling limitations
    on real-cell HKL data).

    Cost: ~10ms per candidate at default batch_size on CPU; ~50s for 5000 cands.

    Returns
    -------
    list of RotationPeak (sorted by LLG descending)
        `score = LLG`, `sigma = Z-score across the candidate set`.
    """
    R_all = _uniform_random_rotations(n_candidates, seed=seed)
    peaks_in = []
    for k in range(n_candidates):
        a, b, g = edmonds_euler_from_rotation_matrix(R_all[k])
        peaks_in.append(RotationPeak(a, b, g, score=0.0, sigma=0.0))
    return sim_mlrf_rescore(
        peaks_in, F_obs, hkl_real, s_mag, centric, interpolator, real_cell,
        n_shells=n_shells, n_refine=n_candidates, batch_size=batch_size,
        verbose=verbose,
    )
