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
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from .frf.rotation_utils import (
    axis_angle_to_matrix,
    edmonds_euler_from_rotation_matrix,
    rotation_matrix_from_edmonds_euler,
    rotation_matrix_from_edmonds_euler_batch,
)
from .frf.types import RotationPeak
from .distributions import (
    phaser_log_rel_rice,
    phaser_log_rel_woolfson,
    rice_log_likelihood,
    woolfson_log_likelihood,
)
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
    return F / _per_shell_sqrt_mean(F, shell_idx, n_shells)


def _normalize_to_e_epsilon(
    F: torch.Tensor, shell_idx: torch.Tensor, n_shells: int, eps: torch.Tensor,
) -> torch.Tensor:
    """ε-corrected Wilson E: ``E²_h = (F²_h/ε_h) / ⟨F²/ε⟩_shell``.

    Matches Phaser's obs E (``E = F/sqrt(ε·Σ_N)``) and the FRF's
    :func:`torchref.experimental.alignment.frf.preprocessing.wilson_normalise_epsilon`. The
    plain :func:`_normalize_to_e` (no ε) over-counts axial reflections (ε>1) on
    high-symmetry spacegroups, letting them dominate the ``-(E²+eImove)/V`` term
    and blind the m_LETF1 orientation discrimination.
    """
    I_corr = (F * F) / eps.clamp(min=1.0)
    sum_shell = torch.zeros(n_shells, dtype=I_corr.dtype, device=F.device)
    sum_shell.scatter_add_(0, shell_idx, I_corr)
    count = torch.bincount(shell_idx, minlength=n_shells).to(I_corr.dtype)
    mean_shell = (sum_shell / count.clamp(min=1.0)).clamp(min=1e-30)
    return (I_corr / mean_shell.index_select(0, shell_idx)).clamp(min=0.0).sqrt()


def _per_shell_sqrt_mean(F: torch.Tensor, shell_idx: torch.Tensor,
                         n_shells: int) -> torch.Tensor:
    """Per-reflection ``sqrt(<F²>_shell)``. Wilson-normalisation denominator.

    Use this when you need to normalise *another* tensor by the same per-shell
    statistic computed from F — e.g. converting rotated |F_calc| to E_calc
    using the reference |F_calc|'s shell means (rotation-invariant).
    """
    F2 = F ** 2
    sum_per_shell = torch.zeros(n_shells, dtype=F2.dtype, device=F.device)
    sum_per_shell.scatter_add_(0, shell_idx, F2)
    count_per_shell = torch.bincount(shell_idx, minlength=n_shells).to(F2.dtype)
    mean_per_shell = (sum_per_shell / count_per_shell.clamp(min=1.0)).clamp(min=1e-30)
    return mean_per_shell.sqrt().index_select(0, shell_idx)


def _shell_ll(
    E_obs: torch.Tensor,
    E_calc: torch.Tensor,
    centric: torch.Tensor,
    D: float,
    interp_var: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-reflection log-likelihood at a given σA = D for one shell, in E-value space.

    Acentric: Rice with F_mean = D · E_calc, variance = (1 − D²) + interp_var.
    Centric:  Woolfson with F_mean = D · E_calc, variance = (1 − D²) + interp_var.

    `interp_var` (Phaser totvar_search analogue) inflates the variance to absorb
    interpolation / model error and prevents the Rice tail from over-penalising
    slightly-noisy true peaks.
    """
    base = max(1.0 - D * D, 1e-4)
    if interp_var is None:
        var = torch.full_like(E_obs, base)
    else:
        var = (interp_var + base).clamp(min=1e-4)
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


def compute_sigma_a_luzzati(
    s_mag: torch.Tensor,
    delta_vrms_A: float = 1.0,
) -> torch.Tensor:
    """
    Phaser-style Luzzati σA(s) = exp(−2π²·s²·ΔVRMS²).

    Closed-form, rotation-independent estimate of the per-reflection (or
    per-shell) σA, derived from the search model's RMS coordinate
    deviation ΔVRMS (in Å). At s=0 returns 1.0 (perfect agreement);
    falls off monotonically with resolution. Matches the Phaser FastRot
    Eterm/Vterm weighting (LERF1 §2.1.2): `Eterm = exp(−2π²s²ΔVRMS)` and
    `Vterm = Eterm²`.

    Parameters
    ----------
    s_mag : torch.Tensor
        Reciprocal magnitudes in Å⁻¹ (any shape).
    delta_vrms_A : float
        Estimated RMS coordinate error of the search model, Å. Default
        1.0 Å is a reasonable starting point for MR search models;
        tune via `frf_delta_vrms_A` kwarg in `align_model_to_data`.

    Returns
    -------
    sigma_a : torch.Tensor, same shape and dtype as `s_mag`.
    """
    return torch.exp(
        -2.0 * (math.pi ** 2) * (s_mag ** 2) * (float(delta_vrms_A) ** 2)
    )


def fit_sigma_a_per_shell(
    E_obs: torch.Tensor,
    E_calc: torch.Tensor,
    centric: torch.Tensor,
    shell_idx: torch.Tensor,
    n_shells: int,
    n_grid: int = 81,
    interp_var: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Vectorised per-shell σA = D fit. Single source of truth for D across the
    alignment stages (rotation rescore + likelihood TF).

    For each shell, scans D ∈ [0, 0.99] on a fine grid and returns the
    grid maximum. With n_grid=81 the resolution is ~0.012, comparable to the
    golden-section result in `_optimize_D_in_shell` for downstream LLG purposes.

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
    interp_var: Optional[torch.Tensor] = None,
    sigma_a: Optional[torch.Tensor] = None,
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
    interp_var : torch.Tensor, shape (N,), optional
        Per-reflection interpolation variance (Phaser totvar_search analogue).
        Added to the model variance term. None ⇒ original Rice/Woolfson.
    sigma_a : torch.Tensor, shape (n_shells,), optional
        Externally-fitted per-shell σA to reuse across candidates. When given,
        the per-(D, B) grid maximisation is bypassed and the LLG is computed
        at this fixed sigma_a (one D per shell, broadcast per reflection).
        This is the "shared D" path used when an external single-source σA
        is available (see ``fit_sigma_a_per_shell``).

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

    if sigma_a is not None:
        # --- Shared per-shell σA path: skip the D-grid, evaluate LL once. ---
        D_per_refl = sigma_a.to(dtype).to(device).index_select(0, shell_idx)  # (N,)
        var_d = (1.0 - D_per_refl * D_per_refl).clamp(min=1e-4)               # (N,)
        if interp_var is not None:
            var_per_refl = (var_d + interp_var.to(dtype).to(device)).clamp(min=1e-4)
        else:
            var_per_refl = var_d
        F_mean = D_per_refl.view(1, N) * E_calc                               # (B, N)
        var_full = var_per_refl.view(1, N).expand(B, N)
        E_obs_full = E_obs.view(1, N).expand(B, N)
        ll_acent = rice_log_likelihood(E_obs_full, F_mean, var_full)
        ll_cent = woolfson_log_likelihood(E_obs_full, F_mean, var_full)
        cent_full = centric.view(1, N)
        ll = torch.where(cent_full, ll_cent, ll_acent)                        # (B, N)
        ll_per_shell = torch.zeros((B, n_shells), dtype=dtype, device=device)
        ll_per_shell.scatter_add_(1, shell_idx_b, ll)
        ll_sim_per_shell = ll_per_shell                                       # (B, n_shells)
    else:
        # --- Joint (D, B, N) likelihood evaluation, original behaviour. ---
        # Memory: D · B · N · 8 B. For default args (D=41, B≤100, N≈3 k) this is
        # ~100 MB, comparable to what the per-shell loop already built per
        # iteration. For dense-R we typically pass n_D_grid=11, so cost is small.
        D_grid = torch.linspace(0.0, 0.99, n_D_grid, device=device, dtype=dtype)
        F_mean = D_grid.view(-1, 1, 1) * E_calc.unsqueeze(0)              # (D, B, N)
        var_d = (1.0 - D_grid * D_grid).clamp(min=1e-4)
        if interp_var is None:
            var_full = var_d.view(-1, 1, 1).expand(n_D_grid, B, N)
        else:
            iv = interp_var.to(dtype).to(device).view(1, 1, N)
            var_full = (var_d.view(-1, 1, 1) + iv).clamp(min=1e-4).expand(n_D_grid, B, N)
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
    interp_var: Optional[torch.Tensor] = None,
    sigma_a: Optional[torch.Tensor] = None,
) -> List[RotationPeak]:
    """
    Rescore a list of FRF peaks by the per-shell-fitted
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
    # FRF synthetic-test convention). For ML scoring, we need "the rotation to apply to
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
            interp_var=interp_var, sigma_a=sigma_a,
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
# Phaser-faithful m_LETF1 rescore: unique-orbit calc sum + V(h) budget +
# Rice/Woolfson logRel formulas (DataMR.cc:1326-1429).
#
# The per-orientation LLG evaluator is factored out of `m_letf1_rescore` into a
# reusable `_LLGContext` + `_llg_for_orientations`, so the sub-peak refiner
# (`quadratic_llg_refine`) optimises the *same* likelihood the rescore ranks on.
# =============================================================================


@dataclass
class _LLGContext:
    """Rotation-independent context for the m_LETF1 per-orientation LLG.

    Built once by :func:`_build_llg_context`; consumed by
    :func:`_llg_for_orientations` (rescore) and :func:`quadratic_llg_refine`
    (sub-peak optimiser). Everything here depends only on the data + σ_A model,
    not on the candidate orientation.
    """

    interpolator: LattmanLoveInterpolator
    real_cell: object
    unrolled_hkl: torch.Tensor   # (M, 3) float64 — distinct orbit mates
    asu_idx: torch.Tensor        # (M,) long — ASU reflection each mate maps to
    N: int                       # number of ASU reflections
    E_obs_b: torch.Tensor        # (1, N)
    V_b: torch.Tensor            # (1, N)
    eImove_prefac: torch.Tensor  # (1, N) = ε·σ_A²/n_ops
    sqrt_mean_per_m: torch.Tensor   # (M,) per-mate E-normaliser
    centric_b: torch.Tensor      # (1, N) bool
    dw_per_m: Optional[torch.Tensor]  # (M,) or None — Wilson-B Debye-Waller
    dtype: torch.dtype
    batch_size: int


def _build_llg_context(
    F_obs: torch.Tensor,
    hkl_real: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    interpolator: LattmanLoveInterpolator,
    real_cell,
    spacegroup,
    *,
    n_shells: int = 20,
    batch_size: int = 50,
    sigma_a: Optional[torch.Tensor] = None,
    eps_factor: Optional[torch.Tensor] = None,
    apply_bulk_solvent: bool = False,
    solvent_fsol: float = 0.95,
    solvent_bsol: float = 300.0,
    vrms_strategy: str = "fixed",
    vrms_n_residues: Optional[int] = None,
    vrms_identity: float = 1.0,
    apply_wilson_b: bool = False,
    wilson_b_value: Optional[float] = None,
    scat_mode: str = "legacy",
) -> _LLGContext:
    """Build the rotation-independent m_LETF1 LLG context (DataMR.cc:1326-1429).

    Two corrections vs. the original implementation, both borrowed from the FRF's
    own high-symmetry fixes:

    * **Unique-orbit calc sum.** The moving-model intensity sums ``|E_calc|²`` over
      the **distinct** orbit mates via
      :func:`torchref.experimental.alignment.frf.preprocessing.epsilon_aware_unroll`
      (Phaser's ``if(!duplicate(isym))``), not all ``n_ops`` raw mates. Summing all
      mates over-weights axial reflections (ε>1) by ε(h) and orientation-blinds
      high-symmetry spacegroups (the 4BX9/6G9X rank-360+ failure).
    * **σ_A Eterm convention.** σ_A uses ``eterm_sigma_a`` (the ``2π²/3`` isotropic
      Eterm, Ensemble.cc:42 — matching the FRF/Phaser), not the ``2π²`` Luzzati
      form which falls off ~3× too fast.
    """
    from .frf.preprocessing import (
        compute_v_budget,
        epsilon_aware_unroll,
        eterm_sigma_a,
    )

    device = F_obs.device
    dtype = F_obs.dtype
    sym_mats = spacegroup.matrices.to(torch.float64).to(device)
    n_ops = int(sym_mats.shape[0])
    N = hkl_real.shape[0]

    # 1. ε(h) per reflection (needed for the ε-corrected obs normalisation).
    if eps_factor is None:
        # `friedel=False`: the conventional count. The variance budget
        # `V = eps - sigma_A**2` wants operations that add coherently and set the
        # mean; operations mapping h -> -h change the DISTRIBUTION instead, which
        # the Woolfson branch below already handles. Counting them here doubles
        # epsilon on every centric reflection -- 6680 of them on 3K7M -- and
        # inflates exactly those reflections' variance.
        eps_factor = spacegroup.epsilon(
            hkl_real.to(torch.long), friedel=False,
        ).to(dtype)
    eps_factor = eps_factor.to(device)

    # 2. Per-shell ε-corrected Wilson E_obs (Phaser E = F/sqrt(ε·Σ_N)). Dividing
    #    ε out of the obs is the obs-side analog of the unique-orbit calc dedup:
    #    both stop axial reflections (ε>1) from being over-weighted on
    #    high-symmetry spacegroups.
    shell_idx = _equal_count_shell_idx(s_mag, n_shells)
    E_obs = _normalize_to_e_epsilon(F_obs, shell_idx, n_shells, eps_factor)

    # 3. Identity-rotation calc reference → E-normalisation scale for F_calc
    #    (rotation-invariant: sphere permutation, shell sums preserved).
    I_eye = torch.eye(3, dtype=torch.float32, device=device)
    F_calc_ref = interpolator.evaluate(
        I_eye, hkl_real, real_cell, return_amplitude=True,
    ).to(dtype).squeeze(0)  # (N,)
    if scat_mode == "legacy":
        # Per-shell unit-variance normalisation: forces <E_calc²>_shell = 1 in
        # EVERY shell, flattening F_calc's inter-shell amplitude shape.
        calc_norm_per_h = _per_shell_sqrt_mean(
            F_calc_ref, shell_idx, n_shells,
        ).to(device)
    elif scat_mode == "absolute":
        # Single GLOBAL scale: preserves F_calc's inter-shell shape (how much the
        # model actually scatters per resolution) instead of flattening it to 1.
        # Phaser keeps E_calc physically scaled and carries the model's fraction
        # of the cell in scatFactor = AtomScatRatio·SCATTERING/TOTAL_SCAT/NSYMP;
        # for a search model that IS the full ASU (the benchmark case) scatFactor
        # reduces to 1/n_ops, so the prefactor is unchanged and the only change
        # here is dropping the per-shell flatten.
        global_rms = F_calc_ref.pow(2).mean().clamp(min=1e-30).sqrt()
        calc_norm_per_h = torch.full(
            (N,), float(global_rms), dtype=dtype, device=device,
        )
    else:
        raise ValueError(
            f"scat_mode={scat_mode!r}; expected 'legacy' or 'absolute'."
        )

    # Optional Wilson-B match (EnsemblePDB.cc:793-851), applied as a per-reflection
    # Debye-Waller multiplier on F_calc.
    if apply_wilson_b and wilson_b_value is None:
        from .frf.preprocessing import fit_relative_wilson_b
        wilson_b_value = fit_relative_wilson_b(
            F_obs, F_calc_ref, s_mag, n_shells=n_shells,
        )
    wilson_b_value = float(wilson_b_value or 0.0)
    if apply_wilson_b and abs(wilson_b_value) > 1e-6:
        dw = torch.exp(-wilson_b_value * (s_mag * s_mag) / 4.0).to(dtype).to(device)
    else:
        dw = None

    # 4. σ_A per reflection — FRF/Phaser Eterm (2π²/3 isotropic form), not the
    #    2π² Luzzati form. Rotation-independent, no aligned model required.
    if sigma_a is None:
        if vrms_strategy == "oeffner":
            if vrms_n_residues is None:
                raise ValueError(
                    "vrms_strategy='oeffner' requires vrms_n_residues=<n>."
                )
            from .frf.preprocessing import oeffner_vrms
            delta_vrms_A = oeffner_vrms(int(vrms_n_residues), float(vrms_identity))
        elif vrms_strategy == "fixed":
            delta_vrms_A = 0.5  # legacy default
        else:
            raise ValueError(
                f"vrms_strategy={vrms_strategy!r}; expected 'fixed' or 'oeffner'."
            )
        sigma_a = eterm_sigma_a(s_mag, delta_vrms_A=delta_vrms_A).to(dtype).to(device)
        if apply_bulk_solvent:
            from .frf.preprocessing import bulk_solvent_factor
            sol = bulk_solvent_factor(
                s_mag, fsol=solvent_fsol, bsol=solvent_bsol,
            ).to(dtype).to(device)
            sigma_a = sigma_a * sol
    sigma_a = sigma_a.to(device)
    sigma_a2 = sigma_a * sigma_a

    # 5. V(h) — rotation-independent variance budget V = ε − σ_A² (n_mol=1).
    V = compute_v_budget(eps_factor, sigma_a, n_mol=1)  # (N,)

    # 6. Unique-orbit unroll: distinct mates only (Phaser duplicate-skip). Each
    #    ASU reflection appears n_ops/ε(h) times, NOT n_ops times.
    unrolled_hkl, asu_idx = epsilon_aware_unroll(hkl_real, sym_mats)
    unrolled_hkl = unrolled_hkl.to(torch.float64).to(device)
    asu_idx = asu_idx.to(device)

    # 7. Broadcastable per-reflection tensors.
    E_obs_b = E_obs.unsqueeze(0)        # (1, N)
    V_b = V.unsqueeze(0)                # (1, N)
    # eImove = ε(h)·σ_A²·(1/n_ops)·Σ_{distinct mates} |E_calc(R^T·S_k·h)|²
    # (Phaser DataMR.cc:1397: thisEsqr *= repsn·scatFactor, scatFactor∝1/NSYMP).
    eImove_prefac = (eps_factor * sigma_a2 / float(n_ops)).unsqueeze(0)  # (1, N)
    centric_b = centric.to(torch.bool).to(device).unsqueeze(0)

    # Per-mate normaliser + DW (rotation preserves |h| → same shell across the
    # orbit, so the per-h scale broadcasts to every mate via asu_idx).
    sqrt_mean_per_m = calc_norm_per_h[asu_idx]                    # (M,)
    dw_per_m = dw[asu_idx] if dw is not None else None

    return _LLGContext(
        interpolator=interpolator, real_cell=real_cell,
        unrolled_hkl=unrolled_hkl, asu_idx=asu_idx, N=N,
        E_obs_b=E_obs_b, V_b=V_b, eImove_prefac=eImove_prefac,
        sqrt_mean_per_m=sqrt_mean_per_m, centric_b=centric_b,
        dw_per_m=dw_per_m, dtype=dtype, batch_size=batch_size,
    )


def _llg_for_orientations(
    ctx: _LLGContext,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Per-orientation m_LETF1 LLG for a batch of Edmonds-ZYZ Euler angles.

    Returns a ``(n_orient,)`` tensor of LLG values. The calc orbit-sum is over
    the deduped mates: evaluate ``|E_calc|²`` on ``ctx.unrolled_hkl`` then
    ``scatter_add`` back per ASU reflection. Same Phaser logRel math as before.
    """
    R_all = rotation_matrix_from_edmonds_euler_batch(
        alpha.to(torch.float64), beta.to(torch.float64), gamma.to(torch.float64),
    ).transpose(-1, -2).to(torch.float32)                       # (n_orient, 3, 3)
    n_orient = R_all.shape[0]
    sqrt_mean_b = ctx.sqrt_mean_per_m.unsqueeze(0)              # (1, M)
    dw_b = ctx.dw_per_m.unsqueeze(0) if ctx.dw_per_m is not None else None

    chunks: List[torch.Tensor] = []
    for start in range(0, n_orient, ctx.batch_size):
        R_batch = R_all[start:start + ctx.batch_size]           # (B, 3, 3)
        F_calc_m = ctx.interpolator.evaluate(
            R_batch, ctx.unrolled_hkl, ctx.real_cell, return_amplitude=True,
        ).to(ctx.dtype)                                         # (B, M)
        if dw_b is not None:
            F_calc_m = F_calc_m * dw_b
        E_calc_m = F_calc_m / sqrt_mean_b                       # (B, M)
        Esq_m = E_calc_m * E_calc_m                             # (B, M)
        B = Esq_m.shape[0]
        sum_per_h = torch.zeros(
            B, ctx.N, dtype=Esq_m.dtype, device=Esq_m.device,
        )
        idx = ctx.asu_idx.unsqueeze(0).expand(B, -1)            # (B, M)
        sum_per_h.scatter_add_(1, idx, Esq_m)                   # (B, N)
        eImove = ctx.eImove_prefac * sum_per_h                  # (B, N)
        sqrt_eImove = eImove.clamp(min=1e-30).sqrt()
        ll_acen = phaser_log_rel_rice(ctx.E_obs_b, sqrt_eImove, ctx.V_b)
        ll_cen = phaser_log_rel_woolfson(ctx.E_obs_b, sqrt_eImove, ctx.V_b)
        ll = torch.where(ctx.centric_b, ll_cen, ll_acen)        # (B, N)
        chunks.append(ll.sum(dim=-1))                           # (B,)
    return torch.cat(chunks)


@dataclass
class _SimLLGContext:
    """Context for the per-candidate-σ_A Sim-LLG surface (no orbit sum).

    Unlike :class:`_LLGContext` (fixed σ_A m_LETF1), this surface FITS σ_A per
    shell for each orientation via :func:`llg_for_rotation_batch`. The fixed-σ_A
    m_LETF1 surface is locally mis-peaked on high-sym/tNCS cases; the per-candidate
    fit re-shapes it so the local maximum sits at the true orientation (the
    property `sim_mlrf_rescore` already exhibits). Used as a ``llg_fn`` for
    :func:`quadratic_llg_refine`.
    """

    interpolator: LattmanLoveInterpolator
    real_cell: object
    hkl: torch.Tensor            # (N, 3)
    F_obs: torch.Tensor          # (N,)
    shell_idx: torch.Tensor      # (N,) int64
    n_shells: int
    E_obs: torch.Tensor          # (N,)
    centric: torch.Tensor        # (N,) bool
    shell_weights: Optional[torch.Tensor]
    n_D_grid: int
    interp_var: Optional[torch.Tensor]
    batch_size: int


def _build_sim_llg_context(
    F_obs: torch.Tensor,
    hkl_real: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    interpolator: LattmanLoveInterpolator,
    real_cell,
    *,
    n_shells: int = 10,
    n_D_grid: int = 21,
    batch_size: int = 64,
    auto_variance_weights: bool = True,
    interp_var: Optional[torch.Tensor] = None,
) -> _SimLLGContext:
    """Build the rotation-independent context for the Sim-LLG surface."""
    shell_idx = _equal_count_shell_idx(s_mag, n_shells)
    E_obs = _normalize_to_e(F_obs, shell_idx, n_shells)
    shell_weights = None
    if auto_variance_weights:
        from .sh import compute_patterson_shell_variance
        patt_obs = (E_obs.to(torch.float64) ** 2) - 1.0
        var_p = compute_patterson_shell_variance(patt_obs, shell_idx, P=n_shells)
        w = 1.0 / var_p.sqrt()
        w = w * (n_shells / w.sum().clamp(min=1e-30))
        shell_weights = w.to(F_obs.dtype)
    return _SimLLGContext(
        interpolator=interpolator, real_cell=real_cell, hkl=hkl_real,
        F_obs=F_obs, shell_idx=shell_idx, n_shells=n_shells, E_obs=E_obs,
        centric=centric.to(torch.bool), shell_weights=shell_weights,
        n_D_grid=n_D_grid, interp_var=interp_var, batch_size=batch_size,
    )


def _sim_llg_for_orientations(
    ctx: _SimLLGContext,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Per-orientation Sim-LLG (per-candidate σ_A fit), drop-in ``llg_fn``."""
    R_all = rotation_matrix_from_edmonds_euler_batch(
        alpha.to(torch.float64), beta.to(torch.float64), gamma.to(torch.float64),
    ).transpose(-1, -2).to(torch.float32)
    n_orient = R_all.shape[0]
    chunks: List[torch.Tensor] = []
    for start in range(0, n_orient, ctx.batch_size):
        R_batch = R_all[start:start + ctx.batch_size]
        F_calc = ctx.interpolator.evaluate(
            R_batch, ctx.hkl, ctx.real_cell, return_amplitude=True,
        ).to(ctx.F_obs.dtype)                                  # (B, N)
        llg = llg_for_rotation_batch(
            F_obs=ctx.F_obs, shell_idx=ctx.shell_idx, n_shells=ctx.n_shells,
            E_obs=ctx.E_obs, centric=ctx.centric, F_calc=F_calc,
            shell_weights=ctx.shell_weights, n_D_grid=ctx.n_D_grid,
            interp_var=ctx.interp_var,
        )
        chunks.append(llg)
    return torch.cat(chunks)


def _euler_batch_from_matrices(R: torch.Tensor):
    """(K,3,3) → three (K,) float64 Euler-angle tensors (Edmonds ZYZ).

    Loops the scalar :func:`edmonds_euler_from_rotation_matrix` (K is small —
    the top-K refine set), returning tensors ready for
    :func:`_llg_for_orientations`.
    """
    a, b, g = [], [], []
    for k in range(R.shape[0]):
        aa, bb, gg = edmonds_euler_from_rotation_matrix(R[k])
        a.append(aa)
        b.append(bb)
        g.append(gg)
    return (
        torch.tensor(a, dtype=torch.float64),
        torch.tensor(b, dtype=torch.float64),
        torch.tensor(g, dtype=torch.float64),
    )


def quadratic_llg_refine(
    peaks: List[RotationPeak],
    ctx: _LLGContext,
    *,
    k_refine: int = 20,
    step_deg: float = 1.5,
    n_grid: int = 3,
    iterations: int = 1,
    max_move_deg: Optional[float] = None,
    llg_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    verbose: int = 0,
) -> List[RotationPeak]:
    """Sub-grid refinement of the top-``k_refine`` peaks on the ML-LLG surface.

    For each peak, sample the LLG on a local **axis-angle** grid around the
    orientation, fit a 3-D paraboloid in the tangent space, and step to the vertex
    (a Newton step on the LLG). Axis-angle (not Euler α,β,γ) perturbation keeps the
    local metric isotropic and avoids the β→0/π gimbal degeneracy where the FRF
    returns many peaks. Guards (Hessian negative-definite + vertex inside the
    sampled box) fall back to the best sampled grid point, so the refined peak can
    never score below its grid value. Refined peaks are re-ranked by their (truly
    re-evaluated) LLG; peaks beyond ``k_refine`` are appended unchanged.

    Reliable (sub-degree recovery from grid-resolution hits) on well-behaved
    crystals; on high-symmetry / tNCS cases the m_LETF1 surface is locally
    mis-peaked (~3–4° off truth) so refinement can WALK AWAY from a good hit —
    use ``max_move_deg`` to bound that.

    Parameters
    ----------
    peaks
        Rescored candidate orientations (Edmonds ZYZ), best-first.
    ctx
        The :class:`_LLGContext` built for the same data (its
        :func:`_llg_for_orientations` defines the surface being optimised).
    k_refine
        Number of leading peaks to refine.
    step_deg
        Half-width of the local axis-angle grid (degrees) and the per-iteration
        capture radius. Default 1.5 ≈ grid_sampling/2 (the FRF grid half-step).
    n_grid
        Samples per tangent axis (3 → 27 orientations per peak).
    iterations
        Newton iterations; each re-centres and halves the grid half-width.
    max_move_deg
        Safety cap: if the refined orientation moves more than this (geodesic
        degrees) from the input peak, keep the input peak instead. ``None``
        disables the cap. Protects against the mis-peaked-surface failure mode.
    llg_fn
        Surface to optimise: a callable ``(alpha,beta,gamma) -> (M,)`` LLG. If
        ``None``, uses the m_LETF1 surface ``_llg_for_orientations(ctx, ...)``.
        Pass a per-candidate-σ_A Sim surface (:func:`_sim_llg_for_orientations`)
        when the fixed-σ_A m_LETF1 surface is locally mis-peaked (high-sym/tNCS).
    """
    if not peaks:
        return []
    if llg_fn is None:
        def llg_fn(a, b, g):
            return _llg_for_orientations(ctx, a, b, g)
    k = min(k_refine, len(peaks))
    head = peaks[:k]
    tail = peaks[k:]

    # R0 for the head peaks (Edmonds ZYZ, un-transposed — the convention
    # `_llg_for_orientations` consumes after its own transpose).
    a0 = torch.tensor([p.alpha for p in head], dtype=torch.float64)
    b0 = torch.tensor([p.beta for p in head], dtype=torch.float64)
    g0 = torch.tensor([p.gamma for p in head], dtype=torch.float64)
    R0 = rotation_matrix_from_edmonds_euler_batch(a0, b0, g0)    # (k, 3, 3)
    R0_orig = R0.clone()                                         # for the move cap

    radius = math.radians(step_deg)
    for _ in range(max(1, iterations)):
        # Local tangent grid (G, 3), shared across peaks; rebuilt per iteration
        # so a 2nd pass zooms in.
        lin = torch.linspace(-radius, radius, n_grid, dtype=torch.float64)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        omegas = torch.stack(
            [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1,
        )                                                       # (G, 3)
        G = omegas.shape[0]
        x, y, z = omegas[:, 0], omegas[:, 1], omegas[:, 2]
        ones = torch.ones_like(x)
        # Design matrix Φ (G,10): [1, x,y,z, x²,y²,z², xy,xz,yz].
        Phi = torch.stack(
            [ones, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z], dim=-1,
        )                                                       # (G, 10)

        # Grid orientations: R = rodrigues(ω) @ R0, for every (peak, grid point).
        Rloc = axis_angle_to_matrix(omegas)                     # (G, 3, 3)
        R_grid = torch.einsum("gij,kjl->kgil", Rloc, R0)        # (k, G, 3, 3)
        a_t, b_t, g_t = _euler_batch_from_matrices(
            R_grid.reshape(k * G, 3, 3),
        )
        llg = llg_fn(a_t, b_t, g_t).reshape(k, G)                     # (k, G)

        # Batched quadratic fit via ridge-stabilised normal equations:
        # θ = (ΦᵀΦ + λI)⁻¹ Φᵀ llg.  ΦᵀΦ is shared; only the RHS varies per peak.
        PtP = Phi.t() @ Phi                                     # (10, 10)
        PtP = PtP + 1e-9 * torch.eye(10, dtype=PtP.dtype)
        rhs = torch.einsum("gd,kg->kd", Phi, llg.to(torch.float64))   # (k, 10)
        theta = torch.linalg.solve(
            PtP.unsqueeze(0).expand(k, -1, -1), rhs.unsqueeze(-1),
        ).squeeze(-1)                                           # (k, 10)

        # Gradient b and Hessian H of the paraboloid (in tangent coords).
        bvec = theta[:, 1:4]                                    # (k, 3)
        H = torch.zeros(k, 3, 3, dtype=torch.float64)
        H[:, 0, 0] = 2.0 * theta[:, 4]
        H[:, 1, 1] = 2.0 * theta[:, 5]
        H[:, 2, 2] = 2.0 * theta[:, 6]
        H[:, 0, 1] = H[:, 1, 0] = theta[:, 7]
        H[:, 0, 2] = H[:, 2, 0] = theta[:, 8]
        H[:, 1, 2] = H[:, 2, 1] = theta[:, 9]

        best_grid = llg.argmax(dim=1)                           # (k,)
        new_R0 = R0.clone()
        n_accept = 0
        for kk in range(k):
            accept = False
            try:
                eig = torch.linalg.eigvalsh(H[kk])
                if bool((eig < 0).all()):                       # genuine maximum
                    xstar = torch.linalg.solve(H[kk], -bvec[kk])  # (3,)
                    if torch.isfinite(xstar).all() and float(xstar.norm()) <= radius:
                        new_R0[kk] = axis_angle_to_matrix(xstar) @ R0[kk]
                        accept = True
                        n_accept += 1
            except Exception:
                accept = False
            if not accept:
                new_R0[kk] = R_grid[kk, best_grid[kk]]
        R0 = new_R0
        radius = radius / 2.0
        if verbose > 1:
            print(
                f"  quadratic_llg_refine: {n_accept}/{k} vertices accepted "
                f"(rest fell back to grid max)",
                flush=True,
            )

    # Safety cap: revert any peak whose total move from the input exceeds
    # ``max_move_deg`` (geodesic). On a locally mis-peaked surface (high-sym /
    # tNCS) the refinement walks toward a spurious LLG max ~3–4° away; capping
    # the move means a good hit can never be degraded by more than the cap.
    if max_move_deg is not None:
        cos_cap = math.cos(math.radians(max_move_deg))
        n_revert = 0
        for kk in range(k):
            trace = torch.einsum("ij,ij->", R0[kk], R0_orig[kk])
            cos_move = float(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
            if cos_move < cos_cap:                  # moved further than the cap
                R0[kk] = R0_orig[kk]
                n_revert += 1
        if verbose > 1 and n_revert:
            print(
                f"  quadratic_llg_refine: reverted {n_revert}/{k} peaks that "
                f"moved > {max_move_deg}° (mis-peaked-surface guard).",
                flush=True,
            )

    # Final TRUE LLG at the refined orientations (never trust the paraboloid).
    af, bf, gf = _euler_batch_from_matrices(R0)
    llg_final = llg_fn(af, bf, gf)                             # (k,)
    if llg_final.numel() > 1:
        std_t = llg_final.std().clamp(min=1e-30)
        sig = (llg_final - llg_final.mean()) / std_t
    else:
        sig = torch.zeros_like(llg_final)

    af_l, bf_l, gf_l = af.tolist(), bf.tolist(), gf.tolist()
    llg_l, sig_l = llg_final.tolist(), sig.tolist()
    refined = [
        RotationPeak(
            alpha=af_l[i], beta=bf_l[i], gamma=gf_l[i],
            score=llg_l[i], sigma=sig_l[i],
        )
        for i in range(k)
    ]
    refined.sort(key=lambda p: p.score, reverse=True)
    return refined + tail


def m_letf1_rescore(
    peaks: List[RotationPeak],
    F_obs: torch.Tensor,
    hkl_real: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    interpolator: LattmanLoveInterpolator,
    real_cell,
    spacegroup,
    *,
    n_shells: int = 20,
    n_refine: Optional[int] = None,
    batch_size: int = 50,
    sigma_a: Optional[torch.Tensor] = None,
    eps_factor: Optional[torch.Tensor] = None,
    verbose: int = 0,
    # --- Phaser model-prep knobs (all default OFF; see frf/preprocessing.py) ---
    apply_bulk_solvent: bool = False,
    solvent_fsol: float = 0.95,
    solvent_bsol: float = 300.0,
    vrms_strategy: str = "fixed",          # "fixed" (legacy delta_vrms=0.5) or "oeffner"
    vrms_n_residues: Optional[int] = None,  # required if vrms_strategy="oeffner"
    vrms_identity: float = 1.0,
    apply_wilson_b: bool = False,
    wilson_b_value: Optional[float] = None,  # if None and apply_wilson_b=True, fitted from data
    scat_mode: str = "legacy",  # "legacy" (per-shell calc norm) | "absolute" (global)
) -> List[RotationPeak]:
    """Phaser-faithful ``m_LETF1`` rescore (DataMR.cc:1326-1429).

    Thin wrapper around :func:`_build_llg_context` + :func:`_llg_for_orientations`.

    Upgrades over :func:`sim_mlrf_rescore`:

    1. **Unique-orbit symmetry sum on calc** — for each obs reflection ``h``, the
       expected moving-model intensity is
       ``eImove(h) = ε(h)·σ_A²·(1/n_ops)·Σ_{distinct mates} |E_calc(R^T·S_k·h)|²``
       summed over the **distinct** orbit mates only (Phaser's
       ``if(!duplicate(isym))``, DataMR.cc:1371-1404), via
       :func:`torchref.experimental.alignment.frf.preprocessing.epsilon_aware_unroll` +
       ``scatter_add``. Summing all ``n_ops`` raw mates over-weights axial
       reflections by ε(h) and orientation-blinds high-symmetry spacegroups.

    2. **Per-reflection variance budget** ``V(h) = ε(h) − σ_A²(s)·n_mol`` from
       :func:`torchref.experimental.alignment.frf.preprocessing.compute_v_budget`
       (DataMR.cc:949,1411). For cross-rotation with no fixed model.

    3. **Phaser ``logRelRice`` / ``logRelWoolfson``** as the per-reflection LL
       formula (RiceWoolfson.cc:25-74), commensurable with Phaser's m_LETF1
       output. Different normalisation from our generic
       :func:`rice_log_likelihood` (factor of 2 in the Bessel argument; ``V`` is
       twice the standard Rice variance for acentric).

    Returns peaks ranked by descending LL with ``score = LL`` and
    ``sigma = (LL − μ_batch) / σ_batch``, drop-in for downstream consumers.

    Parameters
    ----------
    peaks
        Candidate orientations from the FRF, ZYZ Edmonds Euler.
    F_obs, hkl_real, s_mag, centric
        Per-reflection obs arrays (anisotropy-corrected F_obs is fine).
    interpolator, real_cell
        ``LattmanLoveInterpolator`` for the model molecular transform and the
        crystal real cell.
    spacegroup : SpaceGroup
        The crystal's space group. Passed as the object rather than its
        ``matrices`` because the multiplicity this needs is a method on it:
        ``epsilon(hkl, friedel=False)``, the conventional count, which a bare
        tensor of rotations cannot answer.
    sigma_a : (N,) tensor, optional
        Per-reflection σ_A. If ``None``, fitted on-the-fly from the identity
        rotation's |F_calc| via :func:`fit_sigma_a_per_shell` and interpolated
        per shell.
    eps_factor : (N,) tensor, optional
        Per-reflection multiplicity ε(h). If ``None``, computed via
        :meth:`torchref.symmetry.symmetry.Symmetry.epsilon` with ``friedel=False``.
    n_refine, batch_size, verbose
        As in :func:`sim_mlrf_rescore`.
    """
    if not peaks:
        return []
    if n_refine is None:
        n_refine = len(peaks)
    head = peaks[:n_refine]
    tail = peaks[n_refine:]

    ctx = _build_llg_context(
        F_obs, hkl_real, s_mag, centric, interpolator, real_cell, spacegroup,
        n_shells=n_shells, batch_size=batch_size, sigma_a=sigma_a,
        eps_factor=eps_factor, apply_bulk_solvent=apply_bulk_solvent,
        solvent_fsol=solvent_fsol, solvent_bsol=solvent_bsol,
        vrms_strategy=vrms_strategy, vrms_n_residues=vrms_n_residues,
        vrms_identity=vrms_identity, apply_wilson_b=apply_wilson_b,
        wilson_b_value=wilson_b_value, scat_mode=scat_mode,
    )

    alpha_t = torch.tensor([p.alpha for p in head], dtype=torch.float64)
    beta_t = torch.tensor([p.beta for p in head], dtype=torch.float64)
    gamma_t = torch.tensor([p.gamma for p in head], dtype=torch.float64)
    llgs_t = _llg_for_orientations(ctx, alpha_t, beta_t, gamma_t)
    if verbose > 1:
        print(f"  m_LETF1 scored {len(head)} peaks", flush=True)

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


