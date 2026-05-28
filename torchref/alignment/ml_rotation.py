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

from .frf.ball_search import (
    RotationPeak,
    edmonds_euler_from_rotation_matrix,
    rotation_matrix_from_edmonds_euler,
    rotation_matrix_from_edmonds_euler_batch,
)
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
# Phaser-faithful m_LETF1 rescore: NSYMP calc sum + V(h) budget + Rice/Woolfson
# logRel formulas (DataMR.cc:1326-1429).
# =============================================================================


def m_letf1_rescore(
    peaks: List[RotationPeak],
    F_obs: torch.Tensor,
    hkl_real: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    interpolator: LattmanLoveInterpolator,
    real_cell,
    sym_mats: torch.Tensor,
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
) -> List[RotationPeak]:
    """Phaser-faithful ``m_LETF1`` rescore (DataMR.cc:1326-1429).

    Upgrades over :func:`sim_mlrf_rescore`:

    1. **NSYMP symmetry sum on calc** — for each obs reflection ``h``, the
       expected moving-model intensity is
       ``eImove(h) = Σ_isym σ_A²(s) · |F_calc(R^T · S_isym · h)|²``
       summed over the ``NSYMP`` spacegroup rotation operators
       (DataMR.cc:1371-1404). Implemented vectorised: pre-compute the orbit
       ``hkl_unroll`` of shape ``(N, n_ops, 3)``, flatten to
       ``(N·n_ops, 3)``, evaluate the LL interpolator once per orientation,
       view back, square + sum over the symop dim.

    2. **Per-reflection variance budget** ``V(h) = ε(h) − σ_A²(s)·n_mol`` from
       :func:`torchref.alignment.frf.preprocessing.compute_v_budget`
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
    sym_mats : (n_ops, 3, 3) tensor
        Spacegroup rotation operators in the reciprocal (hkl) basis.
    sigma_a : (N,) tensor, optional
        Per-reflection σ_A. If ``None``, fitted on-the-fly from the identity
        rotation's |F_calc| via :func:`fit_sigma_a_per_shell` and interpolated
        per shell.
    eps_factor : (N,) tensor, optional
        Per-reflection multiplicity ε(h). If ``None``, computed via
        :func:`torchref.alignment.frf.preprocessing.compute_epsilon`.
    n_refine, batch_size, verbose
        As in :func:`sim_mlrf_rescore`.
    """
    if not peaks:
        return []
    if n_refine is None:
        n_refine = len(peaks)
    head = peaks[:n_refine]
    tail = peaks[n_refine:]

    device = F_obs.device
    dtype = F_obs.dtype
    n_ops = int(sym_mats.shape[0])
    N = hkl_real.shape[0]

    # 1. Per-shell binning + Wilson-normalised E_obs (E in Phaser notation).
    shell_idx = _equal_count_shell_idx(s_mag, n_shells)
    E_obs = _normalize_to_e(F_obs, shell_idx, n_shells)

    # 2. ε(h) per reflection.
    if eps_factor is None:
        from .frf.preprocessing import compute_epsilon
        eps_factor = compute_epsilon(hkl_real.to(torch.long), sym_mats).to(dtype)
    eps_factor = eps_factor.to(device)

    # 3. σ_A per reflection — Luzzati formula from a coordinate-error parameter
    #    ``delta_vrms_A`` (default 0.5 Å, matching the validated FRF v19 config).
    #    This is the Phaser-faithful choice: Phaser's σ_A per shell is pre-fit
    #    via a Wilson/Luzzati-style formula that depends only on (s, ΔVRMS) and
    #    does NOT require an aligned model. Data-fit alternatives
    #    (`fit_sigma_a_per_shell`) need a meaningful obs-calc alignment, which
    #    we don't have a priori — at any misaligned reference (identity or
    #    even the top FRF peak when truth is buried) the fit returns ~0 and
    #    the LL becomes orientation-blind (the 4BX9 rank-481 / 2DQ6 rank-235
    #    failure modes in v22/v23).
    #
    #    Always need the calc shell-scale for E-normalisation; use identity
    #    (rotation-invariant — sphere permutation, shell sums preserved).
    I_eye = torch.eye(3, dtype=torch.float32, device=device)
    F_calc_ref = interpolator.evaluate(
        I_eye, hkl_real, real_cell, return_amplitude=True,
    ).to(dtype).squeeze(0)  # (N,)
    sqrt_mean_F2_calc_per_h = _per_shell_sqrt_mean(F_calc_ref, shell_idx, n_shells).to(device)

    # Optional Wilson-B match (EnsemblePDB.cc:793-851). Compute once from the
    # identity-rotation F_calc reference (rotation-invariant shell statistic),
    # apply as Debye-Waller multiplier `exp(-B·s²/4)` to F_calc inside the batch.
    if apply_wilson_b and wilson_b_value is None:
        from .frf.preprocessing import fit_relative_wilson_b
        wilson_b_value = fit_relative_wilson_b(
            F_obs, F_calc_ref, s_mag, n_shells=n_shells,
        )
    wilson_b_value = float(wilson_b_value or 0.0)
    if apply_wilson_b and abs(wilson_b_value) > 1e-6:
        dw = torch.exp(-wilson_b_value * (s_mag * s_mag) / 4.0).to(dtype).to(device)
    else:
        dw = None  # skip the elementwise mul if a no-op

    if sigma_a is None:
        # σ_A: Phaser-faithful Luzzati with optional Oeffner vrms + bulk solvent.
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
        sigma_a = compute_sigma_a_luzzati(s_mag, delta_vrms_A=delta_vrms_A).to(dtype).to(device)
        if apply_bulk_solvent:
            from .frf.preprocessing import bulk_solvent_factor
            sol = bulk_solvent_factor(
                s_mag, fsol=solvent_fsol, bsol=solvent_bsol,
            ).to(dtype).to(device)
            sigma_a = sigma_a * sol
    sigma_a = sigma_a.to(device)
    sigma_a2 = sigma_a * sigma_a

    # 4. V(h) — rotation-independent variance budget. Phaser-faithful per
    #    DataMR.cc:1342-1345: `thisV = scatFactor · NSYMP · DFAC² · σ_A²`.
    #    With `scatFactor = 1/NSYMP` (single ensemble, fracMove=1) this collapses
    #    to `thisV = σ_A²`, so V = ε − σ_A² — **independent of NSYMP**. Using
    #    `n_mol=n_ops` (as the initial implementation did) drove V negative on
    #    high-symmetry spacegroups (4BX9 NSYMP=8, V_clamp blew up the LL).
    from .frf.preprocessing import compute_v_budget
    V = compute_v_budget(eps_factor, sigma_a, n_mol=1)  # (N,)

    # 5. Orbit hkl pre-compute (rotation-independent; reused per batch).
    sym_mats_f = sym_mats.to(torch.float64).to(device)
    hkl_f = hkl_real.to(torch.float64).to(device)
    # (N, n_ops, 3) — for each obs h, all n_ops sym-equivalents in hkl space.
    hkl_unroll = torch.einsum("kij,nj->nki", sym_mats_f, hkl_f)
    hkl_flat = hkl_unroll.reshape(-1, 3)                              # (N·n_ops, 3)

    # 6. Build candidate rotation matrices. Same convention as sim_mlrf_rescore:
    #    transpose the Edmonds Euler matrix because the peak encodes "rotation
    #    applied to model coords"; we need the inverse for evaluating
    #    F_calc(R^T · h).
    alpha_t = torch.tensor([p.alpha for p in head], dtype=torch.float64)
    beta_t = torch.tensor([p.beta for p in head], dtype=torch.float64)
    gamma_t = torch.tensor([p.gamma for p in head], dtype=torch.float64)
    R_all = rotation_matrix_from_edmonds_euler_batch(
        alpha_t, beta_t, gamma_t,
    ).transpose(-1, -2).to(torch.float32)                              # (M, 3, 3)
    M = R_all.shape[0]

    # 7. Batched score: eImove via NSYMP-summed calc, LL via Phaser logRel.
    E_obs_b = E_obs.unsqueeze(0)        # (1, N) — broadcasts over batch dim
    V_b = V.unsqueeze(0)                # (1, N)
    # eImove pre-factor per Phaser DataMR.cc:1397: `thisEsqr *= repsn * scatFactor`
    # → `eImove = ε(h) · σ_A² · (1/n_ops) · Σ_k |E_calc(S_k h)|²`. We collapse the
    # `1/n_ops` into the pre-factor so the batch loop just does the raw sum.
    eImove_prefac = (eps_factor * sigma_a2 / float(n_ops)).unsqueeze(0)  # (1, N)
    centric_b = centric.to(torch.bool).unsqueeze(0)

    # Normaliser to convert rotated |F_calc| → |E_calc| (per-shell sqrt mean
    # from the identity-rotation reference; broadcasts over symops via the
    # last unsqueeze since rotation preserves |h| → same shell for all symops).
    sqrt_mean_b = sqrt_mean_F2_calc_per_h.unsqueeze(0).unsqueeze(-1)     # (1, N, 1)

    llg_chunks: List[torch.Tensor] = []
    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        R_batch = R_all[start:stop]                                     # (B, 3, 3)
        F_calc_flat = interpolator.evaluate(
            R_batch, hkl_flat, real_cell, return_amplitude=True,
        ).to(dtype)                                                      # (B, N·n_ops)
        F_calc = F_calc_flat.view(-1, N, n_ops)                          # (B, N, n_ops)
        # Optional Wilson-B Debye-Waller multiplier (per-reflection, broadcasts
        # over batch + symops). Applied PRE-normalisation so the per-shell sqrt
        # mean (computed from un-DW'd identity F_calc_ref) stays the right scale
        # for normalisation; the DW shifts E_calc relative to that scale, which
        # is exactly what Wilson-B matching is supposed to do.
        if dw is not None:
            F_calc = F_calc * dw.unsqueeze(0).unsqueeze(-1)
        # Normalise to E_calc (same Wilson scale as E_obs); Rice/Woolfson LL
        # only makes sense with both sides on the same per-shell scale.
        E_calc = F_calc / sqrt_mean_b
        # eImove(h) = ε(h) · σ_A²(s) · (1/n_ops) · Σ_isym |E_calc(R^T·S_isym·h)|²
        # (Phaser DataMR.cc:1371-1404; scatFactor = 1/NSYMP folds the symop sum
        # into a mean, ε(h) is the multiplicity factor `repsn`).
        eImove = eImove_prefac * (E_calc * E_calc).sum(dim=-1)           # (B, N)
        sqrt_eImove = eImove.clamp(min=1e-30).sqrt()                     # (B, N)
        ll_acen = phaser_log_rel_rice(E_obs_b, sqrt_eImove, V_b)         # (B, N)
        ll_cen = phaser_log_rel_woolfson(E_obs_b, sqrt_eImove, V_b)      # (B, N)
        ll = torch.where(centric_b, ll_cen, ll_acen)                     # (B, N)
        llg_chunks.append(ll.sum(dim=-1))                                # (B,)
        if verbose > 1:
            print(f"  m_LETF1 batch {start}-{stop}/{M}", flush=True)

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


# =============================================================================
# BRF — Brute Rotation Function (Phaser stage that refines top FRF peaks via a
# denser local rotation sampling + full Rice/Woolfson LL).
# =============================================================================


def _random_rotation_in_cone(
    n: int, radius_rad: float, generator: torch.Generator,
) -> torch.Tensor:
    """``n`` random rotation matrices uniformly inside an angular cone of given
    radius from identity. Axis ∼ uniform-on-sphere, angle ∼ uniform on
    ``[0, radius_rad]`` (proper Haar measure on the cone would weight angle by
    sin²(θ/2); for small radii this approximation is essentially uniform).

    Returns ``(n, 3, 3)`` float64 tensor.
    """
    # Random unit axes via Gaussian normalization.
    axes = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    angles = torch.rand(n, generator=generator, dtype=torch.float64) * radius_rad
    # Rodrigues: R = I + sin(θ)·K + (1−cos(θ))·K²   with K skew of axis.
    cos = angles.cos().view(-1, 1, 1)
    sin = angles.sin().view(-1, 1, 1)
    K = torch.zeros(n, 3, 3, dtype=torch.float64)
    K[:, 0, 1] = -axes[:, 2]; K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2];  K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]; K[:, 2, 1] = axes[:, 0]
    I = torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
    K2 = K @ K
    return I + sin * K + (1.0 - cos) * K2


def brf_refine(
    peaks: List[RotationPeak],
    F_obs: torch.Tensor,
    hkl_real: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    interpolator: LattmanLoveInterpolator,
    real_cell,
    sym_mats: torch.Tensor,
    *,
    n_top: int = 100,
    n_perturb: int = 10,
    angular_radius_deg: float = 3.0,
    seed: int = 42,
    verbose: int = 0,
    **m_letf1_kwargs,
) -> List[RotationPeak]:
    """Brute Rotation Function — Phaser's BRF stage (post-FRF rotation refinement).

    For each of the top ``n_top`` FRF peaks, sample ``n_perturb`` random
    rotations within an angular cone of radius ``angular_radius_deg`` from the
    peak; score all (original + perturbations) via :func:`m_letf1_rescore`.
    Returns the rescored set sorted by descending LL — the truth orientation
    typically lurks ≤ FRF-grid-spacing degrees from the FRF peak nearest to it,
    so this local refinement can recover peaks the coarse FRF grid missed.

    Total LL evaluations: ``n_top · (n_perturb + 1)``. With defaults this is
    ``100 × 11 = 1100`` — ~2× the cost of the standard top-500 rescore.

    Parameters
    ----------
    peaks
        FRF peak list (sorted by descending FRF score).
    n_top
        Refine around the top ``n_top`` FRF peaks. Should be ≥ the expected
        rank of the truth peak — for hard cases (e.g., 2DQ6 FRF rank ≈50), use
        ``n_top ≥ 100``.
    n_perturb
        Number of random rotations sampled per peak (in addition to the
        peak itself, which is always included).
    angular_radius_deg
        Cone half-angle for perturbations. Match the FRF grid resolution
        (~3° at ``grid_sampling_deg=3.0``).
    seed
        RNG seed for reproducibility.
    **m_letf1_kwargs
        Forwarded to :func:`m_letf1_rescore` — e.g. ``apply_bulk_solvent=True``,
        ``vrms_strategy="oeffner"``, ``vrms_n_residues=...``, etc.
    """
    if not peaks:
        return []
    n_top = min(n_top, len(peaks))
    top = peaks[:n_top]
    g = torch.Generator().manual_seed(int(seed))
    radius_rad = math.radians(float(angular_radius_deg))

    # One big batch of perturbations: (n_top * n_perturb, 3, 3).
    pert_R = _random_rotation_in_cone(n_top * n_perturb, radius_rad, g)
    pert_R = pert_R.view(n_top, n_perturb, 3, 3)

    out_peaks: List[RotationPeak] = []
    for i, peak in enumerate(top):
        # Include the original peak (perturbation theta=0).
        out_peaks.append(RotationPeak(
            alpha=peak.alpha, beta=peak.beta, gamma=peak.gamma,
            score=peak.score, sigma=peak.sigma,
        ))
        R_orig = rotation_matrix_from_edmonds_euler(
            peak.alpha, peak.beta, peak.gamma,
        ).to(torch.float64)
        for j in range(n_perturb):
            R_pert = R_orig @ pert_R[i, j]
            a, b, c = edmonds_euler_from_rotation_matrix(R_pert)
            out_peaks.append(RotationPeak(
                alpha=a, beta=b, gamma=c, score=peak.score, sigma=peak.sigma,
            ))

    if verbose > 0:
        print(
            f"  BRF: {len(out_peaks)} candidates "
            f"({n_top} top × ({n_perturb}+1) perturbations, ±{angular_radius_deg}°)",
            flush=True,
        )

    return m_letf1_rescore(
        out_peaks, F_obs, hkl_real, s_mag, centric, interpolator, real_cell,
        sym_mats,
        n_refine=len(out_peaks),
        verbose=verbose,
        **m_letf1_kwargs,
    )
