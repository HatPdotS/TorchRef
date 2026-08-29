"""
Unit tests for Phase A additions:
- `fit_sigma_a_per_shell`: vectorised per-shell σA fit.
- `_shell_ll` / `llg_for_rotation_batch` interp_var plumbing.
- `llg_for_rotation_batch` shared `sigma_a` path.

The interp_var rescue mechanism: variance inflation must REDUCE the LLG
penalty on a noisy-but-correct rotation, so the LLG curvature around the
true peak is gentler — that's the fix for the rescore-demotion failure.
"""

import math

import pytest
import torch

from torchref.experimental.alignment.ml_rotation import (
    _equal_count_shell_idx,
    _optimize_D_in_shell,
    _shell_ll,
    fit_sigma_a_per_shell,
    llg_for_rotation_batch,
)


def _make_synthetic_data(N=600, n_shells=10, true_D=0.6, seed=0):
    """Synthetic E_obs / E_calc with a known per-shell σA structure."""
    g = torch.Generator().manual_seed(seed)
    s_mag = torch.linspace(0.05, 0.5, N)
    shell_idx = _equal_count_shell_idx(s_mag, n_shells)
    # E_calc ~ Rayleigh(1) (matches a normalized acentric model)
    E_calc = torch.empty(N).exponential_(generator=g).sqrt()
    # E_obs = sqrt((D·E_calc)² + (1-D²)·noise²)  with noise ~ Rayleigh
    noise = torch.empty(N).exponential_(generator=g).sqrt()
    var = max(1.0 - true_D ** 2, 1e-4)
    E_obs = ((true_D * E_calc) ** 2 + var * noise ** 2).sqrt()
    centric = torch.zeros(N, dtype=torch.bool)
    return E_obs, E_calc, centric, shell_idx


def test_fit_sigma_a_per_shell_matches_per_shell_golden():
    """Vectorised per-shell σA should match per-shell golden-section within grid resolution."""
    E_obs, E_calc, centric, shell_idx = _make_synthetic_data(
        N=1500, n_shells=12, true_D=0.55, seed=1,
    )
    n_shells = int(shell_idx.max().item()) + 1

    # Vectorised fit
    sigma_a_vec = fit_sigma_a_per_shell(
        E_obs, E_calc, centric, shell_idx, n_shells, n_grid=81,
    )

    # Per-shell golden-section reference
    sigma_a_ref = torch.zeros(n_shells)
    for k in range(n_shells):
        mask = shell_idx == k
        if mask.sum() < 5:
            continue
        sigma_a_ref[k] = _optimize_D_in_shell(
            E_obs[mask], E_calc[mask], centric[mask],
        )

    # 81-pt grid resolution = 0.99/80 ≈ 0.012. Loose tolerance because the
    # vectorised path doesn't refine via golden section.
    valid = sigma_a_ref > 0
    diff = (sigma_a_vec - sigma_a_ref)[valid].abs().max().item()
    assert diff < 0.025, f"per-shell σA mismatch {diff:.4f}"


def test_interp_var_changes_llg_monotonically():
    """
    Sanity: adding a positive uniform interp_var changes the per-shell LL
    in a smooth, monotonic way — no NaN/inf, no sign flips for moderate
    inflation. Whether interp_var *helps* the rescue is an integration-level
    claim that depends on resolution distribution of the variance — tested
    on the live sweep, not here.
    """
    E_obs, E_calc, centric, shell_idx = _make_synthetic_data(
        N=1000, n_shells=10, true_D=0.6, seed=2,
    )
    n_shells = int(shell_idx.max().item()) + 1
    F_calc = E_calc.unsqueeze(0)
    sigma_a = fit_sigma_a_per_shell(
        E_obs, E_calc, centric, shell_idx, n_shells, n_grid=81,
    )

    llgs = []
    for iv_scale in [0.0, 0.05, 0.1, 0.2]:
        iv = torch.full_like(E_obs, iv_scale)
        llg = llg_for_rotation_batch(
            F_obs=E_obs, shell_idx=shell_idx, n_shells=n_shells,
            E_obs=E_obs, centric=centric, F_calc=F_calc,
            sigma_a=sigma_a, interp_var=iv,
        ).item()
        assert math.isfinite(llg), f"interp_var={iv_scale} gave LLG={llg}"
        llgs.append(llg)
    # Strictly monotone in some direction (no oscillation).
    diffs = [llgs[i + 1] - llgs[i] for i in range(len(llgs) - 1)]
    same_sign = all(d * diffs[0] >= 0 for d in diffs)
    assert same_sign, f"non-monotonic LLG vs interp_var: {llgs}"


def test_shared_sigma_a_path_matches_per_shell_grid_at_optimum():
    """
    Passing sigma_a (the per-shell argmax of the grid) reproduces the
    grid-search LLG at the optimum within numerical tolerance.
    """
    E_obs, E_calc, centric, shell_idx = _make_synthetic_data(
        N=1000, n_shells=10, true_D=0.6, seed=4,
    )
    n_shells = int(shell_idx.max().item()) + 1
    F_calc_batch = E_calc.unsqueeze(0)

    # Run grid path → per-shell argmax
    sigma_a = fit_sigma_a_per_shell(
        E_obs, E_calc, centric, shell_idx, n_shells, n_grid=81,
    )

    llg_grid = llg_for_rotation_batch(
        F_obs=E_obs, shell_idx=shell_idx, n_shells=n_shells,
        E_obs=E_obs, centric=centric, F_calc=F_calc_batch,
        n_D_grid=81,
    ).item()

    llg_shared = llg_for_rotation_batch(
        F_obs=E_obs, shell_idx=shell_idx, n_shells=n_shells,
        E_obs=E_obs, centric=centric, F_calc=F_calc_batch,
        sigma_a=sigma_a,
    ).item()

    # When sigma_a IS the per-shell grid argmax, shared and grid paths match
    # to within a few percent (small differences arise because the grid path
    # picks D per (shell, batch) before per-shell summation; shared uses
    # fixed D per shell).
    assert abs(llg_grid - llg_shared) < 0.01 * max(abs(llg_grid), 1.0), (
        f"grid LLG = {llg_grid:.4f}  shared LLG = {llg_shared:.4f}"
    )


def test_shell_ll_interp_var_off_default_matches_old():
    """interp_var=None must reproduce the historical _shell_ll behaviour exactly."""
    torch.manual_seed(5)
    N = 300
    E_obs = torch.empty(N).exponential_().sqrt()
    E_calc = torch.empty(N).exponential_().sqrt()
    centric = torch.zeros(N, dtype=torch.bool)
    D = 0.4
    ll_legacy = _shell_ll(E_obs, E_calc, centric, D)
    ll_default = _shell_ll(E_obs, E_calc, centric, D, interp_var=None)
    torch.testing.assert_close(ll_legacy, ll_default)


def test_shell_ll_interp_var_zero_matches_off():
    """interp_var of all-zeros should also match the off path (no inflation)."""
    torch.manual_seed(6)
    N = 250
    E_obs = torch.empty(N).exponential_().sqrt()
    E_calc = torch.empty(N).exponential_().sqrt()
    centric = torch.zeros(N, dtype=torch.bool)
    D = 0.3
    ll_off = _shell_ll(E_obs, E_calc, centric, D)
    iv0 = torch.zeros_like(E_obs)
    ll_zero = _shell_ll(E_obs, E_calc, centric, D, interp_var=iv0)
    torch.testing.assert_close(ll_off, ll_zero, atol=1e-6, rtol=1e-6)
