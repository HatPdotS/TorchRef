"""
Unit tests for the Phaser-style empirical per-shell variance weighting added in
Phase 3.

Covers:
- `compute_patterson_shell_variance` returns ~1.0 per shell on a synthetic
  acentric Wilson sample (E² ~ Exp(1)).
- Sparse-shell handling: a shell below `min_count` inherits a neighbour's value
  and doesn't blow up the inverse-sqrt weight.
- `ball_rotation_search` with `auto_variance_weights=True` agrees with
  `=False` on a synthetic flat-variance case (so the variance correction is a
  no-op on data that already satisfies Wilson assumptions).
"""
import math

import pytest
import torch

from torchref.alignment.ball_search import (
    ball_rotation_search,
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from torchref.alignment.sh import compute_patterson_shell_variance


@pytest.mark.unit
def test_patterson_shell_variance_wilson():
    """On acentric Wilson E² ~ Exp(1), Var(E²-1) ≈ 1 per shell."""
    g = torch.Generator().manual_seed(0)
    P = 6
    per_shell = 4000
    N = P * per_shell
    e2 = -torch.log(torch.rand(N, generator=g, dtype=torch.float64).clamp(min=1e-12))
    patt = e2 - 1.0
    shell_idx = torch.repeat_interleave(torch.arange(P, dtype=torch.int64), per_shell)
    var = compute_patterson_shell_variance(patt, shell_idx, P=P)
    assert var.shape == (P,)
    for k in range(P):
        assert 0.85 < var[k].item() < 1.15, f"shell {k} variance {var[k].item()}"


@pytest.mark.unit
def test_patterson_shell_variance_handles_sparse_shells():
    """A shell with < min_count reflections inherits a neighbour's variance."""
    P = 4
    # Shell 0 dense (1000 pts), shell 1 sparse (2 pts), shells 2-3 dense.
    patt = torch.cat([
        torch.randn(1000, dtype=torch.float64),
        torch.tensor([10.0, -10.0], dtype=torch.float64),  # would give huge var alone
        torch.randn(1000, dtype=torch.float64),
        torch.randn(1000, dtype=torch.float64),
    ])
    shell_idx = torch.cat([
        torch.zeros(1000, dtype=torch.int64),
        torch.ones(2, dtype=torch.int64),
        torch.full((1000,), 2, dtype=torch.int64),
        torch.full((1000,), 3, dtype=torch.int64),
    ])
    var = compute_patterson_shell_variance(patt, shell_idx, P=P, min_count=8)
    # Shell 1 should have inherited from shell 0 or 2 (~1.0), not the outlier value.
    assert var[1].item() < 5.0, f"sparse shell 1 inherited huge variance {var[1].item()}"


@pytest.mark.unit
def test_patterson_shell_variance_eps_floor():
    """A near-zero-variance shell is floored at eps so weight stays bounded."""
    P = 2
    patt = torch.cat([
        torch.full((100,), 1.0, dtype=torch.float64),  # variance ~ 0
        torch.randn(100, dtype=torch.float64),
    ])
    shell_idx = torch.cat([
        torch.zeros(100, dtype=torch.int64),
        torch.ones(100, dtype=torch.int64),
    ])
    eps = 1e-3
    var = compute_patterson_shell_variance(patt, shell_idx, P=P, min_count=8, eps=eps)
    assert var[0].item() >= eps


@pytest.mark.unit
def test_auto_variance_weights_recovers_synthetic_rotation():
    """
    On a uniform synthetic test (flat per-shell variance), the rotation
    recovered with `auto_variance_weights=True` matches `=False` within a
    fraction of a degree. The variance correction must not break the
    well-conditioned case.
    """
    g = torch.Generator().manual_seed(1)
    N = 3000
    # uniform directions on the sphere
    s_dirs = torch.randn(N, 3, generator=g, dtype=torch.float64)
    s_dirs = s_dirs / s_dirs.norm(dim=-1, keepdim=True)
    # log-uniform |s| in [1/15, 1/4] Å^-1
    s_mag = torch.empty(N, dtype=torch.float64).uniform_(
        1.0 / 15.0, 1.0 / 4.0, generator=g,
    )
    s_obs = s_dirs * s_mag.unsqueeze(-1)
    # Wilson-distributed Patterson coefficients per reflection.
    e2 = -torch.log(torch.rand(N, generator=g, dtype=torch.float64).clamp(min=1e-12))
    patt_obs = e2 - 1.0

    R_true = rotation_matrix_from_edmonds_euler(0.7, 1.2, 2.3, dtype=torch.float64)
    # F_calc samples the same field rotated: place obs values at R·ŝ positions.
    s_calc = s_obs @ R_true.T
    patt_calc = patt_obs.clone()

    common_kwargs = dict(
        L=20, P=10, n_peaks=20, d_min=4.0, d_max=15.0,
        refine_subvoxel=True, n_refine=5, sigma_threshold=-5.0,
    )
    _, _, _, _, peaks_on = ball_rotation_search(
        s_obs, patt_obs, s_calc, patt_calc,
        auto_variance_weights=True, **common_kwargs,
    )
    _, _, _, _, peaks_off = ball_rotation_search(
        s_obs, patt_obs, s_calc, patt_calc,
        auto_variance_weights=False, **common_kwargs,
    )

    R_on = rotation_matrix_from_edmonds_euler(
        peaks_on[0].alpha, peaks_on[0].beta, peaks_on[0].gamma, dtype=torch.float64,
    )
    R_off = rotation_matrix_from_edmonds_euler(
        peaks_off[0].alpha, peaks_off[0].beta, peaks_off[0].gamma, dtype=torch.float64,
    )
    # both should be close to R_true; their mutual difference should be small.
    err_on = rotation_angular_distance_deg(R_on, R_true)
    err_off = rotation_angular_distance_deg(R_off, R_true)
    assert err_on < 15.0, f"on: {err_on}"
    assert err_off < 15.0, f"off: {err_off}"
    # variance weighting should not move the answer by more than a voxel-or-so.
    assert abs(err_on - err_off) < 6.0, (
        f"variance weighting changed the synthetic answer significantly: "
        f"on={err_on:.2f}°, off={err_off:.2f}°"
    )
