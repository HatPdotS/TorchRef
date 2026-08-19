"""Unit tests for the Phaser-faithful m_LETF1 rescore.

Covers:
- ``phaser_log_rel_rice`` / ``phaser_log_rel_woolfson`` exact formula parity
  with hand-computed values from RiceWoolfson.cc:25-74.
- ``compute_v_budget`` per DataMR.cc:949,1411 (clamping, degenerate cases).
- ``m_letf1_rescore`` discriminates the truth rotation over random rotations
  on a synthetic obs+calc set.
"""
from __future__ import annotations

import math

import pytest
import torch

from torchref.experimental.alignment.distributions import (
    phaser_log_rel_rice,
    phaser_log_rel_woolfson,
)
from torchref.experimental.alignment.frf.preprocessing import compute_v_budget


def test_phaser_log_rel_rice_hand_computed():
    """logRelRice(F1, DF2, V) = logI₀(2·F1·DF2/V) − log V − (F1²+DF2²)/V."""
    # F1=1, DF2=1, V=1: logI₀(2) − 0 − 2 ≈ 0.8237 − 2 = −1.1763
    val = phaser_log_rel_rice(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
    assert abs(val.item() - (-1.1763)) < 1e-3

    # F1=2, DF2=0.5, V=2: logI₀(1.0) − log 2 − (4 + 0.25)/2
    #                   = 0.2359 − 0.6931 − 2.125 ≈ −2.582
    val = phaser_log_rel_rice(torch.tensor(2.0), torch.tensor(0.5), torch.tensor(2.0))
    assert abs(val.item() - (-2.582)) < 1e-3


def test_phaser_log_rel_woolfson_hand_computed():
    """logRelWoolfson(F1, DF2, V) = log cosh(F1·DF2/V) − ½·log V − (F1²+DF2²)/(2V)."""
    # F1=1, DF2=1, V=1: log cosh(1) − 0 − 1 ≈ 0.4339 − 1 = −0.5661
    val = phaser_log_rel_woolfson(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
    assert abs(val.item() - (-0.5661)) < 1e-3

    # F1=3, DF2=2, V=4: log cosh(1.5) − ½·log 4 − (9+4)/8
    #                 = 0.8553 − 0.6931 − 1.625 ≈ −1.463
    val = phaser_log_rel_woolfson(torch.tensor(3.0), torch.tensor(2.0), torch.tensor(4.0))
    assert abs(val.item() - (-1.463)) < 1e-3


def test_phaser_log_rel_woolfson_large_arg_stable():
    """log cosh(x) ≈ |x| − log 2 for large |x|; no overflow."""
    val = phaser_log_rel_woolfson(
        torch.tensor(50.0), torch.tensor(50.0), torch.tensor(1.0),
    )
    assert torch.isfinite(val)
    # log cosh(2500) ≈ 2500 − log 2; minus 0 minus (2500+2500)/2 = 2500
    # → ≈ 2500 − 0.693 − 2500 = −0.693
    assert abs(val.item() - (-math.log(2.0))) < 1e-2


def test_compute_v_budget_basic():
    """V(h) = ε(h) − σ_A²(s)·n_mol, clamped > 0."""
    eps = torch.tensor([1.0, 2.0, 1.0])
    sa = torch.tensor([0.3, 0.5, 0.4])
    V = compute_v_budget(eps, sa, n_mol=2)
    # h0: 1 − 0.09·2 = 0.82
    # h1: 2 − 0.25·2 = 1.50
    # h2: 1 − 0.16·2 = 0.68
    assert torch.allclose(V, torch.tensor([0.82, 1.50, 0.68]), atol=1e-6)


def test_compute_v_budget_clamps_degenerate():
    """Non-positive V (σ_A²·n_mol overshoots ε) clamps to 1e-6, not negative."""
    eps = torch.tensor([1.0])
    sa = torch.tensor([0.9])  # 0.81·2 = 1.62 > 1.0
    V = compute_v_budget(eps, sa, n_mol=2)
    assert V.item() > 0
    assert V.item() <= 1e-5  # clamp floor


def test_compute_v_budget_with_totvar_known():
    """totvar_known subtracts further from V."""
    eps = torch.tensor([1.0, 1.0])
    sa = torch.tensor([0.1, 0.1])
    totvar = torch.tensor([0.05, 0.30])
    V = compute_v_budget(eps, sa, n_mol=1, totvar_known=totvar)
    # h0: 1 − 0.01 − 0.05 = 0.94
    # h1: 1 − 0.01 − 0.30 = 0.69
    assert torch.allclose(V, torch.tensor([0.94, 0.69]), atol=1e-6)


def test_m_letf1_rescore_runs_and_ranks_truth_top():
    """Synthetic test: m_letf1_rescore should rank the truth rotation at #1
    (or near #1) over random rotations when given matching obs/calc.

    Setup: build a fake LattmanLoveInterpolator-like callable that returns
    ``|F_obs|`` at the identity rotation and noisy values at others, then check
    that the identity-rotation peak comes top of the rescored list.
    """
    pytest.importorskip("torchref.experimental.alignment.lattman_love")
    from torchref.experimental.alignment.ml_rotation import m_letf1_rescore
    from torchref.experimental.alignment.frf.types import RotationPeak

    N = 200
    torch.manual_seed(0)
    s_mag = torch.linspace(0.05, 0.4, N, dtype=torch.float64)
    F_obs = (1.0 + 0.1 * torch.randn(N, dtype=torch.float64)).abs()
    hkl = torch.randint(-10, 10, (N, 3), dtype=torch.long)
    centric = torch.zeros(N, dtype=torch.bool)
    sym_mats = torch.eye(3, dtype=torch.float64).unsqueeze(0)  # P1: only identity

    # Stub interpolator: returns F_obs (perfectly correlated) for R = identity,
    # uncorrelated noise for any other R.
    class StubLL:
        def evaluate(self, R, hkl_real, real_cell, return_amplitude=True):
            if R.dim() == 2:
                R = R.unsqueeze(0)
            B = R.shape[0]
            out = torch.empty(B, hkl_real.shape[0], dtype=torch.float64)
            for b in range(B):
                if torch.allclose(R[b].to(torch.float64), torch.eye(3, dtype=torch.float64), atol=1e-3):
                    out[b] = F_obs
                else:
                    out[b] = (1.0 + 0.1 * torch.randn(hkl_real.shape[0], dtype=torch.float64)).abs()
            return out

    class StubCell:
        @property
        def reciprocal_basis_matrix(self):
            return torch.eye(3, dtype=torch.float64)

    # Truth peak (identity rotation: α=β=γ=0) plus 9 random peaks.
    truth = RotationPeak(alpha=0.0, beta=0.0, gamma=0.0, score=1.0, sigma=5.0)
    rng = torch.Generator().manual_seed(1)
    random_peaks = [
        RotationPeak(
            alpha=float(torch.rand(1, generator=rng).item() * 2 * math.pi),
            beta=float(torch.rand(1, generator=rng).item() * math.pi),
            gamma=float(torch.rand(1, generator=rng).item() * 2 * math.pi),
            score=0.5, sigma=2.0,
        )
        for _ in range(9)
    ]
    peaks = [truth] + random_peaks

    rescored = m_letf1_rescore(
        peaks, F_obs, hkl, s_mag, centric, StubLL(), StubCell(), sym_mats,
        n_shells=5, batch_size=4,
    )
    # Truth (identity) should be among the top 3 rescored peaks (truth=identity
    # gives perfect calc match; noisy candidates should have lower LL).
    top3_eulers = [(p.alpha, p.beta, p.gamma) for p in rescored[:3]]
    assert (0.0, 0.0, 0.0) in top3_eulers, (
        f"truth (identity) not in top 3 after rescore; top3 = {top3_eulers}"
    )


def test_scat_mode_absolute_preserves_calc_intershell_shape():
    """scat_mode='absolute' uses a single GLOBAL calc scale, so a shell where the
    model scatters weakly keeps a small eImove; 'legacy' flattens every shell to
    unit variance. Verify the two modes give different (and predictable) eImove
    inter-shell weighting on a synthetic with a strong resolution-dependent
    F_calc falloff."""
    from torchref.experimental.alignment.ml_rotation import _build_llg_context, _llg_for_orientations
    from torchref.experimental.alignment.frf.preprocessing import compute_epsilon

    N = 300
    torch.manual_seed(3)
    s_mag = torch.linspace(0.05, 0.45, N, dtype=torch.float64)
    F_obs = (1.0 + 0.1 * torch.randn(N, dtype=torch.float64)).abs()
    hkl = torch.randint(-12, 12, (N, 3), dtype=torch.long)
    centric = torch.zeros(N, dtype=torch.bool)
    sym_mats = torch.eye(3, dtype=torch.float64).unsqueeze(0)  # P1

    # F_calc with a strong B-factor falloff → big inter-shell amplitude variation.
    decay = torch.exp(-40.0 * s_mag * s_mag)

    class StubLL:
        def evaluate(self, R, hkl_real, real_cell, return_amplitude=True):
            if R.dim() == 2:
                R = R.unsqueeze(0)
            B = R.shape[0]
            # Amplitude depends only on |s| (resolution) → deterministic per hkl;
            # uses the input hkl rows' index range to map back to s via norm.
            out = decay.unsqueeze(0).expand(B, hkl_real.shape[0]).clone()
            return out

    class StubCell:
        @property
        def reciprocal_basis_matrix(self):
            return torch.eye(3, dtype=torch.float64)

    common = dict(
        interpolator=StubLL(), real_cell=StubCell(), sym_mats=sym_mats,
        n_shells=6, batch_size=64,
    )
    ctx_leg = _build_llg_context(F_obs, hkl, s_mag, centric, scat_mode="legacy", **common)
    ctx_abs = _build_llg_context(F_obs, hkl, s_mag, centric, scat_mode="absolute", **common)

    # Legacy per-shell normaliser varies across shells (tracks the F_calc decay);
    # absolute is a single constant. This is the definitional difference.
    assert ctx_leg.sqrt_mean_per_m.std() > 1e-6, "legacy should vary per shell"
    assert torch.allclose(
        ctx_abs.sqrt_mean_per_m, ctx_abs.sqrt_mean_per_m[0]
    ), "absolute should be a single global scale"
    # Both modes still produce finite LLGs.
    a = torch.zeros(1, dtype=torch.float64)
    llg_leg = _llg_for_orientations(ctx_leg, a, a, a)
    llg_abs = _llg_for_orientations(ctx_abs, a, a, a)
    assert torch.isfinite(llg_leg).all() and torch.isfinite(llg_abs).all()
    assert float(llg_leg[0]) != float(llg_abs[0]), "modes should differ"


def _so3_angle_deg(R1: torch.Tensor, R2: torch.Tensor) -> float:
    """Geodesic distance on SO(3) in degrees."""
    R = R1.to(torch.float64) @ R2.to(torch.float64).T
    tr = (R[0, 0] + R[1, 1] + R[2, 2]).item()
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def test_quadratic_refine_lands_closer_to_known_max(monkeypatch):
    """The vertex of the quadratic fit on a concave LLG surface lands strictly
    closer to the known maximum orientation than the starting grid peak."""
    from torchref.experimental.alignment import ml_rotation
    from torchref.experimental.alignment.ml_rotation import quadratic_llg_refine
    from torchref.experimental.alignment.frf.types import RotationPeak
    from torchref.experimental.alignment.frf.rotation_utils import (
        axis_angle_to_matrix,
        rotation_matrix_from_edmonds_euler,
        rotation_matrix_from_edmonds_euler_batch,
    )

    # Start orientation (away from the β poles) and a known max 0.5° away.
    a0, b0, g0 = 0.5, 0.8, 1.2
    R0 = rotation_matrix_from_edmonds_euler(a0, b0, g0)
    d = torch.tensor([math.radians(0.5), 0.0, 0.0], dtype=torch.float64)
    R_true = axis_angle_to_matrix(d) @ R0

    # Synthetic concave surface: LLG = -k · geodesic_angle(R, R_true)².
    def stub(ctx, alpha, beta, gamma):
        R = rotation_matrix_from_edmonds_euler_batch(alpha, beta, gamma)
        Rrel = torch.einsum("mij,lj->mil", R, R_true)         # R · R_trueᵀ
        tr = Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2]
        ang = torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))
        return -100.0 * ang * ang

    monkeypatch.setattr(ml_rotation, "_llg_for_orientations", stub)

    peak = RotationPeak(alpha=a0, beta=b0, gamma=g0, score=0.0, sigma=0.0)
    refined = quadratic_llg_refine([peak], ctx=None, k_refine=1, step_deg=0.75, n_grid=3)
    Rref = rotation_matrix_from_edmonds_euler(
        refined[0].alpha, refined[0].beta, refined[0].gamma,
    )
    d_start = _so3_angle_deg(R0, R_true)        # ≈ 0.5°
    d_ref = _so3_angle_deg(Rref, R_true)
    assert d_ref < d_start, f"refine did not improve: {d_ref:.3f} vs {d_start:.3f}"
    assert d_ref < 0.1, f"vertex did not land on the max: {d_ref:.3f}°"
    assert refined[0].score > -1.0   # true LLG at the vertex ≈ 0 (near max)


def test_quadratic_refine_guard_falls_back_on_minimum(monkeypatch):
    """On a convex surface (a minimum, not a max) the negative-definite Hessian
    guard rejects the vertex and falls back to the best sampled grid point — so
    the refined orientation moves AWAY from R_true, never toward the minimum."""
    from torchref.experimental.alignment import ml_rotation
    from torchref.experimental.alignment.ml_rotation import quadratic_llg_refine
    from torchref.experimental.alignment.frf.types import RotationPeak
    from torchref.experimental.alignment.frf.rotation_utils import (
        axis_angle_to_matrix,
        rotation_matrix_from_edmonds_euler,
        rotation_matrix_from_edmonds_euler_batch,
    )

    a0, b0, g0 = 0.5, 0.8, 1.2
    R0 = rotation_matrix_from_edmonds_euler(a0, b0, g0)
    d = torch.tensor([math.radians(0.5), 0.0, 0.0], dtype=torch.float64)
    R_true = axis_angle_to_matrix(d) @ R0

    # CONVEX surface: +k · angle² (minimum at R_true → Hessian positive-definite).
    def stub(ctx, alpha, beta, gamma):
        R = rotation_matrix_from_edmonds_euler_batch(alpha, beta, gamma)
        Rrel = torch.einsum("mij,lj->mil", R, R_true)
        tr = Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2]
        ang = torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))
        return +100.0 * ang * ang

    monkeypatch.setattr(ml_rotation, "_llg_for_orientations", stub)

    peak = RotationPeak(alpha=a0, beta=b0, gamma=g0, score=0.0, sigma=0.0)
    refined = quadratic_llg_refine([peak], ctx=None, k_refine=1, step_deg=0.75, n_grid=3)
    Rref = rotation_matrix_from_edmonds_euler(
        refined[0].alpha, refined[0].beta, refined[0].gamma,
    )
    d_start = _so3_angle_deg(R0, R_true)
    d_ref = _so3_angle_deg(Rref, R_true)
    # Fell back to the grid sample with the largest (convex) LLG = farthest from
    # R_true: the guard prevented stepping to the minimum.
    assert d_ref > d_start, (
        f"guard failed — moved toward the minimum ({d_ref:.3f} vs {d_start:.3f})"
    )


def test_quadratic_refine_max_move_cap(monkeypatch):
    """max_move_deg reverts a peak that the (mis-peaked) surface would drag far
    from the input — bounding degradation on high-sym/tNCS cases."""
    from torchref.experimental.alignment import ml_rotation
    from torchref.experimental.alignment.ml_rotation import quadratic_llg_refine
    from torchref.experimental.alignment.frf.types import RotationPeak
    from torchref.experimental.alignment.frf.rotation_utils import (
        axis_angle_to_matrix,
        rotation_matrix_from_edmonds_euler,
        rotation_matrix_from_edmonds_euler_batch,
    )

    a0, b0, g0 = 0.5, 0.8, 1.2
    R0 = rotation_matrix_from_edmonds_euler(a0, b0, g0)
    # Surface peaks 6° away (a spurious far maximum): concave around a far point.
    d = torch.tensor([math.radians(6.0), 0.0, 0.0], dtype=torch.float64)
    R_far = axis_angle_to_matrix(d) @ R0

    def stub(ctx, alpha, beta, gamma):
        R = rotation_matrix_from_edmonds_euler_batch(alpha, beta, gamma)
        Rrel = torch.einsum("mij,lj->mil", R, R_far)
        tr = Rrel[:, 0, 0] + Rrel[:, 1, 1] + Rrel[:, 2, 2]
        ang = torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))
        return -50.0 * ang * ang

    monkeypatch.setattr(ml_rotation, "_llg_for_orientations", stub)
    peak = RotationPeak(alpha=a0, beta=b0, gamma=g0, score=0.0, sigma=0.0)
    # With a 3° step and 2 iters the surface would drag the peak several degrees;
    # cap at 1° must revert it to (essentially) the input orientation.
    refined = quadratic_llg_refine(
        [peak], ctx=None, k_refine=1, step_deg=3.0, n_grid=3, iterations=2,
        max_move_deg=1.0,
    )
    Rref = rotation_matrix_from_edmonds_euler(
        refined[0].alpha, refined[0].beta, refined[0].gamma,
    )
    moved = _so3_angle_deg(Rref, R0)
    assert moved <= 1.0 + 1e-6, f"move-cap failed: moved {moved:.3f}° > 1.0°"
