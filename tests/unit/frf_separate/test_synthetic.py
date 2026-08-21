"""Tier 2 synthetic golden-input tests for frf_separate.

End-to-end: feed a known-rotation pair into ``phaser_rotation_search``
and check the top peak is at the right Euler.
"""
from __future__ import annotations

import math

import pytest
import torch

from torchref.experimental.alignment.frf.api import phaser_rotation_search


def _random_rotation_matrix(seed: int) -> torch.Tensor:
    """Uniform random SO(3) rotation as a 3×3 matrix (Edmonds ZYZ)."""
    g = torch.Generator().manual_seed(seed)
    # Shoemake's quaternion method
    u = torch.rand(3, generator=g, dtype=torch.float64)
    q0 = math.sqrt(1 - u[0]) * math.sin(2 * math.pi * u[1])
    q1 = math.sqrt(1 - u[0]) * math.cos(2 * math.pi * u[1])
    q2 = math.sqrt(u[0])     * math.sin(2 * math.pi * u[2])
    q3 = math.sqrt(u[0])     * math.cos(2 * math.pi * u[2])
    q = torch.tensor([q0, q1, q2, q3], dtype=torch.float64)
    w, x, y, z = q.tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _euler_to_matrix_edmonds_zyz(a: float, b: float, g: float) -> torch.Tensor:
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    return torch.tensor([
        [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
        [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
        [-sb*cg, sb*sg, cb],
    ], dtype=torch.float64)


def _so3_angular_distance_deg(R1: torch.Tensor, R2: torch.Tensor) -> float:
    tr = torch.einsum("ij,ij->", R1, R2).item()
    cos_t = max(min((tr - 1.0) * 0.5, 1.0), -1.0)
    return math.acos(cos_t) * 180.0 / math.pi


def _make_random_reflections(seed: int, n: int = 800) -> tuple:
    """Build a P1 reflection set on a uniform sphere in [s_min, s_max].

    Returns (s_vec, F, centric) torch tensors.
    """
    g = torch.Generator().manual_seed(seed)
    # Uniform on shell |s| ∈ [0.07, 0.25] (~4-15 Å)
    s_mag = 0.07 + 0.18 * torch.rand(n, generator=g, dtype=torch.float64)
    # Uniform on sphere
    theta = torch.acos(2 * torch.rand(n, generator=g, dtype=torch.float64) - 1)
    phi = 2 * math.pi * torch.rand(n, generator=g, dtype=torch.float64)
    s_vec = torch.stack(
        [s_mag * torch.sin(theta) * torch.cos(phi),
         s_mag * torch.sin(theta) * torch.sin(phi),
         s_mag * torch.cos(theta)], dim=-1,
    )
    # Random positive amplitudes (Wilson-distributed, roughly)
    F = torch.randn(n, generator=g, dtype=torch.float64).abs() + 0.1
    centric = torch.zeros(n, dtype=torch.bool)
    return s_vec, F, centric


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_synthetic_rotation_recovery(seed: int):
    """Apply a known rotation to obs; FRF top peak must recover it (within Δ)."""
    torch.manual_seed(seed)
    s_calc, F_calc, centric = _make_random_reflections(seed=seed, n=2000)
    R_truth = _random_rotation_matrix(seed=seed + 100)

    # s_obs = R_truth · s_calc (apply rotation on the same intensities)
    s_obs = (R_truth @ s_calc.T).T
    F_obs = F_calc.clone()

    sym_mats = torch.eye(3, dtype=torch.float64).unsqueeze(0)   # P1

    arf, peaks = phaser_rotation_search(
        s_obs, F_obs, centric,
        s_calc, F_calc,
        sym_mats=sym_mats,
        L=16,
        d_min=4.0, d_max=15.0,
        delta_vrms_A=0.5,
        grid_sampling_deg=5.0,
        n_peaks=20,
        sigma_threshold=-100.0,
        use_french_wilson=False,
        use_m_symmetry_filter=False,
    )

    assert len(peaks) > 0, "no peaks returned"
    # Convention (phaser_frf.py:1140): the peak Euler R satisfies s_calc = R · s_obs.
    # We set s_obs = R_truth · s_calc, so the expected peak is at R_truth^{-1}.
    R_expected = R_truth.T
    # Top peak might not be exactly top-1; allow top-3 (Phaser bracket).
    best_err = min(
        _so3_angular_distance_deg(
            _euler_to_matrix_edmonds_zyz(p.alpha, p.beta, p.gamma),
            R_expected,
        )
        for p in peaks[:3]
    )
    assert best_err < 15.0, (
        f"seed={seed}: best-of-top-3 {best_err:.2f}° from truth; "
        f"top peak Euler "
        f"({math.degrees(peaks[0].alpha):.1f}, {math.degrees(peaks[0].beta):.1f}, "
        f"{math.degrees(peaks[0].gamma):.1f})"
    )


def test_api_returns_correct_types():
    """Type check on the drop-in API."""
    from torchref.experimental.alignment.frf.types import AdaptiveRotationFunction, RotationPeak

    s_calc, F_calc, centric = _make_random_reflections(seed=0, n=500)
    sym_mats = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    arf, peaks = phaser_rotation_search(
        s_calc, F_calc, centric,
        s_calc, F_calc, sym_mats,
        L=8, d_min=4.0, d_max=15.0,
        grid_sampling_deg=10.0,
        n_peaks=5, sigma_threshold=-100.0,
        use_m_symmetry_filter=False,
    )
    assert isinstance(arf, AdaptiveRotationFunction)
    assert len(peaks) > 0
    assert all(isinstance(p, RotationPeak) for p in peaks)


def test_search_is_bit_reproducible_single_threaded():
    """The engine itself is deterministic: no RNG, no order-dependent atomics.

    Repeat runs at the default thread count are NOT bit-identical -- the
    structure-factor reduction is float32 and its parallel summation order
    varies, which on a real high-symmetry case (3GR5, P 6_5 2 2) perturbs peak
    scores by ~5e-8 relative and reorders roughly a dozen of 500 peaks. Truth
    rank was stable there, but nothing guarantees it for two peaks that close.

    Pinning one thread removes the only source of variation, so any future
    difference under this test is a real change in the maths. It is also the
    configuration to use when comparing peak lists across a refactor.
    """
    import torch as _torch

    s_calc, F_calc, centric = _make_random_reflections(seed=3, n=800)
    sym_mats = _torch.eye(3, dtype=_torch.float64).unsqueeze(0)
    kwargs = dict(
        sym_mats=sym_mats, L=12, d_min=4.0, d_max=15.0, delta_vrms_A=0.5,
        grid_sampling_deg=8.0, n_peaks=25, sigma_threshold=-100.0,
        use_french_wilson=False, use_m_symmetry_filter=False,
    )

    n_threads = _torch.get_num_threads()
    _torch.set_num_threads(1)
    try:
        _, first = phaser_rotation_search(s_calc, F_calc, centric,
                                          s_calc, F_calc, **kwargs)
        _, second = phaser_rotation_search(s_calc, F_calc, centric,
                                           s_calc, F_calc, **kwargs)
    finally:
        _torch.set_num_threads(n_threads)

    assert len(first) == len(second) > 0
    for i, (a, b) in enumerate(zip(first, second)):
        assert a.alpha == b.alpha and a.beta == b.beta and a.gamma == b.gamma, (
            f"peak {i} moved between two identical single-threaded runs"
        )
        assert a.score == b.score, (
            f"peak {i} score changed by {abs(a.score - b.score):.3g} between "
            f"two identical single-threaded runs"
        )
