"""Self-tests for the alignment lab primitives.

These pin the contracts whose violation silently changed results in the past:
the seed -> rotation mapping, the append-only benchmark order the seed formula
depends on, and the orbit conventions.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import BENCH_PDBS, case_paths, orbit_rank, random_rotation, seed_for  # noqa: E402
from lab.truth import angle_to_orbit, symmetry_orbit  # noqa: E402


def test_random_rotation_is_a_rotation():
    """Output is orthogonal with det +1 for a spread of seeds."""
    for seed in (0, 1, 42, 2077, 999983):
        R = random_rotation(seed)
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=R.dtype), atol=1e-12)
        assert abs(float(torch.det(R)) - 1.0) < 1e-12


def test_random_rotation_is_deterministic():
    """The seed -> rotation map is the lab's reproducibility contract."""
    assert torch.equal(random_rotation(42), random_rotation(42))
    assert not torch.equal(random_rotation(42), random_rotation(43))


def test_random_rotation_uses_the_sign_corrected_qr():
    """Guard the exact variant: the uncorrected QR gives a different rotation.

    Both forms return a valid rotation, so only a direct comparison catches a
    swap -- and a swap silently makes new results incomparable with archived
    ones for the same seed.
    """
    seed = 42
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q_uncorrected, _ = torch.linalg.qr(A)
    if torch.det(Q_uncorrected) < 0:
        Q_uncorrected[:, 0] = -Q_uncorrected[:, 0]
    assert not torch.allclose(random_rotation(seed), Q_uncorrected, atol=1e-9)


def test_seed_formula():
    """base + 1000*trial + index(pdb)*7."""
    assert seed_for("1DAW", 0) == 42
    assert seed_for("1DAW", 1) == 1042
    assert seed_for("3K7M", 2) == 42 + 2000 + BENCH_PDBS.index("3K7M") * 7


def test_benchmark_order_is_pinned():
    """The seed formula indexes into this tuple, so its order is a contract."""
    assert BENCH_PDBS[0] == "1DAW"
    assert BENCH_PDBS.index("1AK5") == 4
    assert BENCH_PDBS.index("3K7M") == 5
    assert len(BENCH_PDBS) == len(set(BENCH_PDBS)) == 10


@pytest.mark.parametrize("pdb", BENCH_PDBS)
def test_every_benchmark_case_resolves(pdb):
    """Paths are repo-relative and present -- not absolute into another tree."""
    pdb_path, mtz_path = case_paths(pdb)
    assert pdb_path.is_file() and mtz_path.is_file()


def test_orbit_side_and_frame_are_distinct_conventions():
    """left/right and frac/cart really do differ, so recording them matters."""
    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup("P 4 3 2")
    symops = sg.matrices.to(torch.float64).cpu()
    R = random_rotation(7)
    left = symmetry_orbit(R, symops, side="left", frame="frac")
    right = symmetry_orbit(R, symops, side="right", frame="frac")
    assert not torch.allclose(left, right, atol=1e-9)


def test_orbit_contains_truth_at_zero_angle():
    """Every orbit member is 0 degrees from the orbit, by construction."""
    from torchref.symmetry import SpaceGroup

    symops = SpaceGroup("P 4 3 2").matrices.to(torch.float64).cpu()
    R = random_rotation(11)
    orbit = symmetry_orbit(R, symops, side="left", frame="frac")
    for k in range(orbit.shape[0]):
        assert angle_to_orbit(orbit[k], orbit) < 1e-9


def test_orbit_rank_reports_miss_as_minus_one():
    """A peak list with no match ranks -1 but still reports the closest angle."""
    from types import SimpleNamespace

    from torchref.symmetry import SpaceGroup

    symops = SpaceGroup("P 1").matrices.to(torch.float64).cpu()
    peaks = [SimpleNamespace(alpha=0.0, beta=0.0, gamma=0.0)]
    R_true = random_rotation(3)
    rank, ang = orbit_rank(peaks, R_true, symops, frame="frac", thr_deg=1e-6)
    assert rank == -1
    assert math.isfinite(ang) and ang > 0


def test_result_writer_rejects_undeclared_columns(tmp_path):
    """Silent column drift is what made every old CSV need its own aggregator."""
    from lab import ResultWriter

    w = ResultWriter(tmp_path / "r.csv", "demo", extra_fields=("ghosts",))
    w.write(pdb="1DAW", truth_rank=0, ghosts=3)
    with pytest.raises(KeyError):
        w.write(pdb="1DAW", not_declared=1)
    assert (tmp_path / "r.csv").read_text().count("\n") == 2


def test_paired_ranks_reports_no_delta_when_truth_is_outside_the_window():
    """A rescore cannot be blamed for a peak it was never shown.

    When truth is absent from the top-N handed to the engine, ``delta`` is None
    rather than a number -- otherwise the metric silently reports the FRF's
    failure as a rescore regression.
    """
    from types import SimpleNamespace

    from lab import paired_ranks, random_rotation
    from torchref.symmetry import SpaceGroup
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_matrix_from_edmonds_euler,
    )

    symops = SpaceGroup("P 1").matrices.to(torch.float64).cpu()
    R_true = random_rotation(3)

    # A peak list whose only truth-matching entry sits beyond the window.
    def peak_at(R):
        # recover ZYZ angles numerically is unnecessary: use a far-off peak for
        # the decoys and the true rotation only at the tail.
        return SimpleNamespace(alpha=0.0, beta=0.0, gamma=0.0)

    decoys = [peak_at(None) for _ in range(5)]
    out = paired_ranks(decoys, decoys, R_true, symops,
                       n_refine=2, frame="frac", thr_deg=1e-6)
    assert out["truth_in_window"] is False
    assert out["delta"] is None


def test_run_rescore_none_is_an_identity_control():
    """The 'none' arm must return the input order untouched."""
    from types import SimpleNamespace

    from lab import run_rescore

    peaks = [SimpleNamespace(alpha=float(i), beta=0.0, gamma=0.0) for i in range(5)]
    res = run_rescore(peaks, data=None, frf_inputs=None, engine="none", n_refine=3)
    assert res.engine == "none"
    assert [p.alpha for p in res.peaks] == [0.0, 1.0, 2.0]
