"""Refinement trajectories from AlphaFold starts, against a committed reference.

An AlphaFold model is far from convergence, so an early-cycle change in restraint
weighting or connectivity shows up in the trajectory rather than being washed out by the
endpoint. That makes these five structures a sharper probe of a topology or restraint
change than a converged-structure comparison would be.

The comparison carries a **null-control arm**. TorchRef refinement is not bitwise
reproducible -- even two forward passes in one process differ -- so a deviation from the
reference means nothing until it is measured against the deviation between two runs of
the *same* build. A regression is a deviation larger than that null spread, not a
deviation from zero.

Regenerate the reference deliberately, after a change meant to move these numbers::

    ./.dev/bin/python tests/functional/test_af_trajectory.py
"""

import json
from pathlib import Path

import pytest
import torch

#: AlphaFold-start structures, chosen so all five actually descend under refinement and
#: the space groups span centred tetragonal, centred monoclinic, hexagonal and trigonal.
CODES = ["1VER", "6VHI", "1BYW", "6JZA", "6SXW"]

#: Macro-cycles per trajectory. Two is enough for the descent to be visible while
#: keeping the whole test at a few seconds per structure.
CYCLES = 2

REFERENCE = Path(__file__).with_name("af_trajectory_reference.json")

#: Largest same-build spread tolerated before the test reports that nondeterminism
#: itself has grown. Measured spread when the reference was written: 0.0001 to 0.0014.
NULL_CEILING = 0.006

#: Deviation floor, used when the measured null spread is very small. Keeps a structure
#: whose two runs happen to agree closely from being held to too tight a bound.
TOLERANCE_FLOOR = 0.004

#: How many times the measured null spread a deviation may reach before it counts as a
#: real change rather than run-to-run noise.
NULL_MULTIPLE = 4.0


def trajectory(pdb_path, mtz_path, cycles=CYCLES):
    """Per-stage ``(r_work, r_free)`` series of a short refinement.

    Returns
    -------
    list of tuple of float
        Two entries per macro-cycle -- after scaling and after refinement.
    """
    from torchref.refinement.lbfgs_refinement import LBFGSRefinement

    refinement = LBFGSRefinement(
        data_file=str(mtz_path),
        pdb=str(pdb_path),
        verbose=0,
        device=torch.device("cpu"),
    )
    history = refinement.refine_everything(macro_cycles=cycles)
    key = next(k for k in history if k.startswith("refinement_everything"))

    series = []
    for cycle in history[key]:
        for stage in ("after_scaling", "after_refinement"):
            metrics = cycle.get(stage) or {}
            series.append((float(metrics["rwork"]), float(metrics["rfree"])))
    return series


def _max_deviation(a, b):
    """Largest absolute difference between two trajectories, over both R-factors."""
    return max(max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(a, b))


@pytest.fixture(scope="module")
def reference():
    """The committed reference trajectories."""
    if not REFERENCE.exists():
        pytest.fail(
            f"{REFERENCE.name} is missing. Regenerate it with "
            f"`./.dev/bin/python {Path(__file__).name}`."
        )
    return json.loads(REFERENCE.read_text())


@pytest.mark.integration
@pytest.mark.parametrize("code", CODES)
def test_af_trajectory_matches_reference(code, reference, test_files_dir):
    """The trajectory sits within the run-to-run noise of the committed reference."""
    pdb_path = test_files_dir / "pdb" / f"{code}_af.pdb"
    mtz_path = test_files_dir / "mtz" / f"{code}.mtz"
    assert pdb_path.exists(), f"{pdb_path.name} is not bundled"
    assert mtz_path.exists(), f"{mtz_path.name} is not bundled"

    assert (
        reference["cycles"] == CYCLES
    ), f"reference was written for {reference['cycles']} cycles, test runs {CYCLES}"
    expected = [tuple(p) for p in reference["structures"][code]]

    # The null-control arm: two runs of this build, to size the noise.
    run_a = trajectory(pdb_path, mtz_path)
    run_b = trajectory(pdb_path, mtz_path)
    null = _max_deviation(run_a, run_b)

    assert null <= NULL_CEILING, (
        f"{code}: two runs of the same build differ by {null:.6f}, above the "
        f"{NULL_CEILING} ceiling. Refinement nondeterminism has grown, which has to be "
        f"understood before any comparison against the reference means anything."
    )

    assert len(run_a) == len(
        expected
    ), f"{code}: trajectory has {len(run_a)} stages, reference has {len(expected)}"

    observed = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(run_a, run_b)]
    deviation = _max_deviation(observed, expected)
    tolerance = max(TOLERANCE_FLOOR, NULL_MULTIPLE * null)

    assert deviation <= tolerance, (
        f"{code}: trajectory deviates from the reference by {deviation:.6f}, above the "
        f"{tolerance:.6f} tolerance ({NULL_MULTIPLE}x the {null:.6f} null spread). "
        f"observed={observed}\nreference={expected}"
    )


@pytest.mark.integration
def test_af_starts_actually_descend(reference):
    """Each reference trajectory improves R-work, so a regression has signal to lose.

    A structure that barely moves under refinement cannot show a trajectory regression,
    so the set is only useful while every member descends.
    """
    for code in CODES:
        series = reference["structures"][code]
        first_rwork, last_rwork = series[0][0], series[-1][0]
        assert last_rwork < first_rwork - 0.02, (
            f"{code} only moves R-work from {first_rwork:.4f} to {last_rwork:.4f}; "
            f"it is too flat to serve as a trajectory probe"
        )


def _write_reference():
    """Regenerate the reference from the mean of two runs per structure."""
    root = Path(__file__).resolve().parents[1] / "files"
    structures = {}
    for code in CODES:
        pdb_path = root / "pdb" / f"{code}_af.pdb"
        mtz_path = root / "mtz" / f"{code}.mtz"
        run_a = trajectory(pdb_path, mtz_path)
        run_b = trajectory(pdb_path, mtz_path)
        structures[code] = [
            [round((a[0] + b[0]) / 2, 6), round((a[1] + b[1]) / 2, 6)]
            for a, b in zip(run_a, run_b)
        ]
        print(f"{code}: null={_max_deviation(run_a, run_b):.6f}")
    REFERENCE.write_text(
        json.dumps({"cycles": CYCLES, "structures": structures}, indent=2) + "\n"
    )
    print(f"wrote {REFERENCE}")


if __name__ == "__main__":
    _write_reference()
