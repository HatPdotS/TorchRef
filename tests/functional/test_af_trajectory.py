"""Refinement trajectories from AlphaFold starts, against a committed reference.

An AlphaFold model is far from convergence, so an early-cycle change in restraint
weighting or connectivity shows up in the trajectory rather than being washed out by the
endpoint. That makes these five structures a sharper probe of a topology or restraint
change than a converged-structure comparison would be.

TorchRef refinement is not reproducible run to run, so a deviation from the reference
means nothing on its own. Each structure therefore carries its **own** tolerance,
measured over :data:`SPREAD_RUNS` independent runs when the reference was written, and
committed alongside it. Sizing the tolerance from a couple of runs at test time does
not work: 6JZA is bimodal at this cycle count -- its trajectories land in one of two
basins about 0.0074 apart -- and two runs that happen to pick the same basin report a
spread 140 times too small.

R-work and R-free are held to different bounds. R-work is reproducible to a few parts in
ten thousand, so it keeps the measured tolerance and is what catches a change that
actually moves the refinement. R-free is computed on the small free set and has a rare
second basin of its own, a few thousandths wide, that a handful of runs will usually
miss; it therefore carries :data:`RFREE_TOLERANCE_FLOOR`, wide enough to sit outside
that basin and still far inside the descent the trajectory shows.

Regenerate deliberately, after a change meant to move these numbers::

    ./.dev/bin/python tests/functional/test_af_trajectory.py
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

#: AlphaFold-start structures, chosen so all five actually descend under refinement and
#: the space groups span centred tetragonal, centred monoclinic, hexagonal and trigonal.
CODES = ["1VER", "6VHI", "1BYW", "6JZA", "6SXW"]

#: Macro-cycles per trajectory. Two is enough for the descent to be visible while
#: keeping the whole test at a few seconds per structure.
CYCLES = 2

#: Independent runs used to size each structure's tolerance at regeneration time. Enough
#: to see a second basin if there is one; two is not.
SPREAD_RUNS = 5

#: Multiple of the measured spread a deviation may reach before it counts as a change.
SPREAD_MULTIPLE = 3.0

#: Tolerance floor for R-work, so a structure whose runs agree very closely is not held
#: to an unreasonably tight bound.
TOLERANCE_FLOOR = 0.002

#: Tolerance floor for R-free, which moves in discrete basins rather than jitter. Set
#: above the widest basin separation seen on this set and kept well inside every
#: structure's own R-work descent, so a change large enough to matter still fails.
#: ``test_tolerances_are_tight_enough_to_detect_something`` enforces the second half.
RFREE_TOLERANCE_FLOOR = 0.01

REFERENCE = Path(__file__).with_name("af_trajectory_reference.json")


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


def _deviations(a, b):
    """Largest absolute difference between two trajectories, ``(R-work, R-free)``."""
    return (
        max(abs(x[0] - y[0]) for x, y in zip(a, b)),
        max(abs(x[1] - y[1]) for x, y in zip(a, b)),
    )


def _rfree_tolerance(entry):
    """The R-free bound: this structure's measured tolerance, floored."""
    return max(float(entry["tolerance"]), RFREE_TOLERANCE_FLOOR)


@pytest.fixture(scope="module")
def reference():
    """The committed reference trajectories and their tolerances."""
    if not REFERENCE.exists():
        pytest.fail(
            f"{REFERENCE.name} is missing. Regenerate it with "
            f"`./.dev/bin/python {Path(__file__).name}`."
        )
    data = json.loads(REFERENCE.read_text())
    assert (
        data["cycles"] == CYCLES
    ), f"reference was written for {data['cycles']} cycles, test runs {CYCLES}"
    return data


@pytest.mark.integration
@pytest.mark.parametrize("code", CODES)
def test_af_trajectory_matches_reference(code, reference, test_files_dir):
    """The trajectory sits inside this structure's own measured run-to-run spread."""
    pdb_path = test_files_dir / "pdb" / f"{code}_af.pdb"
    mtz_path = test_files_dir / "mtz" / f"{code}.mtz"
    assert pdb_path.exists(), f"{pdb_path.name} is not bundled"
    assert mtz_path.exists(), f"{mtz_path.name} is not bundled"

    entry = reference["structures"][code]
    expected = [tuple(point) for point in entry["trajectory"]]
    rwork_tolerance = float(entry["tolerance"])
    rfree_tolerance = _rfree_tolerance(entry)

    observed = trajectory(pdb_path, mtz_path)
    assert len(observed) == len(
        expected
    ), f"{code}: trajectory has {len(observed)} stages, reference has {len(expected)}"

    dev_work, dev_free = _deviations(observed, expected)
    for label, deviation, tolerance in (
        ("R-work", dev_work, rwork_tolerance),
        ("R-free", dev_free, rfree_tolerance),
    ):
        assert deviation <= tolerance, (
            f"{code}: {label} deviates from the reference by {deviation:.6f}, above its "
            f"tolerance of {tolerance:.6f}.\nobserved={observed}\n"
            f"reference={expected}"
        )


@pytest.mark.integration
def test_af_starts_actually_descend(reference):
    """Each reference trajectory improves R-work, so a regression has signal to lose.

    A structure that barely moves under refinement cannot show a trajectory regression,
    so the set is only useful while every member descends.
    """
    for code in CODES:
        series = reference["structures"][code]["trajectory"]
        first, last = series[0][0], series[-1][0]
        assert last < first - 0.02, (
            f"{code} only moves R-work from {first:.4f} to {last:.4f}; it is too flat "
            f"to serve as a trajectory probe"
        )


@pytest.mark.integration
def test_tolerances_are_tight_enough_to_detect_something(reference):
    """A tolerance so wide it would accept any change is not a test.

    Guards against a future regeneration quietly widening a bound until the structure
    stops constraining anything. The bound has to stay well inside the descent the
    trajectory itself shows.
    """
    for code in CODES:
        entry = reference["structures"][code]
        series = entry["trajectory"]
        descent = series[0][0] - series[-1][0]
        # Checks the widest bound actually applied, not the stored one: flooring R-free
        # would otherwise widen the real bound without this guard seeing it.
        widest = max(float(entry["tolerance"]), _rfree_tolerance(entry))
        assert widest < descent / 4.0, (
            f"{code}: tolerance {widest:.4f} is not small against its "
            f"own R-work descent of {descent:.4f}"
        )


def _write_reference():
    """Regenerate the reference, sizing each tolerance from independent runs."""
    root = Path(__file__).resolve().parents[1] / "files"
    structures = {}
    for code in CODES:
        pdb_path = root / "pdb" / f"{code}_af.pdb"
        mtz_path = root / "mtz" / f"{code}.mtz"

        runs = [trajectory(pdb_path, mtz_path) for _ in range(SPREAD_RUNS)]
        mean = [
            tuple(float(np.mean([run[stage][i] for run in runs])) for i in (0, 1))
            for stage in range(len(runs[0]))
        ]
        spread = max(_max_deviation(run, mean) for run in runs)
        tolerance = max(TOLERANCE_FLOOR, SPREAD_MULTIPLE * spread)

        structures[code] = {
            "trajectory": [[round(w, 6), round(f, 6)] for w, f in mean],
            "spread": round(spread, 6),
            "tolerance": round(tolerance, 6),
        }
        print(f"{code}: spread={spread:.6f} tolerance={tolerance:.6f}", flush=True)

    REFERENCE.write_text(
        json.dumps(
            {"cycles": CYCLES, "spread_runs": SPREAD_RUNS, "structures": structures},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {REFERENCE}")


if __name__ == "__main__":
    _write_reference()
