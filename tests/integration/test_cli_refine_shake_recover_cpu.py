"""
CPU end-to-end test for ``torchref.refine``: shake -> refine -> recover.

The point of this test is not just that the CLI doesn't crash — that is
already covered by the smoke tests in ``test_cli_smoke.py``. The point
here is that refinement *actually does something useful*: when we
perturb the deposited coordinates by ~0.1 Å and run three macro cycles
of joint XYZ/ADP/scale refinement, R_work must come back down
appreciably from its post-shake starting point.

If this test breaks, refinement is broken: it ran without errors but
failed to improve the model.

3GR5 is the smallest available pair in ``tests/files/`` and is therefore
the cheapest realistic input for a default-CI integration test.

Wall-clock budget: ~2 minutes on a 4-core CPU. Intentionally left
unmarked (no ``slow``) so it runs in default CI — it is the highest-
signal of the three new test files and the one that proves refinement
actually works end to end.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from torchref.model import ModelFT


# Margin by which final R_work must beat initial R_work. A 0.1 Å shake
# on a refined deposition normally costs many percent of R; a healthy
# refinement recovers most of it in 3 cycles. The 0.01 threshold is
# generous enough to absorb run-to-run noise while still catching a
# refinement that did nothing.
R_FACTOR_RECOVERY_MARGIN = 0.01


@pytest.fixture
def refine_cli_script(project_root) -> Path:
    script = project_root / "torchref" / "cli" / "refine.py"
    if not script.exists():
        pytest.skip(f"refine CLI script not found: {script}")
    return script


@pytest.fixture
def small_structure_pair(test_files_dir):
    pdb_file = test_files_dir / "pdb" / "3GR5.pdb"
    mtz_file = test_files_dir / "mtz" / "3GR5.mtz"
    if not pdb_file.exists() or not mtz_file.exists():
        pytest.skip("3GR5 test files not found")
    return {"pdb": pdb_file, "mtz": mtz_file}


def _initial_rwork(history: dict) -> float:
    """Extract the initial (post-shake, pre-refinement) R_work.

    ``refine_everything`` records ``history["initial"] = collect_metrics()``
    on entry, which contains lowercase ``rwork``/``rfree`` keys. Fall back
    to the first cycle's ``after_scaling`` block if for some reason
    ``initial`` is absent.
    """
    initial = history.get("initial") or {}
    if "rwork" in initial:
        return float(initial["rwork"])

    for key, value in history.items():
        if not isinstance(value, list) or not value:
            continue
        first = value[0]
        if isinstance(first, dict):
            after_scaling = first.get("after_scaling") or {}
            if "rwork" in after_scaling:
                return float(after_scaling["rwork"])

    pytest.fail(
        "Could not locate initial R_work in refinement_history.json; "
        f"keys present: {list(history.keys())}"
    )


@pytest.mark.integration
def test_refine_recovers_from_shake_cpu(
    refine_cli_script, small_structure_pair, tmp_path
):
    """Shake the deposited structure, run CLI refine on CPU, assert recovery.

    Pipeline:
        1. Load 3GR5.pdb via ModelFT and perturb XYZ by 0.1 Å RMS.
        2. Write the shaken structure to ``tmp_path/shaken.pdb``.
        3. Invoke ``torchref.refine -m shaken.pdb -sf 3GR5.mtz -o tmp_path/out
           -n 3 --device cpu`` as a subprocess.
        4. Verify exit code, output files, and that the refined R_work is
           meaningfully below the initial (post-shake) R_work.
    """
    # ------------------------------------------------------------ shake
    model = ModelFT()
    model.load_pdb(str(small_structure_pair["pdb"]))
    model.shake_coords(0.1)

    shaken_pdb = tmp_path / "shaken.pdb"
    model.write_pdb(str(shaken_pdb))
    assert shaken_pdb.exists(), "Failed to write shaken PDB"

    # ----------------------------------------------------------- refine
    outdir = tmp_path / "refine_out"
    result = subprocess.run(
        [
            sys.executable,
            str(refine_cli_script),
            "-m", str(shaken_pdb),
            "-sf", str(small_structure_pair["mtz"]),
            "-o", str(outdir),
            "-n", "3",
            "--device", "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout[-3000:])
        print("STDERR:", result.stderr[-3000:])

    assert result.returncode == 0, (
        f"refine CLI exited with {result.returncode}. "
        f"stderr tail: {result.stderr[-500:]}"
    )

    # ------------------------------------------------------ output files
    refined_pdb = outdir / "refined.pdb"
    assert refined_pdb.exists(), "Refined PDB not created"
    assert "ATOM" in refined_pdb.read_text(), "Refined PDB has no ATOM records"

    history_json = outdir / "refinement_history.json"
    assert history_json.exists(), "refinement_history.json not created"
    with open(history_json) as f:
        history_data = json.load(f)

    # ----------------------------------------------- device & sanity
    assert history_data["parameters"]["device"] == "cpu", (
        f"Refinement did not run on CPU: "
        f"device={history_data['parameters']['device']}"
    )

    stats = history_data.get("final_statistics") or {}
    assert "R_work" in stats and "R_free" in stats, (
        "Final R-factors missing from history JSON; refinement likely "
        "failed mid-run."
    )
    for key in ("R_work", "R_free"):
        val = stats[key]
        assert math.isfinite(val), f"{key} is not finite: {val}"
        assert 0.0 < val < 1.0, f"{key} outside plausible range: {val}"

    # -------------------------------------------- shake+recover assertion
    initial_rwork = _initial_rwork(history_data.get("history") or {})
    final_rwork = float(stats["R_work"])

    assert final_rwork < initial_rwork - R_FACTOR_RECOVERY_MARGIN, (
        f"Refinement failed to recover from 0.1 Å shake: "
        f"initial R_work={initial_rwork:.4f}, final R_work={final_rwork:.4f} "
        f"(expected drop of at least {R_FACTOR_RECOVERY_MARGIN})."
    )
