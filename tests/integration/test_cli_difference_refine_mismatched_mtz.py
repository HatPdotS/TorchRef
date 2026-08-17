"""CPU end-to-end test for ``torchref.difference-refine`` with MISMATCHED MTZ files.

Difference refinement builds a ``DatasetCollection`` from two independent MTZ
files and aligns the light dataset onto the dark reference via
``ReflectionData.validate_hkl``. When the two files had different reflection
sets, alignment left ``hkl_anomalous`` (read by ``_hkl_for_sf``) at the
pre-alignment length and the run crashed inside the collection difference target
(``RuntimeError: The size of tensor a (...) must match the size of tensor b``).

This test generates two MTZ files with genuinely different reflection sets from
the smallest fixture (3GR5) and asserts the CLI now completes. It is the
end-to-end guard for the 0.6.2 fix; pre-fix it exited non-zero.

Wall-clock budget: a couple of minutes on CPU (1 macro-cycle, 1 step).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def diff_refine_cli_script(project_root) -> Path:
    script = project_root / "torchref" / "cli" / "collection_difference_refine.py"
    if not script.exists():
        pytest.skip(f"difference-refine CLI script not found: {script}")
    return script


@pytest.fixture
def mismatched_mtz_pair(test_files_dir, tmp_path):
    """Two MTZ files sharing a cell/spacegroup but with different reflections.

    Built from 3GR5: ``dark`` drops the last 10% of reflections, ``light`` drops
    the first 15%, so each contains reflections the other lacks and the two
    counts differ -- exactly the condition that triggered the crash.
    """
    pdb_file = test_files_dir / "pdb" / "3GR5.pdb"
    mtz_file = test_files_dir / "mtz" / "3GR5.mtz"
    if not pdb_file.exists() or not mtz_file.exists():
        pytest.skip("3GR5 test files not found")

    import torch

    from torchref.io.datasets.reflection_data import ReflectionData

    full = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz_file))
    n = len(full.hkl)
    idx = torch.arange(n)
    dark = full.__select__(idx < int(n * 0.90))
    light = full.__select__(idx >= int(n * 0.15))
    assert len(dark.hkl) != len(light.hkl), "subsets must differ in size"

    dark_mtz = tmp_path / "dark.mtz"
    light_mtz = tmp_path / "light.mtz"
    dark.write_mtz(str(dark_mtz))
    light.write_mtz(str(light_mtz))
    return {"pdb": pdb_file, "dark": dark_mtz, "light": light_mtz}


@pytest.mark.integration
def test_difference_refine_mismatched_mtz_cpu(
    diff_refine_cli_script, mismatched_mtz_pair, tmp_path
):
    outdir = tmp_path / "diff_out"
    result = subprocess.run(
        [
            sys.executable,
            str(diff_refine_cli_script),
            "-dm", str(mismatched_mtz_pair["pdb"]),
            "-lm", str(mismatched_mtz_pair["pdb"]),
            "-dsf", str(mismatched_mtz_pair["dark"]),
            "-lsf", str(mismatched_mtz_pair["light"]),
            "--fraction", "0.3",
            "--n-cycles", "1",
            "--n-steps", "1",
            "--max-iter", "5",
            "-o", str(outdir),
            "--device", "cpu",
            "--verbose", "1",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout[-3000:])
        print("STDERR:", result.stderr[-3000:])

    assert result.returncode == 0, (
        f"difference-refine exited with {result.returncode} on mismatched MTZ "
        f"files. stderr tail: {result.stderr[-800:]}"
    )

    # The exact stale-tensor shape mismatch must not reappear.
    assert "must match the size of tensor" not in result.stderr

    prefix = "fractions_70_30"
    summary = outdir / f"{prefix}_summary.json"
    diff_mtz = outdir / f"{prefix}_difference_data.mtz"
    assert summary.exists(), "summary JSON not written"
    assert diff_mtz.exists(), "difference MTZ not written"

    with open(summary) as f:
        data = json.load(f)
    assert "results" in data and "r_factor_light" in data["results"]
