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

import gemmi
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
            "--refine-fractions",
            "--title", "Dark and light states",
            "--output-remarks", "Refined against mismatched halves",
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

    # The merged deposition CIF scales each atom's own occupancy by the refined
    # population of its state; 3GR5's HOH A224 sits on a special position at 0.50.
    def occupancies(path):
        structure = gemmi.read_structure(str(path))
        return {
            (chain.name, str(residue.seqid), atom.name, atom.altloc): atom.occ
            for chain in structure[0]
            for residue in chain
            for atom in residue
        }

    merged = occupancies(outdir / f"{prefix}_merged.cif")
    populations = data["results"]["fractions"]
    assert merged[("A", "224", "O", "A")] == pytest.approx(
        0.5 * populations[0], abs=0.01
    )
    for state, altloc, w in zip(("dark", "light"), "AB", populations):
        own = occupancies(outdir / f"{prefix}_{state}.pdb")
        # Both files round occupancies to two decimals.
        for (chain, seqid, name, _), occ in own.items():
            assert merged[(chain, seqid, name, altloc)] == pytest.approx(
                occ * w, abs=0.005 * (1 + w)
            )

    # Header statistics: atom counts as deposited for 3GR5 (its 15 sulfate atoms are
    # neither protein nor solvent), and the merged file reports the reflections its
    # R-factors (mixed model against the light data) were computed on.
    def block(state):
        return gemmi.cif.read(str(outdir / f"{prefix}_{state}.cif")).sole_block()

    dark, light, merged_block = block("dark"), block("light"), block("merged")
    assert dark.find_value("_refine_hist.pdbx_number_atoms_protein") == "1237"
    assert dark.find_value("_refine_hist.number_atoms_solvent") == "77"
    for tag in ("_refine.ls_number_reflns_R_work", "_refine.ls_number_reflns_R_free"):
        assert light.find_value(tag) is not None
        assert merged_block.find_value(tag) == light.find_value(tag)

    # Geometry deviations of the refined light model in the header's units: bond
    # lengths in Å, angles in degrees (near 0.01 Å and 1-2 degrees; radians would
    # read ~0.03).
    devs = {
        gemmi.cif.as_string(row[0]): float(row[1])
        for row in light.find("_refine_ls_restr.", ["type", "dev_ideal"])
    }
    assert 0.001 < devs["f_bond_d"] < 0.1
    assert 0.3 < devs["f_angle_d"] < 5.0

    # --title and --output-remarks reach the per-state files and the merged CIF, whose
    # ensemble note is only the default title.
    for b in (light, merged_block):
        title = gemmi.cif.as_string(b.find_value("_struct.title"))
        details = gemmi.cif.as_string(b.find_value("_refine.details"))
        assert title == "Dark and light states"
        assert details == "Refined against mismatched halves"
