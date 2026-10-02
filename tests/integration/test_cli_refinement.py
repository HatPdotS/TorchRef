"""
End-to-end integration tests for CLI refinement scripts.

These tests run the actual CLI scripts as subprocesses to verify
the complete refinement pipeline works from command line to output files.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch


class TestCLIRefine:
    """End-to-end tests for the refine.py CLI script."""

    @pytest.fixture
    def cli_script(self, project_root):
        """Path to the refine.py CLI script."""
        script = project_root / "torchref" / "cli" / "refine.py"
        if not script.exists():
            pytest.skip(f"CLI script not found: {script}")
        return script

    @pytest.fixture
    def h_structure_pair(self, test_files_dir):
        """Hydrogen-bearing structure pair used to exercise NonBondedHTarget."""
        pdb_file = test_files_dir / "pdb" / "1AK5_with_H.pdb"
        mtz_file = test_files_dir / "mtz" / "1AK5.mtz"
        if not pdb_file.exists() or not mtz_file.exists():
            pytest.skip("1AK5_with_H test files not found")
        return {"pdb": pdb_file, "mtz": mtz_file}

    @pytest.mark.integration
    def test_cli_refine_help(self, cli_script):
        """Test CLI --help works."""
        result = subprocess.run(
            [sys.executable, str(cli_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "--structure" in result.stdout or "-s" in result.stdout

    @pytest.mark.integration
    def test_cli_refine_missing_inputs(self, cli_script, tmp_path):
        """CLI fails gracefully when the model / structure-factor files don't exist."""
        outdir = tmp_path / "refine_missing"

        result = subprocess.run(
            [
                sys.executable,
                str(cli_script),
                "-m", "/nonexistent/file.pdb",
                "-sf", "/nonexistent/file.mtz",
                "-o", str(outdir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        combined = (result.stderr + result.stdout).lower()
        assert "not found" in combined or "error" in combined

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.cuda
    def test_cli_refine_cuda_end_to_end(self, cli_script, h_structure_pair, tmp_path):
        """Run the refine CLI on CUDA end-to-end on an H-bearing structure.

        Exercises the full LBFGS pipeline with ``--device cuda`` against
        1AK5_with_H so that ``NonBondedHTarget`` and its riding-hydrogen
        VDW path are live. Two macro cycles give ``maintenance()`` a
        chance to fire ``rebuild_vdw_restraints`` — the path that
        previously crashed when VDW buffers were left on CPU after a
        mid-refinement rebuild (PR #19).
        """

        outdir = tmp_path / "refine_cuda"

        result = subprocess.run(
            [
                sys.executable,
                str(cli_script),
                "-m", str(h_structure_pair["pdb"]),
                "-sf", str(h_structure_pair["mtz"]),
                "-o", str(outdir),
                "-n", "2",
                "-v", "1",
                "--device", "cuda",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )

        if result.returncode != 0:
            print("STDOUT:", result.stdout[-3000:])
            print("STDERR:", result.stderr[-3000:])

        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}. "
            f"stderr tail: {result.stderr[-500:]}"
        )

        refined_pdb = outdir / "refined.pdb"
        assert refined_pdb.exists(), "Refined PDB not created"
        pdb_text = refined_pdb.read_text()
        assert "ATOM" in pdb_text, "Refined PDB has no ATOM records"

        history_json = outdir / "refinement_history.json"
        assert history_json.exists(), "refinement_history.json not created"
        with open(history_json) as f:
            history = json.load(f)

        assert history["parameters"]["device"].startswith("cuda"), (
            f"Refinement did not run on CUDA: device={history['parameters']['device']}"
        )

        stats = history.get("final_statistics") or {}
        assert "R_work" in stats and "R_free" in stats, (
            "Final R-factors missing from history JSON; refinement likely "
            "failed mid-run (device mismatch?)"
        )
        for key in ("R_work", "R_free"):
            val = stats[key]
            assert math.isfinite(val), f"{key} is not finite: {val}"
            assert 0.0 < val < 1.0, f"{key} outside plausible range: {val}"




class TestWavelengthFlag:
    """The ``wavelength`` constructor / ``--wavelength`` flag coupling."""

    @pytest.fixture
    def small_pair(self, test_files_dir):
        pdb = test_files_dir / "pdb" / "1DAW.pdb"
        mtz = test_files_dir / "mtz" / "1DAW.mtz"
        if not pdb.exists() or not mtz.exists():
            pytest.skip("1DAW test files not found")
        return {"pdb": str(pdb), "mtz": str(mtz)}

    @pytest.fixture
    def anomalous_mtz(self, small_pair, tmp_path):
        """1DAW written with FP(+)/FP(-) columns, so the file offers Bijvoet pairs."""
        import reciprocalspaceship as rs

        ds = rs.read_mtz(small_pair["mtz"])
        out = tmp_path / "1DAW_anomalous.mtz"
        ds[["FP", "SIGFP", "FreeR_flag"]].copy().unstack_anomalous(
            columns=["FP", "SIGFP"]
        ).write_mtz(str(out))
        return str(out)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "kwargs", [{}, {"wavelength": 0}], ids=["default", "wavelength_zero"]
    )
    def test_no_wavelength_means_no_anomalous(self, small_pair, anomalous_mtz, kwargs):
        """Without a wavelength the model has no f'/f'' and F(+)/F(-) are merged."""
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement

        ref = LBFGSRefinement(
            data_file=anomalous_mtz,
            pdb=small_pair["pdb"],
            device=torch.device("cpu"),
            verbose=0,
            **kwargs,
        )
        assert ref.wavelength is None
        assert ref.anomalous is False
        assert bool(ref.reflection_data.friedel_merged) is True
        assert ref.model.wavelength is None
        assert bool(ref.model.anomalous_bijvoet) is False

    @pytest.mark.integration
    def test_wavelength_reads_bijvoet_pairs(self, small_pair, anomalous_mtz):
        """A wavelength gives the model f'/f'' and reads F(+)/F(-) as Bijvoet pairs."""
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement

        ref = LBFGSRefinement(
            data_file=anomalous_mtz,
            pdb=small_pair["pdb"],
            device=torch.device("cpu"),
            verbose=0,
            wavelength=1.54,
        )
        assert bool(ref.reflection_data.friedel_merged) is False
        assert ref.model.wavelength == 1.54
        assert bool(ref.model.anomalous_bijvoet) is True

    @pytest.mark.integration
    def test_anomalous_without_wavelength_raises(self, small_pair):
        """``anomalous=True`` needs the wavelength its f'' term is computed at."""
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement

        with pytest.raises(ValueError, match="wavelength"):
            LBFGSRefinement(
                data_file=small_pair["mtz"],
                pdb=small_pair["pdb"],
                device=torch.device("cpu"),
                verbose=0,
                anomalous=True,
            )
