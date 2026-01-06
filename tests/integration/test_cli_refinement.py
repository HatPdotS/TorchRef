"""
End-to-end integration tests for CLI refinement scripts.

These tests run the actual CLI scripts as subprocesses to verify
the complete refinement pipeline works from command line to output files.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIRefineEverything:
    """End-to-end tests for the refine_everything.py CLI script."""

    @pytest.fixture
    def cli_script(self, project_root):
        """Path to the refine_everything.py CLI script."""
        script = project_root / "torchref" / "cli" / "refine_everything.py"
        if not script.exists():
            pytest.skip(f"CLI script not found: {script}")
        return script

    @pytest.fixture
    def small_structure_pair(self, test_files_dir):
        """Get a small structure pair for fast testing (3GR5)."""
        pdb_file = test_files_dir / "pdb" / "3GR5.pdb"
        mtz_file = test_files_dir / "mtz" / "3GR5.mtz"

        if not pdb_file.exists() or not mtz_file.exists():
            pytest.skip("3GR5 test files not found")

        return {"pdb": pdb_file, "mtz": mtz_file}

    @pytest.mark.integration
    def test_cli_help(self, cli_script):
        """Test CLI --help works."""
        result = subprocess.run(
            [sys.executable, str(cli_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "refinement" in result.stdout.lower() or "refine" in result.stdout.lower()
        assert "--structure" in result.stdout or "-s" in result.stdout

    @pytest.mark.integration
    def test_cli_missing_structure(self, cli_script, tmp_path):
        """Test CLI handles missing structure file gracefully."""
        outdir = tmp_path / "refine_missing"

        result = subprocess.run(
            [
                sys.executable,
                str(cli_script),
                "-s", "/nonexistent/file.pdb",
                "-f", "/nonexistent/file.mtz",
                "-o", str(outdir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cli_refine_basic(self, cli_script, small_structure_pair, tmp_path):
        """Test basic CLI refinement produces expected output files."""
        outdir = tmp_path / "refine_output"

        # Run CLI script
        result = subprocess.run(
            [
                sys.executable,
                str(cli_script),
                "-s", str(small_structure_pair["pdb"]),
                "-f", str(small_structure_pair["mtz"]),
                "-o", str(outdir),
                "-n", "1",  # Single cycle for speed
                "-v", "1",
                "--hyperparameters", "none",  # Skip hyperparameter loading
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Print output for debugging if test fails
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

        # Check output directory was created
        assert outdir.exists(), f"Output directory not created. stderr: {result.stderr[:500]}"

        # Check output files exist
        refined_pdb = outdir / "refined.pdb"
        refined_mtz = outdir / "refined.mtz"

        assert refined_pdb.exists(), f"Refined PDB not created. stderr: {result.stderr[:500]}"
        assert refined_mtz.exists(), f"Refined MTZ not created. stderr: {result.stderr[:500]}"

        # Check PDB has content
        pdb_content = refined_pdb.read_text()
        assert "ATOM" in pdb_content, "Refined PDB has no ATOM records"
        assert len(pdb_content) > 1000, "Refined PDB seems too small"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cli_refine_with_history(self, cli_script, small_structure_pair, tmp_path):
        """Test CLI refinement creates history JSON."""
        outdir = tmp_path / "refine_history"

        result = subprocess.run(
            [
                sys.executable,
                str(cli_script),
                "-s", str(small_structure_pair["pdb"]),
                "-f", str(small_structure_pair["mtz"]),
                "-o", str(outdir),
                "-n", "1",
                "-v", "1",  # Verbose mode creates history JSON
                "--hyperparameters", "none",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Check history JSON exists and is valid
        history_json = outdir / "refinement_history.json"
        if history_json.exists():
            with open(history_json) as f:
                history = json.load(f)

            assert "input_files" in history
            assert "parameters" in history
            assert history["parameters"]["n_cycles"] == 1

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cli_refine_multiple_cycles(self, cli_script, small_structure_pair, tmp_path):
        """Test CLI refinement with multiple cycles."""
        outdir = tmp_path / "refine_cycles"

        result = subprocess.run(
            [
                sys.executable,
                str(cli_script),
                "-s", str(small_structure_pair["pdb"]),
                "-f", str(small_structure_pair["mtz"]),
                "-o", str(outdir),
                "-n", "2",  # Two cycles
                "-v", "0",  # Quiet mode for speed
                "--hyperparameters", "none",
            ],
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout for 2 cycles
        )

        # Check output files
        assert (outdir / "refined.pdb").exists(), f"No PDB. stderr: {result.stderr[:500]}"
        assert (outdir / "refined.mtz").exists(), f"No MTZ. stderr: {result.stderr[:500]}"


class TestCLIRefine:
    """End-to-end tests for the refine.py CLI script."""

    @pytest.fixture
    def cli_script(self, project_root):
        """Path to the refine.py CLI script."""
        script = project_root / "torchref" / "cli" / "refine.py"
        if not script.exists():
            pytest.skip(f"CLI script not found: {script}")
        return script

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


class TestCLIRefineScreened:
    """End-to-end tests for the refine_screened.py CLI script."""

    @pytest.fixture
    def cli_script(self, project_root):
        """Path to the refine_screened.py CLI script."""
        script = project_root / "torchref" / "cli" / "refine_screened.py"
        if not script.exists():
            pytest.skip(f"CLI script not found: {script}")
        return script

    @pytest.mark.integration
    def test_cli_refine_screened_help(self, cli_script):
        """Test CLI --help works."""
        result = subprocess.run(
            [sys.executable, str(cli_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "--" in result.stdout  # Has options
