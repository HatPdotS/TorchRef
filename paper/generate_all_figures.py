#!/usr/bin/env python
"""
Regenerate all paper figures from pre-computed data.

Usage:
    python generate_all_figures.py
"""

import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent


def run_script(script_path: Path, expected_output: Path):
    """Run a plotting script and verify the output file was created."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path.relative_to(PAPER_DIR)}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
    )
    if result.returncode != 0:
        print(f"  ERROR: Script exited with code {result.returncode}")
        return False
    if not expected_output.exists():
        print(f"  ERROR: Expected output not found: {expected_output}")
        return False
    print(f"  OK: {expected_output.relative_to(PAPER_DIR)}")
    return True


def main():
    scripts = [
        (
            PAPER_DIR / "figure2_validation" / "plot_figure2.py",
            PAPER_DIR / "figure2_validation" / "output" / "figure2.png",
        ),
        (
            PAPER_DIR / "figure3_performance" / "plot_figure3a.py",
            PAPER_DIR / "figure3_performance" / "output" / "figure3a_fcalc.png",
        ),
        (
            PAPER_DIR / "figure3_performance" / "plot_figure3b.py",
            PAPER_DIR / "figure3_performance" / "output" / "figure3b_profiling.png",
        ),
    ]

    successes = 0
    for script, expected in scripts:
        if run_script(script, expected):
            successes += 1

    print(f"\n{'='*60}")
    print(f"Results: {successes}/{len(scripts)} figures generated successfully.")
    print()
    print("Note: Figure 4 panels are PyMOL renders in")
    print("  figure4_difference_refinement/panels/")
    print("  They cannot be auto-regenerated (require PyMOL + manual positioning).")
    print("  CCP4 maps can be regenerated via:")
    print("  figure4_difference_refinement/validation/run_validation.sh")
    print(f"{'='*60}")

    if successes < len(scripts):
        sys.exit(1)


if __name__ == "__main__":
    main()
