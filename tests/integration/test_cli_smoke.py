"""
Smoke tests for every registered ``torchref.*`` CLI entry point.

For each script declared in ``pyproject.toml`` under ``[project.scripts]``
we verify two cheap invariants:

1. ``--help`` exits 0 with non-empty stdout — proves the module imports
   cleanly and its argparse setup is well-formed.
2. Running with no arguments (or a clearly invalid one) exits non-zero
   — proves required-argument validation is wired up.

These tests are deliberately cheap and parametrized at collection time
so that adding a new CLI entry point picks up automatic smoke coverage
without anyone touching this file.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

PYPROJECT_SCRIPTS_PATTERN = re.compile(
    r'^"(?P<name>[^"]+)"\s*=\s*"(?P<target>[^"]+)"\s*$'
)


def _parse_project_scripts(pyproject_path: Path):
    """Extract ``[project.scripts]`` entries from ``pyproject.toml``.

    A regex parser (rather than ``tomllib``) keeps this test working on
    Python 3.10, where ``tomllib`` is not yet in the stdlib and we don't
    want to add ``tomli`` as a test dependency.
    """
    text = pyproject_path.read_text(encoding="utf-8")
    in_section = False
    scripts = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == "[project.scripts]"
            continue
        if not in_section:
            continue
        m = PYPROJECT_SCRIPTS_PATTERN.match(line)
        if m:
            scripts[m.group("name")] = m.group("target")
    return scripts


def _entry_point_to_module(target: str) -> str:
    """``torchref.cli.refine:main`` -> ``torchref.cli.refine``."""
    return target.split(":", 1)[0]


def _collect_clis(project_root: Path):
    scripts = _parse_project_scripts(project_root / "pyproject.toml")
    return [(name, _entry_point_to_module(target)) for name, target in scripts.items()]


def _project_root_for_collection() -> Path:
    """Resolve project root at collection time (no fixtures available yet)."""
    return Path(__file__).resolve().parents[2]


CLI_ENTRY_POINTS = _collect_clis(_project_root_for_collection())


@pytest.mark.integration
def test_pyproject_scripts_section_found():
    """Guard against the parser silently finding nothing."""
    assert len(CLI_ENTRY_POINTS) >= 5, (
        f"Expected to discover several CLI entry points in pyproject.toml, "
        f"found {len(CLI_ENTRY_POINTS)}: {CLI_ENTRY_POINTS}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "cli_name,module_name",
    CLI_ENTRY_POINTS,
    ids=[name for name, _ in CLI_ENTRY_POINTS],
)
def test_cli_help(cli_name, module_name):
    """``python -m <module> --help`` exits 0 with non-empty stdout."""
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{cli_name}: --help exited {result.returncode}. "
        f"stderr tail: {result.stderr[-500:]}"
    )
    assert result.stdout.strip(), (
        f"{cli_name}: --help produced empty stdout."
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "cli_name,module_name",
    CLI_ENTRY_POINTS,
    ids=[name for name, _ in CLI_ENTRY_POINTS],
)
def test_cli_rejects_missing_args(cli_name, module_name):
    """Running with no arguments must produce a non-zero exit.

    Every CLI in torchref takes at least one required argument (an input
    file or similar). Invoking with nothing should therefore fail; an exit
    code of 0 would mean we silently no-op'd or accepted bogus inputs.
    """
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, (
        f"{cli_name}: expected non-zero exit on no-args invocation, "
        f"got 0. stdout tail: {result.stdout[-500:]}"
    )
