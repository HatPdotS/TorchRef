"""The console-script launcher and the process settings it applies.

The settings only take effect if they are in place before torch loads its OpenMP
runtime, so the launcher must not import torch, every console script must go through it,
and a library import of torchref must leave the user's process alone. Each check that
depends on import order runs in a fresh interpreter.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import _torchref_cli

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


def _fresh(code, **env_vars):
    """Run ``code`` in a new interpreter with ``OMP_WAIT_POLICY`` unset unless given."""
    env = {k: v for k, v in os.environ.items() if k != "OMP_WAIT_POLICY"}
    env.update(env_vars)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT), env.get("PYTHONPATH")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip()


def test_launcher_imports_neither_torch_nor_torchref():
    out = _fresh(
        "import sys, _torchref_cli\n"
        "print('torch' in sys.modules, 'torchref' in sys.modules)"
    )
    assert out == "False False"


def test_cli_process_gets_a_passive_wait_policy_unless_the_user_chose():
    code = (
        "import os, _torchref_cli\n"
        "_torchref_cli.configure_process()\n"
        "print(os.environ['OMP_WAIT_POLICY'])"
    )
    assert _fresh(code) == "PASSIVE"
    assert _fresh(code, OMP_WAIT_POLICY="ACTIVE") == "ACTIVE"


def test_library_import_leaves_the_wait_policy_alone():
    out = _fresh("import os, torchref\nprint(os.environ.get('OMP_WAIT_POLICY'))")
    assert out.splitlines()[-1] == "None"


def test_every_console_script_goes_through_the_launcher():
    tomllib = pytest.importorskip("tomllib")
    with open(ROOT / "pyproject.toml", "rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]
    assert scripts
    for name, target in scripts.items():
        module, attr = target.split(":")
        assert module == "_torchref_cli", f"{name} bypasses the launcher: {target}"
        entry = getattr(_torchref_cli, attr)
        cli_module = importlib.import_module(f"torchref.cli.{entry.__name__}")
        assert callable(cli_module.main), f"{name}: torchref.cli.{entry.__name__}"
