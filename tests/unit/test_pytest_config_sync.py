"""The two pytest configurations agree on the options that change test outcomes.

A bare ``pytest`` at the repo root reads ``pyproject.toml``; any path under ``tests/``
reads ``tests/pytest.ini``. Options such as ``--strict-markers`` or the
``TorchRefDegradationWarning`` error filter must not depend on which one applies.
"""

import shlex
from pathlib import Path

import iniconfig
import pytest

tomllib = pytest.importorskip("tomllib")

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


def _lines(value: str) -> list:
    return [line.strip() for line in value.splitlines() if line.strip()]


def test_pytest_ini_matches_pyproject():
    """Pin equal addopts, markers and filterwarnings in both configuration files."""
    ini = iniconfig.IniConfig(str(_REPO / "tests" / "pytest.ini"))["pytest"]
    with open(_REPO / "pyproject.toml", "rb") as fh:
        toml = tomllib.load(fh)["tool"]["pytest"]["ini_options"]

    toml_addopts = toml.get("addopts", [])
    if isinstance(toml_addopts, str):
        toml_addopts = shlex.split(toml_addopts)

    assert shlex.split(ini.get("addopts", "")) == toml_addopts
    assert _lines(ini["markers"]) == toml["markers"]
    assert _lines(ini["filterwarnings"]) == toml["filterwarnings"]
