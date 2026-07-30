"""
Smoke test: every torchref subpackage/module parses and imports.

This catches parse-time failures (SyntaxError, IndentationError) in any
file under ``torchref/`` — the kind of bug that historically slipped past
CI because only the top-level ``torchref`` package was imported during
the unit-test run, leaving subpackages like ``torchref.experimental.kinetic`` and
``torchref.experimental.alignment`` untested at import time.

Modules that legitimately depend on optional packages (JAX, CCTBX,
OpenMM, torchmd-net, pdbfixer, ihm) are allowed to raise ``ImportError``
*only when the missing dependency is one of those known optionals*.
Any other failure mode — syntax errors, name errors, attribute errors,
or an ImportError naming a non-optional package — is a hard failure.
"""

import importlib
import os
import pkgutil

import pytest

import torchref


# Optional dependencies whose absence is tolerated by individual modules.
# An ImportError whose message mentions one of these is treated as "skip",
# not "fail".
OPTIONAL_DEP_HINTS = (
    "jax",
    "jaxlib",
    "s2fft",
    "s2ball",
    "spherical",
    "quaternionic",
    "iotbx",
    "cctbx",
    "torchmdnet",
    "openmm",
    "pdbfixer",
    "ihm",
    "triton",  # CUDA-only; not installable on macOS / CPU-only environments
)


def _discover_modules():
    """Walk torchref and return every importable module name."""
    found = []
    for module_info in pkgutil.walk_packages(
        torchref.__path__, prefix=f"{torchref.__name__}."
    ):
        # Skip private modules; they're internal and not part of the public surface
        if any(part.startswith("_") for part in module_info.name.split(".")[1:]):
            continue
        found.append(module_info.name)
    return found


MODULES = _discover_modules()


@pytest.mark.unit
@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    """Each torchref module must at minimum parse cleanly.

    Optional-dependency stack ImportErrors are tolerated; everything else
    (SyntaxError, NameError, ImportError on a non-optional package, …)
    fails the test.
    """
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        message = str(exc).lower()
        if any(hint in message for hint in OPTIONAL_DEP_HINTS):
            pytest.skip(f"optional dependency missing: {exc}")
        raise


@pytest.mark.unit
def test_discovery_found_modules():
    """Guard against a silent regression where walk_packages finds nothing."""
    assert len(MODULES) >= 20, (
        f"Expected to discover at least 20 torchref modules, found {len(MODULES)}. "
        "Did the package layout change, or is walk_packages misconfigured?"
    )


@pytest.mark.unit
def test_kinetic_subpackage_imports():
    """Explicit guard for the bug that motivated this whole test file.

    ``torchref.experimental.kinetic.refinement`` once shipped with a misplaced import
    line that broke ``import torchref.experimental.kinetic`` at parse time. Pin the
    invariant directly so that regression is impossible to miss.
    """
    importlib.import_module("torchref.experimental.kinetic")
    importlib.import_module("torchref.experimental.kinetic.refinement")


@pytest.mark.unit
def test_import_survives_a_broken_triton_install(tmp_path):
    """A Triton that *raises* on import must not take ``import torchref`` down with it.

    Absent Triton was always handled; a **broken** one was not. The optional-kernel guard
    caught only ``ImportError``, so a Triton whose import raises anything else -- a driver
    or LLVM version skew, which is the common real-world failure -- propagated straight out
    of ``import torchref``, making the whole package unusable on a host whose GPU stack had
    drifted. ``triton_available()`` was written to absorb exactly this and never got the
    chance.

    Run in a subprocess with a deliberately-exploding ``triton`` shadowing the real one:
    the failure happens at *import* time, so it cannot be provoked in-process once
    ``torchref`` is already loaded.
    """
    import subprocess
    import sys

    pkg = tmp_path / "triton"
    pkg.mkdir()
    boom = 'raise RuntimeError("simulated driver/LLVM version skew")\n'
    (pkg / "__init__.py").write_text(boom)
    (pkg / "language.py").write_text(boom)

    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torchref\n"
            "from torchref.utils import triton_available\n"
            "assert triton_available() is False, 'a broken triton must read as absent'\n"
            "print('ok')\n",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        "import torchref failed with a broken triton on the path:\n"
        f"{proc.stderr[-2000:]}"
    )
    assert "ok" in proc.stdout
