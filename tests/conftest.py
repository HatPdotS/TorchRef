"""Register shared fixture plugins and gate tests on host capabilities."""

import importlib.util
import shutil
import warnings

import pytest

# Apply process-wide settings before the device fixtures import torch.
import torchref  # noqa: F401

pytest_plugins = (
    "tests.fixtures.paths",
    "tests.fixtures.files",
    "tests.fixtures.devices",
    "tests.fixtures.precision",
    "tests.fixtures.objects",
)

_HAS_OPENMM = importlib.util.find_spec("openmm") is not None
_HAS_AMBERTOOLS = bool(shutil.which("antechamber") and shutil.which("tleap"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-cuda",
        action="store_true",
        default=False,
        help=(
            "Require CUDA: fail instead of skipping if it is unavailable. "
            "CUDA tests already run automatically when a CUDA device is "
            "present; use this in CI to catch a runner that lost its GPU."
        ),
    )
    parser.addoption(
        "--run-mps",
        action="store_true",
        default=False,
        help=(
            "Require MPS: fail instead of skipping if it is unavailable. "
            "MPS tests already run automatically on Apple silicon."
        ),
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Deprecated no-op: accelerator tests now run automatically.",
    )
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no I/O)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, real I/O)"
    )
    config.addinivalue_line(
        "markers", "gpu: Needs any accelerator (CUDA or MPS); auto-skipped if none"
    )
    config.addinivalue_line(
        "markers", "cuda: Needs CUDA specifically (e.g. Triton); auto-skipped if absent"
    )
    config.addinivalue_line(
        "markers", "mps: Needs MPS specifically (Metal kernels); auto-skipped if absent"
    )
    config.addinivalue_line("markers", "cuda_only: Deprecated alias for 'cuda'")
    config.addinivalue_line("markers", "slow: Slow tests (skipped by default)")
    config.addinivalue_line(
        "markers", "openmm: Needs OpenMM (the [amber] extra); skipped if absent"
    )
    config.addinivalue_line(
        "markers",
        "amber: Needs OpenMM + AmberTools (antechamber/tleap); skipped if absent",
    )

    if config.getoption("--run-gpu"):
        # UserWarning, not DeprecationWarning: pytest.ini filters the latter,
        # and a silently-swallowed notice is worse than none when the whole
        # point is telling someone their flag no longer does anything.
        warnings.warn(
            "--run-gpu is deprecated and does nothing: accelerator tests now "
            "run automatically wherever the backend is available. Use "
            "--run-cuda / --run-mps to *require* a backend (fail rather than "
            "skip when it is missing).",
            UserWarning,
            stacklevel=2,
        )


def pytest_collection_modifyitems(config, items):
    """Gate tests on what this host can actually do.

    Accelerator tests are **not** opt-in: a ``cuda``-marked test runs whenever
    CUDA is present, an ``mps``-marked test whenever MPS is, and a ``gpu``-marked
    (backend-agnostic) test whenever either is. Anything the host cannot run is
    skipped with a reason naming the missing backend.

    ``--run-cuda`` / ``--run-mps`` invert the *absence* case from skip to
    error, for CI that expects a specific backend and would otherwise go green
    on a runner that quietly lost its GPU. They do it by *not* adding the skip
    marker, so the backend tests run and fail with the real error from torch.

    This function is the **only** place that decides what runs. Tests must not
    re-check availability themselves: a second layer of ``pytest.skip`` can only
    mask a forgotten marker, and turns "this host cannot run it" into a silent
    pass instead of the visible skip or the real error.
    """
    from tests.fixtures.devices import _cuda_available, _mps_available

    has_cuda = _cuda_available()
    has_mps = _mps_available()

    # Forced-but-absent is a warning, not a ``pytest.UsageError``. A UsageError
    # aborts the entire session -- every unrelated test with it -- and says
    # nothing about which backend call actually broke. Warning and letting the
    # marked tests run gives a precise per-test failure and still runs the rest
    # of the suite. UserWarning, not DeprecationWarning: pytest.ini filters the
    # latter (see the --run-gpu note in pytest_configure).
    require_cuda = config.getoption("--run-cuda")
    require_mps = config.getoption("--run-mps")
    if require_cuda and not has_cuda:
        warnings.warn(
            "--run-cuda given but no CUDA device is available: running the "
            "cuda-marked tests anyway so they error with the real backend "
            "error instead of being skipped.",
            UserWarning,
            stacklevel=2,
        )
    if require_mps and not has_mps:
        warnings.warn(
            "--run-mps given but MPS is not available: running the mps-marked "
            "tests anyway so they error with the real backend error instead of "
            "being skipped.",
            UserWarning,
            stacklevel=2,
        )

    skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
    skip_openmm = pytest.mark.skip(
        reason="OpenMM not installed (pip install '.[amber]')"
    )
    skip_amber = pytest.mark.skip(
        reason="AmberTools (antechamber/tleap) not on PATH (conda install ambertools)"
    )
    skip_cuda = pytest.mark.skip(reason="No CUDA device on this host")
    skip_mps = pytest.mark.skip(reason="No MPS device on this host")
    skip_gpu = pytest.mark.skip(reason="No accelerator (CUDA or MPS) on this host")

    for item in items:
        keywords = item.keywords
        # ``cuda_only`` is the retired spelling of ``cuda``.
        wants_cuda = "cuda" in keywords or "cuda_only" in keywords
        wants_mps = "mps" in keywords
        if wants_cuda and not has_cuda and not require_cuda:
            item.add_marker(skip_cuda)
        if wants_mps and not has_mps and not require_mps:
            item.add_marker(skip_mps)
        # A bare ``gpu`` mark means "any accelerator"; a test that also names a
        # specific backend has already been gated on the stricter condition.
        if (
            "gpu" in keywords
            and not (wants_cuda or wants_mps)
            and not (has_cuda or has_mps)
        ):
            item.add_marker(skip_gpu)
        if "slow" in keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        # Amber stack gates: "amber" needs OpenMM + AmberTools; "openmm" needs
        # just OpenMM. Skip with the most specific missing-dependency reason.
        if "amber" in item.keywords:
            if not _HAS_OPENMM:
                item.add_marker(skip_openmm)
            elif not _HAS_AMBERTOOLS:
                item.add_marker(skip_amber)
        elif "openmm" in item.keywords and not _HAS_OPENMM:
            item.add_marker(skip_openmm)
