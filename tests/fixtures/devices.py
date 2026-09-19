"""Provide explicit backend fixtures and the configured default device.

Capability probes are shared with collection hooks and structure-factor cases.
Device parametrization is constructed at import so collection can see its marks.
"""

import pytest
import torch


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _accelerator() -> "torch.device | None":
    """The canonical accelerator this host can actually use, or ``None``.

    Indices are filled in (``cuda:0`` / ``mps:0``) so the value compares equal
    to a device read back off a real tensor -- ``torch.device('mps')`` and
    ``torch.device('mps:0')`` are *not* equal even though they name the same
    physical device.
    """
    if _cuda_available():
        return torch.device("cuda", torch.cuda.current_device())
    if _mps_available():
        return torch.device("mps", 0)
    return None


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """CPU torch device."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_device() -> torch.device:
    """Select CUDA, then MPS, for tests marked ``gpu``.

    Skip if neither backend is available. Use ``cuda_device`` or ``mps_device``
    when the test exercises a backend-specific contract.
    """
    accel = _accelerator()
    if accel is None:
        pytest.skip("No accelerator (CUDA or MPS) on this host")
    return accel


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    """Canonical CUDA device for ``cuda``-marked tests.

    Deliberately unguarded. What runs is decided by the ``cuda`` marker in
    :func:`pytest_collection_modifyitems` and nowhere else, so this fixture does
    not re-check availability: on a host without CUDA the test is *meant* to
    error with the real backend error rather than be quietly skipped here.
    """
    return torch.device("cuda", 0)


@pytest.fixture(scope="session")
def mps_device() -> torch.device:
    """Canonical MPS device for ``mps``-marked tests.

    Unguarded for the same reason as :func:`cuda_device` -- the ``mps`` marker
    owns the decision.
    """
    return torch.device("mps", 0)


# Built at import time so the ``gpu`` mark is attached during *collection*.
# Adding it later (e.g. via ``request.node.add_marker`` inside the fixture) is
# too late for ``pytest_collection_modifyitems`` to gate on.
_DEVICE_PARAMS = [pytest.param(torch.device("cpu"), id="cpu")]
_ACCELERATOR = _accelerator()
if _ACCELERATOR is not None:
    _DEVICE_PARAMS.append(
        pytest.param(
            _ACCELERATOR,
            id=_ACCELERATOR.type,
            # Backend-specific mark, so a CUDA-less host skips the cuda leg and
            # a non-Mac skips the mps leg, each with an accurate reason.
            marks=getattr(pytest.mark, _ACCELERATOR.type),
        )
    )


@pytest.fixture(params=_DEVICE_PARAMS)
def any_device(request: pytest.FixtureRequest) -> torch.device:
    """Every device this host can actually use, one test run per device.

    The CPU leg always runs. An available accelerator runs automatically and
    carries its backend-specific marker. No accelerator leg is created on a
    CPU-only host.
    """
    return request.param


@pytest.fixture
def device(request: pytest.FixtureRequest) -> torch.device:
    """Default test device.

    Uses the package-wide auto-detected default (``torchref.device.current``)
    so tests run on whichever device the user's machine resolved to at
    import time: cuda -> mps -> cpu. Tests marked ``@pytest.mark.cuda_only``
    are skipped when CUDA is not available.
    """
    from torchref.config import get_default_device

    markers = {m.name for m in request.node.iter_markers()}
    if "cuda_only" in markers and not torch.cuda.is_available():
        pytest.skip("Test requires CUDA")
    if "gpu" in markers and not (_cuda_available() or _mps_available()):
        pytest.skip("No GPU (CUDA or MPS) available")
    return get_default_device()
