"""
Root pytest configuration and shared fixtures for torchref tests.

This module provides fixtures that are automatically available to all test files.
"""
import importlib.util
import shutil
import warnings

import pytest
import torchref
import torch
import numpy as np
from pathlib import Path


# Optional Amber/ensemble stack: OpenMM (pip ``[amber]`` extra) and AmberTools
# (antechamber/tleap — conda-only, detected on PATH). Tests that need them are
# tagged ``@pytest.mark.openmm`` (OpenMM only) or ``@pytest.mark.amber`` (OpenMM
# + AmberTools) and auto-skipped below when the stack is absent.
_HAS_OPENMM = importlib.util.find_spec("openmm") is not None
_HAS_AMBERTOOLS = bool(shutil.which("antechamber") and shutil.which("tleap"))


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


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
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no I/O)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, real I/O)")
    config.addinivalue_line("markers", "gpu: Needs any accelerator (CUDA or MPS); auto-skipped if none")
    config.addinivalue_line("markers", "cuda: Needs CUDA specifically (e.g. Triton); auto-skipped if absent")
    config.addinivalue_line("markers", "mps: Needs MPS specifically (Metal kernels); auto-skipped if absent")
    config.addinivalue_line("markers", "cuda_only: Deprecated alias for 'cuda'")
    config.addinivalue_line("markers", "slow: Slow tests (skipped by default)")
    config.addinivalue_line("markers", "openmm: Needs OpenMM (the [amber] extra); skipped if absent")
    config.addinivalue_line("markers", "amber: Needs OpenMM + AmberTools (antechamber/tleap); skipped if absent")

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
    skip_openmm = pytest.mark.skip(reason="OpenMM not installed (pip install '.[amber]')")
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


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def tests_root() -> Path:
    """Root of the tests directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Root of the project."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_files_dir(tests_root) -> Path:
    """Path to test files directory."""
    return tests_root / "files"


@pytest.fixture(scope="session")
def cif_dir(test_files_dir) -> Path:
    """Path to CIF model files."""
    return test_files_dir / "cif"


@pytest.fixture(scope="session")
def cif_sf_dir(test_files_dir) -> Path:
    """Path to CIF structure factor files."""
    return test_files_dir / "cif_sf"


@pytest.fixture(scope="session")
def mtz_dir(test_files_dir) -> Path:
    """Path to MTZ reflection files."""
    return test_files_dir / "mtz"


@pytest.fixture(scope="session")
def pdb_dir(test_files_dir) -> Path:
    """Path to PDB model files."""
    return test_files_dir / "pdb"


@pytest.fixture(scope="session")
def external_monomer_library(project_root) -> Path:
    """Path to external monomer library."""
    return project_root / "external_monomer_library"


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """CPU torch device."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_device() -> torch.device:
    """GPU torch device (only use with @pytest.mark.gpu).

    Prefers CUDA, falls back to MPS; skips if neither is available. Prefer the
    backend-specific ``cuda_device`` / ``mps_device`` below when a test needs one
    particular backend -- this fixture's preference order means a
    ``cuda``-marked test asking for it on a dual-backend host could be handed
    MPS, which is why the MPS tests used to carry a ``type != 'mps'`` skip to
    undo it.
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
def any_device(request) -> torch.device:
    """Every device this host can actually use, one test run per device.

    The CPU leg always runs. The accelerator leg is ``gpu``-marked, so a plain
    ``pytest`` run skips it and ``pytest --run-gpu`` picks up CUDA on a CUDA
    box or MPS on a Mac. On a CPU-only host the accelerator parameter does not
    exist at all, so there is no skip noise.
    """
    return request.param


@pytest.fixture(scope="session")
def _device_model_cache() -> dict:
    """``{device_str: ModelFT}`` built at most once per device, per session."""
    return {}


@pytest.fixture
def device_model_bundle(_device_model_cache, pdb_dir, any_device):
    """A loaded model on ``any_device``, for target conformance tests.

    The existing ``loaded_model`` / ``model_and_data`` fixtures are
    function-scoped and construct on the process default, so a
    device-parametrized sweep over them would reload the structure once per
    test per device. This caches one model per device instead.

    Shared mutable state: callers must treat the bundle as read-only. A test
    that moves a *target* will drag the borrowed model with it, poisoning every
    later test on that device -- see ``test_target_device_round_trip``, which
    deliberately builds its own.
    """
    key = str(any_device)
    if key not in _device_model_cache:
        pdb = pdb_dir / "1DAW.pdb"
        if not pdb.exists():
            pytest.skip("1DAW.pdb fixture not present")
        from torchref.model import ModelFT

        _device_model_cache[key] = ModelFT(device=any_device, verbose=0).load_pdb(
            str(pdb)
        )
    return {"model": _device_model_cache[key]}


@pytest.fixture
def device(request) -> torch.device:
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


# =============================================================================
# Numerical Fixtures
# =============================================================================

@pytest.fixture
def rtol() -> float:
    """Relative tolerance for floating point comparisons."""
    return 1e-5


@pytest.fixture
def atol() -> float:
    """Absolute tolerance for floating point comparisons."""
    return 1e-8


# =============================================================================
# Sample File Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_cif_file(cif_dir):
    """Return a sample CIF file for testing."""
    cif_file = cif_dir / "1DAW.cif"
    if cif_file.exists():
        return cif_file
    # Try any available CIF file
    cif_files = list(cif_dir.glob("*.cif"))
    if cif_files:
        return cif_files[0]
    pytest.skip("No CIF files found in test data")


@pytest.fixture(scope="session")
def sample_mtz_file(mtz_dir):
    """Return a sample MTZ file for testing."""
    mtz_file = mtz_dir / "1DAW.mtz"
    if mtz_file.exists():
        return mtz_file
    # Try any available MTZ file
    mtz_files = list(mtz_dir.glob("*.mtz"))
    if mtz_files:
        return mtz_files[0]
    pytest.skip("No MTZ files found in test data")


@pytest.fixture(scope="session")
def sample_pdb_file(pdb_dir):
    """Return a sample PDB file for testing."""
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        pytest.skip("No PDB files found in test data directory")
    return pdb_files[0]


@pytest.fixture(scope="session")
def sample_structure_factor_cif(cif_sf_dir):
    """Return a sample structure factor CIF file."""
    sf_files = sorted(cif_sf_dir.glob("*.cif"))
    if not sf_files:
        pytest.skip("No structure factor CIF files found")
    return sf_files[0]


@pytest.fixture(scope="session")
def sample_structure_pair(cif_dir, mtz_dir):
    """Return a matching pair of CIF model and MTZ reflections."""
    # Try to find matching files
    pdb_id = "1DAW"
    cif_file = cif_dir / f"{pdb_id}.cif"
    mtz_file = mtz_dir / f"{pdb_id}.mtz"

    if cif_file.exists() and mtz_file.exists():
        return {"model": cif_file, "reflections": mtz_file}

    # Try to find any matching pair
    cif_files = {f.stem: f for f in cif_dir.glob("*.cif")}
    mtz_files = {f.stem: f for f in mtz_dir.glob("*.mtz")}

    common_ids = set(cif_files.keys()) & set(mtz_files.keys())
    if common_ids:
        pdb_id = sorted(common_ids)[0]
        return {"model": cif_files[pdb_id], "reflections": mtz_files[pdb_id]}

    pytest.skip("No matching CIF/MTZ pairs found in test data")


@pytest.fixture(scope="session")
def all_structure_pairs(cif_dir, mtz_dir):
    """Return all matching pairs of CIF models and MTZ reflections."""
    cif_files = {f.stem: f for f in cif_dir.glob("*.cif")}
    mtz_files = {f.stem: f for f in mtz_dir.glob("*.mtz")}

    common_ids = set(cif_files.keys()) & set(mtz_files.keys())

    if not common_ids:
        pytest.skip("No matching CIF/MTZ pairs found in test data")

    return [
        {"pdb_id": pdb_id, "model": cif_files[pdb_id], "reflections": mtz_files[pdb_id]}
        for pdb_id in sorted(common_ids)
    ]


@pytest.fixture(scope="session")
def all_cif_files(cif_dir):
    """Return all available CIF test structure files."""
    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        pytest.skip("No CIF files found in test data directory")
    return cif_files


@pytest.fixture(scope="session")
def all_test_structures(all_structure_pairs):
    """Return all loaded model/data pairs for comprehensive testing."""
    from torchref.model.model import Model
    from torchref.io import ReflectionData

    structures = []
    for pair in all_structure_pairs:
        try:
            model = Model()
            model.load_cif(str(pair["model"]))

            data = ReflectionData()
            data.load_mtz(str(pair["reflections"]))

            structures.append({
                "pdb_id": pair["pdb_id"],
                "model": model,
                "data": data,
                "model_path": pair["model"],
                "data_path": pair["reflections"]
            })
        except Exception:
            # Skip structures that fail to load
            continue

    if not structures:
        pytest.skip("No structures could be loaded")

    return structures


@pytest.fixture(scope="session")
def monomer_library_path(project_root):
    """Get path to the monomer library as a string.

    Returns
    -------
    str
        Absolute path to the external_monomer_library directory.
    """
    lib_path = project_root / "external_monomer_library"
    if not lib_path.exists():
        pytest.skip("Monomer library not found")
    return str(lib_path)


# =============================================================================
# Real Object Fixtures
# =============================================================================

@pytest.fixture
def loaded_model(sample_cif_file):
    """Fixture providing a fully loaded Model from a real CIF file."""
    from torchref.model.model import Model

    model = Model()
    model.load_cif(str(sample_cif_file))
    return model


@pytest.fixture
def loaded_reflection_data(sample_mtz_file):
    """Fixture providing fully loaded ReflectionData from a real MTZ file."""
    from torchref.io import ReflectionData

    data = ReflectionData()
    data.load_mtz(str(sample_mtz_file))
    return data


@pytest.fixture
def model_and_data(sample_structure_pair):
    """Fixture providing matching model and reflection data."""
    from torchref.model.model import Model
    from torchref.io import ReflectionData

    model = Model()
    model.load_cif(str(sample_structure_pair["model"]))

    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    return {"model": model, "data": data}


@pytest.fixture
def model_with_symmetry(loaded_model):
    """Fixture providing model with initialized symmetry."""
    from torchref.symmetry import SpaceGroup

    sg = SpaceGroup(loaded_model.spacegroup)
    return {"model": loaded_model, "symmetry": sg}


@pytest.fixture
def initialized_scaler(model_and_data):
    """Fixture providing initialized Scaler with model and data."""
    from torchref.scaling.scaler import Scaler

    model = model_and_data["model"]
    data = model_and_data["data"]

    scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
    return scaler


@pytest.fixture
def model_with_restraints(loaded_model):
    """Fixture providing model with built restraints."""
    from torchref.restraints import Restraints

    restraints = Restraints(
        pdb=loaded_model.pdb,
        xyz_fn=loaded_model.xyz,
        vdw_radii_fn=loaded_model.get_vdw_radii,
        verbose=0
    )
    restraints.build_restraints()
    return {"model": loaded_model, "restraints": restraints}

@pytest.fixture
def double_cpu():
    """float64/complex128 on CPU for the duration of a test; restore afterwards.

    Required rather than cosmetic for anything touching eager structure factors:
    ``iso_structure_factor_torched`` casts ``hkl`` to the *global* ``dtypes.float``
    (``torchref/base/direct_summation/isotropic.py:121``), so under the default float32
    config a float64 leaf produces a dtype-mismatched matmul.

    Promoted here from three byte-similar copies in ``tests/unit/test_kernel_fixes.py``,
    ``tests/unit/test_gradient_correctness.py`` and
    ``tests/integration/test_dtype_config_float64.py``. This version also restores
    ``sigma_cutoff_ed``, which none of those did -- so a test that changed the cutoff
    leaked it into everything that ran afterwards.
    """
    import torchref
    from torchref.config import device as _device, dtypes as _dtypes

    f0, c0, d0 = _dtypes.float, _dtypes.complex, _device.current
    s0 = torchref.sigma_cutoff_ed.value
    _dtypes.float = torch.float64
    _dtypes.complex = torch.complex128
    _device.current = torch.device("cpu")
    try:
        yield
    finally:
        _dtypes.float = f0
        _dtypes.complex = c0
        _device.current = d0
        torchref.sigma_cutoff_ed.value = s0
