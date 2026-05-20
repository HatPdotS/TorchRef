"""
Root pytest configuration and shared fixtures for torchref tests.

This module provides fixtures that are automatically available to all test files.
"""
import pytest
import torch
import numpy as np
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests"
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
    config.addinivalue_line("markers", "gpu: GPU-requiring tests (CUDA or MPS; skipped by default)")
    config.addinivalue_line("markers", "cuda_only: Tests that specifically require CUDA (e.g. Triton)")
    config.addinivalue_line("markers", "slow: Slow tests (skipped by default)")


def pytest_collection_modifyitems(config, items):
    """Skip GPU/slow tests unless explicitly requested."""
    skip_gpu = pytest.mark.skip(reason="Need --run-gpu option to run")
    skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
    
    for item in items:
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)


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


def _gpu_available() -> bool:
    """Return True if any GPU backend (CUDA or MPS) is available."""
    if torch.cuda.is_available():
        return True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True
    return False


@pytest.fixture(scope="session")
def gpu_device() -> torch.device:
    """GPU torch device (only use with @pytest.mark.gpu).

    Prefers CUDA, falls back to MPS; skips if neither is available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("No GPU (CUDA or MPS) available")


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
    if "gpu" in markers and not _gpu_available():
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
