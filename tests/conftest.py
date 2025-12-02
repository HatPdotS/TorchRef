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
    config.addinivalue_line("markers", "gpu: GPU-requiring tests (skipped by default)")
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


@pytest.fixture(scope="session")
def gpu_device() -> torch.device:
    """GPU torch device (only use with @pytest.mark.gpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA not available")


@pytest.fixture
def device(request) -> torch.device:
    """Default device - CPU unless in GPU-marked test."""
    if "gpu" in [m.name for m in request.node.iter_markers()]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        pytest.skip("CUDA not available")
    return torch.device("cpu")


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
