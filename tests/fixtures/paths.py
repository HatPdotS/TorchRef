"""Locate bundled test data and optional monomer-library installations."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tests_root() -> Path:
    """Return the root of the test tree."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def test_files_dir(tests_root: Path) -> Path:
    """Return the bundled test-data directory."""
    return tests_root / "files"


@pytest.fixture(scope="session")
def cif_dir(test_files_dir: Path) -> Path:
    """Return the model CIF directory."""
    return test_files_dir / "cif"


@pytest.fixture(scope="session")
def cif_sf_dir(test_files_dir: Path) -> Path:
    """Return the structure-factor CIF directory."""
    return test_files_dir / "cif_sf"


@pytest.fixture(scope="session")
def mtz_dir(test_files_dir: Path) -> Path:
    """Return the MTZ reflection directory."""
    return test_files_dir / "mtz"


@pytest.fixture(scope="session")
def pdb_dir(test_files_dir: Path) -> Path:
    """Return the model PDB directory."""
    return test_files_dir / "pdb"


@pytest.fixture(scope="session")
def external_monomer_library(project_root: Path) -> Path:
    """Return the optional external monomer-library path without checking it."""
    return project_root / "external_monomer_library"


@pytest.fixture(scope="session")
def monomer_library_path(project_root: Path) -> str:
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
