"""
Integration test specific fixtures.
Integration tests use real file I/O and test the full pipeline.
"""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def sample_cif_file(test_files_dir):
    """Get a sample CIF file from the test files directory."""
    cif_dir = test_files_dir / "cif"
    if not cif_dir.exists():
        pytest.skip("Test data directory 'cif' not found")
    
    # Get first available CIF file (sorted for deterministic ordering)
    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        pytest.skip("No CIF files found in test data directory")

    return cif_files[0]


@pytest.fixture
def sample_pdb_file(test_files_dir):
    """Get a sample PDB file from the test files directory."""
    pdb_dir = test_files_dir / "pdb"
    if not pdb_dir.exists():
        pytest.skip("Test data directory 'pdb' not found")
    
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        pytest.skip("No PDB files found in test data directory")

    return pdb_files[0]


@pytest.fixture
def sample_mtz_file(test_files_dir):
    """Get a sample MTZ file from the test files directory."""
    mtz_dir = test_files_dir / "mtz"
    if not mtz_dir.exists():
        pytest.skip("Test data directory 'mtz' not found")
    
    mtz_files = sorted(mtz_dir.glob("*.mtz"))
    if not mtz_files:
        pytest.skip("No MTZ files found in test data directory")

    return mtz_files[0]


@pytest.fixture
def sample_structure_factor_cif(test_files_dir):
    """Get a sample structure factor CIF file."""
    sf_dir = test_files_dir / "cif_sf"
    if not sf_dir.exists():
        pytest.skip("Test data directory 'cif_sf' not found")
    
    sf_files = sorted(sf_dir.glob("*.cif"))
    if not sf_files:
        pytest.skip("No structure factor CIF files found")

    return sf_files[0]


@pytest.fixture
def sample_structure_pair(test_files_dir):
    """Get a matching pair of model CIF and reflection data."""
    cif_dir = test_files_dir / "cif"
    mtz_dir = test_files_dir / "mtz"
    
    if not cif_dir.exists() or not mtz_dir.exists():
        pytest.skip("Test data directories not found")
    
    cif_files = sorted(cif_dir.glob("*.cif"))
    mtz_files = sorted(mtz_dir.glob("*.mtz"))
    
    if not cif_files or not mtz_files:
        pytest.skip("No matching structure files found")
    
    # Try to find matching pair by PDB ID
    for cif_file in cif_files:
        pdb_id = cif_file.stem.lower()
        for mtz_file in mtz_files:
            if pdb_id in mtz_file.stem.lower():
                return {"model": cif_file, "reflections": mtz_file, "pdb_id": pdb_id}
    
    # Fallback: just return first of each
    return {"model": cif_files[0], "reflections": mtz_files[0], "pdb_id": cif_files[0].stem}


@pytest.fixture
def all_test_structures(test_files_dir):
    """Get all available test structures."""
    cif_dir = test_files_dir / "cif"
    if not cif_dir.exists():
        pytest.skip("Test data directory 'cif' not found")
    
    return list(cif_dir.glob("*.cif"))


@pytest.fixture(scope="session")
def monomer_library_path():
    """Get path to the monomer library as a string.
    
    Returns
    -------
    str
        Absolute path to the external_monomer_library directory.
    """
    # Path: conftest.py -> integration -> tests -> torchref (repo root)
    lib_path = Path(__file__).parent.parent.parent / "external_monomer_library"
    if not lib_path.exists():
        pytest.skip("Monomer library not found")
    return str(lib_path)
