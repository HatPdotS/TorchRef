"""Select sample paths and matching model/reflection pairs without loading them."""

from pathlib import Path

import pytest

from tests.helpers.structure_cases import EXTENDED_PAIR_CODES


@pytest.fixture(
    params=[pytest.param(code, marks=pytest.mark.slow) for code in EXTENDED_PAIR_CODES]
)
def compatibility_structure_pair(
    cif_dir: Path, mtz_dir: Path, request: pytest.FixtureRequest
) -> dict:
    """Select one named extended crystal without loading its model or observations."""
    code = request.param
    return {
        "pdb_id": code,
        "model": cif_dir / f"{code}.cif",
        "reflections": mtz_dir / f"{code}.mtz",
    }


@pytest.fixture(scope="session")
def sample_cif_file(cif_dir: Path) -> Path:
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
def sample_mtz_file(mtz_dir: Path) -> Path:
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
def sample_pdb_file(pdb_dir: Path) -> Path:
    """Return a sample PDB file for testing."""
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        pytest.skip("No PDB files found in test data directory")
    return pdb_files[0]


@pytest.fixture(scope="session")
def sample_structure_factor_cif(cif_sf_dir: Path) -> Path:
    """Return a sample structure factor CIF file."""
    sf_files = sorted(cif_sf_dir.glob("*.cif"))
    if not sf_files:
        pytest.skip("No structure factor CIF files found")
    return sf_files[0]


@pytest.fixture(scope="session")
def sample_structure_pair(cif_dir: Path, mtz_dir: Path) -> dict[str, Path]:
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
        pdb_id = min(common_ids)
        return {"model": cif_files[pdb_id], "reflections": mtz_files[pdb_id]}

    pytest.skip("No matching CIF/MTZ pairs found in test data")
