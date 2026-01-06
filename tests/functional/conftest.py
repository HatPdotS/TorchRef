"""
Functional test fixtures.

These fixtures provide real objects initialized from test files
for functional testing.
"""
import pytest
import torch
from pathlib import Path


# =============================================================================
# Base Path Fixtures (for files that don't exist in root conftest)
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
    from torchref.symmetrie.symmetrie import Symmetry
    
    sym = Symmetry(loaded_model.spacegroup)
    return {"model": loaded_model, "symmetry": sym}


@pytest.fixture
def initialized_scaler(model_and_data):
    """Fixture providing initialized Scaler with model and data."""
    from torchref.scaling.scaler import Scaler
    
    model = model_and_data["model"]
    data = model_and_data["data"]
    
    scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
    return scaler


@pytest.fixture
def model_with_restraints(loaded_model, external_monomer_library):
    """Fixture providing model with built restraints."""
    from torchref.restraints.restraints import Restraints
    
    restraints = Restraints(
        model=loaded_model,
        cif_path=str(external_monomer_library),
        verbose=0
    )
    restraints.build_restraints()
    return {"model": loaded_model, "restraints": restraints}
