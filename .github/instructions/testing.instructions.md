# Testing Framework Implementation Guide

This document outlines the testing framework architecture for `torchref`, designed for easy extensibility, clear separation of unit and integration tests, GitHub Actions CI/CD integration, and SLURM-based local testing.

---

## 1. Directory Structure

The test directory mirrors the `torchref/` module structure:

```
torchref/
├── tests/                              # All tests live here
│   ├── conftest.py                     # Root fixtures (shared across all tests)
│   ├── pytest.ini                      # Pytest configuration
│   │
│   ├── unit/                           # Unit tests (fast, no I/O, mocked)
│   │   ├── conftest.py                 # Unit-specific fixtures
│   │   ├── io/                         # Tests for torchref/io/
│   │   │   ├── test_cif_readers.py
│   │   │   ├── test_data_router.py
│   │   │   └── test_file_writers.py
│   │   ├── math_functions/             # Tests for torchref/math_functions/
│   │   │   ├── test_french_wilson.py
│   │   │   ├── test_math_torch.py
│   │   │   ├── test_math_numpy.py
│   │   │   ├── test_scattering_factors.py
│   │   │   └── test_optimized_kernels.py
│   │   ├── model/                      # Tests for torchref/model/
│   │   │   ├── test_model.py
│   │   │   ├── test_model_ft.py
│   │   │   └── test_parameter_wrappers.py
│   │   ├── refinement/                 # Tests for torchref/refinement/
│   │   │   ├── test_targets.py
│   │   │   ├── test_loss_weighting.py
│   │   │   └── test_optimizers.py
│   │   ├── restraints/                 # Tests for torchref/restraints/
│   │   │   ├── test_restraints.py
│   │   │   └── test_restraints_helper.py
│   │   ├── scaling/                    # Tests for torchref/scaling/
│   │   │   ├── test_scaler.py
│   │   │   └── test_solvent.py
│   │   ├── symmetrie/                  # Tests for torchref/symmetrie/
│   │   │   ├── test_symmetrie.py
│   │   │   ├── test_map_symmetry.py
│   │   │   └── test_grid_utils.py
│   │   └── utils/                      # Tests for torchref/utils/
│   │       ├── test_utils.py
│   │       └── test_gradnorm.py
│   │
│   ├── integration/                    # Integration tests (slower, real I/O)
│   │   ├── conftest.py                 # Integration-specific fixtures
│   │   ├── test_data_loading.py        # Full data loading pipeline
│   │   ├── test_refinement_pipeline.py # End-to-end refinement
│   │   ├── test_cli.py                 # CLI commands
│   │   └── test_full_workflow.py       # Complete workflows
│   │
│   ├── files/                          # Test data (COMMITTED to git)
│   │   ├── README.md                   # Documentation of test files
│   │   ├── cif/                        # Model coordinates in CIF format
│   │   │   ├── 1DAW.cif
│   │   │   ├── 2DQ6.cif
│   │   │   ├── 3A5V.cif
│   │   │   ├── ... (10 structures)
│   │   │   └── 6G9X.cif
│   │   ├── cif_sf/                     # Structure factors in CIF format
│   │   │   ├── 1DAW-sf.cif
│   │   │   ├── ... (10 structures)
│   │   │   └── 6G9X-sf.cif
│   │   ├── mtz/                        # Reflection data in MTZ format
│   │   │   ├── 1DAW.mtz
│   │   │   ├── ... (10 structures)
│   │   │   └── 6G9X.mtz
│   │   └── pdb/                        # Model coordinates in PDB format
│   │       ├── 1DAW.pdb
│   │       ├── ... (10 structures)
│   │       └── 6G9X.pdb
│   │
│   └── scripts/                        # Test runner scripts
│       ├── run_all.sh                  # Run all tests locally
│       ├── run_unit.sh                 # Run unit tests only
│       ├── run_integration.sh          # Run integration tests only
│       └── submit_tests.sbatch         # SLURM submission script
│
├── .github/
│   └── workflows/
│       └── tests.yml                   # GitHub Actions CI configuration
│
├── torchref/                           # Main package
│   ├── io/
│   ├── math_functions/
│   ├── model/
│   ├── refinement/
│   ├── restraints/
│   ├── scaling/
│   ├── symmetrie/
│   └── utils/
│
└── pyproject.toml                      # Updated with test configuration
```

---

## 2. Test Categories and Markers

### 2.1 Unit Tests (`tests/unit/`)
- **Characteristics:**
  - Fast execution (<1 second per test)
  - No file I/O (use mocks or minimal in-memory data)
  - Test single functions/classes in isolation
  - Run on every commit via GitHub Actions
  
- **Example markers:**
  ```python
  @pytest.mark.unit
  def test_calculate_gradient():
      ...
  ```

### 2.2 Integration Tests (`tests/integration/`)
- **Characteristics:**
  - Slower execution (may take minutes)
  - Real file I/O with test data
  - Test complete workflows and pipelines
  - Run on pull requests and nightly
  
- **Example markers:**
  ```python
  @pytest.mark.integration
  def test_full_refinement_pipeline():
      ...
  ```

### 2.3 GPU Tests
- **NOT run by default** (require explicit flag)
- Can be in either unit or integration
- Marked with `@pytest.mark.gpu`

```python
@pytest.mark.gpu
@pytest.mark.unit
def test_cuda_structure_factor_calculation():
    ...
```

### 2.4 Slow Tests
- Long-running tests that should be skipped in quick runs
- Marked with `@pytest.mark.slow`

```python
@pytest.mark.slow
@pytest.mark.integration
def test_large_structure_refinement():
    ...
```

---

## 3. Configuration Files

### 3.1 `tests/pytest.ini`

```ini
[pytest]
# Test discovery
python_files = test_*.py
python_classes = Test*
python_functions = test_*
testpaths = .

# Add parent to path for imports
pythonpath = ..

# Markers
markers =
    unit: Unit tests (fast, no I/O, isolated)
    integration: Integration tests (slower, real I/O, pipelines)
    gpu: Tests requiring CUDA GPU (skipped by default)
    slow: Long-running tests (skipped in quick runs)

# Default options (exclude GPU tests by default)
addopts =
    -v
    --tb=short
    --strict-markers
    -m "not gpu"
    -ra

# Ignore warnings from dependencies
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### 3.2 `tests/conftest.py` (Root)

```python
"""
Root pytest configuration and shared fixtures.
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
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "gpu: GPU-requiring tests")
    config.addinivalue_line("markers", "slow: Slow tests")


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
```

### 3.3 `tests/unit/conftest.py`

```python
"""
Unit test specific fixtures.
Unit tests should NOT use real file I/O.
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def random_coordinates():
    """Generate random atomic coordinates."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms, 3) * 10, dtype=torch.float32)
    return _generate


@pytest.fixture
def random_b_factors():
    """Generate random B-factors."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms) * 50 + 10, dtype=torch.float32)
    return _generate


@pytest.fixture
def mock_unit_cell():
    """Mock unit cell parameters [a, b, c, alpha, beta, gamma]."""
    return torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float32)


@pytest.fixture
def mock_hkl_indices():
    """Generate mock HKL indices."""
    def _generate(n_reflections: int = 100, seed: int = 42):
        np.random.seed(seed)
        h = np.random.randint(-10, 11, n_reflections)
        k = np.random.randint(-10, 11, n_reflections)
        l = np.random.randint(-10, 11, n_reflections)
        return torch.tensor(np.stack([h, k, l], axis=1), dtype=torch.int32)
    return _generate
```

### 3.4 `tests/integration/conftest.py`

```python
"""
Integration test specific fixtures.
These fixtures load real test data from tests/files/.
"""
import pytest
from pathlib import Path


# Available test structures (PDB IDs)
TEST_STRUCTURES = ["1DAW", "2DQ6", "3A5V", "3E98", "3GR5", "3K7M", "3VRJ", "4BX9", "5BOV", "6G9X"]


@pytest.fixture(scope="module", params=TEST_STRUCTURES[:3])  # Use first 3 by default
def structure_id(request):
    """Parameterized fixture for test structure IDs."""
    return request.param


@pytest.fixture(scope="module")
def structure_paths(test_files_dir, structure_id):
    """Get paths for a test structure."""
    return {
        "pdb_id": structure_id,
        "model_cif": test_files_dir / "cif" / f"{structure_id}.cif",
        "model_pdb": test_files_dir / "pdb" / f"{structure_id}.pdb",
        "sf_cif": test_files_dir / "cif_sf" / f"{structure_id}-sf.cif",
        "mtz": test_files_dir / "mtz" / f"{structure_id}.mtz",
    }


@pytest.fixture(scope="module")
def structure_1daw(test_files_dir):
    """Load 1DAW test structure (small, fast tests)."""
    return {
        "pdb_id": "1DAW",
        "model_cif": test_files_dir / "cif" / "1DAW.cif",
        "model_pdb": test_files_dir / "pdb" / "1DAW.pdb",
        "sf_cif": test_files_dir / "cif_sf" / "1DAW-sf.cif",
        "mtz": test_files_dir / "mtz" / "1DAW.mtz",
    }


@pytest.fixture(scope="module")
def loaded_model_1daw(structure_1daw):
    """Load 1DAW model using torchref."""
    from torchref.io import Data
    data = Data(
        model_path=str(structure_1daw["model_cif"]),
        sf_path=str(structure_1daw["sf_cif"]),
    )
    return data
```

---

## 4. GitHub Actions CI Configuration

### 4.1 `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [main, master, develop]
  pull_request:
    branches: [main, master, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run unit tests with coverage
        run: |
          pytest tests/unit/ -v --cov=torchref --cov-report=xml --cov-report=term-missing
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests  # Only run if unit tests pass
    
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true  # If using Git LFS for test data
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v --tb=short
```

---

## 5. SLURM Scripts for Local Testing

### 5.1 `tests/scripts/submit_tests.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=torchref-tests
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --output=test_results_%j.log
#SBATCH --error=test_errors_%j.log

# =============================================================================
# SLURM test submission script for torchref
# Usage: sbatch submit_tests.sbatch [unit|integration|all|gpu]
# =============================================================================

set -e

# Load environment
module load anaconda
conda activate /das/work/p17/p17490/CONDA/torchref

# Navigate to project root
cd /das/work/p17/p17490/Peter/Library/torchref

# Parse test type argument (default: unit)
TEST_TYPE="${1:-unit}"

echo "========================================"
echo "Running torchref tests: ${TEST_TYPE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "========================================"

case "$TEST_TYPE" in
    unit)
        echo "Running unit tests..."
        pytest tests/unit/ -v --tb=short --cov=torchref --cov-report=term-missing
        ;;
    integration)
        echo "Running integration tests..."
        pytest tests/integration/ -v --tb=short
        ;;
    all)
        echo "Running all tests..."
        pytest tests/ -v --tb=short --cov=torchref --cov-report=term-missing
        ;;
    gpu)
        echo "Running GPU tests..."
        pytest tests/ -v --tb=short --run-gpu -m gpu
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Usage: sbatch submit_tests.sbatch [unit|integration|all|gpu]"
        exit 1
        ;;
esac

echo "========================================"
echo "Tests completed at: $(date)"
echo "========================================"
```

### 5.2 `tests/scripts/run_unit.sh`

```bash
#!/bin/bash
# Quick unit test runner for interactive use with srun

set -e

cd "$(dirname "$0")/../.."

echo "Running unit tests..."
pytest tests/unit/ -v --tb=short "$@"
```

### 5.3 `tests/scripts/run_integration.sh`

```bash
#!/bin/bash
# Integration test runner

set -e

cd "$(dirname "$0")/../.."

echo "Running integration tests..."
pytest tests/integration/ -v --tb=short "$@"
```

---

## 6. Test Data Management

### 6.1 Test Files Location

All test files are located in `tests/files/` with the following structure:

```
tests/files/
├── cif/          # Model coordinates (CIF format)
├── cif_sf/       # Structure factors (CIF format)
├── mtz/          # Reflection data (MTZ format)
└── pdb/          # Model coordinates (PDB format)
```

### 6.2 Available Test Structures

The following 10 PDB structures are available for testing:

| PDB ID | Description |
|--------|-------------|
| 1DAW   | Small structure - recommended for fast unit tests |
| 2DQ6   | |
| 3A5V   | |
| 3E98   | |
| 3GR5   | |
| 3K7M   | |
| 3VRJ   | |
| 4BX9   | |
| 5BOV   | |
| 6G9X   | |

Each structure has 4 files available:
- `{PDB_ID}.cif` - Model coordinates in mmCIF format
- `{PDB_ID}.pdb` - Model coordinates in PDB format
- `{PDB_ID}-sf.cif` - Structure factors in mmCIF format
- `{PDB_ID}.mtz` - Reflection data in MTZ format

### 6.3 Usage Guidelines

- **Unit tests**: Use mock/synthetic data where possible. If real files are needed, use `1DAW` (smallest).
- **Integration tests**: Can use any of the 10 structures.
- **Parameterized tests**: Use the `structure_id` fixture to run tests across multiple structures.

### 6.4 `tests/files/README.md`

```markdown
# Test Files

This directory contains crystallographic test data for torchref tests.

## Directory Structure

- `cif/` - Model coordinates in mmCIF format
- `cif_sf/` - Structure factors in mmCIF format  
- `mtz/` - Reflection data in MTZ format
- `pdb/` - Model coordinates in PDB format

## Available Structures

All files are from the Protein Data Bank (public domain):
- 1DAW, 2DQ6, 3A5V, 3E98, 3GR5, 3K7M, 3VRJ, 4BX9, 5BOV, 6G9X

## Usage

Unit tests should prefer mock data. Integration tests can use these files via fixtures:

```python
def test_loading(structure_1daw):
    model_path = structure_1daw["model_cif"]
    ...
```
```

---

## 7. Writing Tests - Examples

### 7.1 Unit Test Example

```python
# tests/unit/math_functions/test_math_torch.py
"""Unit tests for torchref.math_functions.math_torch"""

import pytest
import torch
import numpy as np


class TestStructureFactorCalculation:
    """Tests for structure factor calculation functions."""

    @pytest.mark.unit
    def test_calculate_phase_basic(self, random_coordinates, mock_hkl_indices):
        """Test phase calculation with known values."""
        from torchref.math_functions.math_torch import calculate_phase
        
        coords = random_coordinates(n_atoms=5)
        hkl = mock_hkl_indices(n_reflections=10)
        
        phases = calculate_phase(coords, hkl)
        
        assert phases.shape == (10, 5)
        assert torch.all(torch.isfinite(phases))

    @pytest.mark.unit
    def test_calculate_phase_empty(self):
        """Test phase calculation with empty input."""
        from torchref.math_functions.math_torch import calculate_phase
        
        coords = torch.empty((0, 3))
        hkl = torch.tensor([[1, 0, 0]], dtype=torch.int32)
        
        phases = calculate_phase(coords, hkl)
        
        assert phases.shape == (1, 0)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_calculate_phase_gpu(self, random_coordinates, mock_hkl_indices, gpu_device):
        """Test phase calculation on GPU."""
        from torchref.math_functions.math_torch import calculate_phase
        
        coords = random_coordinates(n_atoms=100).to(gpu_device)
        hkl = mock_hkl_indices(n_reflections=1000).to(gpu_device)
        
        phases = calculate_phase(coords, hkl)
        
        assert phases.device.type == "cuda"
        assert phases.shape == (1000, 100)
```

### 7.2 Integration Test Example

```python
# tests/integration/test_data_loading.py
"""Integration tests for data loading pipeline."""

import pytest
from pathlib import Path


class TestDataLoading:
    """Tests for loading crystallographic data."""

    @pytest.mark.integration
    def test_load_cif_model(self, structure_1daw):
        """Test loading a CIF model file."""
        from torchref.io import Data
        
        data = Data(model_path=str(structure_1daw["model_cif"]))
        
        assert data.model is not None
        assert len(data.model.atoms) > 0

    @pytest.mark.integration
    def test_load_pdb_model(self, structure_1daw):
        """Test loading a PDB model file."""
        from torchref.io import Data
        
        data = Data(model_path=str(structure_1daw["model_pdb"]))
        
        assert data.model is not None
        assert len(data.model.atoms) > 0

    @pytest.mark.integration
    def test_load_complete_dataset_cif(self, structure_1daw):
        """Test loading model with structure factors (CIF)."""
        from torchref.io import Data
        
        data = Data(
            model_path=str(structure_1daw["model_cif"]),
            sf_path=str(structure_1daw["sf_cif"]),
        )
        
        assert data.model is not None
        assert data.reflections is not None
        assert len(data.reflections) > 0

    @pytest.mark.integration
    def test_load_complete_dataset_mtz(self, structure_1daw):
        """Test loading model with structure factors (MTZ)."""
        from torchref.io import Data
        
        data = Data(
            model_path=str(structure_1daw["model_cif"]),
            sf_path=str(structure_1daw["mtz"]),
        )
        
        assert data.model is not None
        assert data.reflections is not None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_refinement_cycle(self, loaded_model_1daw):
        """Test a complete refinement cycle."""
        from torchref.refinement import LBFGSRefinement
        
        # This is a slow test - only runs with --run-slow
        refiner = LBFGSRefinement(loaded_model_1daw)
        result = refiner.run(max_iterations=10)
        
        assert result.converged or result.iterations == 10


class TestMultipleStructures:
    """Tests that run across multiple structures."""

    @pytest.mark.integration
    def test_load_all_structures(self, structure_paths):
        """Test loading works for all test structures."""
        from torchref.io import Data
        
        data = Data(
            model_path=str(structure_paths["model_cif"]),
            sf_path=str(structure_paths["sf_cif"]),
        )
        
        assert data.model is not None
        assert data.reflections is not None
```

---

## 8. pyproject.toml Updates

Add/update the following sections in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
markers = [
    "unit: Unit tests (fast, no I/O)",
    "integration: Integration tests (slower, real I/O)",
    "gpu: GPU tests (skipped by default)",
    "slow: Slow tests (skipped by default)",
]
addopts = "-v --tb=short --strict-markers -m 'not gpu'"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]

[tool.coverage.run]
source = ["torchref"]
branch = true
omit = [
    "torchref/__init__.py",
    "*/tests/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
show_missing = true
```

---

## 9. Implementation Checklist

### Phase 1: Setup Structure
- [ ] Create `tests/` directory structure
- [ ] Create `tests/conftest.py` (root fixtures)
- [ ] Create `tests/pytest.ini`
- [ ] Create `tests/unit/conftest.py`
- [ ] Create `tests/integration/conftest.py`
- [ ] Update `pyproject.toml` with test configuration

### Phase 2: Test Data
- [ ] Verify `tests/files/` directory structure is correct
- [ ] Create `tests/files/README.md`
- [ ] Ensure all 10 structures have all 4 file formats

### Phase 3: Initial Tests
- [ ] Create 1 unit test file per major module
- [ ] Create 1-2 integration tests for core workflows
- [ ] Verify all tests pass locally

### Phase 4: CI/CD
- [ ] Create `.github/workflows/tests.yml`
- [ ] Set up Codecov integration (optional)
- [ ] Create SLURM scripts in `tests/scripts/`

### Phase 5: Coverage & Documentation
- [ ] Add coverage reporting to CI
- [ ] Document test conventions
- [ ] Add badges to README

---

## 10. Running Tests

### Local Development (Interactive)
```bash
# Quick unit tests
srun -c 8 -p day -t 1:00:00 bash -c "cd /das/work/p17/p17490/Peter/Library/torchref && pytest tests/unit/ -v"

# With coverage
srun -c 8 -p day -t 1:00:00 bash -c "cd /das/work/p17/p17490/Peter/Library/torchref && pytest tests/unit/ -v --cov=torchref"
```

### Batch Submission
```bash
# Unit tests
sbatch tests/scripts/submit_tests.sbatch unit

# Integration tests
sbatch tests/scripts/submit_tests.sbatch integration

# All tests
sbatch tests/scripts/submit_tests.sbatch all

# GPU tests (submit to GPU partition)
sbatch -p gpu tests/scripts/submit_tests.sbatch gpu
```

### Specific Test Selection
```bash
# Run specific test file
pytest tests/unit/math_functions/test_math_torch.py -v

# Run specific test class
pytest tests/unit/math_functions/test_math_torch.py::TestStructureFactorCalculation -v

# Run specific test
pytest tests/unit/math_functions/test_math_torch.py::TestStructureFactorCalculation::test_calculate_phase_basic -v

# Run by marker
pytest -m unit           # Only unit tests
pytest -m integration    # Only integration tests
pytest -m "not slow"     # Skip slow tests
pytest --run-gpu -m gpu  # Run GPU tests
```

---

## 11. Best Practices Summary

1. **Mirror module structure** in test directories for easy navigation
2. **Use markers** consistently (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. **Keep unit tests fast** - mock I/O, use minimal data
4. **Document test data** - track sources and licenses
5. **Run unit tests on every commit** via GitHub Actions
6. **Skip GPU tests by default** - require explicit `--run-gpu` flag
7. **Use fixtures** for common setup - reduce code duplication
8. **Name tests descriptively** - `test_<what>_<condition>_<expected>`
9. **One assertion per test** when possible - easier debugging
10. **Clean up after tests** - use fixtures with proper teardown
