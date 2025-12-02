# torchref Test Suite

This directory contains the complete test suite for torchref.

## Directory Structure

```
tests/
├── conftest.py              # Root fixtures (paths, devices, skip decorators)
├── pytest.ini               # Pytest configuration
├── __init__.py
├── files/                   # Test data files (CIF, PDB, MTZ)
│   ├── cif/                 # Model CIF files
│   ├── pdb/                 # PDB files  
│   ├── mtz/                 # Reflection MTZ files
│   └── cif_sf/              # Structure factor CIF files
├── unit/                    # Unit tests (fast, no I/O)
│   ├── conftest.py          # Unit test fixtures (mock data)
│   ├── math_functions/      # Math module tests
│   ├── model/               # Model module tests
│   ├── refinement/          # Refinement module tests
│   ├── scaling/             # Scaling module tests
│   ├── symmetrie/           # Symmetry module tests
│   ├── io/                  # I/O module tests
│   ├── restraints/          # Restraints module tests
│   └── utils/               # Utils module tests
├── integration/             # Integration tests (real I/O)
│   ├── conftest.py          # Integration fixtures
│   ├── test_io_cif.py       # CIF loading tests
│   ├── test_io_reflections.py # Reflection data tests
│   └── test_refinement_pipeline.py # Pipeline tests
└── scripts/                 # Test runner scripts
    ├── submit_tests.sbatch  # SLURM job for CPU tests
    ├── submit_gpu_tests.sbatch # SLURM job for GPU tests
    ├── run_tests.sh         # Interactive test runner
    └── logs/                # Test output logs
```

## Running Tests

### Quick Local Run (on login node, for small tests only)

```bash
# Run all unit tests
python -m pytest tests/unit -v

# Run specific test file
python -m pytest tests/unit/math_functions/test_math_torch.py -v

# Run tests matching a pattern
python -m pytest tests/unit -k "test_coordinate" -v
```

### Interactive Run (on compute node)

```bash
# Start an interactive session
srun -c 8 -p day -t 1-00:00:00 --pty bash

# Then run tests
./tests/scripts/run_tests.sh unit
./tests/scripts/run_tests.sh integration
./tests/scripts/run_tests.sh --cov  # With coverage
```

Or use the convenience script directly:
```bash
srun -c 8 -p day -t 1-00:00:00 tests/scripts/run_tests.sh unit
```

### Detached Run (via SLURM)

```bash
# Submit CPU tests
sbatch tests/scripts/submit_tests.sbatch unit
sbatch tests/scripts/submit_tests.sbatch integration
sbatch tests/scripts/submit_tests.sbatch all

# Submit GPU tests (requires GPU partition)
sbatch tests/scripts/submit_gpu_tests.sbatch

# Check job status
squeue -u $USER

# View logs
tail -f tests/scripts/logs/test_<job_id>.out
```

## Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.unit` - Fast unit tests, no I/O
- `@pytest.mark.integration` - Integration tests with file I/O
- `@pytest.mark.gpu` - Tests requiring GPU (not run by default)
- `@pytest.mark.slow` - Slow tests (>30 seconds)

### Running by marker

```bash
# Unit tests only (fast)
pytest -m "unit"

# Skip GPU tests
pytest -m "not gpu"

# Integration tests only
pytest -m "integration"

# Fast tests only (no slow, no gpu)
pytest -m "not slow and not gpu"
```

## Coverage

Generate coverage report:

```bash
# Terminal report
pytest tests/unit --cov=torchref --cov-report=term-missing

# HTML report
pytest tests/unit --cov=torchref --cov-report=html

# Open HTML report
open htmlcov/index.html
```

## GitHub Actions

Tests are automatically run on GitHub via Actions:
- Unit tests run on Python 3.9, 3.10, 3.11
- Coverage is uploaded to Codecov
- Integration tests run after unit tests pass

See `.github/workflows/tests.yml` for configuration.

## Adding New Tests

1. **Unit tests**: Add to `tests/unit/<module>/test_<feature>.py`
   - Use mock data from fixtures
   - No file I/O operations
   - Mark with `@pytest.mark.unit`

2. **Integration tests**: Add to `tests/integration/test_<feature>.py`
   - Can use real files from `tests/files/`
   - Mark with `@pytest.mark.integration`

3. **GPU tests**: Add `@pytest.mark.gpu` marker
   - Will be skipped if no GPU available

## Test Data

Test data files are in `tests/files/`. Available structures:
- 1DAW, 2DQ6, 3A5V, 3E98, 3GR5, 3K7M, 3VRJ, 4BX9, 5BOV, 6G9X

Each structure has:
- `cif/` - Model coordinates
- `pdb/` - PDB format (if available)
- `mtz/` - Reflection data
- `cif_sf/` - Structure factor CIF (if available)
