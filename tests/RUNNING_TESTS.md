# Running Tests Guide

This guide explains how to run different test classes and subsets of tests.

**Current Test Status:**
- **217 total tests** (140 unit + 77 integration)
- **26% code coverage** 
- All tests passing (13 skipped - GPU/slow markers)

---

## Quick Reference

```bash
# Activate environment first
module load anaconda
conda activate /das/work/p17/p17490/CONDA/torchref
cd /das/work/p17/p17490/Peter/Library/torchref
```

---

## Running by Test Type (Markers)

### Unit Tests Only
```bash
# All unit tests (fast, no I/O)
pytest tests/unit -m "unit" -v

# Unit tests excluding GPU tests
pytest tests/unit -m "unit and not gpu" -v
```

### Integration Tests Only
```bash
# All integration tests (with file I/O)
pytest tests/integration -m "integration" -v

# Integration tests excluding GPU
pytest tests/integration -m "integration and not gpu" -v
```

### All Tests (Unit + Integration)
```bash
# Everything except GPU tests
pytest tests/ -m "not gpu" -v

# Fast tests only (no slow, no gpu)
pytest tests/ -m "not slow and not gpu" -v
```

### GPU Tests Only
```bash
# Requires GPU node
pytest tests/ -m "gpu" -v
```

---

## Running by Module

### Math Functions
```bash
# All math tests
pytest tests/unit/math_functions/ -v

# PyTorch math only
pytest tests/unit/math_functions/test_math_torch.py -v

# NumPy math only
pytest tests/unit/math_functions/test_math_numpy.py -v
```

### Model
```bash
# All model tests
pytest tests/unit/model/ -v

# Model class only
pytest tests/unit/model/test_model.py -v

# Parameter wrappers (MixedTensor, etc.)
pytest tests/unit/model/test_parameter_wrappers.py -v
```

### Refinement
```bash
# All refinement tests
pytest tests/unit/refinement/ -v

# Loss weighting schemes
pytest tests/unit/refinement/test_loss_weighting.py -v

# Target/loss functions
pytest tests/unit/refinement/test_targets.py -v
```

### Scaling
```bash
pytest tests/unit/scaling/ -v
pytest tests/unit/scaling/test_scaler.py -v
```

### Symmetry
```bash
pytest tests/unit/symmetrie/ -v
pytest tests/unit/symmetrie/test_symmetrie.py -v
```

### I/O (Input/Output)
```bash
# Unit tests (mock data)
pytest tests/unit/io/ -v

# Integration tests (real files)
pytest tests/integration/test_io_cif.py -v
pytest tests/integration/test_io_reflections.py -v
```

### Restraints
```bash
pytest tests/unit/restraints/ -v
pytest tests/unit/restraints/test_restraints.py -v
```

### Utils
```bash
pytest tests/unit/utils/ -v
pytest tests/unit/utils/test_gradnorm.py -v
pytest tests/unit/utils/test_utils.py -v
```

---

## Running by Test Class

### Syntax
```bash
pytest path/to/test_file.py::TestClassName -v
```

### Examples

#### Math Functions Classes
```bash
# Coordinate transformations (PyTorch)
pytest tests/unit/math_functions/test_math_torch.py::TestCoordinateTransformations -v

# R-factor calculations
pytest tests/unit/math_functions/test_math_torch.py::TestRFactorCalculations -v

# Grid functions
pytest tests/unit/math_functions/test_math_torch.py::TestGridFunctions -v

# Outlier detection
pytest tests/unit/math_functions/test_math_torch.py::TestOutlierDetection -v

# Rotation functions
pytest tests/unit/math_functions/test_math_torch.py::TestRotation -v

# Alignment
pytest tests/unit/math_functions/test_math_torch.py::TestAlignment -v
```

#### Model Classes
```bash
# Model initialization
pytest tests/unit/model/test_model.py::TestModelInitialization -v

# Model device handling
pytest tests/unit/model/test_model.py::TestModelDeviceHandling -v

# MixedTensor initialization
pytest tests/unit/model/test_parameter_wrappers.py::TestMixedTensorInitialization -v

# MixedTensor operations
pytest tests/unit/model/test_parameter_wrappers.py::TestMixedTensorOperations -v
```

#### Refinement Classes
```bash
# Fixed weighting
pytest tests/unit/refinement/test_loss_weighting.py::TestFixedWeighting -v

# Resolution-dependent weighting
pytest tests/unit/refinement/test_loss_weighting.py::TestResolutionDependentWeighting -v

# Gaussian NLL loss
pytest tests/unit/refinement/test_targets.py::TestGaussianNLL -v

# Least squares target
pytest tests/unit/refinement/test_targets.py::TestLeastSquaresTarget -v
```

#### Symmetry Classes
```bash
# Symmetry initialization
pytest tests/unit/symmetrie/test_symmetrie.py::TestSymmetryInitialization -v

# Symmetry matrices
pytest tests/unit/symmetrie/test_symmetrie.py::TestSymmetryMatrices -v

# Symmetry application
pytest tests/unit/symmetrie/test_symmetrie.py::TestSymmetryApplication -v

# Space group mapping
pytest tests/unit/symmetrie/test_symmetrie.py::TestSpaceGroupMapping -v
```

#### Scaling Classes
```bash
# Scaler initialization
pytest tests/unit/scaling/test_scaler.py::TestScalerInitialization -v

# Scaling calculations
pytest tests/unit/scaling/test_scaler.py::TestScalingCalculations -v

# B-factor scaling
pytest tests/unit/scaling/test_scaler.py::TestBFactorScaling -v

# Anisotropic scaling
pytest tests/unit/scaling/test_scaler.py::TestAnisotropicScaling -v
```

#### I/O Classes
```bash
# ReflectionData initialization
pytest tests/unit/io/test_data.py::TestReflectionDataInitialization -v

# ReflectionData device handling
pytest tests/unit/io/test_data.py::TestReflectionDataDeviceMovement -v

# CIF loading (integration)
pytest tests/integration/test_io_cif.py::TestCIFLoading -v

# MTZ loading (integration)
pytest tests/integration/test_io_reflections.py::TestMTZLoading -v
```

#### Restraints Classes
```bash
# Restraints initialization
pytest tests/unit/restraints/test_restraints.py::TestRestraintsInitialization -v

# Bond restraint calculations
pytest tests/unit/restraints/test_restraints.py::TestBondRestraintCalculations -v

# Angle restraint calculations
pytest tests/unit/restraints/test_restraints.py::TestAngleRestraintCalculations -v

# Torsion restraint calculations
pytest tests/unit/restraints/test_restraints.py::TestTorsionRestraintCalculations -v
```

---

## Running Individual Tests

### Syntax
```bash
pytest path/to/test_file.py::TestClassName::test_function_name -v
```

### Examples
```bash
# Single test
pytest tests/unit/math_functions/test_math_torch.py::TestRFactorCalculations::test_rfactor_identical -v

# Pattern matching with -k
pytest tests/unit -k "test_rfactor" -v
pytest tests/unit -k "coordinate" -v
pytest tests/unit -k "initialization" -v
```

---

## With Coverage

```bash
# Coverage for specific module
pytest tests/unit/math_functions/ --cov=torchref.math_functions --cov-report=term-missing -v

# Coverage for all unit tests
pytest tests/unit --cov=torchref --cov-report=term-missing -v

# HTML coverage report
pytest tests/unit --cov=torchref --cov-report=html
# Then open htmlcov/index.html
```

---

## On Compute Nodes (SLURM)

### Interactive Session
```bash
# Start interactive session
srun -c 8 -p day -t 1-00:00:00 --pty bash

# Then run any pytest command above
pytest tests/unit -v
```

### Batch Job
```bash
# Submit unit tests
sbatch tests/scripts/submit_tests.sbatch unit

# Submit integration tests
sbatch tests/scripts/submit_tests.sbatch integration

# Submit all tests
sbatch tests/scripts/submit_tests.sbatch all

# Check status
squeue -u $USER

# View output
tail -f tests/scripts/logs/test_<jobid>.out
```

---

## Useful Options

| Option | Description |
|--------|-------------|
| `-v` | Verbose output |
| `-vv` | More verbose |
| `-q` | Quiet mode |
| `-x` | Stop on first failure |
| `--tb=short` | Short traceback |
| `--tb=long` | Full traceback |
| `-k "pattern"` | Run tests matching pattern |
| `--collect-only` | List tests without running |
| `-n auto` | Parallel execution (needs pytest-xdist) |
| `--lf` | Run last failed tests |
| `--ff` | Run failed tests first |

### Examples
```bash
# Stop on first failure with short traceback
pytest tests/unit -x --tb=short -v

# Run tests matching "coordinate" or "transform"
pytest tests/unit -k "coordinate or transform" -v

# List all tests without running
pytest tests/unit --collect-only

# Run only previously failed tests
pytest tests/unit --lf -v
```

---

## Available Test Classes Summary

### Unit Tests (`tests/unit/`)

| File | Classes |
|------|---------|
| `math_functions/test_math_torch.py` | `TestCoordinateTransformations`, `TestRFactorCalculations`, `TestOutlierDetection`, `TestGridFunctions`, `TestTransformationMatrices`, `TestAlignment`, `TestSmallestDiff`, `TestRotation` |
| `math_functions/test_math_numpy.py` | `TestCoordinateTransformations`, `TestScatteringVectors`, `TestRFactorCalculations`, `TestRotation`, `TestOutlierDetection` |
| `model/test_model.py` | `TestModelInitialization`, `TestModelDeviceHandling` |
| `model/test_parameter_wrappers.py` | `TestMixedTensorInitialization`, `TestMixedTensorOperations`, `TestMixedTensorDeviceHandling`, `TestOccupancyTensor`, `TestPositiveMixedTensor` |
| `refinement/test_loss_weighting.py` | `TestFixedWeighting`, `TestResolutionDependentWeighting`, `TestLossWeightingModule` |
| `refinement/test_targets.py` | `TestTargetBase`, `TestGaussianNLL`, `TestLeastSquaresTarget`, `TestRiceNLL`, `TestTargetDeviceHandling`, `TestNumericStability` |
| `scaling/test_scaler.py` | `TestScalerInitialization`, `TestScalerDeviceHandling`, `TestScalingCalculations`, `TestBFactorScaling`, `TestAnisotropicScaling` |
| `symmetrie/test_symmetrie.py` | `TestSymmetryInitialization`, `TestSymmetryMatrices`, `TestSymmetryApplication`, `TestSymmetryDeviceHandling`, `TestSpaceGroupMapping` |
| `io/test_data.py` | `TestReflectionDataInitialization`, `TestReflectionDataDeviceMovement`, `TestReflectionDataAttributes`, `TestReflectionDataProperties`, `TestMockReflectionData` |
| `restraints/test_restraints.py` | `TestRestraintsInitialization`, `TestBondRestraintCalculations`, `TestAngleRestraintCalculations`, `TestTorsionRestraintCalculations`, `TestRestraintDeviceHandling`, `TestRestraintNumericStability` |
| `utils/test_gradnorm.py` | `TestGradNorm` |
| `utils/test_utils.py` | `TestModuleReference`, `TestCIFReader` |

### Integration Tests (`tests/integration/`)

| File | Classes |
|------|---------|
| `test_io_cif.py` | `TestCIFLoading`, `TestMultipleCIFFiles`, `TestCIFSaving` |
| `test_io_reflections.py` | `TestMTZLoading`, `TestSFCIFLoading`, `TestReflectionDataProperties`, `TestMatchingDataPairs` |
| `test_refinement_pipeline.py` | `TestRefinementSetup`, `TestStructureFactorCalculation`, `TestGradientFlow`, `TestDeviceConsistency` |
