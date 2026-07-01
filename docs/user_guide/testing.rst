Testing
=======

I know its annoying but very much necessary. Please write tests for any new functionality you would like to add.
This guide covers the torchref testing framework, including how to run tests,
write new tests, and understand the test organization.

Overview
--------

The testing framework is organized into four categories:

- **Unit tests**: Fast, isolated tests with mock data (no file I/O)
- **Integration tests**: Tests with real file I/O and complete pipelines
- **Functional tests**: Complex workflows combining multiple components
- **Manual tests**: Exploratory and diagnostic tests for development

Directory Structure
-------------------

.. code-block:: text

    torchref/
    ├── tests/                          # Automated test suite
    │   ├── conftest.py                 # Root fixtures and configuration
    │   ├── pytest.ini                  # Pytest configuration
    │   │
    │   ├── unit/                       # Unit tests
    │   │   ├── conftest.py             # Mock data generators
    │   │   ├── base/                   # Base math functions
    │   │   ├── model/                  # Model module
    │   │   ├── io/                     # I/O module
    │   │   ├── refinement/             # Refinement/targets
    │   │   ├── scaling/                # Scaling module
    │   │   ├── symmetry/               # Symmetry module
    │   │   └── restraints/             # Restraints module
    │   │
    │   ├── integration/                # Integration tests
    │   │   ├── conftest.py             # Real file fixtures
    │   │   └── test_*.py               # Integration test files
    │   │
    │   ├── functional/                 # Functional tests
    │   │   ├── conftest.py             # Loaded object fixtures
    │   │   └── test_*.py               # Functional test files
    │   │
    │   └── files/                      # Test data files
    │       ├── cif/                    # Model CIF files
    │       ├── pdb/                    # PDB format files
    │       ├── mtz/                    # Reflection data files
    │       └── cif_sf/                 # Structure factor CIF files
    │
    ├── tests_manual/                   # Manual/exploratory tests
    │   ├── alignment/                  # Alignment algorithm tests
    │   ├── bulk_solvent/               # Bulk solvent model tests
    │   ├── refinement/                 # Refinement algorithm tests
    │   └── ...                         # Other exploratory tests
    │
    └── tox.ini                         # CI/CD configuration

Running Tests
-------------

Quick Start
~~~~~~~~~~~

Run all fast tests (recommended for development):

.. code-block:: bash

    cd tests
    pytest -m "not gpu and not slow" -v

Run by test type:

.. code-block:: bash

    pytest tests/unit -v                # Unit tests only
    pytest tests/integration -v         # Integration tests only
    pytest tests/functional -v          # Functional tests only

Run by module:

.. code-block:: bash

    pytest tests/unit/model/ -v         # Model tests
    pytest tests/unit/refinement/ -v    # Refinement tests
    pytest tests/unit/symmetry/ -v      # Symmetry tests

Run specific test class or function:

.. code-block:: bash

    # Run a specific test class
    pytest tests/unit/model/test_model.py::TestModelInitialization -v

    # Run a specific test function
    pytest tests/unit/model/test_model.py::TestModelInitialization::test_model_empty_initialization -v

Test Markers
~~~~~~~~~~~~

Tests are organized with pytest markers:

- ``@pytest.mark.unit``: Fast tests with mock data
- ``@pytest.mark.integration``: Tests requiring file I/O
- ``@pytest.mark.gpu``: Tests requiring a GPU (CUDA or MPS); skipped by default
- ``@pytest.mark.cuda_only``: Tests that specifically require CUDA (e.g. Triton kernels)
- ``@pytest.mark.slow``: Long-running tests
- ``@pytest.mark.openmm``: Tests needing OpenMM (the ``[amber]`` extra); auto-skipped if absent
- ``@pytest.mark.amber``: Tests needing OpenMM **and** AmberTools (``antechamber``/``tleap`` on PATH); auto-skipped if absent

The Amber target ships with v0.6.0. To run its tests, install the optional
stack and AmberTools:

.. code-block:: bash

    pip install -e ".[amber]"           # OpenMM + Amber helpers
    conda install -c conda-forge ambertools   # antechamber / tleap

``openmm`` and ``amber`` tests are skipped automatically when their
dependencies are missing, so the default suite stays green without them.

Run tests by marker:

.. code-block:: bash

    pytest -m "unit" -v                 # Unit tests only
    pytest -m "integration" -v          # Integration tests only
    pytest -m "not gpu" -v              # Skip GPU tests
    pytest -m "not slow" -v             # Skip slow tests
    pytest -m "amber" -v                # Amber target tests (needs the stack above)

GPU and Slow Tests
~~~~~~~~~~~~~~~~~~

GPU and slow tests are skipped by default. Enable them with flags:

.. code-block:: bash

    pytest --run-gpu -v                 # Include GPU tests
    pytest --run-slow -v                # Include slow tests
    pytest --run-gpu --run-slow -v      # Include both

Coverage Reporting
~~~~~~~~~~~~~~~~~~

Generate coverage reports:

.. code-block:: bash

    pytest tests/unit --cov=torchref --cov-report=term-missing -v
    pytest tests/unit --cov=torchref --cov-report=html -v

Available Fixtures
------------------

Root Fixtures (tests/conftest.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path fixtures (session-scoped):

.. code-block:: python

    def test_something(tests_root, project_root, test_files_dir):
        """Access test directories."""
        cif_path = test_files_dir / "cif" / "1DAW.cif"

    def test_with_files(cif_dir, mtz_dir, pdb_dir, cif_sf_dir):
        """Access specific file type directories."""
        model_file = cif_dir / "3GR5.cif"
        reflection_file = mtz_dir / "3GR5.mtz"

Device fixtures:

.. code-block:: python

    def test_cpu(cpu_device):
        """Test on CPU."""
        tensor = torch.zeros(10, device=cpu_device)

    @pytest.mark.gpu
    def test_gpu(gpu_device):
        """Test on GPU (requires --run-gpu flag)."""
        tensor = torch.zeros(10, device=gpu_device)

Numerical tolerance fixtures:

.. code-block:: python

    def test_numerical(rtol, atol):
        """Use standard tolerances."""
        # rtol = 1e-5, atol = 1e-8
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

Unit Test Fixtures (tests/unit/conftest.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mock data generators (all return callables):

.. code-block:: python

    def test_with_mock_data(
        random_coordinates,
        random_fractional_coordinates,
        random_adp,
        random_occupancies,
        mock_cell,
        mock_hkl_indices,
        mock_structure_factors,
    ):
        """Generate mock crystallographic data."""
        # Coordinates
        coords = random_coordinates(n_atoms=100)           # Shape: (100, 3)
        frac_coords = random_fractional_coordinates(n_atoms=100)

        # Thermal parameters
        adp = random_adp(n_atoms=100)                      # B-factors
        occ = random_occupancies(n_atoms=100)              # Occupancies

        # Unit cell
        cell = mock_cell()                                 # [50, 60, 70, 90, 90, 90]

        # Reflection data
        hkl = mock_hkl_indices(n_reflections=1000)         # Miller indices
        sf = mock_structure_factors(n_reflections=1000)    # Complex F

Integration Test Fixtures (tests/integration/conftest.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real file fixtures (session-scoped):

.. code-block:: python

    def test_with_real_files(
        sample_cif_file,
        sample_pdb_file,
        sample_mtz_file,
        sample_structure_factor_cif,
    ):
        """Load real crystallographic files."""
        model = Model()
        model.load_cif(str(sample_cif_file))

    def test_matched_pair(sample_structure_pair):
        """Get matching model and reflection data."""
        pdb_id = sample_structure_pair["pdb_id"]
        cif_file = sample_structure_pair["cif"]
        mtz_file = sample_structure_pair["mtz"]

Functional Test Fixtures (tests/functional/conftest.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-loaded object fixtures:

.. code-block:: python

    def test_with_loaded_objects(
        loaded_model,
        loaded_reflection_data,
        model_and_data,
        initialized_scaler,
    ):
        """Work with pre-initialized objects."""
        # Model already loaded from CIF
        xyz = loaded_model.xyz()

        # Matching model and data
        model = model_and_data["model"]
        data = model_and_data["data"]

        # Scaler ready to use
        scaled = initialized_scaler.scale()

Writing Tests
-------------

Unit Test Pattern
~~~~~~~~~~~~~~~~~

Unit tests should be fast and isolated:

.. code-block:: python

    import pytest
    import torch

    @pytest.mark.unit
    class TestMyFunction:
        """Unit tests for my_function."""

        def test_basic_functionality(self, mock_cell, random_coordinates):
            """Test with mock data - no file I/O."""
            coords = random_coordinates(n_atoms=10)
            cell = mock_cell()

            result = my_function(coords, cell)

            assert result is not None
            assert result.shape == (10, 3)

        def test_edge_case_empty(self):
            """Test empty input handling."""
            result = my_function(torch.empty(0, 3), [50, 60, 70, 90, 90, 90])
            assert result.shape[0] == 0

        def test_numerical_stability(self, rtol, atol):
            """Test numerical precision."""
            result = my_function(...)
            expected = ...
            torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

Integration Test Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

Integration tests use real files:

.. code-block:: python

    import pytest
    from torchref.model import Model

    @pytest.mark.integration
    class TestModelLoading:
        """Integration tests for model loading."""

        def test_load_cif(self, sample_cif_file):
            """Test CIF file loading."""
            model = Model()
            model.load_cif(str(sample_cif_file))

            assert model.xyz() is not None
            assert model.n_atoms > 0

        def test_load_all_structures(self, all_test_structures):
            """Test loading all available structures."""
            for cif_file in all_test_structures:
                model = Model()
                model.load_cif(str(cif_file))
                assert model.initialized

Functional Test Pattern
~~~~~~~~~~~~~~~~~~~~~~~

Functional tests combine multiple components:

.. code-block:: python

    import pytest

    @pytest.mark.integration
    class TestRefinementWorkflow:
        """Functional tests for refinement workflows."""

        def test_complete_pipeline(self, model_and_data):
            """Test complete refinement pipeline."""
            model = model_and_data["model"]
            data = model_and_data["data"]

            # Initialize scaler
            scaler = Scaler(model, data)
            scaler.initialize()

            # Run refinement
            target = MLTarget(model, data, scaler)
            loss = target.compute()

            assert torch.isfinite(loss)

Test Data Files
---------------

The ``tests/files/`` directory contains 10 test structures:

============  ===========  ==========
PDB ID        Resolution   Space Group
============  ===========  ==========
1DAW          1.8 A        P 21 21 21
2DQ6          1.5 A        P 21
3A5V          2.0 A        C 2 2 21
3E98          1.9 A        P 21
3GR5          1.6 A        P 21 21 21
3K7M          1.7 A        P 1 21 1
3VRJ          2.1 A        P 21 21 21
4BX9          1.4 A        P 21
5BOV          1.8 A        C 2 2 21
6G9X          1.5 A        P 21 21 21
============  ===========  ==========

Each structure has four file types:

- ``cif/*.cif``: Model coordinates (mmCIF format)
- ``pdb/*.pdb``: Model coordinates (PDB format)
- ``mtz/*.mtz``: Reflection data (MTZ format)
- ``cif_sf/*.cif``: Structure factors (SF-CIF format)

CI/CD Configuration
-------------------

The ``tox.ini`` file defines test environments for continuous integration:

.. code-block:: ini

    # Test across Python versions (project requires Python >= 3.10)
    py310-latest    # Python 3.10
    py311-latest    # Python 3.11
    py312-latest    # Python 3.12

    # Boundary testing
    py311-lowerbounds  # Declared minimum dependency versions

Refer to the actual ``tox.ini`` in the repository for the authoritative,
up-to-date environment list.

Run tox locally:

.. code-block:: bash

    # Run all environments
    tox

    # Run specific environment
    tox -e py311-latest

    # Run with parallel execution
    tox -p auto

Best Practices
--------------

1. **Use appropriate markers**: Mark tests with ``@pytest.mark.unit``,
   ``@pytest.mark.integration``, ``@pytest.mark.gpu``, or ``@pytest.mark.slow``.

2. **Use fixtures**: Leverage existing fixtures rather than duplicating setup code.

3. **Test isolation**: Unit tests should not depend on file I/O or external state.

4. **Numerical testing**: Use ``rtol`` and ``atol`` fixtures for consistent tolerances.

5. **Descriptive names**: Use clear test names that describe what is being tested.

6. **Test edge cases**: Include tests for empty inputs, boundary conditions, and error cases.

7. **Keep tests fast**: Unit tests should complete in milliseconds. Use ``@pytest.mark.slow``
   for longer tests.

Troubleshooting
---------------

**Tests not found**

Ensure you're running from the correct directory:

.. code-block:: bash

    cd /path/to/torchref/tests
    pytest -v

**GPU tests skipped**

GPU tests require the ``--run-gpu`` flag and CUDA availability:

.. code-block:: bash

    pytest --run-gpu -v

**Import errors**

Ensure torchref is installed or the path is set:

.. code-block:: bash

    pip install -e .
    # or
    export PYTHONPATH=/path/to/torchref:$PYTHONPATH

**Fixture not found**

Check that conftest.py files are in place and the fixture scope is appropriate.
