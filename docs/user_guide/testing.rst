Testing
=======

Please write tests for any new functionality. This page covers running the
suite, the markers that decide what actually executes, and the fixtures
available to a new test.

Test Categories
---------------

- ``tests/unit/`` — fast, isolated, mock data, no file I/O. Mirrors the package
  layout (``model/``, ``io/``, ``refinement/``, ``scaling/``, ``symmetry/``,
  ``restraints/``, ``structure_factor/``, ``scattering/``, ``maps/``, ``base/``,
  ``utils/``, ``experimental/``).
- ``tests/integration/`` — real file I/O and complete pipelines.
- ``tests/functional/`` — multi-component workflows on pre-loaded objects.
- ``tests/files/`` — test data: ``cif/``, ``pdb/``, ``mtz/``, ``cif_sf/``.

Test Data
~~~~~~~~~

Ten structures carry all four file types, spanning triclinic to cubic:

======  ==========  ============
PDB ID  d_min (Å)   Space group
======  ==========  ============
5BOV    1.60        P 1
2DQ6    1.50        P 31 2 1
3E98    2.43        P 1 21 1
3VRJ    1.90        P 1 21 1
1DAW    2.20        C 1 2 1
6G9X    2.30        P 21 21 2
3GR5    2.05        P 65 2 2
4BX9    2.60        P 43 21 2
3A5V    2.00        I 4 2 2
3K7M    1.95        P 4 3 2
======  ==========  ============

``tests/files/`` also holds partial sets — ``1AK5_with_H.pdb`` + ``1AK5.mtz``
(no CIF), ``7L84.pdb`` + ``7L84-sf.cif`` (no MTZ), ``test_ihm_ensemble.cif`` —
so a test that globs one directory and assumes a matching file in another will
fail on those. Use ``sample_structure_pair`` / ``all_test_structures``.

Running Tests
-------------

.. code-block:: bash

    pytest -m "not gpu and not slow" -v     # recommended during development
    pytest tests/unit -v                    # by category
    pytest tests/unit/model/ -v             # by module
    pytest tests/unit/model/test_model.py::TestModelInitialization -v

Coverage:

.. code-block:: bash

    pytest tests/unit --cov=torchref --cov-report=term-missing -v

Markers
~~~~~~~

- ``@pytest.mark.unit`` — fast tests with mock data
- ``@pytest.mark.integration`` — tests requiring file I/O
- ``@pytest.mark.gpu`` — needs any accelerator (CUDA or MPS)
- ``@pytest.mark.cuda`` — needs CUDA specifically (e.g. Triton kernels);
  ``cuda_only`` is the retired spelling and still works
- ``@pytest.mark.mps`` — needs MPS specifically (Metal kernels)
- ``@pytest.mark.slow`` — long-running
- ``@pytest.mark.openmm`` — needs OpenMM (the ``[amber]`` extra)
- ``@pytest.mark.amber`` — needs OpenMM **and** AmberTools (``antechamber`` /
  ``tleap`` on ``PATH``)

Skipped by Default
~~~~~~~~~~~~~~~~~~

``slow`` tests need ``--run-slow``. Accelerator tests are **not** opt-in:
``tests/conftest.py`` runs a ``cuda`` / ``mps`` / ``gpu`` test whenever the host
has that backend and skips it when it does not, so no flag can make one run
without the hardware (``--run-gpu`` is a deprecated no-op that only warns). Use
``--run-cuda`` / ``--run-mps`` in CI to *fail* rather than skip when the backend
is missing. ``openmm`` / ``amber`` tests self-skip when their dependencies are
absent. A green suite is therefore not evidence that the GPU or Amber paths were
exercised — check the skip count.

.. code-block:: bash

    pytest --run-slow -v                # include the slow tests
    pytest --run-cuda -v                # CI: require CUDA, error if absent
    pytest -m "amber" -v                # Amber target tests

The Amber stack, if you want it:

.. code-block:: bash

    pip install -e ".[amber]"                  # OpenMM + Amber helpers
    conda install -c conda-forge ambertools     # antechamber / tleap

Fixtures
--------

Almost everything lives in the root ``tests/conftest.py`` and is therefore
available from every category — the ``integration/`` and ``functional/``
conftests are docstrings only. Mock data is the exception:
``tests/unit/conftest.py``. Read those two files for the authoritative list; the
ones you will reach for most:

- Paths (session-scoped): ``tests_root``, ``project_root``, ``test_files_dir``,
  the per-format ``cif_dir``, ``mtz_dir``, ``pdb_dir``, ``cif_sf_dir``, and
  ``external_monomer_library``.
- Devices: ``cpu_device``, ``gpu_device``, ``cuda_device``, ``mps_device``, and
  ``any_device`` / ``device`` which parametrise a test across what is present.
- Tolerances: ``rtol`` (1e-5) and ``atol`` (1e-8). Use these instead of
  hand-picked numbers so a dtype change moves every test at once.
- Mock data: ``random_coordinates``, ``random_fractional_coordinates``,
  ``random_adp``, ``random_occupancies``, ``mock_hkl_indices``,
  ``mock_structure_factors``, ``mock_F_obs``, ``mock_F_sigma``,
  ``mock_aniso_u``, ``mock_scattering_factors``, ``mock_weights``.
- Real files: ``sample_cif_file``, ``sample_pdb_file``, ``sample_mtz_file``,
  ``sample_structure_factor_cif``, ``sample_structure_pair`` (matched model +
  data), ``all_structure_pairs``, ``all_test_structures``.
- Loaded objects: ``loaded_model``, ``loaded_reflection_data``,
  ``model_and_data``, ``initialized_scaler``.

The mock-data fixtures yield a *factory* taking ``n_atoms`` / ``n_reflections``
and ``seed``; ``mock_cell`` and ``mock_cell_triclinic`` yield the tensor
directly. Mixing the two conventions up is the usual first failure:

.. code-block:: python

    def test_with_mock_data(random_coordinates, mock_cell, rtol, atol):
        coords = random_coordinates(n_atoms=100)  # factory: call it
        cell = mock_cell                          # tensor: do NOT call it
        torch.testing.assert_close(f(coords, cell), expected, rtol=rtol, atol=atol)

Writing Tests
-------------

Mark the test, take fixtures rather than duplicating setup, and keep unit tests
free of file I/O:

.. code-block:: python

    import pytest
    import torch

    @pytest.mark.unit
    class TestMyFunction:
        def test_basic_functionality(self, mock_cell, random_coordinates):
            result = my_function(random_coordinates(n_atoms=10), mock_cell)
            assert result.shape == (10, 3)

        def test_edge_case_empty(self):
            result = my_function(torch.empty(0, 3), [50, 60, 70, 90, 90, 90])
            assert result.shape[0] == 0

    @pytest.mark.integration
    class TestModelLoading:
        def test_load_cif(self, sample_cif_file):
            model = Model()
            model.load_cif(str(sample_cif_file))
            assert model.initialized
            assert len(model.pdb) > 0

Cover the edge cases that actually bite here: empty selections, degenerate
geometry (which is where gradients go NaN), and non-default dtype/device.

CI
--

``tox.ini`` defines the environments: ``py310``–``py313`` against current
dependencies, plus boundary environments pinning NumPy, Numba, PyTorch, Pandas
and Gemmi versions and a ``lowerbounds`` pair at the declared minimums. Refer to
the file itself for the authoritative list.

.. code-block:: bash

    tox                     # all environments
    tox -e py311-latest     # one
    tox -p auto             # parallel

Troubleshooting
---------------

**Tests not found** — run from the repository root, where
``[tool.pytest.ini_options]`` in ``pyproject.toml`` sets ``testpaths``, or from
inside ``tests/``, where ``tests/pytest.ini`` takes over. The two are kept in
step deliberately.

**GPU tests skipped** — the host has no CUDA/MPS device. The marks are gated on
real availability, not on a flag, and a ``cuda``-marked test also skips on an
MPS-only host. ``--run-cuda`` / ``--run-mps`` turn the skip into the real
backend error.

**Import errors** — install the package (``pip install -e .``) rather than
relying on the working directory.
