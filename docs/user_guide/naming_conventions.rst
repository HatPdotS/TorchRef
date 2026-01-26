Naming Conventions
==================

There have been many problems with naming conventions in this project. To avoid this in the future,
this guide documents standardized variable naming conventions used throughout TorchRef.
Consistent naming improves code readability and reduces confusion when working with
crystallographic data.

General Principles
------------------

TorchRef follows these general conventions:

- **snake_case** for variables, functions, and methods
- **CamelCase** for class names only
- Domain-specific conventions for crystallographic quantities (see below)

Structure Factors
-----------------

Structure factors use case to distinguish between complex and amplitude representations:

.. code-block:: python

   # Complex structure factors (lowercase f)
   f_calc = model.forward()      # Complex F_calc with phase

   # Amplitude structure factors (uppercase F)
   F_calc = torch.abs(f_calc)    # |F_calc| amplitude

   # Getting observed data from dataset
   hkl, F_obs, F_sigma, rfree = dataset()  # Central data access method

   # Dataset attributes
   F_obs = dataset.F              # Structure factor amplitudes
   F_sigma = dataset.F_sigma      # Uncertainties on F
   I = dataset.I                  # Intensities (if available)
   I_sigma = dataset.I_sigma      # Uncertainties on I

**Summary**:

- ``f_calc``, ``f_obs`` → complex structure factors (with phase)
- ``F_calc``, ``F_obs``, ``F`` → amplitudes (absolute values)

Atomic Displacement Parameters
------------------------------

.. code-block:: python

   # Model-level: use 'adp'
   adp = model.adp()                    # Isotropic ADPs (B-factors)
   u = model.u()                        # Anisotropic U tensor (6 components)

   # Dataset-level: use 'b' for overall scaling
   b_overall = scaler.b                 # Overall B-factor for scaling

**Summary**:

- ``adp`` → model atomic displacement parameters
- ``u`` → anisotropic U tensor
- ``b`` → dataset/scaling B-factor

Coordinates
-----------

.. code-block:: python

   # Cartesian coordinates (Angstroms)
   xyz = model.xyz()                    # Real-space coordinates

   # Fractional coordinates (0-1 within unit cell)
   xyz_fractional = model.xyz_fractional()

**Summary**:

- ``xyz`` → Cartesian coordinates in Angstroms
- ``xyz_fractional`` → fractional coordinates

Unit Cell
---------

.. code-block:: python

   # Cell object
   cell = model.cell                    # Cell object with methods

   # Raw parameters tensor
   cell_params = torch.tensor([a, b, c, alpha, beta, gamma])

Uncertainties
-------------

Uncertainties follow the ``{quantity}_sigma`` pattern:

.. code-block:: python

   # From dataset
   F_sigma = dataset.F_sigma            # Uncertainty on F
   I_sigma = dataset.I_sigma            # Uncertainty on intensity

Space Group
-----------

Use ``spacegroup`` as a single word:

.. code-block:: python

   spacegroup = model.spacegroup        # Not 'space_group'

Resolution
----------

.. code-block:: python

   # Resolution bounds (Angstroms)
   d_min = 2.0                          # High-resolution limit
   d_max = 50.0                         # Low-resolution limit

   # Per-reflection resolution from dataset
   resolution = dataset.resolution      # Resolution for each reflection
