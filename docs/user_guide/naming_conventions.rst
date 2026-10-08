Naming Conventions
==================

Standardized variable names used throughout TorchRef. The crystallographic ones
matter more than usual here, because ``F`` and ``f`` mean genuinely different
things and mixing them up produces a plausible-looking wrong answer rather than
an error.

General Principles
------------------

- **snake_case** for variables, functions, and methods
- **CamelCase** for class names only
- ``spacegroup`` as one word — not ``space_group``

Structure Factors
-----------------

Case distinguishes complex from amplitude:

- ``f_calc``, ``f_obs`` — **complex** structure factors, with phase
- ``F_calc``, ``F_obs``, ``F`` — **amplitudes**, absolute values

.. code-block:: python

   f_calc = model(hkl)              # complex, with phase
   F_calc = torch.abs(f_calc)       # amplitude

   F_obs = dataset.F                # amplitudes
   F_sigma = dataset.F_sigma        # uncertainty on F
   I, I_sigma = dataset.I, dataset.I_sigma      # intensities, if present

These are plain, full-length tensors aligned with ``dataset.hkl`` and
``dataset.rfree_flags``, masked-out reflections included. For split data use the
subset views ``dataset.work``, ``dataset.free`` and ``dataset.validation``
(``.F``, ``.sigF``, ``.hkl``, ``.select(t)``), which also drop masked-out
reflections. A ``ScaledDataset`` in a ``DatasetCollection`` exposes its live
scale correction through the same attributes.

Atomic Displacement Parameters
------------------------------

- ``adp`` — isotropic model ADPs (B-factors, Ų): ``model.adp()``
- ``u`` — anisotropic U tensor, 6 components per atom: ``model.u()``
- ``b`` — a B-factor reported for *scaling*, not a model parameter (e.g.
  ``SolventModel.b_solvent_equivalent()``, the B fitted to the solvent falloff)

Coordinates and Occupancy
-------------------------

- ``xyz`` — Cartesian, Ångströms: ``model.xyz()``
- ``xyz_fractional`` — fractional, 0–1 within the cell:
  ``model.xyz_fractional()``
- occupancies: ``model.occupancy()``

Note that ``freeze()`` / ``unfreeze()`` take the *parameter-type* names —
``'xyz'``, ``'adp'``, ``'u'``, ``'occupancy'`` — and raise ``ValueError`` on
anything else, such as the abbreviation ``'b'`` or ``'occ'``.

Unit Cell
---------

- ``cell`` — a :class:`~torchref.symmetry.cell.Cell` object, ``model.cell``
- ``cell_params`` — the raw ``[a, b, c, alpha, beta, gamma]`` tensor

Uncertainties
-------------

``{quantity}_sigma``: ``F_sigma``, ``I_sigma``.

Resolution
----------

- ``d_min`` — high-resolution limit, Å
- ``d_max`` — low-resolution limit, Å
- ``dataset.resolution`` — per-reflection resolution
