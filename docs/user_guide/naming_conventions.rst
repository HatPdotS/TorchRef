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

   hkl, F, F_sigma, rfree = dataset()           # legacy accessor (deprecated)

The property and the call are not interchangeable. ``dataset.F`` is a plain
tensor of everything as read; ``dataset()`` returns ``F`` and ``F_sigma`` as
``torch.masked.MaskedTensor``, scaled, with invalid reflections marked rather
than removed — so aggregations skip them but indices still line up with ``hkl``.
Pass ``mask=False`` / ``scale=False`` to opt out.

Two traps on the call. The masked ``F`` / ``F_sigma`` are **detached clones**, so
no gradient flows through them; and the call itself is deprecated (it emits a
``DeprecationWarning``). Prefer the subset accessor — ``dataset.work.F``,
``dataset.free.F``, ``.sigF`` / ``.hkl`` / ``.select(...)`` — or
``dataset.get_corrected_data()`` for the full scaled ``(F, F_sigma)`` with the
graph intact.

Atomic Displacement Parameters
------------------------------

- ``adp`` — isotropic model ADPs (B-factors, Ų): ``model.adp()``
- ``u`` — anisotropic U tensor, 6 components per atom: ``model.u()``
- ``b`` — a B-factor used for *scaling*, not a model parameter (e.g. the
  scaler's ``b_solvent``)

Coordinates and Occupancy
-------------------------

- ``xyz`` — Cartesian, Ångströms: ``model.xyz()``
- ``xyz_fractional`` — fractional, 0–1 within the cell:
  ``model.xyz_fractional()``
- occupancies: ``model.occupancy()``

Note that ``freeze()`` / ``unfreeze()`` take the *parameter-type* names —
``'xyz'``, ``'adp'``, ``'u'``, ``'occupancy'`` — and silently ignore anything
else, so an abbreviation like ``'b'`` or ``'occ'`` is a no-op rather than an
error.

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
