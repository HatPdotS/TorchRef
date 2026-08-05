Refinement Guide
================

Overview
--------

:class:`~torchref.refinement.base_refinement.Refinement` is the central
component. It coordinates the atomic model (coordinates, ADPs, occupancies), the
reflection data, scaling (overall scale, anisotropy, bulk solvent), geometry
restraints, and the target functions.

Basic Usage
-----------

.. code-block:: python

   from torchref import LBFGSRefinement
   import torch

   refinement = LBFGSRefinement(
       data_file="data.mtz",
       pdb="model.pdb",
       device=torch.device("cuda"),
   )

   # Alternate parameter groups per macro cycle: xyz -> ADP -> scaler
   refinement.refine(macro_cycles=5)

   # Or optimize xyz, ADP, U and occupancy jointly in one step per cycle
   refinement.refine_everything(macro_cycles=5)

If the PDB's ``CRYST1`` cell disagrees with the MTZ cell, the model cell is
synced to the data and a warning is emitted — a stale ``CRYST1`` otherwise puts
the refinement in the wrong basin rather than failing outright.

Refinement Parameters
---------------------

- **Coordinates** — ``model.xyz()``, Ångströms
- **Isotropic ADPs** — ``model.adp()``, B-factors in Ų
- **Anisotropic U** — ``model.u()``, 6 components per atom. Used automatically
  when the input model carries ``ANISOU`` records
- **Occupancies** — ``model.occupancy()``, 0–1

Anisotropic ADPs are six parameters per atom, so refining them against
low-resolution data overfits badly. Check that the resolution supports it before
handing in a model with ``ANISOU``.

Parameter Selection
-------------------

.. code-block:: python

   # By parameter type. Valid names are exactly 'xyz', 'adp', 'u', 'occupancy';
   # anything else ('b', 'occ', ...) is silently ignored rather than rejected.
   refinement.model.freeze('xyz')
   refinement.model.unfreeze('adp')

   # By selection (Phenix-style syntax)
   refinement.model.freeze_selection("chain A and resseq 10:20")
   refinement.model.unfreeze_selection("all")

Residues for which no restraints could be built are frozen in ``xyz``
automatically, so an unrecognised ligand ends up immobile rather than distorted.

Monitoring Progress
-------------------

.. code-block:: python

   r_work, r_free = refinement.get_rfactor()
   print(refinement.collect_metrics())      # every metric, as one dict

``collect_metrics`` is unfiltered — filtering by verbosity happens at display
time. So ``rwork`` / ``rfree`` / ``rfree_gap`` come back as floats, but the
nested ``geometry`` and ``adp`` entries hold
:class:`~torchref.utils.stats.StatEntry` wrappers; take ``.value`` before doing
arithmetic on those.

``get_rfactor`` evaluates under ``no_grad``, which leaves the model's shared
``F_calc`` cache holding a *detached* tensor. Scoring in the middle of an
optimization step therefore silently zeroes the gradients that follow — call
``refinement.model.reset_cache()`` afterwards if you score inside a loop rather
than between cycles.
