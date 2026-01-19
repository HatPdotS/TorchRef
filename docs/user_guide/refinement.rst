Refinement Guide
================

This guide covers the refinement framework in TorchRef.

Overview
--------

The :class:`~torchref.refinement.base_refinement.Refinement` class is the central 
component for crystallographic refinement. It coordinates:

- Atomic model (coordinates, B-factors, occupancies)
- Reflection data (observed structure factors)
- Scaling (overall scale, anisotropy, bulk solvent)
- Geometry restraints (bonds, angles, torsions, planes)
- Target functions (least squares, maximum likelihood)

Basic Usage
-----------

.. code-block:: python

   from torchref import LBFGSRefinement
   import torch

   # Initialize
   refinement = Refinement(
       data_file="data.mtz",
       pdb="model.pdb",
       device=torch.device("cuda")
   )

   # Run refinement xyz and B-factors separate
   refinement.refine(
       macro_cycles=5,
   )

   # Or refine all parameters together (recommended currently)
   refinement.refine_everything(
       macro_cycles=5,
   )

Refinement Parameters
---------------------

The refinement optimizes these parameters:

- **Coordinates** (``xyz``): Atomic positions in Ångströms
- **B-factors** (``b``): Isotropic displacement parameters in Ų
- **Anisotropic U** (``u``): Anisotropic displacement tensors (6 components)
- The model automatically detects when to use Anisotropic B-factors based the input model.
- **Occupancies**: Atomic occupancies (0-1)

Parameter Selection
-------------------

Control which parameters are refined:

.. code-block:: python
   
   refinement.model.freeze('xyz')      # Freeze coordinates options are: 'xyz', 'b', 'u', 'occ'
   
   refinement.model.unfreeze('b')      # Unfreeze B-factors

   refinement.model.freeze_selection("chain A and resseq 10:20")  # Freeze selection
   refinement.model.unfreeze_selection("all")                     # Unfreeze all
   



Monitoring Progress
-------------------

Access refinement statistics:

.. code-block:: python

   # R-factors
   r_work, rfree = refinement.get_rfactor()

   # Dictionary with all metrics.
   print(refinement.collect_metrics())

