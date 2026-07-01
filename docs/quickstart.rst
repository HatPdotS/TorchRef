Quick Start
===========

This guide walks you through a basic crystallographic refinement with TorchRef.

For interactive examples, see the Jupyter notebooks in ``example_notebooks/``.

For a more detailed explanation check out these Colab notebooks:

- `Quick Start <https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/quickstart.ipynb>`_ - Getting started tutorial
- `Structure Factors <https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/structure_factors.ipynb>`_ - FFT-based F_calc
- `Targets and Weighting <https://colab.research.google.com/github/HatPdotS/TorchRef/blob/main/example_notebooks/targets_and_weighting.ipynb>`_ - Refinement targets and weighting

Basic Refinement
----------------

The simplest way to run a refinement:

.. testcode::

   from torchref import LBFGSRefinement, ROOT_TORCHREF

   # Initialize refinement with data and model
   refinement = LBFGSRefinement(
       data_file="example.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   # Check initial R-factors
   rwork, rfree = refinement.get_rfactor()
   print(f"Initial Rwork: {rwork:.3f}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Initial Rwork: 0...

Loading Data
------------

TorchRef supports multiple file formats:

**MTZ files** (reflection data):

.. testcode::

   from torchref import read_mtz, ROOT_TORCHREF

   data = read_mtz(f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz")

   # Access reflection data
   hkl, F, sigF, rfree_flags = data()
   print(f"Number of reflections: {len(F)}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Number of reflections: ...

**PDB files** (atomic models):

.. testcode::

   from torchref import read_pdb, ROOT_TORCHREF

   model = read_pdb(f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb")

   print(f"Number of atoms: {len(model.pdb)}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Number of atoms: ...

Parameter Freezing
------------------

Model parameters can be selectively frozen during refinement:

.. testcode::

   from torchref import read_pdb, ROOT_TORCHREF

   model = read_pdb(f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb")

   # Freeze/unfreeze by parameter type
   model.freeze('b')      # Freeze all B-factors
   model.unfreeze('b')    # Unfreeze B-factors
   model.freeze('xyz')    # Freeze coordinates
   model.unfreeze('xyz')  # Unfreeze coordinates

   # Freeze/unfreeze by selection (phenix-style syntax)
   model.freeze_selection("chain A and resseq 10:20")
   print(f"Refinable xyz atoms after freezing selection: {model.xyz.refinable_params.shape[0]}")

   # Unfreeze everything
   model.unfreeze_selection("all")
   print(f"Refinable xyz atoms after unfreezing all: {model.xyz.refinable_params.shape[0]}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Refinable xyz atoms after freezing selection: ...
   Refinable xyz atoms after unfreezing all: ...

Computing Structure Factors
---------------------------

Use the model to compute structure factors for given Miller indices:

.. testcode::

   import torch
   from torchref import read_mtz, read_pdb, Scaler, ROOT_TORCHREF
   from torchref.base.metrics.rfactor import get_rfactors

   # Load data and model
   data = read_mtz(f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz")
   model = read_pdb(f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb")

   # Get reflection indices
   hkl, F, sigF, rfree = data()

   # Calculate structure factors
   fcalc = model(hkl)

   # Scale to observed data
   scale = Scaler(model, data)
   scale.initialize()
   scale.refine_lbfgs()

   scaled_fcalc = scale(fcalc)
   Fcalc_abs = torch.abs(scaled_fcalc)

   # Compute R-factors
   rwork, rfree_val = get_rfactors(F, Fcalc_abs, rfree)
   print(f"Rwork: {rwork:.3f}, Rfree: {rfree_val:.3f}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Rwork: 0..., Rfree: 0...

Running Refinement
------------------

Run coordinate refinement with geometry restraints:

.. testcode::

   from torchref import LBFGSRefinement, ROOT_TORCHREF

   refinement = LBFGSRefinement(
       data_file=f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   # Perturb coordinates to simulate errors
   refinement.model.shake_coords(0.1)

   # Run xyz refinement
   refinement.refine_xyz()

   rwork, rfree = refinement.get_rfactor()
   print(f"After refinement - Rwork: {rwork:.3f}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   After refinement - Rwork: 0...

Writing Output Files
--------------------

Save refined structures and structure factors:

.. testcode::

   import os
   import tempfile
   from torchref import LBFGSRefinement, ROOT_TORCHREF

   refinement = LBFGSRefinement(
       data_file=f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   # Write to temporary directory for this example
   with tempfile.TemporaryDirectory() as tmpdir:
       pdb_path = os.path.join(tmpdir, "refined.pdb")
       mtz_path = os.path.join(tmpdir, "refined.mtz")

       # Write refined structure
       refinement.write_out_pdb(pdb_path)
       print(f"PDB written: {os.path.exists(pdb_path)}")

       # Write structure factors with map coefficients
       refinement.write_out_mtz(mtz_path)
       print(f"MTZ written: {os.path.exists(mtz_path)}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   PDB written: True
   ...
   MTZ written: True

Geometry Restraints
-------------------

Access and inspect geometry restraints:

.. testcode::

   from torchref import LBFGSRefinement, ROOT_TORCHREF

   refinement = LBFGSRefinement(
       data_file=f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   # Access restraint counts
   bonds = refinement.restraints.restraints['bond']['intra']
   angles = refinement.restraints.restraints['angle']['intra']

   print(f"Bond restraints: {bonds['indices'].shape[0]}")
   print(f"Angle restraints: {angles['indices'].shape[0]}")

   # Compute geometry losses
   losses = refinement.geometry_target.target_losses()
   print(f"Bond loss: {losses['bond'].item():.4f}")
   print(f"Angle loss: {losses['angle'].item():.4f}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Bond restraints: ...
   Angle restraints: ...
   Bond loss: ...
   Angle loss: ...

Custom Target Functions
-----------------------

Define custom refinement targets with automatic gradient computation:

.. testcode::

   import torch
   from torchref import LBFGSRefinement, ROOT_TORCHREF
   from torchref.refinement.targets import Target

   class CustomTarget(Target):
       """Custom refinement target with automatic gradient computation."""
       name = 'custom_lsq'

       def __init__(self, refinement):
           # The base Target signature is __init__(self, verbose=0, **kwargs);
           # keep a handle on the refinement object yourself.
           super().__init__()
           self.refinement = refinement

       def forward(self):
           # Define your loss - gradients computed automatically!
           F_calc = self.refinement.get_F_calc()
           F_obs = self.refinement.reflection_data.F
           loss = torch.mean((torch.abs(F_calc) - F_obs) ** 2)
           return loss

   # Create refinement and register custom target
   refinement = LBFGSRefinement(
       data_file=f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   loss_state = refinement.create_loss_state()
   custom_target = CustomTarget(refinement)
   loss_state.register_target(custom_target.name, custom_target)
   loss_state.set_weight(custom_target.name, 3.0)

   print(f"Registered targets: {len(loss_state.targets)}")
   print(f"Custom target weight: {loss_state.weights['custom_lsq']}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Registered targets: ...
   Custom target weight: 3.0

LossState Workflow
------------------

The LossState object manages targets, weights, and metadata for refinement:

.. testcode::

   from torchref import LBFGSRefinement, ROOT_TORCHREF

   refinement = LBFGSRefinement(
       data_file=f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   # Create and populate loss state
   loss_state = refinement.create_loss_state()
   refinement.add_target_info_to_state(loss_state)
   refinement.populate_state_meta(loss_state)
   refinement.update_weights(loss_state)

   # Compute total loss
   loss = loss_state.aggregate()
   print(f"Total loss: {loss.item():.2f}")

   # Gradients computed automatically
   loss.backward()
   print(f"Gradients computed for xyz: {refinement.model.xyz.refinable_params.grad is not None}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Total loss: ...
   Gradients computed for xyz: True

Saving and Loading State
------------------------

Save complete refinement state for later continuation:

.. testcode::

   import torch
   from torchref import LBFGSRefinement, ROOT_TORCHREF

   refinement = LBFGSRefinement(
       data_file=f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   # Get state dict (can be saved with torch.save)
   state = refinement.state_dict()
   print(f"State dict keys: {len(state)} entries")

.. testoutput::
   :options: +ELLIPSIS

   ...
   State dict keys: ... entries

GPU Acceleration
----------------

Move computations to GPU for faster refinement:

.. testcode::

   import torch
   from torchref import LBFGSRefinement, ROOT_TORCHREF

   refinement = LBFGSRefinement(
       data_file=f"{ROOT_TORCHREF}/example_notebooks/1DAW.mtz",
       pdb=f"{ROOT_TORCHREF}/example_notebooks/1DAW.pdb",
   )

   if torch.cuda.is_available():
       refinement.cuda()
       print(f"Model on GPU: {refinement.model.xyz.device}")
   else:
       print("CUDA not available, using CPU")

.. testoutput::
   :options: +ELLIPSIS

   ...

Command Line Interface
----------------------

For quick refinements from the command line::

   torchref.refine -m structure.pdb -sf reflections.mtz -o output_dir/

Next Steps
----------

- See ``example_notebooks/quickstart.ipynb`` for a complete tutorial
- See ``example_notebooks/structure_factors.ipynb`` for structure-factor calculation
- See ``example_notebooks/targets_and_weighting.ipynb`` for targets and weighting
