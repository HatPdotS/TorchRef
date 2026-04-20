Target Functions
================

Target functions (loss functions) drive the refinement optimization. TorchRef 
provides several built-in targets and makes it easy to create custom ones.

Targets reference the refinement object to access model parameters, 
reflection data, and restraints.

Targets are registered in a :class:`~torchref.refinement.loss_state.LossState` object, which manages

- Target instances
- Weights for each target
- Metadata for refinement monitoring

Targets can be grouped into composite targets for geometry and ADP restraints or custom combinations.

Built-in Targets
----------------

X-ray Targets
^^^^^^^^^^^^^

- **Least Squares**: Traditional least squares refinement
- **Maximum Likelihood**: ML-based refinement with proper error model. This is the default.
- **Gaussian NLL**: Gaussian negative log-likelihood

Geometry Targets
^^^^^^^^^^^^^^^^

- **Bond restraints**: Penalize deviations from ideal bond lengths
- **Angle restraints**: Penalize deviations from ideal bond angles
- **Torsion restraints**: Penalize deviations from ideal torsion angles
- **Planarity restraints**: Penalize out-of-plane deviations
- **Chirality restraints**: Maintain correct stereochemistry
- **VDW restraints**: Prevent atomic overlaps

ADP Targets
^^^^^^^^^^^

- **SIMU**: Similar ADPs for nearby atoms
- **locality restraint**: Smooth ADP variations among neighbors
- **DELU**: Rigid bond restraints
- **Population restraint**: Restrain ADPs towards a target distribution


Stats functionality
^^^^^^^^^^^^^^^^^^^^^

Each target implements a `stats()` method returning a dictionary of metrics.
These metrics are collected during refinement for monitoring and logging.
They supply useful information about the model.

.. code-block:: python

   # Collect stats from each target
   xray_stats = xray_target.stats()
   geom_stats = geom_target.stats()
   adp_stats = adp_target.stats()

   # Example: print R-factors from X-ray target
   print(f"R-work: {xray_stats['rwork']}, R-free: {xray_stats['rfree']}")

Using Targets
-------------

.. code-block:: python

   from torchref.refinement.targets import (
       create_xray_target,
       TotalGeometryTarget,
       TotalADPTarget
   )

   # Create targets
   xray_target = create_xray_target(data, model, target_type='ml')
   geom_target = TotalGeometryTarget(model)
   adp_target = TotalADPTarget(model)

   # Compute losses
   xray_loss = xray_target()
   geom_loss = geom_target()
   adp_loss = adp_target()

Custom Targets
--------------

Create custom targets by subclassing :class:`~torchref.refinement.targets.Target`:

.. code-block:: python

   from torchref.refinement.targets import Target
   import torch

   class EntropyRegularization(Target):
       """Entropy regularization for B-factors."""
       
       def __init__(self, model, weight=0.01):
           super().__init__(model) 
       
       def forward(self):
           b_factors = self.model.b()
           # Entropy-based regularization
           entropy = -torch.sum(b_factors * torch.log(b_factors + 1e-8))
           return entropy

The key advantage: **you only define the forward pass**. PyTorch's autograd 
automatically computes all necessary gradients for optimization.

LossState
---------

The :class:`~torchref.refinement.loss_state.LossState` class manages multiple targets and their weights.
It is used to track targets during refinement and supplies information for weight calculation.

.. code-block:: python

    from torchref.refinement import LossState

    state = LossState(device=self.device)

    # Register targets
    state.register_target('xray', xray_target)
    state.register_target('geometry', geom_target)
    state.register_target('adp', adp_target)

    # Set weights
    state.set_weight('xray', 1.0)
    state.set_weight('geometry', 0.1)
    state.set_weight('adp', 0.1)

    # Compute total loss
    total_loss = state.aggregate()

    # Compute backward pass
    total_loss.backward()


    # Example optimization step with LBFGS using the LossState
    # This prunes loss evaluation and keeps track of what things need to be recomputed automatically
    from torch.optim import LBFGS
    optimizer = LBFGS(model.parameters(), lr=1.0, max_iter=100)
    state.run(optimizer, n_steps=1) # equivalent to state.step(optimizer)
