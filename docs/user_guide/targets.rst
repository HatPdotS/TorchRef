Target Functions
================

Target functions (loss functions) drive the refinement optimization. TorchRef
ships the standard ones and makes new ones cheap to add: subclass
:class:`~torchref.refinement.targets.Target`, write ``forward()``, and autograd
supplies the derivatives.

The base ``Target`` holds no model or refinement handle of its own. Each target
stores what it needs — a model, a
:class:`~torchref.io.datasets.reflection_data.ReflectionData`, or the refinement
— on ``self`` in its ``__init__``. The base signature is
``__init__(self, verbose=0, **kwargs)``, so a subclass must keep its own handle.

Targets are registered in a
:class:`~torchref.refinement.loss_state.LossState`, which owns the target
instances, their weights, and the metadata used for monitoring, and can be
grouped into composite targets for geometry and ADP restraints.

X-ray Targets
-------------

Seven modes, selected by name. ``XRAY_TARGETS`` (in
``torchref.refinement.targets.xray._specs``) is the single table behind both
:func:`~torchref.refinement.targets.create_xray_target` and
``torchref.refine --help``, so the list below cannot drift from the CLI:

- ``ml`` — **default**. Read MLF: variance :math:`\epsilon\beta`, conditional
  mean :math:`\alpha|F_{calc}|`, with a cross-validated per-shell Luzzati
  :math:`\sigma_A`.
- ``ml_noalpha`` — as ``ml`` with the Luzzati mean coupling fixed at 1.
- ``ml_full`` — full-form MLF: marginalises the unknown error-free amplitude
  instead of inflating the variance. Roughly 4× the cost.
- ``nll_beta`` — Gaussian amplitude NLL on ``ml``'s model-error variance, i.e.
  the large-signal limit of ``ml``. Diagnostic: isolates the variance model from
  the likelihood shape.
- ``nll`` — Gaussian amplitude NLL weighted by the experimental sigma only. No
  model-error term, so it does **not** control overfitting.
- ``ls`` — least squares, unit weights; the scaler owns the overall scale.
- ``ls_wunit_k1`` — Phenix-style least squares: unit weights and a single global
  scale recomputed every gradient call, bypassing the scaler.

Geometry Targets
----------------

Bond, angle, torsion, planarity, chirality, non-bonded (VDW), and Ramachandran,
combined by :class:`~torchref.refinement.targets.TotalGeometryTarget`.
Ramachandran is off by default — give it a non-zero weight to enable it. Set any
component's weight to 0 to disable it. See :doc:`restraints` for the functional
forms.

ADP Targets
-----------

:class:`~torchref.refinement.targets.TotalADPTarget` combines three components.
``locality`` and ``KL`` work in ``log B`` (B is positive and right-skewed, so
log B is the natural scale); ``simu`` restrains the raw ΔB of bonded atoms:

- ``simu`` (:class:`~torchref.refinement.targets.ADPSimilarityTarget`) — bonded atoms should have similar B.
- ``locality`` (:class:`~torchref.refinement.targets.ADPLocalityTarget`) — K-NN spatial smoothness with
  distance-scaled sigma.
- ``KL`` (:class:`~torchref.refinement.targets.ADPEntropyTarget`) — KL divergence against a fixed-spread
  Gaussian, which controls the *spread* of the B distribution. Despite the class
  name it is not an entropy term.

:class:`~torchref.refinement.targets.RigidBondTarget` (``adp/delu``, the DELU rigid-bond restraint) exists but
is not part of ``TotalADPTarget``; register it yourself if you want it.

Statistics
----------

Every target implements ``stats()``, returning a dict of metrics collected during
refinement for monitoring and logging:

.. code-block:: python

   xray_stats = xray_target.stats()
   print(xray_stats["rwork"].value, xray_stats["rfree"].value)

   # Composite targets nest by component name
   geom_stats = geom_target.stats()
   print(geom_stats["bond"]["rms_z"].value)

The values are :class:`~torchref.utils.stats.StatEntry` wrappers carrying a
verbosity level, not floats. ``__repr__`` prints the bare value, so a print looks
right while arithmetic on the entry fails — take ``.value``.

Using Targets
-------------

.. code-block:: python

   from torchref.refinement.targets import (
       create_xray_target,
       TotalGeometryTarget,
       TotalADPTarget,
   )

   xray_target = create_xray_target(data, model, mode='ml')   # 'ml' is the default
   geom_target = TotalGeometryTarget(model)
   adp_target = TotalADPTarget(model)

   xray_loss = xray_target()
   geom_loss = geom_target()
   adp_loss = adp_target()

Custom Targets
--------------

.. code-block:: python

   import torch
   from torchref.refinement.targets import Target

   class EntropyRegularization(Target):
       """Entropy regularization for B-factors."""
       name = 'entropy_reg'

       def __init__(self, model):
           super().__init__()          # base takes (verbose=0, **kwargs) only
           self.model = model          # keep your own handle

       def forward(self):
           b_factors = self.model.adp()      # not model.b(), which does not exist
           return -torch.sum(b_factors * torch.log(b_factors + 1e-8))

You define only the forward pass; autograd handles the rest. See
:doc:`../quickstart` for a runnable version registered against a ``LossState``.

LossState
---------

:class:`~torchref.refinement.loss_state.LossState` tracks the active targets and
their weights, and supplies the information used for weight calculation. Beyond
``register_target`` / ``set_weight`` / ``aggregate`` (see :doc:`../quickstart`),
its value is driving an optimizer: ``run`` handles the closure, prunes loss
evaluations, and tracks what needs recomputing between line-search steps.

.. code-block:: python

    from torch.optim import LBFGS

    state = refinement.complete_loss_state()
    optimizer = LBFGS(refinement.model.parameters(), lr=1.0, max_iter=100)
    state.run(optimizer, n_steps=1)      # equivalent to state.step(optimizer)
