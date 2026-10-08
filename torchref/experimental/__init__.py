"""
Experimental TorchRef modules.

This namespace collects features whose APIs are still under active
development and may change without notice. Import the submodules
directly:

* :mod:`torchref.experimental.alignment` -- Patterson-based molecular
  replacement (rotation and translation search; returns a placement).
* :mod:`torchref.experimental.ensemble` -- multi-member ensemble refinement
  against one dataset, modelling disorder as explicit member spread.
* :mod:`torchref.experimental.kinetic` -- time-resolved (kinetic) refinement.
* :mod:`torchref.experimental.monolithic_refinement` -- macrocycle-free
  refinement with a differentiable, co-refined model-error variance.
* :mod:`torchref.experimental.targets` -- experimental refinement
  targets (AMBER14/GAFF2 force field, real-space, occupancy diagnostics).

Submodules are not imported eagerly, so importing ``torchref.experimental``
stays cheap and pulls in no optional dependency (e.g. OpenMM for the AMBER
targets) until a submodule is requested.
"""


__all__ = ["alignment", "ensemble", "kinetic", "monolithic_refinement", "targets"]
