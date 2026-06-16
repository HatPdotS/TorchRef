"""
Experimental TorchRef modules.

This namespace collects features whose APIs are still under active
development and may change without notice. Import the submodules
directly:

* :mod:`torchref.experimental.alignment` -- Patterson-based molecular
  replacement (rotation/translation search, rigid-body refinement).
* :mod:`torchref.experimental.kinetic` -- time-resolved / kinetic
  refinement against collections of datasets.

* :mod:`torchref.experimental.monolithic_refinement` -- macrocycle-free
  refinement with a differentiable, co-refined model-error variance.


Submodules are not imported eagerly here so that pulling in
``torchref.experimental`` stays cheap and does not trigger the optional
dependencies (e.g. JAX for alignment) until a submodule is requested.
"""


__all__ = ["alignment", "kinetic", "monolithic_refinement"]
