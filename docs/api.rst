API Reference
=============

Generated automatically from the package docstrings. ``torchref.experimental``
is deliberately absent: it holds unvalidated code (alignment, kinetics,
ensemble and monolithic refinement) whose API is expected to move.

.. autosummary::
   :toctree: api/
   :recursive:

   torchref.cli
   torchref.config
   torchref.io
   torchref.maps
   torchref.model
   torchref.refinement
   torchref.scaling
   torchref.symmetry
   torchref.topology
   torchref.base
   torchref.utils

Target and Backend Tables
-------------------------

These tables are the single source of truth for the selectable X-ray targets and for
direct-summation, electron-density and target-math kernel dispatch. They live in
private modules, which the recursive summary above skips, so they are rendered here.

torchref.refinement.targets.xray._specs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: torchref.refinement.targets.xray._specs

.. autodata:: torchref.refinement.targets.xray._specs.XRAY_TARGETS
   :no-value:

torchref.refinement.targets.collection._specs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: torchref.refinement.targets.collection._specs

.. autodata:: torchref.refinement.targets.collection._specs.COLLECTION_XRAY_TARGETS
   :no-value:

torchref.base.direct_summation._backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: torchref.base.direct_summation._backends

.. autodata:: torchref.base.direct_summation._backends.DS_BACKENDS
   :no-value:

torchref.base.electron_density._backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: torchref.base.electron_density._backends

.. autodata:: torchref.base.electron_density._backends.DENSITY_BACKENDS
   :no-value:

torchref.base.targets._dispatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: torchref.base.targets._dispatch

.. autodata:: torchref.base.targets._dispatch.TARGET_BACKENDS
   :no-value:
