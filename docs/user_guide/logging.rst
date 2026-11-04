Logging and Verbosity
=====================

TorchRef reports through the standard :mod:`logging` module, one logger per
module under ``torchref`` (``torchref.io.mtz``, ``torchref.scaling.scaler``, ...).
Nothing in the library prints diagnostics, and no class takes a ``verbose``
argument.

Levels
------

Verbosity levels map onto logging levels:

=====  ===========  ===========================================================
Level  Logging      Shows
=====  ===========  ===========================================================
``0``  ``WARNING``  warnings and errors only
``1``  ``INFO``     progress: files read, R-factors per cycle, what was set up
``2``  ``DEBUG``    detail: per-bin statistics, per-step diagnostics
``3``  ``TRACE``    internals, including per-iteration values
=====  ===========  ===========================================================

As a library
------------

Imported as a library, TorchRef shows warnings only. Raise the level for the
whole package or for one subpackage:

.. code-block:: python

   import torchref

   torchref.set_verbosity(1)                       # progress for everything
   torchref.set_verbosity(2, "torchref.scaling")   # detail for scaling only

   with torchref.verbosity(2):                     # for one block
       refinement.refine(macro_cycles=3)

   with torchref.quiet():                          # nothing below WARNING
       scratch = model.copy()

``TORCHREF_LOG_LEVEL`` (``0``-``3`` or a level name) sets the level at import,
which is convenient for batch jobs.

TorchRef writes to stderr through its own handler only while the application has
not configured logging. Once it has (``logging.basicConfig``, a framework,
pytest), TorchRef's records propagate to the root logger's handlers instead and
are formatted and filtered there; ``torchref.set_verbosity`` still sets which
records TorchRef emits.

:func:`torchref.quiet` is per thread: it does not change any level, so it is
safe to use inside library code and in parallel jobs.

From the command line
---------------------

Every ``torchref.*`` command takes ``-v N`` (default ``1``) and ``-q`` (``-v 0``).
Progress goes to stdout, interleaved with the command's results, and warnings go to
stderr. ``--log-file PATH`` writes the log to a file as well. Python warnings
raised during a run are routed through the same handlers.

Refinement statistics
---------------------

The statistics recorded during refinement (``refinement.logger``, the history
dicts) are data, not log messages. Each statistic still carries its own
verbosity tier (:func:`torchref.utils.stats.stat`), and the tier displayed
follows the level of the ``torchref.refinement`` logger.

Debug checks
------------

Some numerical self-checks are expensive, since each one is a host sync, and are
off by default: finite maps and structure factors in ``ModelFT`` and the
scalers, and a module-state dump when ``Refinement`` setup fails. Turn them on
with ``TORCHREF_DEBUG_CHECKS=1`` or ``torchref.config.debug_checks.value = True``.
They do not depend on the log level.
