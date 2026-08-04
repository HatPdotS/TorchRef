"""The two model-error estimators, one module each.

A refined model disagrees with the data for two reasons that need separating: the
measurement is noisy (``sigma_obs``, which the data carries) and the *model is wrong*.
This package estimates the second, by two independent routes kept behind matching
interfaces so comparing them is a first-class capability:

* :mod:`.sigma_a` -- **data-driven**. Infers the per-shell Luzzati ``sigma_A`` from
  data-model disagreement on the free set and derives ``alpha``/``beta``/``beta_model``
  from it. What every ``sigma_A``-family x-ray target consumes.
* :mod:`.sigma_m` -- **structure-driven**. Predicts a per-reflection variance from the
  *structure alone* via the diagonal Fisher information, never seeing ``F_obs`` or
  ``F_calc``. It agrees with ``beta`` on shape but not magnitude, which is why it takes a
  caller-supplied scale.

**This ``__init__`` deliberately imports nothing.** Both modules are heavy and
``sigma_a`` is imported from inside :mod:`torchref.scaling` methods to avoid closing a
``scaling`` <-> ``refinement`` cycle; pulling them in here would defeat that, and
re-exporting from :mod:`torchref.refinement` would make the import an attribute lookup on
a partially-initialised package. Import submodules by full path::

    from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator
"""
