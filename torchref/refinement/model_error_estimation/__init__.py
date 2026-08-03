"""The two model-error estimators, one module each.

A refined model disagrees with the data for two reasons that need separating: the
measurement is noisy (``sigma_obs``, which the data carries) and the *model is wrong*.
This package estimates the second. Two independent routes, deliberately kept behind
matching interfaces so comparing them is a first-class capability rather than a one-off:

* :mod:`~torchref.refinement.model_error_estimation.sigma_a` -- **data-driven**. Infers the
  per-shell Luzzati ``sigma_A`` from data-model disagreement on the free set, and derives
  ``alpha``/``beta``/``beta_model`` from it. What every ``sigma_A``-family x-ray target
  consumes.
* :mod:`~torchref.refinement.model_error_estimation.sigma_m` -- **structure-driven**.
  Predicts a per-reflection model-error variance from the *structure alone* via the
  diagonal Fisher information; it never sees ``F_obs`` or ``F_calc``, so the estimate
  exists before any comparison with the data.

The two were measured to agree on *shape* (Spearman 0.88-0.99 against ``beta``) but to
differ by ~15x in *magnitude*, which is why the structure-driven one takes a caller-supplied
scale.

**This ``__init__`` deliberately imports nothing.** Both modules are heavy and one of them
(``sigma_a``) is imported from inside :mod:`torchref.scaling` methods to avoid closing a
``scaling`` <-> ``refinement`` cycle -- pulling them into the package namespace would defeat
that, and re-exporting from :mod:`torchref.refinement` would make the import an attribute
lookup on a partially-initialised package. Import the submodules by full path:

    from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator
    from torchref.refinement.model_error_estimation.sigma_m import SigmaMEstimator

Same pattern, and the same reason, as :mod:`torchref.base.targets`, whose ``__init__``
pointedly does not import its two most coupled modules either.
"""
