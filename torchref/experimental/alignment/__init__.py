"""Molecular replacement: a rotation search feeding a translation search.

1. **Fast Rotation Function** (:func:`rotation_search`, over
   :class:`~torchref.experimental.alignment.frf.FastRotationFunction`) -- a
   Phaser-style Bessel-radial x spherical-harmonic expansion that returns a
   shortlist of orientations.
2. **Fast Translation Function** (:mod:`~torchref.experimental.alignment.translation`)
   -- a Crowther-Blow search per rotation candidate, ranked by its likelihood.

The pipeline returns a **placement** -- rotation and translation -- not a refined
model. ``__all__`` re-exports the entry points, the rotation search with its FRF
building blocks, and the translation search; ``search_peaks``, ``fit_anisotropy``
and the spherical-harmonic helpers in ``sh`` stay in their modules.

Example
-------
::

    from torchref.experimental.alignment import MolecularReplacementPipeline
    from torchref.model import ModelFT
    from torchref.io.datasets.reflection_data import ReflectionData

    data = ReflectionData().load_mtz('observed.mtz')
    model = ModelFT().load_pdb('search_model.pdb')

    solutions = MolecularReplacementPipeline(data, model).run()
    print(f"best analytic R: {solutions[0].r_factor:.3f}")
"""

import warnings

warnings.warn(
    "torchref.experimental.alignment is in development. APIs may change.",
    FutureWarning,
)

from .frf import (
    FastRotationFunction,
    RotationPeak,
    dense_calc_via_box,
    edmonds_euler_from_rotation_matrix,
    phaser_lmax_resolution,
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from .rotation_search import (
    RotationSolutions,
    rotation_search,
)
from .translation import (
    CandidateTransform,
    TranslationObs,
    TranslationPeak,
    analytic_r_at,
    fast_translation_function,
    llg_at_translations,
    prepare_candidate,
)
from .pipeline import (
    MolecularReplacementPipeline,
    MRSolution,
    align_model_to_data,
)

__all__ = [
    # Entry points
    "align_model_to_data",
    "MolecularReplacementPipeline",
    "MRSolution",
    # Rotation search
    "rotation_search",
    "RotationSolutions",
    "FastRotationFunction",
    "phaser_lmax_resolution",
    "dense_calc_via_box",
    "RotationPeak",
    "rotation_matrix_from_edmonds_euler",
    "edmonds_euler_from_rotation_matrix",
    "rotation_angular_distance_deg",
    # Translation search
    "TranslationObs",
    "TranslationPeak",
    "fast_translation_function",
    "CandidateTransform",
    "analytic_r_at",
    "prepare_candidate",
    "llg_at_translations",
]
