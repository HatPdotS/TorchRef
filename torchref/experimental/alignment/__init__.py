"""Molecular replacement: a rotation search feeding a translation search.

Two stages and one normalisation between them.

1. **Fast Rotation Function** (:func:`rotation_search`, over
   :class:`~torchref.experimental.alignment.frf.FastRotationFunction`) --
   Phaser-faithful Bessel-radial x spherical-harmonic expansion against a dense
   P1-box calc, with stable Wigner-d. It is a **shortlist generator**: over 30
   seeded cells it puts the true orientation at rank 0 six times, and inside the
   top twenty essentially always. Only the second of those is required.
2. **Fast Translation Function** (:mod:`~torchref.experimental.alignment.translation`)
   -- a Crowther-Blow correlation over the fractional cell, run per rotation
   candidate, then an analytical-R local refine. On the same 30 cells it reaches
   rank 0 in 24, and its likelihood in 27. Rotation ghosts are morphologically
   identical to truth in a Patterson by construction; they stop being identical
   once the crystal lattice is involved.

There is deliberately nothing between the two. An ML rescore used to sit there
and was removed: it reordered a shortlist that already contained truth, and
end-to-end pose recovery was 18/30 with it against 24/30 without.

Both stages normalise through :class:`torchref.scaling.WilsonNormaliser` and
weight through :mod:`torchref.scaling.weighting`, so ``E_obs`` means one thing
across the whole run.

The pipeline returns a **placement** -- rotation and translation -- and stops.
Refining it is downstream refinement's job, and deleting the post-placement
polish that used to be here took pose recovery from 24/30 to 30/30, because on
the hard cases it walked a correct placement away from truth.

Example
-------
::

    from torchref.experimental.alignment import MolecularReplacementPipeline
    from torchref.model import ModelFT
    from torchref.io.datasets.reflection_data import ReflectionData

    data = ReflectionData().load_mtz('observed.mtz')
    model = ModelFT().load_pdb('search_model.pdb')

    solutions = MolecularReplacementPipeline(data, model).run()
    print(f"best R-work: {solutions[0].r_factor:.3f}")
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
    FRFInputs,
    RotationSolutions,
    prepare_frf_inputs,
    rotation_search,
)
from .translation import (
    DirectModelEvaluator,
    TranslationObs,
    TranslationPeak,
    amplitude_translation_search,
    find_translation_peaks,
    fit_model_error,
    llg_translation_rescore,
    local_translation_refine,
    precompute_G_for_rotation,
)
from .pipeline import (
    MolecularReplacementPipeline,
    MRSolution,
    align_model_to_data,
    cluster_rotation_peaks,
    euler_angular_distance,
    rotation_angular_distance,
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
    "FRFInputs",
    "prepare_frf_inputs",
    "phaser_lmax_resolution",
    "dense_calc_via_box",
    "RotationPeak",
    "rotation_matrix_from_edmonds_euler",
    "edmonds_euler_from_rotation_matrix",
    "rotation_angular_distance_deg",
    "cluster_rotation_peaks",
    "rotation_angular_distance",
    "euler_angular_distance",
    # Translation search
    "TranslationObs",
    "TranslationPeak",
    "DirectModelEvaluator",
    "amplitude_translation_search",
    "local_translation_refine",
    "llg_translation_rescore",
    "precompute_G_for_rotation",
    "find_translation_peaks",
    "fit_model_error",
]
