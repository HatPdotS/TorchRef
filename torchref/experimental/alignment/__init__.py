"""
Molecular replacement for TorchRef: a rotation search feeding a translation
search, and one Wilson normalisation shared between them.

1. Fast Rotation Function (``rotation_search``, over
   ``frf.FastRotationFunction``) — Phaser-faithful Bessel-radial × SH
   expansion, stable Wigner-d, dense P1-box calc. A shortlist generator.
2. Fast Translation Function (``translation.amplitude_translation_search`` +
   ``local_translation_refine``) — run per rotation candidate, and where the
   discrimination actually happens.
3. Pipeline (``pipeline.MolecularReplacementPipeline``) — the multi-candidate
   FRF → FTF tree with early stopping, which ``align.align_model_to_data``
   delegates to. It returns a placement; refine it downstream.

Example — full MR pipeline
--------------------------
::

    from torchref.experimental.alignment import MolecularReplacementPipeline
    from torchref.model import ModelFT
    from torchref.io.datasets.reflection_data import ReflectionData

    data = ReflectionData().load_mtz('observed.mtz')
    model = ModelFT().load_pdb('search_model.pdb')

    pipeline = MolecularReplacementPipeline(data, model)
    solutions = pipeline.run(n_rotation_peaks=200, min_tries=3, max_tries=10)
    print(f"Best R-factor: {solutions[0].r_factor:.3f}")
"""

import warnings

warnings.warn(
    "torchref.experimental.alignment is in development. APIs may change.",
    FutureWarning,
)

# =============================================================================
# Fast Rotation Function — Phaser-faithful, single engine
# =============================================================================
from .frf import (
    FastRotationFunction,
    RotationPeak,
    dense_calc_via_box,
    edmonds_euler_from_rotation_matrix,
    phaser_lmax_resolution,
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from .sh import (
    evaluate_ylm,
    sh_expand_ball,
    equal_count_shell_edges,
    assign_shells,
)
from .pipeline import (
    MolecularReplacementPipeline,
    MRSolution,
    cluster_rotation_peaks,
    rotation_angular_distance,
    euler_angular_distance,
)
from .align import align_model_to_data
from .rotation_search import RotationSolutions, rotation_search

# =============================================================================
# Translation search
# =============================================================================
from .translation import (
    TranslationPeak,
    amplitude_translation_search,
    find_translation_peaks,
    fit_sigma_a_per_shell,
    llg_translation_rescore,
    local_translation_refine,
    precompute_G_for_rotation,
)

# =============================================================================
# ML distributions
# =============================================================================
from .distributions import (
    stable_log_bessel_i0,
    rice_log_likelihood,
    woolfson_log_likelihood,
)

__all__ = [
    # Rotation search
    "FastRotationFunction",
    "phaser_lmax_resolution",
    "dense_calc_via_box",
    "RotationPeak",
    "rotation_matrix_from_edmonds_euler",
    "edmonds_euler_from_rotation_matrix",
    "rotation_angular_distance_deg",
    # Rescore + interpolation
    # Low-level math primitives
    "evaluate_ylm",
    "sh_expand_ball",
    "equal_count_shell_edges",
    "assign_shells",
    # Pipeline
    "MolecularReplacementPipeline",
    "MRSolution",
    "cluster_rotation_peaks",
    "rotation_angular_distance",
    "euler_angular_distance",
    "align_model_to_data",
    "rotation_search",
    "RotationSolutions",
    # Translation
    "TranslationPeak",
    "amplitude_translation_search",
    "find_translation_peaks",
    "fit_sigma_a_per_shell",
    "llg_translation_rescore",
    "local_translation_refine",
    "precompute_G_for_rotation",
    # Rigid body refinement
    # Transforms
    # Clash scoring
    # Distributions
    "stable_log_bessel_i0",
    "rice_log_likelihood",
    "woolfson_log_likelihood",
    # Utilities
]
