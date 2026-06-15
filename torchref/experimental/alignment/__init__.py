"""
Alignment module for TorchRef.

Pure-PyTorch Patterson-based molecular replacement:

1. Fast Rotation Function (``frf.phaser_rotation_search`` /
   ``frf.FastRotationFunction``) — Phaser-faithful Bessel-radial × SH
   expansion, stable Wigner-d, dense P1-box calc — then ML rescoring
   (``ml_rotation.m_letf1_rescore``) to rank candidate orientations.
2. Fast Translation Function (``translation.amplitude_translation_search`` +
   ``local_translation_refine``) — run per rotation candidate.
3. Rigid Body Refinement (``rigid_body.RigidBodyRefinement``) — LBFGS on
   rotation and translation (and optional B-factors) with an ML target.
4. Canonical Pipeline (``pipeline.MolecularReplacementPipeline``) — the
   multi-candidate FRF → FTF → post-refine tree with early-stopping; the
   implementation that ``align.align_model_to_data`` /
   ``ModelFT.fit_to_data`` delegate to.

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
    phaser_rotation_search,
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from .lattman_love import LattmanLoveInterpolator
from .ml_rotation import m_letf1_rescore, sim_mlrf_rescore
from .sh import (
    evaluate_ylm,
    sh_expand_ball,
    equal_count_shell_edges,
    assign_shells,
)
from .wigner import (
    small_d_block,
    small_d_packed,
    wigner_D_pointwise,
    evaluate_rotation_function_grid,
    evaluate_rotation_function_pointwise,
)
from .pipeline import (
    MolecularReplacementPipeline,
    MRSolution,
    cluster_rotation_peaks,
    rotation_angular_distance,
    euler_angular_distance,
)
from .align import align_model_to_data

# =============================================================================
# Translation search
# =============================================================================
from .translation import (
    fft_translation_search,
    fft_translation_search_torch,
    TranslationPeak,
    find_translation_peaks,
    apply_translation_to_fcalc,
    apply_translation_to_fcalc_torch,
)

# =============================================================================
# Rigid body refinement
# =============================================================================
from .rigid_body import RigidBodyRefinement, RigidBodyResult

# =============================================================================
# Rigid body transformations
# =============================================================================
from .transform import (
    RigidTransform,
    quaternion_normalize,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_rotate,
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_euler_zyz,
    euler_zyz_to_quaternion,
    rotation_matrix_from_euler,
    sample_angles,
)

# =============================================================================
# Clash scoring
# =============================================================================
from .clashscore import ClashScoreCalculator, AtomSampler, compute_clash_score

# =============================================================================
# ML distributions
# =============================================================================
from .distributions import (
    stable_log_bessel_i0,
    rice_log_likelihood,
    woolfson_log_likelihood,
    combined_log_likelihood,
    acentric_pdf,
    centric_pdf,
)

# =============================================================================
# Sampling utilities
# =============================================================================
from .sampling import VectorSampler, get_rotation_sampling_range

__all__ = [
    # Rotation search
    "FastRotationFunction",
    "phaser_rotation_search",
    "phaser_lmax_resolution",
    "dense_calc_via_box",
    "RotationPeak",
    "rotation_matrix_from_edmonds_euler",
    "edmonds_euler_from_rotation_matrix",
    "rotation_angular_distance_deg",
    # Rescore + interpolation
    "LattmanLoveInterpolator",
    "m_letf1_rescore",
    "sim_mlrf_rescore",
    # Low-level math primitives
    "evaluate_ylm",
    "sh_expand_ball",
    "equal_count_shell_edges",
    "assign_shells",
    "small_d_block",
    "small_d_packed",
    "wigner_D_pointwise",
    "evaluate_rotation_function_grid",
    "evaluate_rotation_function_pointwise",
    # Pipeline
    "MolecularReplacementPipeline",
    "MRSolution",
    "cluster_rotation_peaks",
    "rotation_angular_distance",
    "euler_angular_distance",
    "align_model_to_data",
    # Translation
    "fft_translation_search",
    "fft_translation_search_torch",
    "TranslationPeak",
    "find_translation_peaks",
    "apply_translation_to_fcalc",
    "apply_translation_to_fcalc_torch",
    # Rigid body refinement
    "RigidBodyRefinement",
    "RigidBodyResult",
    # Transforms
    "RigidTransform",
    "quaternion_normalize",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_rotate",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "quaternion_to_euler_zyz",
    "euler_zyz_to_quaternion",
    "rotation_matrix_from_euler",
    "sample_angles",
    # Clash scoring
    "ClashScoreCalculator",
    "AtomSampler",
    "compute_clash_score",
    # Distributions
    "stable_log_bessel_i0",
    "rice_log_likelihood",
    "woolfson_log_likelihood",
    "combined_log_likelihood",
    "acentric_pdf",
    "centric_pdf",
    # Utilities
    "VectorSampler",
    "get_rotation_sampling_range",
]
