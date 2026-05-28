"""
Alignment module for TorchRef.

Pure-PyTorch Patterson-based molecular replacement:

1. Fast Rotation Function (`ball_search.ball_rotation_search`) — ball-harmonic
   SO(3) cross-correlation, evaluated on an Euler-angle grid via inverse Wigner
   transform.
2. Translation Search (`translation.fft_translation_search_torch`).
3. Rigid Body Refinement (`rigid_body.RigidBodyRefinement`) — LBFGS on rotation
   and translation parameters with a maximum-likelihood x-ray target.
4. Unified Pipeline (`pipeline.MolecularReplacementPipeline`) — end-to-end
   workflow with early-stopping.

Example — full MR pipeline
--------------------------
::

    from torchref.alignment import MolecularReplacementPipeline
    from torchref.model import ModelFT
    from torchref.io.datasets.reflection_data import ReflectionData

    data = ReflectionData().load_mtz('observed.mtz')
    model = ModelFT().load_pdb('search_model.pdb')

    pipeline = MolecularReplacementPipeline(data, model)
    solutions = pipeline.run(n_rotation_peaks=200, min_tries=3, max_tries=10)
    print(f"Best R-factor: {solutions[0].r_factor:.3f}")

Example — individual components
-------------------------------
::

    from torchref.alignment import (
        ball_rotation_search,
        fft_translation_search_torch,
        RigidBodyRefinement,
    )

    # 1. Rotation search
    C, alphas, betas, gammas, peaks = ball_rotation_search(
        s_obs, e_obs, s_calc, e_calc, L=48, P=24,
    )

    # 2. Translation search for top rotation
    peak = peaks[0]
    corr_map, best_trans, trans_peaks = fft_translation_search_torch(
        F_obs, F_calc_rotated, hkl
    )

    # 3. Rigid body refinement
    rb = RigidBodyRefinement(model, data,
                             initial_rotation=..., initial_translation=...)
    result = rb.refine()
"""

import warnings

warnings.warn(
    "torchref.alignment is in development. APIs may change.",
    FutureWarning,
)

# =============================================================================
# Fast Rotation Function engines (consolidated in the .frf sub-package)
# =============================================================================
# Ball-harmonic E-value rotation search (engine="ball").
from .frf.ball_search import (
    BallHarmonicCoefficients,
    RotationPeak,
    ball_rotation_search,
    compute_ball_harmonic_coefficients,
    compute_ball_cross_correlation_coefficients,
    find_rotation_peaks,
    refine_peaks_subvoxel,
    rotation_matrix_from_edmonds_euler,
    edmonds_euler_from_rotation_matrix,
    rotation_angular_distance_deg,
)
# Phaser-faithful engine — the production default rotation search.
from .frf.api import (
    FastRotationFunction,
    phaser_lmax_resolution,
    phaser_rotation_search,
)
from .frf.dense_calc import dense_calc_via_box
from .lattman_love import LattmanLoveInterpolator
from .ml_rotation import sim_mlrf_rescore, brute_ml_rotation_search
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
    # -------------------------------------------------------------------------
    # Rotation search (ball-harmonic Patterson, pure torch)
    # -------------------------------------------------------------------------
    "ball_rotation_search",
    "compute_ball_harmonic_coefficients",
    "compute_ball_cross_correlation_coefficients",
    "find_rotation_peaks",
    "refine_peaks_subvoxel",
    "BallHarmonicCoefficients",
    "RotationPeak",
    "rotation_matrix_from_edmonds_euler",
    "edmonds_euler_from_rotation_matrix",
    "rotation_angular_distance_deg",
    # Phaser-faithful engine (production default)
    "FastRotationFunction",
    "phaser_rotation_search",
    "phaser_lmax_resolution",
    "dense_calc_via_box",
    "LattmanLoveInterpolator",
    "sim_mlrf_rescore",
    "brute_ml_rotation_search",
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
    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------
    "MolecularReplacementPipeline",
    "MRSolution",
    "cluster_rotation_peaks",
    "rotation_angular_distance",
    "euler_angular_distance",
    "align_model_to_data",
    # -------------------------------------------------------------------------
    # Translation
    # -------------------------------------------------------------------------
    "fft_translation_search",
    "fft_translation_search_torch",
    "TranslationPeak",
    "find_translation_peaks",
    "apply_translation_to_fcalc",
    "apply_translation_to_fcalc_torch",
    # -------------------------------------------------------------------------
    # Rigid body refinement
    # -------------------------------------------------------------------------
    "RigidBodyRefinement",
    "RigidBodyResult",
    # -------------------------------------------------------------------------
    # Transforms
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # Clash scoring
    # -------------------------------------------------------------------------
    "ClashScoreCalculator",
    "AtomSampler",
    "compute_clash_score",
    # -------------------------------------------------------------------------
    # Distributions
    # -------------------------------------------------------------------------
    "stable_log_bessel_i0",
    "rice_log_likelihood",
    "woolfson_log_likelihood",
    "combined_log_likelihood",
    "acentric_pdf",
    "centric_pdf",
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    "VectorSampler",
    "get_rotation_sampling_range",
]
