"""
Alignment module for TorchRef.

Provides molecular replacement functionality including:

1. Fast Rotation Function: Ball harmonic transform for rotation search
2. Translation Search: FFT-based translation function
3. Rigid Body Refinement: Optimization of rotation and translation
4. Unified Pipeline: Complete MR workflow with early stopping

Example - Full MR Pipeline
--------------------------
::

    from torchref.alignment import MolecularReplacementPipeline
    from torchref.model import ModelFT
    from torchref.io.datasets.reflection_data import ReflectionData

    data = ReflectionData().load_mtz('observed.mtz')
    model = ModelFT().load_pdb('search_model.pdb')

    pipeline = MolecularReplacementPipeline(data, model)
    solutions = pipeline.run(n_rotation_peaks=50, min_tries=3, max_tries=10)
    print(f"Best R-factor: {solutions[0].r_factor:.3f}")

Example - Individual Components
-------------------------------
::

    from torchref.alignment import (
        ball_rotation_search_torch,
        fft_translation_search_torch,
        RigidBodyRefinement,
    )

    # 1. Rotation search
    rf, angles, peaks = ball_rotation_search_torch(
        E_obs, s_obs, E_calc, s_calc, L=32, P=20
    )

    # 2. Translation search for top rotation
    alpha, beta, gamma, score, sigma = peaks[0]
    corr_map, best_trans, trans_peaks = fft_translation_search_torch(
        F_obs, F_calc_rotated, hkl
    )

    # 3. Rigid body refinement
    rb = RigidBodyRefinement(model, data, initial_rotation=..., initial_translation=...)
    result = rb.refine()
"""

import warnings

warnings.warn(
    "torchref.alignment is in development. APIs may change.",
    FutureWarning,
)

# =============================================================================
# Pipeline & Rotation search require JAX + s2fft (dev dependencies)
# =============================================================================
try:
    from .pipeline import (
        MolecularReplacementPipeline,
        MRSolution,
        cluster_rotation_peaks,
        rotation_angular_distance,
        euler_angular_distance,
    )
    from .ball_transform import (
        ball_rotation_search,
        ball_rotation_search_torch,
        rotation_matrix_from_euler_zyz,
        rotation_matrix_to_euler_zyz,
        rotation_matrix_to_quaternion,
        check_rotation_recovery,
        BallHarmonicCoefficients,
        splat_evalues_to_ball,
        compute_ball_harmonic_coefficients,
        compute_ball_cross_correlation_coefficients,
        evaluate_rotation_function,
        find_rotation_peaks,
        reduce_rotation_by_symmetry,
        reduce_peaks_by_symmetry,
        reduce_peaks_by_symmetry_torch,
        cluster_rotation_peaks,
        cluster_rotation_peaks_torch,
        RotationCluster,
    )
    _HAS_BALL_TRANSFORM = True
except ImportError:
    _HAS_BALL_TRANSFORM = False

    _BALL_TRANSFORM_MSG = (
        "The alignment pipeline and rotation search require jax, s2fft, "
        "s2ball, spherical, and quaternionic. "
        "Install with:  pip install torchref[dev]"
    )

    def _missing_dep_factory(name):
        """Create a callable stub that raises ImportError with install hint."""
        def _stub(*args, **kwargs):
            raise ImportError(
                f"{name} is not available. {_BALL_TRANSFORM_MSG}"
            )
        _stub.__name__ = name
        _stub.__qualname__ = name
        return _stub

    # Provide stubs so that attribute access works but calling raises
    MolecularReplacementPipeline = _missing_dep_factory("MolecularReplacementPipeline")
    MRSolution = _missing_dep_factory("MRSolution")
    ball_rotation_search = _missing_dep_factory("ball_rotation_search")
    ball_rotation_search_torch = _missing_dep_factory("ball_rotation_search_torch")

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
    # Translation search
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

if _HAS_BALL_TRANSFORM:
    __all__ += [
        # Pipeline (main entry point)
        "MolecularReplacementPipeline",
        "MRSolution",
        "cluster_rotation_peaks",
        "rotation_angular_distance",
        "euler_angular_distance",
        # Rotation search
        "ball_rotation_search",
        "ball_rotation_search_torch",
        "rotation_matrix_from_euler_zyz",
        "rotation_matrix_to_euler_zyz",
        "rotation_matrix_to_quaternion",
        "check_rotation_recovery",
        "BallHarmonicCoefficients",
        "splat_evalues_to_ball",
        "compute_ball_harmonic_coefficients",
        "compute_ball_cross_correlation_coefficients",
        "evaluate_rotation_function",
        "find_rotation_peaks",
        "reduce_rotation_by_symmetry",
        "reduce_peaks_by_symmetry",
        "reduce_peaks_by_symmetry_torch",
        "cluster_rotation_peaks",
        "cluster_rotation_peaks_torch",
        "RotationCluster",
    ]
