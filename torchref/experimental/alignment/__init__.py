"""
Experimental ball-harmonic molecular-replacement engine for TorchRef.

Experimental / unstable API. This is the opt-in **ball-harmonic MR engine**;
the production / canonical MR entry point is ``torchref.alignment`` (the
consolidated FRF engine, default ``engine="frf_separate"``). Everything in this
package may change without notice (importing it emits a ``FutureWarning``).

Provides molecular replacement functionality including:

1. Fast Rotation Function: Ball harmonic transform for rotation search
2. Translation Search: FFT-based translation function
3. Rigid Body Refinement: Optimization of rotation and translation
4. Unified Pipeline: Complete MR workflow with early stopping

Notes
-----
Optional dependency surface. The pipeline and ball rotation-search symbols
(``MolecularReplacementPipeline``, ``MRSolution``, ``ball_rotation_search``,
``ball_rotation_search_torch``, ``BallHarmonicCoefficients``,
``splat_evalues_to_ball``, ``compute_ball_harmonic_coefficients``,
``compute_ball_cross_correlation_coefficients``, ``evaluate_rotation_function``,
``find_rotation_peaks``, ``reduce_rotation_by_symmetry``, ``RotationCluster``,
``cluster_rotation_peaks`` and the other rotation/Euler helpers) require the
JAX/s2fft stack and are exported only when ``pip install torchref[alignment]``
is present. Without that extra they fall back to stubs (or are absent from
``__all__``); only translation search, rigid-body refinement, transforms, clash
scoring, distributions, and the sampling utilities are unconditionally
available.

The package-level ``cluster_rotation_peaks`` is the ``.ball_transform`` version
(signature ``(peaks, cluster_radius_deg=5.0, symmetry_matrices=None,
return_details=False)`` returning 6-tuples / ``RotationCluster`` objects). The
``.pipeline`` module keeps its own simpler variant for internal use, reachable
as ``torchref.experimental.alignment.pipeline.cluster_rotation_peaks``.

A handful of public ball helpers (``compute_ball_harmonic_coefficients_analytical``,
``refine_peaks_analytical``, ``refine_peaks_subvoxel_wrapper``,
``evaluate_rotation_function_at_angles``, ``build_wigner_index_mapping``) are
documented in their modules but are not re-exported here and are not part of the
supported package API.

Example - Full MR Pipeline
--------------------------
::

    from torchref.experimental.alignment import MolecularReplacementPipeline
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

    from torchref.experimental.alignment import (
        ball_rotation_search_torch,
        fft_translation_search_torch,
        RigidBodyRefinement,
    )

    # E_obs, s_obs, E_calc, s_calc, F_obs, F_calc_rotated, hkl assumed
    # computed beforehand from the data/model.

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
    "torchref.experimental.alignment is in development. APIs may change.",
    FutureWarning,
)

# =============================================================================
# Pipeline & Rotation search require JAX + s2fft (dev dependencies)
# =============================================================================
try:
    from .pipeline import (
        MolecularReplacementPipeline,
        MRSolution,
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
        "Install with:  pip install torchref[alignment]"
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
