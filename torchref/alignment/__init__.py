"""
Alignment module for TorchRef.

Provides two approaches for aligning predicted structures to observed
diffraction data:

1. Patterson-based alignment: Uses Patterson map vector matching for
   initial orientation search.

2. Maximum Likelihood alignment: Uses ML target functions for gradient-based
   refinement of rotation and translation.

Example - Patterson alignment
-----------------------------
::

    from torchref.alignment import PattersonAligner
    from torchref.model import Model
    from torchref.io.datasets.reflection_data import ReflectionData

    data = ReflectionData().load_mtz('data.mtz')
    model = Model().load_pdb('predicted.pdb')

    aligner = PattersonAligner(data, model)
    aligned_model, result = aligner.align(model)

Example - ML alignment
----------------------
::

    from torchref.alignment import MLOrientationAligner
    from torchref.model import ModelFT
    from torchref.io.datasets.reflection_data import ReflectionData

    data = ReflectionData().load_mtz('data.mtz')
    model = ModelFT().load_pdb('predicted.pdb')

    aligner = MLOrientationAligner(data, model)
    aligned_model, result = aligner.align()
    print(f"Final LLG: {result.llg:.2f}")
"""

import warnings

warnings.warn(
    "torchref.alignment is in development and does not have full functionality. "
    "APIs will change in future releases.",
    FutureWarning,
)

from .align import AlignmentResult, PattersonAligner
from .clashscore import AtomSampler, ClashScoreCalculator, compute_clash_score
from .distributions import (
    combined_log_likelihood,
    rice_log_likelihood,
    stable_log_bessel_i0,
    woolfson_log_likelihood,
)
from .fast_rotation_function import (
    FastRotationFunction,
    FRFPeak,
    FRFResult,
    apply_anisotropic_normalization,
    apply_patterson_sharpening,
    compute_anisotropic_scale,
    compute_e_values,
    fast_rotation_function,
    fit_anisotropic_wilson,
)
from .likelihood import MLTargetFunction, compute_d_factors, compute_llg, mltf
from .ml_aligner import (
    InterpolatedMLTarget,
    MLAlignmentResult,
    MLOrientationAligner,
    MolecularReplacementPipeline,
    MRResult,
    RigidBodyMLTarget,
    TranslationSearchTarget,
)
from .rigid_body import (
    RigidBodyRefinement,
    RigidBodyResult,
    compute_r_factor,
    ml_xray_loss,
)
from .sampling import VectorSampler, get_rotation_sampling_range
from .transform import RigidTransform

__all__ = [
    # Main API - Patterson
    "PattersonAligner",
    "AlignmentResult",
    # Main API - Maximum Likelihood
    "MLOrientationAligner",
    "RigidBodyMLTarget",
    "MLAlignmentResult",
    "MLTargetFunction",
    # MR Pipeline
    "MolecularReplacementPipeline",
    "MRResult",
    "TranslationSearchTarget",
    "InterpolatedMLTarget",
    # Fast Rotation Function
    "FastRotationFunction",
    "FRFPeak",
    "FRFResult",
    "fast_rotation_function",
    # Rigid body refinement
    "RigidBodyRefinement",
    "RigidBodyResult",
    "compute_r_factor",
    "ml_xray_loss",
    # Rigid body transformations
    "RigidTransform",
    # Clash scoring
    "ClashScoreCalculator",
    "AtomSampler",
    "compute_clash_score",
    # ML likelihood functions
    "mltf",
    "compute_d_factors",
    "compute_llg",
    # Distribution primitives
    "stable_log_bessel_i0",
    "rice_log_likelihood",
    "woolfson_log_likelihood",
    "combined_log_likelihood",
    # Lower-level utilities (for advanced users)
    "VectorSampler",
    "get_rotation_sampling_range",
    "params_to_matrix",
    "matrix_to_params",
    "random_rotation_params",
    "random_rotation_matrix",
]
