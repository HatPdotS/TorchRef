"""Fast Rotation Function — single, validated implementation.

Phaser-faithful engine: chunked Bessel-SH expansion, Wigner-d from the J_y
eigendecomposition (``wigner_d.wigner_contraction_per_beta``),
resolution↔bandwidth coupling (``phaser_lmax_resolution``, default cap 64,
``rotation_search.LMAX_CAP``), dense P1-box calc, all under ``no_grad``.

Shared leaf math (``..sh``) lives in the parent ``alignment`` package; this
sub-package imports it "up".
"""
from .api import FastRotationFunction, phaser_lmax_resolution
from .dense_calc import dense_calc_via_box, model_sf_abs
from .rotation_utils import (
    edmonds_euler_from_rotation_matrix,
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from .types import (
    AdaptiveRotationFunction,
    BesselSHCoefficients,
    RotationPeak,
)

__all__ = [
    # Engine
    "FastRotationFunction",
    "phaser_lmax_resolution",
    "dense_calc_via_box",
    "model_sf_abs",
    # Rotation geometry helpers
    "rotation_matrix_from_edmonds_euler",
    "edmonds_euler_from_rotation_matrix",
    "rotation_angular_distance_deg",
    # Types
    "AdaptiveRotationFunction",
    "BesselSHCoefficients",
    "RotationPeak",
]
