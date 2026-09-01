"""Fast Rotation Function — single, validated implementation.

Phaser-faithful engine: chunked Bessel-SH expansion, stable Wigner-d
(``wigner_d.small_d_stable``), resolution↔bandwidth coupling
(``phaser_lmax_resolution``, default cap=48), dense P1-box calc, all under
``no_grad``. Solved the high-symmetry cases that broke the earlier ball
and Phaser-mimic engines (4BX9 342→4–7, 6G9X 77→1–4).

Shared leaf math (``..sh``, ``..wigner``) lives in the parent ``alignment``
package; this sub-package imports it "up".
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
