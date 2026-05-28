"""Fast Rotation Function engines (consolidated).

This sub-package collects every Fast Rotation Function (FRF) implementation in
one place. Three engines coexist (see ``FRF_CONSOLIDATION.md``):

- **``api`` (the production default)** — the Phaser-faithful engine validated in
  the high-symmetry investigation: chunked Bessel-SH expansion, stable Wigner-d
  (``wigner_d.small_d_stable``), resolution↔bandwidth coupling
  (``phaser_lmax_resolution``, cap=48), dense P1-box calc (``dense_calc``), all
  under ``no_grad``. Reached 4BX9 342→4-7, 6G9X 77→1-4.
- **``ball_search`` (``engine="ball"``)** — the original pure-torch ball-harmonic
  E-value rotation search.
- **``phaser_frf`` (``legacy_phaser_rotation_search``)** — the earlier 1164-line
  Phaser-mimic, superseded by ``api`` but kept for reference/benchmarking.

Shared leaf math (``..sh``, ``..wigner``) stays in the parent ``alignment``
package; this sub-package imports it "up".
"""
from .api import (
    FastRotationFunction,
    phaser_lmax_resolution,
    phaser_rotation_search,
)
from .ball_search import (
    BallHarmonicCoefficients,
    RotationPeak as BallRotationPeak,
    ball_rotation_search,
    compute_ball_cross_correlation_coefficients,
    compute_ball_harmonic_coefficients,
    edmonds_euler_from_rotation_matrix,
    find_rotation_peaks,
    refine_peaks_subvoxel,
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from .dense_calc import dense_calc_via_box, model_sf_abs
from .phaser_frf import phaser_rotation_search as legacy_phaser_rotation_search
from .types import (
    AdaptiveRotationFunction,
    BesselSHCoefficients,
    RotationPeak,
    WignerContraction,
)

__all__ = [
    # Validated engine (production default)
    "FastRotationFunction",
    "phaser_rotation_search",
    "phaser_lmax_resolution",
    "dense_calc_via_box",
    "model_sf_abs",
    # Ball-harmonic engine (engine="ball")
    "ball_rotation_search",
    "BallHarmonicCoefficients",
    "BallRotationPeak",
    "compute_ball_harmonic_coefficients",
    "compute_ball_cross_correlation_coefficients",
    "find_rotation_peaks",
    "refine_peaks_subvoxel",
    "rotation_matrix_from_edmonds_euler",
    "edmonds_euler_from_rotation_matrix",
    "rotation_angular_distance_deg",
    # Legacy Phaser-mimic
    "legacy_phaser_rotation_search",
    # Types (validated engine)
    "AdaptiveRotationFunction",
    "BesselSHCoefficients",
    "RotationPeak",
    "WignerContraction",
]
