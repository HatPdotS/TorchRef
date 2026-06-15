"""Monolithic, macrocycle-free refinement (EXPERIMENTAL).

Refines the structure against a Read-MLF Rice likelihood whose model-error
variance is the *differentiable* Fisher ``sigma_m**2`` (from the atomic B-factor
distribution) with a small co-refined calibration ``c`` — so the error model is a
live function of the model, optimized jointly in one LBFGS step with no free-set
``beta`` estimation and no macrocycle.

Key Classes
-----------
RiceSigmaMXrayTarget
    Read-MLF Rice target with ``beta = c * sigma_m**2`` (differentiable, co-refined).
MonolithicRefinement
    LBFGSRefinement that wires the target and exposes ``refine_monolithic``.
    Uses the differentiable density-derived bulk solvent by default
    (``use_density_solvent=True``).
DensitySolventModel
    Differentiable bulk solvent derived from the model electron density.
DensitySolventScaler
    Scaler subclass that drives the solvent from ``DensitySolventModel`` (live,
    differentiable) instead of the static vdW mask.
"""

from torchref.experimental.monolithic_refinement.refinement import MonolithicRefinement
from torchref.experimental.monolithic_refinement.targets import RiceSigmaMXrayTarget
from torchref.experimental.monolithic_refinement.density_solvent import (
    DensitySolventModel,
)
from torchref.experimental.monolithic_refinement.density_scaler import (
    DensityDerivedSolvent,
    DensitySolventScaler,
)

__all__ = [
    "MonolithicRefinement",
    "RiceSigmaMXrayTarget",
    "DensitySolventModel",
    "DensityDerivedSolvent",
    "DensitySolventScaler",
]
