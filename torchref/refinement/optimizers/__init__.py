"""
Optimizers for crystallographic refinement.

This module provides custom optimizers:
- AdamWithAdaptiveNoise: Adam with scale-invariant noise injection
- ExploratoryLBFGS: LBFGS with automatic landscape exploration via Lanczos
- LangevinSA: BAOAB Langevin dynamics with simulated annealing
- MomentumStochasticSA: Phenix-style SA (gradient descent + momentum + noise)
- SeededLBFGS: L-BFGS whose first step is a diagonal-Newton step
"""

from .adam_noise import AdamWithAdaptiveNoise
from .curvature import (
    hessian_diagonal,
    hessian_diagonal_preconditioner,
    preconditioner_from_diagonal,
)
from .exploratory_lbfgs import ExploratoryLBFGS
from .langevin_sa import LangevinSA
from .momentum_sa import MomentumStochasticSA
from .seeded_lbfgs import SeededLBFGS


__all__ = [
    "AdamWithAdaptiveNoise",
    "ExploratoryLBFGS",
    "LangevinSA",
    "MomentumStochasticSA",
    "SeededLBFGS",
    "hessian_diagonal",
    "hessian_diagonal_preconditioner",
    "preconditioner_from_diagonal",
]
