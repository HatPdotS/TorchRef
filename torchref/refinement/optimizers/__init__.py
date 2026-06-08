"""
Optimizers for crystallographic refinement.

This module provides custom optimizers:
- AdamWithAdaptiveNoise: Adam with scale-invariant noise injection
- ExploratoryLBFGS: LBFGS with automatic landscape exploration via Lanczos
- LangevinSA: BAOAB Langevin dynamics with simulated annealing
- MomentumStochasticSA: Phenix-style SA (gradient descent + momentum + noise)
"""

from .adam_noise import AdamWithAdaptiveNoise
from .exploratory_lbfgs import ExploratoryLBFGS
from .langevin_sa import LangevinSA
from .momentum_sa import MomentumStochasticSA


__all__ = [
    "AdamWithAdaptiveNoise",
    "ExploratoryLBFGS",
    "LangevinSA",
    "MomentumStochasticSA",
]
