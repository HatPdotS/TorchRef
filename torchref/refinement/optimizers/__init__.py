"""
Optimizers for crystallographic refinement.

This module provides custom optimizers and optimization functions:
- AdamWithAdaptiveNoise: Adam with scale-invariant noise injection
- optimize_simulated_annealing: Simulated annealing optimization
- optimize_stochastic_sa: Stochastic SA for internal coordinates (per-parameter)
- optimize_stochastic_sa_batch: Stochastic SA for internal coordinates (batch)
- optimize_internal_coord_sa: Universal SA for internal coordinates with auto-calibration
- optimize_gradient_sa: Gradient-based SA with per-parameter acceptance
- refine_sa_lbfgs: Combined Metropolis SA + LBFGS pipeline
- optimize_momentum_sa: Phenix-style SA (gradient descent + momentum + noise)
- refine_momentum_sa_lbfgs: Combined Phenix-style SA + LBFGS pipeline
"""

from .adam_noise import AdamWithAdaptiveNoise
from .momentum_sa import MomentumStochasticSA


__all__ = [
    "AdamWithAdaptiveNoise",
    "MomentumStochasticSA",

]
