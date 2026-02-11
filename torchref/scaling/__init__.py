"""
Structure factor scaling module for TorchRef.

This module provides classes for scaling calculated structure factors
to match observed data, including:
- Overall and anisotropic scale factors
- Bulk solvent contribution modeling

Classes
-------
ScalerBase
    Base scaler class that does not require a Model object.
    All methods that need F_calc take it as an input argument.
Scaler
    Full-featured scaler with Model integration.
    Extends ScalerBase with convenience methods that auto-compute F_calc.
SolventModel
    Models bulk solvent contribution to structure factors using
    flat solvent model with k_sol and B_sol parameters.

Example
-------
::

    from torchref.scaling import Scaler, ScalerBase, SolventModel

    # Using Scaler with a model (auto-computes F_calc)
    scaler = Scaler(model, data, nbins=20)
    scaler.initialize()
    F_calc_scaled = scaler(F_calc)

    # Using ScalerBase without a model (requires F_calc as input)
    scaler_base = ScalerBase(data=data, nbins=20)
    scaler_base.initialize(fcalc)
    F_calc_scaled = scaler_base(fcalc)
"""

from torchref.scaling.scaler import Scaler
from torchref.scaling.scaler_base import ScalerBase
from torchref.scaling.solvent import SolventModel

__all__ = [
    "Scaler",
    "ScalerBase",
    "SolventModel",
]
