"""
Structure factor scaling module for TorchRef.

This module provides classes for scaling calculated structure factors
to match observed data, including:
- Overall and anisotropic scale factors
- Bulk solvent contribution modeling

Classes
-------
Scaler
    Computes overall and anisotropic scale factors between F_calc and F_obs.
SolventModel
    Models bulk solvent contribution to structure factors using
    flat solvent model with k_sol and B_sol parameters.

Example
-------
>>> from torchref.scaling import Scaler, SolventModel
>>>
>>> # Scale structure factors
>>> scaler = Scaler(device='cuda')
>>> F_calc_scaled = scaler(F_calc, F_obs, s_squared)
>>>
>>> # Add bulk solvent contribution
>>> solvent = SolventModel(device='cuda')
>>> F_calc_total = solvent(F_calc, F_mask, s_squared)
"""

from torchref.scaling.scaler import Scaler
from torchref.scaling.solvent_new import SolventModel

__all__ = [
    'Scaler',
    'SolventModel',
]
