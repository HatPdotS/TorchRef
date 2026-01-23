"""
Atomic scattering factor functions.

This submodule provides functions for computing atomic scattering factors
using the ITC92 parameterization.
"""

from .itc92 import (
    get_scattering_factors_unique,
    get_scattering_factors,
    get_scattering_itc92,
    calc_scattering_factors_paramtetrization,
    get_parameterization,
    get_parameterization_extended,
    get_parametrization_for_elements,
    get_parametrization_atom,
    linear_interpolation,
)

__all__ = [
    "get_scattering_factors_unique",
    "get_scattering_factors",
    "get_scattering_itc92",
    "calc_scattering_factors_paramtetrization",
    "get_parameterization",
    "get_parameterization_extended",
    "get_parametrization_for_elements",
    "get_parametrization_atom",
    "linear_interpolation",
]
