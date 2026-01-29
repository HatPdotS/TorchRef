"""
Atomic scattering factor functions.

This submodule provides functions for computing atomic scattering factors
using the ITC92 parameterization.

Two approaches are available:
1. Table-based lookup (recommended): Fast, vectorized, no gemmi dependency at runtime
2. Runtime gemmi calls (legacy): Slower, requires gemmi

Example using table lookup::

    from torchref.base.scattering import (
        load_scattering_table,
        get_scattering_params_by_z,
        elements_to_z,
    )

    z = elements_to_z(['C', 'N', 'O'])
    A, B = get_scattering_params_by_z(z)
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

from .scattering_table import (
    load_scattering_table,
    get_scattering_params_by_z,
    get_element_to_z_mapping,
    get_z_to_element_mapping,
    get_scattering_params_for_ion,
    elements_to_z,
)

__all__ = [
    # Legacy gemmi-based functions
    "get_scattering_factors_unique",
    "get_scattering_factors",
    "get_scattering_itc92",
    "calc_scattering_factors_paramtetrization",
    "get_parameterization",
    "get_parameterization_extended",
    "get_parametrization_for_elements",
    "get_parametrization_atom",
    "linear_interpolation",
    # Table-based lookup (recommended)
    "load_scattering_table",
    "get_scattering_params_by_z",
    "get_element_to_z_mapping",
    "get_z_to_element_mapping",
    "get_scattering_params_for_ion",
    "elements_to_z",
]
