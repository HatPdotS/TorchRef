"""Atomic scattering factors from the ITC92 parameterization.

Table-based lookup (:func:`get_scattering_params_by_z` after
:func:`elements_to_z`) is the recommended path -- vectorized and gemmi-free at
runtime. Anomalous f'/f'' corrections come from ``anomalous_table``.
"""

from .scattering_table import (
    load_scattering_table,
    get_scattering_params_by_z,
    get_element_to_z_mapping,
    get_z_to_element_mapping,
    get_scattering_params_for_ion,
    elements_to_z,
)

from .anomalous_table import (
    get_anomalous_correction,
    get_significant_elements,
    get_anomalous_corrections_by_indices,
)

__all__ = [
    # Table-based lookup (recommended)
    "load_scattering_table",
    "get_scattering_params_by_z",
    "get_element_to_z_mapping",
    "get_z_to_element_mapping",
    "get_scattering_params_for_ion",
    "elements_to_z",
    # Anomalous scattering (f' and f'')
    "get_anomalous_correction",
    "get_significant_elements",
    "get_anomalous_corrections_by_indices",
]
