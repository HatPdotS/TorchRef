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
