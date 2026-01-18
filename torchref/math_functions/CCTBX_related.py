"""
CCTBX/iotbx related utilities for crystallographic calculations.

This module requires the iotbx package from CCTBX, which must be installed via conda:
    conda install -c conda-forge cctbx-base

Functions in this module will raise ImportError if iotbx is not available.
"""
import numpy as np

try:
    from iotbx import pdb
    IOTBX_AVAILABLE = True
except ImportError:
    IOTBX_AVAILABLE = False
    pdb = None


def calculate_scattering_factor_cctbx(pdb_file, hkls=None, d_min=2.0):
    """
    Calculate structure factors using CCTBX/iotbx.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file.
    hkls : array-like, optional
        HKL indices to filter results.
    d_min : float, optional
        Minimum d-spacing resolution (default: 2.0).

    Returns
    -------
    tuple
        (f_calc, idx) - calculated structure factors and their HKL indices.

    Raises
    ------
    ImportError
        If iotbx is not installed.
    """
    if not IOTBX_AVAILABLE:
        raise ImportError(
            "iotbx is required for this function but is not installed. "
            "Install via conda: conda install -c conda-forge cctbx-base"
        )

    pdb_input = pdb.input(file_name=pdb_file)
    xray_structure = pdb_input.xray_structure_simple()
    f_calc = xray_structure.structure_factors(d_min=d_min).f_calc()
    idx = np.array(f_calc.indices())
    f_calc = np.array(f_calc.data())
    if hkls is not None:
        hkls = np.array(hkls, dtype=int)
        hkls = set([tuple(hkl) for hkl in hkls])
        f_calc_new = []
        idx_new = []
        for hkl, val in zip(idx, f_calc):
            if tuple(hkl) in hkls:
                f_calc_new.append(val)
                idx_new.append(hkl)
        f_calc = np.array(f_calc_new)
        idx = np.array(idx_new)
    return f_calc, idx
