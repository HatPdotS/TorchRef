"""
Utilities for determining FFT-compatible grid sizes for crystallographic symmetry.

For interpolation-free symmetry expansion, grid dimensions must be compatible
with the symmetry operations. Specifically:
- Screw axes require specific divisibility constraints
- Grid sizes should also be FFT-friendly (factors of 2, 3, 5)

This module provides the canonical implementations of FFT-friendly grid utilities.
The spacegroup module imports and re-exports these functions for convenience.
"""

import numpy as np
import torch

from torchref.config import NYQUIST_OVERSAMPLING


def get_symmetry_grid_requirements(space_group: str) -> dict:
    """
    Get grid size requirements for a given space group.

    This is a convenience wrapper around spacegroup.get_grid_requirements().

    Returns a dict with keys 'nx_mod', 'ny_mod', 'nz_mod' indicating
    the required divisibility for each axis.

    Parameters
    ----------
    space_group : str
        Space group symbol (e.g., 'P21', 'P212121', 'P41', etc.).

    Returns
    -------
    dict
        {'nx_mod': int, 'ny_mod': int, 'nz_mod': int}
        Required divisibility for each axis.
    """
    # Import here to avoid circular imports
    from torchref.symmetry.spacegroup import get_grid_requirements

    return get_grid_requirements(space_group)


def find_fft_friendly_size(n: int, divisibility: int = 1) -> int:
    """
    Find the nearest FFT-friendly size >= n that satisfies divisibility constraint.

    FFT-friendly means factors only of 2, 3, and 5 (radix-2,3,5 FFT algorithms).

    Parameters
    ----------
    n : int
        Minimum grid size.
    divisibility : int, default 1
        Required divisibility (e.g., 2 for screw axes).

    Returns
    -------
    int
        Optimal grid size.
    """
    # Start from n and search upward
    candidate = n

    # Make sure it satisfies divisibility
    if candidate % divisibility != 0:
        candidate = ((candidate // divisibility) + 1) * divisibility

    # Now find nearest FFT-friendly size
    while not is_fft_friendly(candidate):
        candidate += divisibility

    return candidate


def is_fft_friendly(n: int) -> bool:
    """
    Check if a number has only factors of 2, 3, and 5.

    These are optimal for radix-2,3,5 FFT algorithms.
    """
    if n <= 0:
        return False

    # Remove all factors of 2, 3, 5
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5

    # If we're left with 1, the number is FFT-friendly
    return n == 1


def calculate_optimal_grid_size(cell_params, max_res: float, space_group: str) -> tuple:
    """
    Calculate optimal grid size for a given unit cell and space group.

    Grid sizes are chosen to:

    1. Satisfy Shannon-Nyquist sampling (``NYQUIST_OVERSAMPLING`` × relative
       to max_res; see :data:`torchref.config.NYQUIST_OVERSAMPLING`)
    2. Respect symmetry requirements (screw axis divisibility)
    3. Be FFT-friendly (factors of 2, 3, 5 only)

    Parameters
    ----------
    cell_params : array-like, shape (6,)
        Unit cell [a, b, c, alpha, beta, gamma].
    max_res : float
        Maximum resolution in Angstroms.
    space_group : str
        Space group symbol.

    Returns
    -------
    tuple
        Optimal grid dimensions (nx, ny, nz).
    """
    # Import here to avoid circular imports
    from torchref.symmetry.spacegroup import suggest_grid_size

    if isinstance(cell_params, torch.Tensor):
        cell_params = cell_params.cpu().numpy()

    a, b, c = cell_params[:3]

    # Shannon-Nyquist: sample at NYQUIST_OVERSAMPLING × the maximum frequency
    nx_min = int(np.floor(a / max_res * NYQUIST_OVERSAMPLING))
    ny_min = int(np.floor(b / max_res * NYQUIST_OVERSAMPLING))
    nz_min = int(np.floor(c / max_res * NYQUIST_OVERSAMPLING))

    # Use spacegroup module to suggest optimal size
    return suggest_grid_size((nx_min, ny_min, nz_min), space_group, make_fft_friendly=True)


def check_grid_compatibility(grid_shape: tuple, space_group: str) -> dict:
    """
    Check if a grid is compatible with the space group symmetry.

    This is a convenience wrapper around spacegroup.check_grid_compatibility().

    Parameters
    ----------
    grid_shape : tuple
        Grid dimensions (nx, ny, nz).
    space_group : str
        Space group symbol.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - 'compatible' : bool
        - 'issues' : list of str (description of problems)
        - 'requirements' : dict (required divisibility)
        - 'can_use_direct_indexing' : bool (True if interpolation not needed)
        - 'fft_friendly' : bool
    """
    # Import here to avoid circular imports
    from torchref.symmetry.spacegroup import (
        check_grid_compatibility as sg_check_grid_compatibility,
    )

    return sg_check_grid_compatibility(grid_shape, space_group)


def recommend_grid_size(current_shape: tuple, space_group: str) -> tuple:
    """
    Recommend a nearby compatible grid size.

    Parameters
    ----------
    current_shape : tuple
        Current (nx, ny, nz).
    space_group : str
        Space group symbol.

    Returns
    -------
    tuple
        Recommended (nx, ny, nz).
    """
    # Import here to avoid circular imports
    from torchref.symmetry.spacegroup import suggest_grid_size

    return suggest_grid_size(current_shape, space_group, make_fft_friendly=True)
