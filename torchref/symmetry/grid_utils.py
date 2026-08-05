"""FFT- and symmetry-compatible grid sizes.

Interpolation-free symmetry expansion needs grid dimensions divisible by what the
screw axes demand, and radix-2,3,5 FFTs want factors of 2, 3, 5 only.

These are thin wrappers over ``spacegroup``, which holds the canonical
implementations -- including its own ``is_fft_friendly`` /
``find_fft_friendly_size`` pair. Prefer ``spacegroup`` for new code.
"""

import numpy as np
import torch

from torchref.config import NYQUIST_OVERSAMPLING


def get_symmetry_grid_requirements(space_group: str) -> dict:
    """Per-axis divisibility ``{'nx_mod', 'ny_mod', 'nz_mod'}`` for ``space_group``.

    Wrapper over :func:`~torchref.symmetry.spacegroup.get_grid_requirements`.
    """
    # Import here to avoid circular imports
    from torchref.symmetry.spacegroup import get_grid_requirements

    return get_grid_requirements(space_group)


def find_fft_friendly_size(n: int, divisibility: int = 1) -> int:
    """Smallest size >= ``n`` factoring into 2, 3, 5 and divisible by ``divisibility``.

    Parameters
    ----------
    n : int
        Minimum grid size.
    divisibility : int, default 1
        Required divisibility (e.g. 2 for a screw axis).

    Returns
    -------
    int
        Optimal grid size.
    """
    candidate = n

    if candidate % divisibility != 0:
        candidate = ((candidate // divisibility) + 1) * divisibility

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
    Optimal grid for a unit cell and space group.

    Satisfies Shannon-Nyquist sampling at
    :data:`torchref.config.NYQUIST_OVERSAMPLING`, the screw-axis divisibility, and
    FFT-friendliness (factors of 2, 3, 5 only).

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
    """Check ``(nx, ny, nz)`` against the space group symmetry and the FFT.

    Wrapper over
    :func:`~torchref.symmetry.spacegroup.check_grid_compatibility`, which
    documents the report dict.
    """
    # Import here to avoid circular imports
    from torchref.symmetry.spacegroup import (
        check_grid_compatibility as sg_check_grid_compatibility,
    )

    return sg_check_grid_compatibility(grid_shape, space_group)


def recommend_grid_size(current_shape: tuple, space_group: str) -> tuple:
    """Smallest symmetry- and FFT-compatible grid at or above ``current_shape``."""
    # Import here to avoid circular imports
    from torchref.symmetry.spacegroup import suggest_grid_size

    return suggest_grid_size(current_shape, space_group, make_fft_friendly=True)
