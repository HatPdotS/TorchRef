"""Real-space and reciprocal-space grid generation.

Grid points sit at cell edges ``i / N`` (CCTBX/gemmi convention). Unless an
explicit ``gridsize`` is given, every helper here sizes as
``floor(cell[:3] / max_res * NYQUIST_OVERSAMPLING)`` -- the same Shannon-Nyquist
factor used by :meth:`~torchref.symmetry.cell.Cell.compute_grid_size`, so changing
:data:`torchref.config.NYQUIST_OVERSAMPLING` moves all of them together.
"""

import numpy as np
import torch

from torchref.config import NYQUIST_OVERSAMPLING, dtypes, get_default_device
from torchref.base.coordinates.transforms_torch import (
    fractional_to_cartesian_torch,
    get_fractional_matrix,
)


def get_real_grid(cell=None, fractional_matrix=None, max_res=0.8, gridsize=None, device=None):
    """
    Generate a real space grid for electron density calculations.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    fractional_matrix : torch.Tensor, optional
        Pre-computed fractionalization matrix.
    max_res : float, optional
        Maximum resolution for automatic grid sizing. Default is 0.8.
    gridsize : torch.Tensor or array-like, optional
        Explicit grid dimensions [nx, ny, nz]. If None, calculated from max_res.
    device : torch.device or str, optional
        Device for tensor placement. If None, inferred from ``fractional_matrix``
        or ``cell`` (whichever tensor is provided); falls back to CPU.

    Returns
    -------
    torch.Tensor
        Real space grid of shape (nx, ny, nz, 3) containing Cartesian coordinates.
    """
    if device is None:
        if isinstance(fractional_matrix, torch.Tensor):
            device = fractional_matrix.device
        elif isinstance(cell, torch.Tensor):
            device = cell.device
        else:
            device = get_default_device()

    if isinstance(gridsize, torch.Tensor):
        nsteps = gridsize.to(dtypes.int).to(device)
    elif gridsize is not None:
        nsteps = torch.tensor(gridsize, dtype=dtypes.int, device=device)
    else:
        nsteps = (
            torch.floor(cell[:3] / max_res * NYQUIST_OVERSAMPLING)
            .to(dtypes.int)
            .to(device)
        )
    x = torch.arange(nsteps[0], device=device, dtype=dtypes.float) / nsteps[0]
    y = torch.arange(nsteps[1], device=device, dtype=dtypes.float) / nsteps[1]
    z = torch.arange(nsteps[2], device=device, dtype=dtypes.float) / nsteps[2]
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = torch.cat((x, y, z), axis=3).reshape(-1, 3)
    cell_float = (
        cell.to(device=device, dtype=dtypes.float) if cell is not None else None
    )
    frac_matrix_float = (
        fractional_matrix.to(device=device, dtype=dtypes.float)
        if fractional_matrix is not None
        else None
    )
    xyz_real_grid = fractional_to_cartesian_torch(xyz, cell_float, frac_matrix_float)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    return xyz_real_grid


def find_grid_size(cell: torch.Tensor, max_res: float):
    """
    Calculate grid size based on unit cell and resolution.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    max_res : float
        Maximum resolution in Angstroms.

    Returns
    -------
    torch.Tensor
        Grid dimensions [nx, ny, nz] as int32, per the module-level sizing rule.
    """
    return torch.floor(cell[:3] / max_res * NYQUIST_OVERSAMPLING).to(dtypes.int)


def put_hkl_on_grid(real_space_grid, diff, hkl):
    """
    Place structure factors on a zero-filled reciprocal space grid.

    Parameters
    ----------
    real_space_grid : numpy.ndarray
        Only its leading three dimensions are used, to size the output.
    diff : numpy.ndarray
        Complex structure factor values to place.
    hkl : numpy.ndarray
        Miller indices, shape (N, 3), used directly as numpy indices -- negative
        h wraps to the end of the axis, giving FFT layout, but an index whose
        magnitude exceeds the grid raises rather than aliasing.

    Returns
    -------
    numpy.ndarray
        Complex reciprocal space grid with shape (nx, ny, nz).
    """
    rec_space = np.zeros(real_space_grid.shape[:3], dtype=np.complex128)
    f = diff
    rec_space[hkl[:, 0], hkl[:, 1], hkl[:, 2]] = f
    return rec_space
