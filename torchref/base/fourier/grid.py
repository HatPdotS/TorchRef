"""Real-space grid generation.

Grid points sit at cell edges ``i / N`` (CCTBX/gemmi convention). The caller
supplies the dimensions; :meth:`~torchref.symmetry.cell.Cell.compute_grid_size` is
the sizing rule.
"""

import torch

from torchref.config import dtypes, get_default_device
from torchref.base.coordinates.transforms_torch import (
    fractional_to_cartesian_torch,
    get_fractional_matrix,
)


def get_real_grid(cell=None, fractional_matrix=None, *, gridsize, device=None):
    """
    Generate a real space grid for electron density calculations.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    fractional_matrix : torch.Tensor, optional
        Pre-computed fractionalization matrix.
    gridsize : torch.Tensor or array-like
        Grid dimensions [nx, ny, nz], e.g. from
        :meth:`~torchref.symmetry.cell.Cell.compute_grid_size`.
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
    else:
        nsteps = torch.tensor(gridsize, dtype=dtypes.int, device=device)
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
