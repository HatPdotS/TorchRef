"""
Grid generation functions for crystallographic calculations.

Functions for creating real-space and reciprocal-space grids.
"""

import numpy as np
import torch

from torchref.config import dtypes, get_default_device
from torchref.base.coordinates.transforms_torch import (
    fractional_to_cartesian_torch,
    get_fractional_matrix,
)
from torchref.base.coordinates.transforms_numpy import (
    fractional_to_cartesian,
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
        nsteps = torch.floor(cell[:3] / max_res * 3).to(dtypes.int).to(device)
    # Place grid points at grid edges: i / N (CCTBX convention)
    # This matches how CCTBX/gemmi create maps
    x = torch.arange(nsteps[0], device=device, dtype=dtypes.float) / nsteps[0]
    y = torch.arange(nsteps[1], device=device, dtype=dtypes.float) / nsteps[1]
    z = torch.arange(nsteps[2], device=device, dtype=dtypes.float) / nsteps[2]
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = torch.cat((x, y, z), axis=3).reshape(-1, 3)
    # Ensure consistent dtype and device for fractional_to_cartesian_torch
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
        Grid dimensions [nx, ny, nz] as int32.
    """
    return torch.floor(cell[:3] / max_res * 2.3).to(dtypes.int)


def get_real_grid_numpy(cell, max_res=0.8, gridsize=None):
    """
    Generate a real-space grid of Cartesian coordinates (NumPy version).

    Creates a 3D grid in fractional coordinates and converts it to Cartesian
    coordinates. Grid points are placed at cell edges following CCTBX convention.

    Parameters
    ----------
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.
    max_res : float, optional
        Maximum resolution in Angstroms for grid spacing. Default is 0.8.
        Ignored if gridsize is provided.
    gridsize : list or numpy.ndarray, optional
        Explicit grid dimensions [nx, ny, nz]. If provided, overrides max_res.

    Returns
    -------
    numpy.ndarray
        Real-space grid coordinates with shape (nx, ny, nz, 3).
    """
    if gridsize is not None:
        nsteps = np.array(gridsize, dtype=int)
    else:
        nsteps = np.astype(np.floor(cell[:3] / max_res * 3), int)
    # Place grid points at grid edges: i / N (CCTBX convention)
    # This matches how CCTBX/gemmi create maps
    x = np.arange(nsteps[0]) / nsteps[0]
    y = np.arange(nsteps[1]) / nsteps[1]
    z = np.arange(nsteps[2]) / nsteps[2]
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = np.concatenate((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_grid = fractional_to_cartesian(xyz, cell)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    return xyz_real_grid


def get_grids(cell, max_res=0.8):
    """
    Generate real-space and reciprocal-space grids for Fourier transforms.

    Creates a 3D grid in fractional coordinates and converts it to Cartesian
    coordinates, along with an empty reciprocal space grid.

    Parameters
    ----------
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.
    max_res : float, optional
        Maximum resolution in Angstroms for grid spacing. Default is 0.8.

    Returns
    -------
    recgrid : numpy.ndarray
        Empty reciprocal space grid with shape determined by resolution.
    xyz_real_grid : numpy.ndarray
        Real-space grid coordinates with shape (nx, ny, nz, 3).
    """
    nsteps = np.astype(np.floor(cell[:3] / max_res * 3), int)
    x = np.arange(nsteps[0]) / nsteps[0]
    y = np.arange(nsteps[1]) / nsteps[1]
    z = np.arange(nsteps[2]) / nsteps[2]
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = np.concatenate((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_grid = fractional_to_cartesian(xyz, cell)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    recgrid = np.zeros(array_shape, dtype=float)
    return recgrid, xyz_real_grid


def put_hkl_on_grid(real_space_grid, diff, hkl):
    """
    Place structure factors on a reciprocal space grid.

    Maps structure factor values to their corresponding positions on a
    3D reciprocal space grid based on Miller indices.

    Parameters
    ----------
    real_space_grid : numpy.ndarray
        Real-space grid used to determine the reciprocal grid dimensions.
        Shape should be (nx, ny, nz, 3) or similar.
    diff : numpy.ndarray
        Structure factor values (complex) to place on the grid.
    hkl : numpy.ndarray
        Miller indices with shape (N, 3), used as grid indices.

    Returns
    -------
    numpy.ndarray
        Complex reciprocal space grid with shape (nx, ny, nz).
    """
    rec_space = np.zeros(real_space_grid.shape[:3], dtype=np.complex128)
    f = diff
    rec_space[hkl[:, 0], hkl[:, 1], hkl[:, 2]] = f
    return rec_space
