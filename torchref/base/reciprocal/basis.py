"""
Reciprocal space basis matrix calculations.

Functions for computing reciprocal lattice vectors and scattering vectors
from unit cell parameters.
"""

import torch

from torchref.base.coordinates.transforms_torch import get_inv_fractional_matrix_torch


def reciprocal_basis_matrix(cell: torch.Tensor):
    """
    Compute the reciprocal space basis matrix from unit cell parameters.

    The rows of the Cartesian-to-fractional matrix are a*, b*, c*, so this is
    the inverse of
    :func:`~torchref.base.coordinates.transforms_torch.get_fractional_matrix` and
    shares its cell metric.

    Parameters
    ----------
    cell : torch.Tensor
        Cell parameters [a, b, c, alpha, beta, gamma], shape (6,), where
        lengths are in Angstroms and angles in degrees. Single cell only
        (not batched).

    Returns
    -------
    torch.Tensor
        Reciprocal basis matrix of shape (3, 3) with a*, b*, c* as rows, in Å⁻¹.
    """
    return get_inv_fractional_matrix_torch(cell)


def get_scattering_vectors(hkl: torch.Tensor, cell: torch.Tensor, recB=None):
    """
    Calculate scattering vectors from Miller indices.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N, 3).
    cell : torch.Tensor
        Cell parameters [a, b, c, alpha, beta, gamma].
    recB : torch.Tensor, optional
        Pre-computed reciprocal basis matrix of shape (3, 3).

    Returns
    -------
    torch.Tensor
        Scattering vectors of shape (N, 3).
    """
    if recB is None:
        recB = reciprocal_basis_matrix(cell)
    s = torch.matmul(hkl.to(cell.dtype), recB)
    return s
