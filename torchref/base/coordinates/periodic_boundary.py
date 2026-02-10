"""
Periodic boundary condition handling functions.

These functions compute minimum image distances with periodic boundary
conditions, essential for crystallographic calculations where atoms
wrap around the unit cell boundaries.
"""

import torch


def smallest_diff(
    diff: torch.Tensor, inv_frac_matrix: torch.Tensor, frac_matrix: torch.Tensor
):
    """
    Compute minimum image squared distances with periodic boundary conditions.

    Parameters
    ----------
    diff : torch.Tensor
        Difference vectors of shape (..., 3).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix of shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix of shape (3, 3).

    Returns
    -------
    torch.Tensor
        Squared distances with shape (...).
    """
    diff_shape = diff.shape
    diff = diff.reshape(-1, 3)
    diff_frac = torch.matmul(inv_frac_matrix, diff.T)
    translation = torch.round(diff_frac)
    diff = diff - torch.matmul(frac_matrix, translation).T
    return torch.sum(diff**2, axis=-1).reshape(diff_shape[:-1])


def smallest_diff_aniso(
    diff: torch.Tensor, inv_frac_matrix: torch.Tensor, frac_matrix: torch.Tensor
):
    """
    Compute minimum image difference vectors for anisotropic calculations.

    Parameters
    ----------
    diff : torch.Tensor
        Difference vectors of shape (..., 3).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix of shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix of shape (3, 3).

    Returns
    -------
    torch.Tensor
        Signed difference vectors with shape (..., 3).
        Note: For anisotropic calculations, the signed vectors are needed
        to correctly compute the quadratic form r^T × B^(-1) × r with
        off-diagonal U tensor terms.
    """
    diff_shape = diff.shape
    diff = diff.reshape(-1, 3)
    diff_frac = torch.matmul(inv_frac_matrix, diff.T)
    translation = torch.round(diff_frac)
    diff -= torch.matmul(frac_matrix, translation).T
    return diff.reshape(diff_shape)
