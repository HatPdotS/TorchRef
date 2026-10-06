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
        Fractionalization matrix B^-1 (Cartesian -> fractional) of shape (3, 3).
    frac_matrix : torch.Tensor
        Orthogonalization matrix B (fractional -> Cartesian) of shape (3, 3).

    Returns
    -------
    torch.Tensor
        Squared distances with shape (...).
    """
    return smallest_diff_aniso(diff, inv_frac_matrix, frac_matrix).pow(2).sum(-1)


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
        Fractionalization matrix B^-1 (Cartesian -> fractional) of shape (3, 3).
    frac_matrix : torch.Tensor
        Orthogonalization matrix B (fractional -> Cartesian) of shape (3, 3).

    Returns
    -------
    torch.Tensor
        Signed minimum-image vectors with shape (..., 3), in Å; the anisotropic
        Gaussian needs the vector, not just its length.
    """
    diff_shape = diff.shape
    diff = diff.reshape(-1, 3)
    diff_frac = torch.matmul(inv_frac_matrix, diff.T)
    translation = torch.round(diff_frac)
    diff = diff - torch.matmul(frac_matrix, translation).T
    return diff.reshape(diff_shape)
