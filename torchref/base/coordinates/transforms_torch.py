"""
PyTorch implementations of coordinate transformation functions.

These functions are GPU-accelerated and support automatic differentiation
for use in optimization and refinement.

.. note::
    ``get_fractional_matrix`` is an exception: it assembles its output via
    ``torch.tensor([...])``, which detaches from the autograd graph, so
    gradients do not flow back to the input ``cell`` parameters.
"""

import torch


def cartesian_to_fractional_torch(xyz, cell, B_inv=None):
    """
    Convert Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    xyz : torch.Tensor
        Cartesian coordinates of shape (N, 3).
    cell : array-like
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    B_inv: torch.Tensor, optional
        Inverse fractionalization matrix. If None, it will be calculated from cell.

    Returns
    -------
    torch.Tensor
        Fractional coordinates of shape (N, 3).
    """
    if B_inv is None:
        # Stay in torch — falling back to numpy here breaks cuda inputs.
        B_inv = get_inv_fractional_matrix_torch(cell).to(
            dtype=xyz.dtype, device=xyz.device
        )
    xyz_fractional = torch.einsum("ik,kj->ij", xyz, B_inv.T)
    return xyz_fractional


def fractional_to_cartesian_torch(xyz_fractional, cell, B=None):
    """
    Convert fractional coordinates to Cartesian coordinates.

    Parameters
    ----------
    xyz_fractional : torch.Tensor
        Fractional coordinates of shape (N, 3).
    cell : array-like
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    B: torch.Tensor, optional
        Fractionalization matrix. If None, it will be calculated from cell.

    Returns
    -------
    torch.Tensor
        Cartesian coordinates of shape (N, 3).
    """
    if B is None:
        B = get_fractional_matrix(cell)
    xyz = torch.einsum("ik,kj->ij", xyz_fractional, B.T)
    return xyz


def get_fractional_matrix(cell):
    """
    Calculate the fractional-to-Cartesian transformation matrix.

    Constructs the matrix B that transforms fractional coordinates to
    Cartesian coordinates based on the unit cell parameters.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    torch.Tensor
        3x3 transformation matrix B such that cart = frac @ B.T.

    Notes
    -----
    The matrix is assembled with ``torch.tensor([...])``, which detaches the
    result from the autograd graph. Gradients therefore do not propagate back
    to ``cell``; this function is non-differentiable in the cell parameters.
    """
    a, b, c = cell[:3]
    alpha, beta, gamma = torch.deg2rad(cell[3:])
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    volume = torch.sqrt(
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    B = torch.tensor(
        [
            [a, b * cos_gamma, c * cos_beta],
            [0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
            [0, 0, c * volume / sin_gamma],
        ], dtype=cell.dtype, device=cell.device
    )
    return B


def get_inv_fractional_matrix_torch(cell):
    """
    Calculate the Cartesian-to-fractional transformation matrix (PyTorch version).

    Computes the inverse of the fractional matrix for converting Cartesian
    coordinates to fractional coordinates.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    torch.Tensor
        3x3 inverse transformation matrix B_inv such that frac = cart @ B_inv.T.
    """
    B = get_fractional_matrix(cell)
    B_inv = torch.linalg.inv(B)
    return B_inv
