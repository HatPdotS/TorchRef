"""
PyTorch implementations of coordinate transformation functions.

These functions are GPU-accelerated and support automatic differentiation
for use in optimization and refinement.

:func:`get_fractional_matrix` is the one place the cell metric is written out; the
reciprocal basis (:func:`~torchref.base.reciprocal.basis.reciprocal_basis_matrix`) and
:attr:`torchref.symmetry.Cell.volume` are derived from it.
"""

import torch


def cartesian_to_fractional_torch(xyz, cell, B_inv=None):
    """
    Convert Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    xyz : torch.Tensor
        Cartesian coordinates of shape (N, 3).
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] of shape (6,), lengths in
        Angstroms and angles in degrees. The matrix derived from it takes ``xyz``'s
        dtype and device.
    B_inv : torch.Tensor, optional
        Fractionalization matrix B^-1 (Cartesian -> fractional); from cell if None.

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
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma] of shape (6,), lengths in
        Angstroms and angles in degrees. The matrix derived from it takes
        ``xyz_fractional``'s dtype and device.
    B : torch.Tensor, optional
        Orthogonalization matrix B (fractional -> Cartesian); from cell if None.

    Returns
    -------
    torch.Tensor
        Cartesian coordinates of shape (N, 3).
    """
    if B is None:
        B = get_fractional_matrix(cell).to(
            dtype=xyz_fractional.dtype, device=xyz_fractional.device
        )
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
        3x3 upper-triangular matrix B such that cart = frac @ B.T, in the PDB
        orientation (a along x, b in the xy plane), on ``cell``'s dtype and device.
        Differentiable in ``cell``.
    """
    a, b, c = cell[0], cell[1], cell[2]
    alpha, beta, gamma = torch.deg2rad(cell[3:])
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    volume_factor = torch.sqrt(
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    zero = torch.zeros_like(a)
    return torch.stack(
        [
            torch.stack([a, b * cos_gamma, c * cos_beta]),
            torch.stack(
                [zero, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma]
            ),
            torch.stack([zero, zero, c * volume_factor / sin_gamma]),
        ]
    )


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
