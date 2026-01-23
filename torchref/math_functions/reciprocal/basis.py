"""
Reciprocal space basis matrix calculations.

Functions for computing reciprocal lattice vectors and scattering vectors
from unit cell parameters.
"""

import numpy as np
import torch


def reciprocal_basis_matrix(cell: torch.Tensor):
    """
    Compute the reciprocal space basis matrix from unit cell parameters.

    Parameters
    ----------
    cell : torch.Tensor
        Cell parameters [a, b, c, alpha, beta, gamma].

    Returns
    -------
    torch.Tensor
        Reciprocal basis matrix of shape (3, 3) with a*, b*, c* as rows.
    """
    # Extract cell parameters
    angles_rad = torch.deg2rad(cell[3:])
    # Compute real-space basis vectors
    angles_cos = torch.cos(angles_rad)
    cos_squared = angles_cos**2
    sin_gamma = torch.sin(angles_rad[2])
    volume = torch.sqrt(
        1
        - cos_squared[0]
        - cos_squared[1]
        - cos_squared[2]
        + 2 * cos_squared[0] * cos_squared[1] * cos_squared[2]
    )
    a_vec = torch.tensor(
        [cell[0], 0, 0], dtype=cell.dtype, device=cell.device
    )
    b_vec = torch.tensor(
        [cell[1] * angles_cos[2], cell[1] * sin_gamma, 0],
        dtype=cell.dtype,
        device=cell.device,
    )
    c_vec = torch.tensor(
        [
            cell[2] * angles_cos[1],
            cell[2] * (angles_cos[0] - angles_cos[1] * angles_cos[2]) / sin_gamma,
            cell[2] * volume / sin_gamma,
        ],
        dtype=cell.dtype,
        device=cell.device,
    )
    # Compute reciprocal basis vectors
    volume_real = torch.dot(a_vec, torch.linalg.cross(b_vec, c_vec))
    a_star = torch.linalg.cross(b_vec, c_vec) / volume_real
    b_star = torch.linalg.cross(c_vec, a_vec) / volume_real
    c_star = torch.linalg.cross(a_vec, b_vec) / volume_real
    # Assemble reciprocal basis matrix
    return torch.stack([a_star, b_star, c_star])


def reciprocal_basis_matrix_numpy(cell):
    """
    Calculate the reciprocal basis matrix from unit cell parameters (NumPy version).

    Computes the reciprocal space basis vectors (a*, b*, c*) that define
    the transformation from Miller indices to scattering vectors.

    Parameters
    ----------
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        3x3 matrix containing reciprocal basis vectors as rows [a*, b*, c*].
    """
    # Extract cell parameters
    a, b, c, alpha, beta, gamma = cell
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    # Compute real-space basis vectors
    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_gamma = np.sin(gamma)
    volume = np.sqrt(
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    a_vec = np.array([a, 0, 0])
    b_vec = np.array([b * cos_gamma, b * sin_gamma, 0])
    c_vec = np.array(
        [
            c * cos_beta,
            c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
            c * volume / sin_gamma,
        ]
    )
    # Compute reciprocal basis vectors
    volume_real = np.dot(a_vec, np.cross(b_vec, c_vec))
    a_star = np.cross(b_vec, c_vec) / volume_real
    b_star = np.cross(c_vec, a_vec) / volume_real
    c_star = np.cross(a_vec, b_vec) / volume_real
    # Assemble reciprocal basis matrix
    return np.array([a_star, b_star, c_star])


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


def get_scattering_vectors_numpy(hkl, cell):
    """
    Calculate scattering vectors from Miller indices and unit cell (NumPy version).

    Transforms Miller indices to reciprocal space scattering vectors
    using the reciprocal basis matrix.

    Parameters
    ----------
    hkl : numpy.ndarray or list
        Miller indices with shape (N, 3).
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Scattering vectors in reciprocal space with shape (N, 3).
    """
    recB = reciprocal_basis_matrix_numpy(cell)
    hkl = np.array(hkl)  # Ensure hkl is a numpy array
    s = np.dot(hkl, recB)
    return s


def get_s(hkl, cell):
    """
    Calculate the magnitude of scattering vectors for given Miller indices.

    Computes |s| = 1/d where d is the interplanar spacing for each reflection.

    Parameters
    ----------
    hkl : numpy.ndarray
        Miller indices with shape (N, 3).
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Magnitude of scattering vectors with shape (N,).
    """
    s = get_scattering_vectors_numpy(hkl, cell)
    s = np.sum(s**2, axis=1) ** 0.5
    return s
