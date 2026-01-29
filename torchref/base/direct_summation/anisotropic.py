"""
Anisotropic structure factor calculations.

Functions for computing structure factors from atomic models with
anisotropic (non-spherical) atomic displacement parameters.
"""

import numpy as np
import torch


def aniso_structure_factor_torched(
    hkl, s_vector, xyz_fractional, occ, scattering_factors, U, spacegroup
):
    """
    Calculate anisotropic structure factors using PyTorch.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N_reflections, 3).
    s_vector : torch.Tensor
        Scattering vectors of shape (N_reflections, 3).
    xyz_fractional : torch.Tensor
        Fractional coordinates of shape (N_atoms, 3).
    occ : torch.Tensor
        Occupancies of shape (N_atoms,).
    scattering_factors : torch.Tensor
        Atomic scattering factors of shape (N_reflections, N_atoms).
    U : torch.Tensor
        Anisotropic displacement parameters of shape (N_atoms, 6).
    spacegroup : callable
        Space group symmetry operator function.

    Returns
    -------
    torch.Tensor
        Complex structure factors of shape (N_reflections,).
    """
    xyz_fractional = spacegroup(xyz_fractional.T)
    fractional_shape = xyz_fractional.shape
    xyz_fractional = xyz_fractional.reshape(3, -1)
    dot_product = torch.matmul(hkl.to(torch.float64), xyz_fractional).reshape(
        hkl.shape[0], fractional_shape[1], -1
    )
    U_row1 = torch.stack([U[:, 0], U[:, 3], U[:, 4]], dim=0)
    U_row2 = torch.stack([U[:, 3], U[:, 1], U[:, 5]], dim=0)
    U_row3 = torch.stack([U[:, 4], U[:, 5], U[:, 2]], dim=0)
    U_matrix = torch.stack([U_row1, U_row2, U_row3], dim=0)
    U_dot_s = torch.einsum("jik,li->jkl", U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum("li,ikl->lk", s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) * StUS
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot), axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))


def aniso_structure_factor_torched_no_complex(
    hkl, s_vector, fractional_coords, occ, scattering_factors, U, space_group
):
    """
    Calculate anisotropic structure factors without complex numbers.

    Returns real and imaginary parts as separate rows.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N_reflections, 3).
    s_vector : torch.Tensor
        Scattering vectors of shape (N_reflections, 3).
    fractional_coords : torch.Tensor
        Fractional coordinates of shape (N_atoms, 3).
    occ : torch.Tensor
        Occupancies of shape (N_atoms,).
    scattering_factors : torch.Tensor
        Atomic scattering factors of shape (N_reflections, N_atoms).
    U : torch.Tensor
        Anisotropic displacement parameters of shape (N_atoms, 6).
    space_group : callable
        Space group symmetry operator function.

    Returns
    -------
    torch.Tensor
        Structure factors as [real, imag] of shape (2, N_reflections).
    """
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3, -1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(
        hkl.shape[0], fractional_shape[1], -1
    )
    U_row1 = torch.stack([U[:, 0], U[:, 3], U[:, 4]], dim=0)
    U_row2 = torch.stack([U[:, 3], U[:, 1], U[:, 5]], dim=0)
    U_row3 = torch.stack([U[:, 4], U[:, 5], U[:, 2]], dim=0)
    U_matrix = torch.stack([U_row1, U_row2, U_row3], dim=0)
    U_dot_s = torch.einsum("jik,li->jkl", U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum("li,ikl->lk", s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) * StUS
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    complex_part = torch.sum(torch.sum(torch.sin(pidot), axis=-1) * terms, axis=1)
    real_part = torch.sum(torch.sum(torch.cos(pidot), axis=-1) * terms, axis=1)
    return torch.vstack((real_part, complex_part))
