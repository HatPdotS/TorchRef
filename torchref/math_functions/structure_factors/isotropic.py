"""
Isotropic structure factor calculations.

Functions for computing structure factors from atomic models with
isotropic (spherical) atomic displacement parameters.
"""

import numpy as np
import torch


def iso_structure_factor_torched(
    hkl, s, xyz_fractional, occ, scattering_factors, adp, spacegroup
):
    """
    Calculate isotropic structure factors using PyTorch.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N_reflections, 3).
    s : torch.Tensor
        Scattering vector magnitudes of shape (N_reflections,).
    xyz_fractional : torch.Tensor
        Fractional coordinates of shape (N_atoms, 3).
    occ : torch.Tensor
        Occupancies of shape (N_atoms,).
    scattering_factors : torch.Tensor
        Atomic scattering factors of shape (N_reflections, N_atoms).
    adp : torch.Tensor
        Atomic displacement parameters (isotropic) of shape (N_atoms,).
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
    adp = adp.reshape(1, -1)
    s = s.reshape(-1, 1)
    B = -adp * (s**2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot), axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))


def iso_structure_factor_torched_no_complex(
    hkl, s, fractional_coords, occ, scattering_factors, tempfactor, space_group
):
    """
    Calculate isotropic structure factors without complex numbers.

    Returns real and imaginary parts as separate rows.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N_reflections, 3).
    s : torch.Tensor
        Scattering vector magnitudes of shape (N_reflections,).
    fractional_coords : torch.Tensor
        Fractional coordinates of shape (N_atoms, 3).
    occ : torch.Tensor
        Occupancies of shape (N_atoms,).
    scattering_factors : torch.Tensor
        Atomic scattering factors of shape (N_reflections, N_atoms).
    tempfactor : torch.Tensor
        Isotropic temperature factors (B-factors) of shape (N_atoms,).
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
    tempfactor = tempfactor.reshape(1, -1)
    s = s.reshape(-1, 1)
    B = -tempfactor * (s**2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    complex_part = torch.sum(torch.sum(torch.sin(pidot), axis=-1) * terms, axis=1)
    real_part = torch.sum(torch.sum(torch.cos(pidot), axis=-1) * terms, axis=1)
    return torch.vstack((real_part, complex_part))
