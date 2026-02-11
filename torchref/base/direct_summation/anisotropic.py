"""
Anisotropic structure factor calculations.

Functions for computing structure factors from atomic models with
anisotropic (non-spherical) atomic displacement parameters.
"""

from typing import Optional

import numpy as np
import torch


def _estimate_batch_size_aniso(n_refl: int, n_atoms: int, n_ops: int, max_memory_gb: float) -> int:
    """
    Estimate optimal batch size for anisotropic calculations.

    Anisotropic has additional memory for U matrix operations.
    Conservative estimate: ~80 bytes per (refl, atom, op) combination.
    """
    bytes_per_refl = n_atoms * n_ops * 80
    max_bytes = max_memory_gb * 1e9
    batch_size = max(1, int(max_bytes / bytes_per_refl))
    return min(batch_size, n_refl)


def aniso_structure_factor_torched(
    hkl, s_vector, xyz_fractional, occ, scattering_factors, U, spacegroup,
    max_memory_gb: Optional[float] = None,
    A: Optional[torch.Tensor] = None,
    B_coeff: Optional[torch.Tensor] = None,
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
    scattering_factors : torch.Tensor or None
        Atomic scattering factors of shape (N_reflections, N_atoms).
        If None, must provide A and B_coeff to compute them in batches.
    U : torch.Tensor
        Anisotropic displacement parameters of shape (N_atoms, 6).
    spacegroup : callable
        Space group symmetry operator function.
    max_memory_gb : float, optional
        Maximum memory to use in GB. If None, no batching is applied.
    A : torch.Tensor, optional
        ITC92 A coefficients (N_atoms, 5) for computing scattering factors.
    B_coeff : torch.Tensor, optional
        ITC92 B coefficients (N_atoms, 5) for computing scattering factors.

    Returns
    -------
    torch.Tensor
        Complex structure factors of shape (N_reflections,).
    """
    # Apply spacegroup to get symmetry-expanded coordinates
    xyz_expanded = spacegroup(xyz_fractional.T)  # (3, N_atoms, n_ops)
    fractional_shape = xyz_expanded.shape
    n_ops = fractional_shape[2] if len(fractional_shape) > 2 else 1
    n_atoms = fractional_shape[1]
    n_refl = hkl.shape[0]

    # Reshape for matrix multiplication
    xyz_flat = xyz_expanded.reshape(3, -1)  # (3, N_atoms * n_ops)

    # Precompute U matrix (doesn't depend on batch)
    U_row1 = torch.stack([U[:, 0], U[:, 3], U[:, 4]], dim=0)
    U_row2 = torch.stack([U[:, 3], U[:, 1], U[:, 5]], dim=0)
    U_row3 = torch.stack([U[:, 4], U[:, 5], U[:, 2]], dim=0)
    U_matrix = torch.stack([U_row1, U_row2, U_row3], dim=0)  # (3, 3, N_atoms)

    # Check if batching is needed
    if max_memory_gb is not None:
        batch_size = _estimate_batch_size_aniso(n_refl, n_atoms, n_ops, max_memory_gb)
        if batch_size < n_refl:
            return _aniso_sf_batched(
                hkl, s_vector, xyz_flat, fractional_shape, occ, scattering_factors,
                U_matrix, batch_size, A, B_coeff
            )
        # Batching not needed but scattering_factors may be None - compute them
        if scattering_factors is None and A is not None and B_coeff is not None:
            s_mag = torch.norm(s_vector, dim=1)
            scattering_factors = _compute_scattering_factors_batch(s_mag, A, B_coeff)

    # No batching - compute all at once
    dot_product = torch.matmul(hkl.to(torch.float64), xyz_flat).reshape(
        n_refl, n_atoms, -1
    )
    U_dot_s = torch.einsum("jik,li->jkl", U_matrix, s_vector)  # (3, N_atoms, N_refl)
    StUS = torch.einsum("li,ikl->lk", s_vector, U_dot_s)  # (N_refl, N_atoms)
    B = -2 * (np.pi**2) * StUS
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot), axis=-1)
    return torch.sum(terms * sin_cos, axis=1)


from torchref.base.direct_summation import (
    compute_scattering_factors_batch as _compute_scattering_factors_batch,
)


def _aniso_sf_batched(
    hkl, s_vector, xyz_flat, fractional_shape, occ, scattering_factors, U_matrix, batch_size,
    A=None, B_coeff=None
):
    """
    Compute anisotropic structure factors in batches over reflections.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices (N_refl, 3).
    s_vector : torch.Tensor
        Scattering vectors (N_refl, 3).
    xyz_flat : torch.Tensor
        Flattened fractional coordinates (3, N_atoms * n_ops).
    fractional_shape : tuple
        Original shape (3, N_atoms, n_ops).
    occ : torch.Tensor
        Occupancies (N_atoms,).
    scattering_factors : torch.Tensor or None
        Scattering factors (N_refl, N_atoms). If None, computed from A/B_coeff.
    U_matrix : torch.Tensor
        Precomputed U matrix (3, 3, N_atoms).
    batch_size : int
        Number of reflections per batch.
    A : torch.Tensor, optional
        ITC92 A coefficients for computing scattering factors.
    B_coeff : torch.Tensor, optional
        ITC92 B coefficients for computing scattering factors.

    Returns
    -------
    torch.Tensor
        Complex structure factors (N_refl,).
    """
    n_refl = hkl.shape[0]
    n_atoms = fractional_shape[1]
    device = hkl.device

    # Pre-allocate output tensor to avoid accumulating in list
    sf_out = torch.zeros(n_refl, dtype=torch.complex128, device=device)

    for start in range(0, n_refl, batch_size):
        end = min(start + batch_size, n_refl)

        # Slice inputs for this batch
        hkl_batch = hkl[start:end]
        s_batch = s_vector[start:end]

        # Get or compute scattering factors for this batch
        if scattering_factors is not None:
            sf_batch = scattering_factors[start:end]
        else:
            s_mag = torch.norm(s_batch, dim=1)
            sf_batch = _compute_scattering_factors_batch(s_mag, A, B_coeff)

        # Compute for this batch
        dot_product = torch.matmul(hkl_batch.to(torch.float64), xyz_flat).reshape(
            end - start, n_atoms, -1
        )
        U_dot_s = torch.einsum("jik,li->jkl", U_matrix, s_batch)  # (3, N_atoms, batch)
        StUS = torch.einsum("li,ikl->lk", s_batch, U_dot_s)  # (batch, N_atoms)
        B = -2 * (np.pi**2) * StUS
        exp_B = torch.exp(B)
        terms = sf_batch * exp_B * occ
        pidot = 2 * np.pi * dot_product
        sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot), axis=-1)
        sf_out[start:end] = torch.sum(terms * sin_cos, axis=1)

    return sf_out


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
