"""
Isotropic structure factor calculations.

Functions for computing structure factors from atomic models with
isotropic (spherical) atomic displacement parameters.
"""

from typing import Optional

import numpy as np
import torch

from torchref.config import dtypes


def _estimate_batch_size(n_refl: int, n_atoms: int, n_ops: int, max_memory_gb: float) -> int:
    """
    Estimate optimal batch size based on memory constraints.

    Main memory consumers per reflection:
    - dot_product: (N_atoms * n_ops) float32 = 4 bytes each
    - pidot: same size
    - sin_cos: (N_atoms * n_ops) complex64 = 8 bytes each
    - terms: (N_atoms) float32 = 4 bytes each
    - scattering_factors slice: (N_atoms) float32 = 4 bytes each

    Conservative estimate: ~50 bytes per (refl, atom, op) combination.
    """
    bytes_per_refl = n_atoms * n_ops * 50  # Conservative estimate
    max_bytes = max_memory_gb * 1e9
    batch_size = max(1, int(max_bytes / bytes_per_refl))
    return min(batch_size, n_refl)


def iso_structure_factor_torched(
    hkl, s, xyz_fractional, occ, scattering_factors, adp, spacegroup,
    max_memory_gb: Optional[float] = None,
    A: Optional[torch.Tensor] = None,
    B_coeff: Optional[torch.Tensor] = None,
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
    scattering_factors : torch.Tensor or None
        Atomic scattering factors of shape (N_reflections, N_atoms).
        If None, must provide A and B_coeff to compute them in batches.
    adp : torch.Tensor
        Atomic displacement parameters (isotropic) of shape (N_atoms,).
    spacegroup : callable
        Space group symmetry operator function.
    max_memory_gb : float, optional
        Maximum memory to use in GB. If None, no batching is applied.
    A : torch.Tensor, optional
        ITC92 A coefficients (N_atoms, 5) for computing scattering factors.
        Required if scattering_factors is None and batching is needed.
    B_coeff : torch.Tensor, optional
        ITC92 B coefficients (N_atoms, 5) for computing scattering factors.
        Required if scattering_factors is None and batching is needed.

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

    # Precompute ADP terms (doesn't depend on batch)
    adp_row = adp.reshape(1, -1)  # (1, N_atoms)

    # Check if batching is needed
    if max_memory_gb is not None:
        batch_size = _estimate_batch_size(n_refl, n_atoms, n_ops, max_memory_gb)
        if batch_size < n_refl:
            return _iso_sf_batched(
                hkl, s, xyz_flat, fractional_shape, occ, scattering_factors,
                adp_row, batch_size, A, B_coeff
            )
        # Batching not needed but scattering_factors may be None - compute them
        if scattering_factors is None and A is not None and B_coeff is not None:
            scattering_factors = _compute_scattering_factors_batch(s, A, B_coeff)

    # No batching - compute all at once
    dot_product = torch.matmul(hkl.to(dtypes.float), xyz_flat).reshape(
        n_refl, n_atoms, -1
    )
    s_col = s.reshape(-1, 1)
    B = -adp_row * (s_col**2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot), axis=-1)
    return torch.sum(terms * sin_cos, axis=1)


def _compute_scattering_factors_batch(
    s_batch: torch.Tensor, A: torch.Tensor, B_coeff: torch.Tensor
) -> torch.Tensor:
    """
    Compute scattering factors for a batch of reflections.

    Parameters
    ----------
    s_batch : torch.Tensor
        Scattering vector magnitudes (batch_size,).
    A : torch.Tensor
        ITC92 A coefficients (N_atoms, 5).
    B_coeff : torch.Tensor
        ITC92 B coefficients (N_atoms, 5).

    Returns
    -------
    torch.Tensor
        Scattering factors (batch_size, N_atoms).
    """
    # s: (batch,) -> (batch, 1, 1)
    s_sq = (s_batch.reshape(-1, 1, 1) ** 2) / 4
    # A, B: (N_atoms, 5) -> (1, N_atoms, 5)
    A_exp = A.unsqueeze(0)
    B_exp = B_coeff.unsqueeze(0)
    # Compute: (batch, N_atoms, 5)
    exp_terms = torch.exp(-B_exp * s_sq)
    # Sum over Gaussians: (batch, N_atoms)
    return torch.sum(A_exp * exp_terms, dim=-1)


def _iso_sf_batched(
    hkl, s, xyz_flat, fractional_shape, occ, scattering_factors, adp_row, batch_size,
    A=None, B_coeff=None
):
    """
    Compute isotropic structure factors in batches over reflections.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices (N_refl, 3).
    s : torch.Tensor
        Scattering vector magnitudes (N_refl,).
    xyz_flat : torch.Tensor
        Flattened fractional coordinates (3, N_atoms * n_ops).
    fractional_shape : tuple
        Original shape (3, N_atoms, n_ops).
    occ : torch.Tensor
        Occupancies (N_atoms,).
    scattering_factors : torch.Tensor or None
        Scattering factors (N_refl, N_atoms). If None, computed from A/B_coeff.
    adp_row : torch.Tensor
        ADP values reshaped to (1, N_atoms).
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
        s_batch = s[start:end]

        # Get or compute scattering factors for this batch
        if scattering_factors is not None:
            sf_batch = scattering_factors[start:end]
        else:
            sf_batch = _compute_scattering_factors_batch(s_batch, A, B_coeff)

        # Compute for this batch
        dot_product = torch.matmul(hkl_batch.to(dtypes.float), xyz_flat).reshape(
            end - start, n_atoms, -1
        )
        s_col = s_batch.reshape(-1, 1)
        B = -adp_row * (s_col**2) / 4
        exp_B = torch.exp(B)
        terms = sf_batch * exp_B * occ
        pidot = 2 * np.pi * dot_product
        sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot), axis=-1)
        sf_out[start:end] = torch.sum(terms * sin_cos, axis=1)

    return sf_out


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
    dot_product = torch.matmul(hkl.to(dtypes.float), fractional_coords).reshape(
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
