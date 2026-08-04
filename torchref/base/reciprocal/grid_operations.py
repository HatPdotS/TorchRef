"""Placing structure factors on reciprocal-space grids and reading them back.

:func:`place_on_grid` and :func:`extract_structure_factor_from_grid` share one
index convention (hkl taken mod the grid dimensions), so they round-trip.
"""

import math

import torch


def place_on_grid(
    hkls, structure_factor, grid_size, enforce_hermitian: bool = True
) -> torch.Tensor:
    """
    Place structure factors on a reciprocal-space grid, wrapping hkl periodically.

    Parameters
    ----------
    hkls : torch.Tensor
        Miller indices of shape (N, 3).
    structure_factor : torch.Tensor
        Structure factors of shape (N,) or (B, N) for batched input.
    grid_size : tuple or torch.Tensor
        Grid dimensions (Nx, Ny, Nz).
    enforce_hermitian : bool, optional
        Whether to enforce Hermitian symmetry. Default is True.

    Returns
    -------
    torch.Tensor
        Complex tensor grid of structure factors of shape (Nx, Ny, Nz)
        or (B, Nx, Ny, Nz) for batched input.

    Notes
    -----
    ``enforce_hermitian=True`` index-adds each reflection's conjugate at ``-hkl``,
    so input already holding both Friedel mates is double-counted -- pass only the
    unique half. Placement is index-*add*, so duplicate hkl accumulate.
    """
    batch_mode = True
    if structure_factor.ndim == 1:
        structure_factor = structure_factor.unsqueeze(0)  # Add batch dimension
        batch_mode = False
    B = structure_factor.shape[0]
    device = structure_factor.device
    dtype = structure_factor.dtype
    Nx, Ny, Nz = [int(x) for x in grid_size]
    hkls = hkls.to(device=device)
    h = hkls[:, 0].to(torch.int64)
    k = hkls[:, 1].to(torch.int64)
    l = hkls[:, 2].to(torch.int64)

    hi = torch.remainder(h, Nx)
    ki = torch.remainder(k, Ny)
    li = torch.remainder(l, Nz)
    lin = (hi * (Ny * Nz) + ki * Nz + li).to(torch.int64)  # (N,)
    grid = torch.zeros((B, Nx * Ny * Nz), dtype=dtype, device=device)
    grid = grid.index_add(1, lin, structure_factor)  # (B, Nx*Ny*Nz)

    if enforce_hermitian:
        hi_sym = torch.remainder(-h, Nx)
        ki_sym = torch.remainder(-k, Ny)
        li_sym = torch.remainder(-l, Nz)
        lin_sym = (hi_sym * (Ny * Nz) + ki_sym * Nz + li_sym).to(torch.int64)
        vals_conj = torch.conj(structure_factor)
        grid = grid.index_add(1, lin_sym, vals_conj)

    grid = grid.view(B, Nx, Ny, Nz)
    if not batch_mode:
        grid = grid.squeeze(0)
    return grid


def extract_structure_factor_from_grid(reciprocal_grid, hkls) -> torch.Tensor:
    """
    Extract structure factors from reciprocal space grid at given Miller indices.

    Parameters
    ----------
    reciprocal_grid : torch.Tensor
        Complex tensor of shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz).
    hkls : torch.Tensor
        Miller indices of shape (N, 3).

    Returns
    -------
    torch.Tensor
        Structure factors of shape (N,) or (B, N) for batched input.
    """
    device = reciprocal_grid.device
    dtype = reciprocal_grid.dtype

    if reciprocal_grid.ndim == 3:
        reciprocal_grid = reciprocal_grid.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False

    B, Nx, Ny, Nz = reciprocal_grid.shape

    # Same wrapping convention as place_on_grid.
    hkls = hkls.to(device=device)
    h = hkls[:, 0].to(torch.int64)
    k = hkls[:, 1].to(torch.int64)
    l = hkls[:, 2].to(torch.int64)

    hi = torch.remainder(h, Nx)
    ki = torch.remainder(k, Ny)
    li = torch.remainder(l, Nz)

    structure_factors = reciprocal_grid[:, hi, ki, li]  # (B, N)

    if squeeze_output:
        structure_factors = structure_factors.squeeze(0)  # (N,)

    return structure_factors


def apply_translation_phase(
    F_calc: torch.Tensor,
    hkl: torch.Tensor,
    translation_frac: torch.Tensor,
) -> torch.Tensor:
    """
    Apply translation phase shift to structure factors.

    For a translation t in fractional coordinates, the structure factor transforms as:
    F'(hkl) = F(hkl) * exp(2πi * hkl · t)

    Parameters
    ----------
    F_calc : torch.Tensor
        Complex structure factors of shape (N,).
    hkl : torch.Tensor
        Miller indices of shape (N, 3).
    translation_frac : torch.Tensor
        Translation vector in fractional coordinates of shape (3,).

    Returns
    -------
    torch.Tensor
        Phase-shifted structure factors of shape (N,).

    Notes
    -----
    The phase is computed in float32 regardless of input dtype and only then cast
    to ``F_calc.dtype``, so a float64 caller does *not* get float64 phases.
    """
    phase = 2.0 * math.pi * (hkl.float() @ translation_frac.float())

    phase_factor = torch.complex(torch.cos(phase), torch.sin(phase))

    return F_calc * phase_factor.to(F_calc.dtype)
