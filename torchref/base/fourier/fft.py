"""
FFT operations for crystallographic calculations.

Functions for forward and inverse Fourier transforms following
crystallographic sign conventions.
"""

import torch


def fft(reciprocal_grid, volume: float = None) -> torch.Tensor:
    """
    Perform FFT to obtain real space electron density.

    Uses fftn with norm="forward" to match crystallographic sign convention
    directly, avoiding expensive flip/roll operations.

    Crystallographic convention: ρ(r) = (1/V) Σ F(h) exp(-2πi h·r)

    PyTorch fftn with norm="forward" gives:
        fftn(x)[n] = (1/N) Σ_k x[k] exp(-2πi k·n/N)

    When input structure factors F are correctly scaled (with V/N factor from ifft),
    we need to multiply by N/V to recover the original electron density:
        ρ = fftn(F) * (N / V)

    Parameters
    ----------
    reciprocal_grid : torch.Tensor
        Reciprocal space grid of shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz).
        Expected to contain correctly scaled structure factors (from ifft with volume).
    volume : float, optional
        Unit cell volume in Å³. If provided, result is scaled by N/V to give
        correctly normalized electron density.

    Returns
    -------
    torch.Tensor
        Real-valued tensor of electron density with same shape as input.
    """
    if reciprocal_grid.ndim == 4:
        rs = torch.fft.fftn(reciprocal_grid, dim=(1, 2, 3), norm="forward").real
    else:
        rs = torch.fft.fftn(reciprocal_grid, dim=(0, 1, 2), norm="forward").real

    if volume is not None:
        # Apply crystallographic normalization: multiply by N/V
        N_total = reciprocal_grid.numel()
        rs = rs * N_total / volume

    return rs


def ifft(real_space_map, volume: float = None) -> torch.Tensor:
    """
    Perform inverse FFT to obtain reciprocal space structure factors.

    Crystallographic convention: F(h) = Σ ρ(r) exp(+2πi h·r) * ΔV
    where ΔV = V_cell / N is the voxel volume.

    PyTorch ifftn with norm="forward" gives unnormalized DFT:
        DFT[k] = Σ x[n] exp(+2πi k·n/N)

    To obtain correctly scaled structure factors, we multiply by voxel volume:
        F(h) = DFT(ρ) * (V_cell / N)

    Parameters
    ----------
    real_space_map : torch.Tensor
        Real space electron density map of shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz).
    volume : float, optional
        Unit cell volume in Å³. If provided, result is scaled by voxel volume
        (V_cell / N_total) to give correctly normalized structure factors.

    Returns
    -------
    torch.Tensor
        Complex-valued tensor of structure factors with same shape as input.
    """
    if real_space_map.ndim == 4:
        rg = torch.fft.ifftn(real_space_map, dim=(1, 2, 3), norm="forward")
    else:
        rg = torch.fft.ifftn(real_space_map, dim=(0, 1, 2), norm="forward")

    if volume is not None:
        # Apply crystallographic normalization: multiply by voxel volume (V/N)
        N_total = real_space_map.numel()
        voxel_volume = volume / N_total
        rg = rg * voxel_volume

    return rg
