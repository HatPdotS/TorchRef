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
    PyTorch fftn: X[k] = Σ x[n] exp(-2πi k·n/N)

    Parameters
    ----------
    reciprocal_grid : torch.Tensor
        Reciprocal space grid of shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz).
    volume : float, optional
        Unit cell volume in Å³. If provided, result is multiplied by volume.

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
        rs = rs * volume

    return rs


def ifft(real_space_map, volume: float = None) -> torch.Tensor:
    """
    Perform inverse FFT to obtain reciprocal space structure factors.

    Crystallographic convention: F(h) = Σ ρ(r) exp(+2πi h·r)
    PyTorch ifftn: x[n] = (1/N) Σ X[k] exp(+2πi k·n/N)

    Parameters
    ----------
    real_space_map : torch.Tensor
        Real space electron density map of shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz).
    volume : float, optional
        Unit cell volume in Å³. If provided, result is divided by volume.

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
        rg = rg / volume

    return rg
