"""Forward and inverse Fourier transforms in the crystallographic convention.

The crystallographic sign convention is the opposite of torch's, so
:func:`fft` (F -> ρ) calls ``fftn`` while :func:`ifft` (ρ -> F) calls ``ifftn``.
Pass ``volume`` to both or to neither -- mixing leaves an N/V scale error.
"""

import torch


def fft(reciprocal_grid, volume: float = None) -> torch.Tensor:
    """
    Transform structure factors to a real-space electron density map.

    Implements ρ(r) = (1/V) Σ F(h) exp(-2πi h·r), which is torch's *forward*
    ``fftn`` under the crystallographic sign convention -- hence the apparent
    name inversion against :func:`ifft`. No flip/roll is needed.

    Parameters
    ----------
    reciprocal_grid : torch.Tensor
        Structure factors, shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz), scaled as
        :func:`ifft` with ``volume`` leaves them.
    volume : float, optional
        Unit cell volume in Å³. Applies the N/V factor that undoes ``ifft``'s
        voxel-volume scaling; omitting it here after passing it there (or the
        reverse) leaves the density off by N/V.

    Returns
    -------
    torch.Tensor
        Real-valued electron density, same shape as the input.
    """
    if reciprocal_grid.ndim == 4:
        rs = torch.fft.fftn(reciprocal_grid, dim=(1, 2, 3), norm="forward").real
    else:
        rs = torch.fft.fftn(reciprocal_grid, dim=(0, 1, 2), norm="forward").real

    if volume is not None:
        N_total = reciprocal_grid.numel()
        rs = rs * N_total / volume

    return rs


def ifft(real_space_map, volume: float = None) -> torch.Tensor:
    """
    Transform an electron density map to reciprocal-space structure factors.

    Implements F(h) = ΔV Σ ρ(r) exp(+2πi h·r) with ΔV = V/N the voxel volume,
    which is torch's *inverse* ``ifftn`` under the crystallographic sign
    convention -- hence the apparent name inversion against :func:`fft`.

    Parameters
    ----------
    real_space_map : torch.Tensor
        Electron density, shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz).
    volume : float, optional
        Unit cell volume in Å³. Applies the ΔV = V/N factor; without it the
        structure factors carry no absolute scale, and :func:`fft` must then also
        be called without ``volume``.

    Returns
    -------
    torch.Tensor
        Complex structure factors, same shape as the input.
    """
    if real_space_map.ndim == 4:
        rg = torch.fft.ifftn(real_space_map, dim=(1, 2, 3), norm="forward")
    else:
        rg = torch.fft.ifftn(real_space_map, dim=(0, 1, 2), norm="forward")

    if volume is not None:
        N_total = real_space_map.numel()
        voxel_volume = volume / N_total
        rg = rg * voxel_volume

    return rg
