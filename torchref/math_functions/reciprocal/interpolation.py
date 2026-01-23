"""
Interpolation functions for reciprocal space grids.

These functions enable structure factor extraction at non-integer HKL positions,
which is essential for rotation and translation searches in molecular replacement.
"""

import numpy as np
import torch


def interpolate_structure_factor_from_grid(
    reciprocal_grid: torch.Tensor,
    hkl_float: torch.Tensor,
    interpolate_amplitude: bool = True,
) -> torch.Tensor:
    """
    Interpolate structure factors from reciprocal space grid at non-integer positions.

    Parameters
    ----------
    reciprocal_grid : torch.Tensor
        Complex tensor of shape (Nx, Ny, Nz).
    hkl_float : torch.Tensor
        Non-integer HKL positions of shape (N, 3).
    interpolate_amplitude : bool, optional
        If True (default), interpolate amplitudes instead of complex values.
        This avoids phase cancellation issues where linear interpolation of
        complex numbers with different phases can give incorrect magnitudes
        (e.g., interpolating F=1 and F=-1 gives 0 instead of 1).

    Returns
    -------
    torch.Tensor
        Interpolated structure factors of shape (N,).
        If interpolate_amplitude=True, returns real-valued amplitudes.
        If interpolate_amplitude=False, returns complex values (use with caution).

    Notes
    -----
    For a rotation R applied to the model, the structure factor at hkl becomes
    F(R^T @ hkl), so you would call this with hkl_float = hkl @ R.

    WARNING: Complex interpolation (interpolate_amplitude=False) can give
    incorrect results when neighboring voxels have very different phases.
    For example, if F1 = A*exp(i*0) and F2 = A*exp(i*π), linear interpolation
    gives magnitude 0 at the midpoint instead of A. Use interpolate_amplitude=True
    for rotation functions where only magnitudes matter.
    """
    device = reciprocal_grid.device
    Nx, Ny, Nz = reciprocal_grid.shape

    hkl_float = hkl_float.to(device=device, dtype=torch.float32)

    # Get the 8 corner indices for trilinear interpolation
    h = hkl_float[:, 0]
    k = hkl_float[:, 1]
    l = hkl_float[:, 2]

    # Floor indices
    h0 = torch.floor(h).long()
    k0 = torch.floor(k).long()
    l0 = torch.floor(l).long()

    # Ceil indices
    h1 = h0 + 1
    k1 = k0 + 1
    l1 = l0 + 1

    # Fractional parts (weights) - use float32 for weights
    hd = h - h0.float()
    kd = k - k0.float()
    ld = l - l0.float()

    # Wrap indices to grid (periodic boundary)
    h0 = torch.remainder(h0, Nx)
    h1 = torch.remainder(h1, Nx)
    k0 = torch.remainder(k0, Ny)
    k1 = torch.remainder(k1, Ny)
    l0 = torch.remainder(l0, Nz)
    l1 = torch.remainder(l1, Nz)

    # Get values at 8 corners
    c000 = reciprocal_grid[h0, k0, l0]
    c001 = reciprocal_grid[h0, k0, l1]
    c010 = reciprocal_grid[h0, k1, l0]
    c011 = reciprocal_grid[h0, k1, l1]
    c100 = reciprocal_grid[h1, k0, l0]
    c101 = reciprocal_grid[h1, k0, l1]
    c110 = reciprocal_grid[h1, k1, l0]
    c111 = reciprocal_grid[h1, k1, l1]

    if interpolate_amplitude:
        # Convert to amplitudes before interpolation to avoid phase cancellation
        a000 = torch.abs(c000)
        a001 = torch.abs(c001)
        a010 = torch.abs(c010)
        a011 = torch.abs(c011)
        a100 = torch.abs(c100)
        a101 = torch.abs(c101)
        a110 = torch.abs(c110)
        a111 = torch.abs(c111)

        # Trilinear interpolation of amplitudes
        a00 = a000 * (1 - ld) + a001 * ld
        a01 = a010 * (1 - ld) + a011 * ld
        a10 = a100 * (1 - ld) + a101 * ld
        a11 = a110 * (1 - ld) + a111 * ld

        a0 = a00 * (1 - kd) + a01 * kd
        a1 = a10 * (1 - kd) + a11 * kd

        result = a0 * (1 - hd) + a1 * hd
        return result

    else:
        # Complex interpolation (original behavior - use with caution)
        # Convert weights to complex dtype for multiplication
        dtype = reciprocal_grid.dtype
        hd = hd.to(dtype)
        kd = kd.to(dtype)
        ld = ld.to(dtype)

        c00 = c000 * (1 - ld) + c001 * ld
        c01 = c010 * (1 - ld) + c011 * ld
        c10 = c100 * (1 - ld) + c101 * ld
        c11 = c110 * (1 - ld) + c111 * ld

        c0 = c00 * (1 - kd) + c01 * kd
        c1 = c10 * (1 - kd) + c11 * kd

        result = c0 * (1 - hd) + c1 * hd
        return result


def interpolate_complex_from_grid(
    reciprocal_grid: torch.Tensor,
    hkl_float: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate complex structure factors from reciprocal space grid.

    Unlike amplitude interpolation, this preserves phase information, which is
    essential for translation searches where phases are used to compute
    correlation functions.

    Parameters
    ----------
    reciprocal_grid : torch.Tensor
        Complex tensor of shape (Nx, Ny, Nz).
    hkl_float : torch.Tensor
        Non-integer HKL positions of shape (N, 3).

    Returns
    -------
    torch.Tensor
        Interpolated complex structure factors of shape (N,).

    Notes
    -----
    This function performs trilinear interpolation of complex values.
    For rotation-only searches (where only magnitudes matter), use
    `interpolate_structure_factor_from_grid(interpolate_amplitude=True)` instead.

    For translation searches, complex interpolation is needed because the
    translation function depends on the phase relationship between F_obs and F_calc.

    WARNING: Complex interpolation can give reduced magnitudes when neighboring
    voxels have very different phases. This is acceptable for translation searches
    where we're computing correlation functions, but not for rotation searches.
    """
    device = reciprocal_grid.device
    Nx, Ny, Nz = reciprocal_grid.shape

    hkl_float = hkl_float.to(device=device, dtype=torch.float32)

    # Get the 8 corner indices for trilinear interpolation
    h = hkl_float[:, 0]
    k = hkl_float[:, 1]
    l = hkl_float[:, 2]

    # Floor indices
    h0 = torch.floor(h).long()
    k0 = torch.floor(k).long()
    l0 = torch.floor(l).long()

    # Ceil indices
    h1 = h0 + 1
    k1 = k0 + 1
    l1 = l0 + 1

    # Fractional parts (weights)
    hd = h - h0.float()
    kd = k - k0.float()
    ld = l - l0.float()

    # Wrap indices to grid (periodic boundary)
    h0 = torch.remainder(h0, Nx)
    h1 = torch.remainder(h1, Nx)
    k0 = torch.remainder(k0, Ny)
    k1 = torch.remainder(k1, Ny)
    l0 = torch.remainder(l0, Nz)
    l1 = torch.remainder(l1, Nz)

    # Get complex values at 8 corners
    c000 = reciprocal_grid[h0, k0, l0]
    c001 = reciprocal_grid[h0, k0, l1]
    c010 = reciprocal_grid[h0, k1, l0]
    c011 = reciprocal_grid[h0, k1, l1]
    c100 = reciprocal_grid[h1, k0, l0]
    c101 = reciprocal_grid[h1, k0, l1]
    c110 = reciprocal_grid[h1, k1, l0]
    c111 = reciprocal_grid[h1, k1, l1]

    # Convert weights to complex dtype for multiplication
    dtype = reciprocal_grid.dtype
    hd = hd.to(dtype)
    kd = kd.to(dtype)
    ld = ld.to(dtype)

    # Trilinear interpolation of complex values
    c00 = c000 * (1 - ld) + c001 * ld
    c01 = c010 * (1 - ld) + c011 * ld
    c10 = c100 * (1 - ld) + c101 * ld
    c11 = c110 * (1 - ld) + c111 * ld

    c0 = c00 * (1 - kd) + c01 * kd
    c1 = c10 * (1 - kd) + c11 * kd

    result = c0 * (1 - hd) + c1 * hd
    return result


def trilinear_interpolate_patterson(
    grid: torch.Tensor, points: torch.Tensor, chunk_size: int = 10_000_000
) -> torch.Tensor:
    """
    Memory-efficient trilinear interpolation on a 3D grid.

    Pure torch implementation for GPU acceleration and gradient flow.
    Replaces scipy.ndimage.map_coordinates for torch tensors.

    Parameters
    ----------
    grid : torch.Tensor
        3D grid of values with shape (nx, ny, nz).
    points : torch.Tensor
        Coordinates to sample with shape (n_points, 3).
        Should be in fractional coordinates [0, 1) for 'wrap' mode.
        Or batch K, n_points, 3 for multiple batches.
    chunk_size : int, optional
        Number of points to process at once. Default is 1M.

    Returns
    -------
    torch.Tensor
        Interpolated values with shape (n_points,) or (batch, n_points).

    Notes
    -----
    Supports automatic differentiation for gradient-based optimization.
    Uses chunked processing and in-place accumulation to reduce memory.
    """
    original_shape = points.shape[:-1]
    points = points.reshape(-1, 3)
    n_total = points.shape[0]

    nx, ny, nz = grid.shape
    device = grid.device
    dtype = grid.dtype

    result = torch.empty(n_total, device=device, dtype=dtype)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        pts = points[start:end] % 1.0  # Wrap to [0, 1)

        # Scale to grid coordinates
        px = pts[:, 0] * nx
        py = pts[:, 1] * ny
        pz = pts[:, 2] * nz

        # Floor indices
        x0 = px.long() % nx
        y0 = py.long() % ny
        z0 = pz.long() % nz
        x1 = (x0 + 1) % nx
        y1 = (y0 + 1) % ny
        z1 = (z0 + 1) % nz

        # Fractional parts (weights)
        xd = (px - px.floor()).to(dtype)
        yd = (py - py.floor()).to(dtype)
        zd = (pz - pz.floor()).to(dtype)

        # Precompute complementary weights
        xd1 = 1 - xd
        yd1 = 1 - yd
        zd1 = 1 - zd

        # Direct accumulation - avoids storing 8 corner arrays
        result[start:end] = (
            grid[x0, y0, z0] * (xd1 * yd1 * zd1)
            + grid[x0, y0, z1] * (xd1 * yd1 * zd)
            + grid[x0, y1, z0] * (xd1 * yd * zd1)
            + grid[x0, y1, z1] * (xd1 * yd * zd)
            + grid[x1, y0, z0] * (xd * yd1 * zd1)
            + grid[x1, y0, z1] * (xd * yd1 * zd)
            + grid[x1, y1, z0] * (xd * yd * zd1)
            + grid[x1, y1, z1] * (xd * yd * zd)
        )

    return result.reshape(original_shape)
