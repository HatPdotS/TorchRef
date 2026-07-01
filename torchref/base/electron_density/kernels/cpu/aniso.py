"""Anisotropic Gaussian box-splat for CPU / MPS (``Engine.AUTO``).

Mirrors the isotropic separable box-splat (center index -> local voxel cube ->
per-axis ``d_frac`` -> structured scatter) but evaluates the full 3D anisotropic
Gaussian over the cube (the cross-terms U12/U13/U23 prevent 1D factorization).
"""

import math

import torch

from torchref.config import dtypes


def _aniso_density_cube(d_frac, frac_matrix, Minv, A_norm):
    """Anisotropic Gaussian density over the local voxel cube.

    The aniso analogue of ``_separable_density`` — but the 3D Gaussian
    does NOT factorize across axes (cross-terms U12/U13/U23), so it builds the
    full Cartesian offset cube and evaluates the quadratic form directly. The
    component loop keeps peak memory at O(C*n^3).

    Parameters
    ----------
    d_frac : (C, 3, n) — PBC-wrapped fractional offsets per axis.
    frac_matrix : (3, 3) — fractional -> Cartesian.
    Minv : (C, 5, 3, 3) — inverse of M_g = (B_g*I + 8*pi^2*U)/4.
    A_norm : (C, 5) — A * occ * pi^1.5 / sqrt(det M_g).

    Returns
    -------
    (C, n, n, n) density cube.
    """
    pi_sq = math.pi * math.pi
    da = d_frac[:, 0, :][:, :, None, None]  # (C, n, 1, 1)
    db = d_frac[:, 1, :][:, None, :, None]  # (C, 1, n, 1)
    dc = d_frac[:, 2, :][:, None, None, :]  # (C, 1, 1, n)
    fm = frac_matrix
    # Cartesian offset cube r = frac_matrix @ d_frac  (each (C, n, n, n))
    cx = fm[0, 0] * da + fm[0, 1] * db + fm[0, 2] * dc
    cy = fm[1, 0] * da + fm[1, 1] * db + fm[1, 2] * dc
    cz = fm[2, 0] * da + fm[2, 1] * db + fm[2, 2] * dc

    C = d_frac.shape[0]
    n = d_frac.shape[2]
    density_cube = d_frac.new_zeros(C, n, n, n)
    for g in range(Minv.shape[1]):
        m00 = Minv[:, g, 0, 0][:, None, None, None]
        m11 = Minv[:, g, 1, 1][:, None, None, None]
        m22 = Minv[:, g, 2, 2][:, None, None, None]
        m01 = Minv[:, g, 0, 1][:, None, None, None]
        m02 = Minv[:, g, 0, 2][:, None, None, None]
        m12 = Minv[:, g, 1, 2][:, None, None, None]
        q = (
            m00 * cx * cx
            + m11 * cy * cy
            + m22 * cz * cz
            + 2.0 * (m01 * cx * cy + m02 * cx * cz + m12 * cy * cz)
        )
        density_cube = density_cube + A_norm[:, g, None, None, None] * torch.exp(
            -pi_sq * q
        )
    return density_cube
