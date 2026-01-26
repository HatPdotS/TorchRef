"""
Ball Harmonic Transform for Fast Rotation Function.

Implements 3D ball transforms that preserve radial (resolution) information
for molecular replacement rotation searches.

The ball function f(r, θ, φ) is expanded using:
- Uniform radial shells (resolution bins)
- Spherical harmonics for angular component (via s2fft)

For rotation correlation of two ball functions f and g:
    C(R) = ∫ f(x) g(R⁻¹x) dx
         = Σ_{p,l,m,n} f*_{p,l,m} g_{p,l,n} D^l_{m,n}(R)
         = Σ_{l,m,n} ξ_{l,m,n} D^l_{m,n}(R)

where ξ_{l,m,n} = Σ_p w_p f*_{p,l,m} g_{p,l,n} sums over radial shells.

Key property: Rotations only affect the angular part - radial indices are summed.
This preserves resolution information while reducing to a standard Wigner transform.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numba import njit

# Enable JAX 64-bit precision before importing s2fft/s2ball
import jax
jax.config.update("jax_enable_x64", True)

import s2fft
import s2ball.transform.wigner as wigner_transform
import spherical
import quaternionic


# =============================================================================
# Numba-accelerated Analytical Spherical Harmonic Computation
# =============================================================================

@njit(cache=True)
def _compute_plm_recurrence(l_max: int, cos_theta: np.ndarray) -> np.ndarray:
    """
    Compute associated Legendre polynomials P_l^m(cos(theta)) using recurrence.

    Parameters
    ----------
    l_max : int
        Maximum l value (exclusive), i.e., computes for l = 0, 1, ..., l_max-1.
    cos_theta : np.ndarray
        Cosine of colatitude angles, shape (N,).

    Returns
    -------
    Plm : np.ndarray
        Associated Legendre polynomials, shape (N, l_max, l_max).
        Plm[i, l, m] = P_l^m(cos_theta[i]) for m >= 0.
    """
    N = len(cos_theta)
    sin_theta = np.sqrt(1 - cos_theta**2)
    Plm = np.zeros((N, l_max, l_max))

    # P_0^0 = 1
    Plm[:, 0, 0] = 1.0

    if l_max > 1:
        # P_1^0 = cos(theta)
        Plm[:, 1, 0] = cos_theta
        # P_1^1 = -sin(theta)
        Plm[:, 1, 1] = -sin_theta

    # Recurrence for P_l^l (diagonal)
    for l in range(2, l_max):
        Plm[:, l, l] = -(2*l - 1) * sin_theta * Plm[:, l-1, l-1]

    # Recurrence for P_l^{l-1} (subdiagonal)
    for l in range(2, l_max):
        Plm[:, l, l-1] = cos_theta * (2*l - 1) * Plm[:, l-1, l-1]

    # Recurrence for P_l^m (general)
    for l in range(2, l_max):
        for m in range(0, l-1):
            Plm[:, l, m] = ((2*l - 1) * cos_theta * Plm[:, l-1, m] -
                           (l + m - 1) * Plm[:, l-2, m]) / (l - m)

    return Plm


@njit(cache=True)
def _compute_sh_normalization_factors(l_max: int) -> np.ndarray:
    """
    Compute normalization factors for spherical harmonics.

    K_l^m = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!)

    Parameters
    ----------
    l_max : int
        Maximum l value (exclusive).

    Returns
    -------
    K : np.ndarray
        Normalization factors, shape (l_max, l_max).
        K[l, m] for m >= 0.
    """
    K = np.zeros((l_max, l_max))
    for l in range(l_max):
        for m in range(l + 1):
            # Compute log((l-m)!/(l+m)!) for numerical stability
            log_factor = 0.0
            for k in range(l - m + 1, l + m + 1):
                log_factor -= np.log(k)
            K[l, m] = np.sqrt((2*l + 1) / (4 * np.pi) * np.exp(log_factor))
    return K


@njit(cache=True)
def _compute_sh_coeffs_analytical_numba(
    E: np.ndarray,
    cos_theta: np.ndarray,
    phi: np.ndarray,
    L: int,
) -> np.ndarray:
    """
    Compute spherical harmonic coefficients analytically using numba.

    Computes a_lm = (4*pi/N) * sum_i(E_i * conj(Y_lm(theta_i, phi_i)))

    where Y_lm = K_l^m * P_l^m(cos(theta)) * exp(i*m*phi)

    Parameters
    ----------
    E : np.ndarray
        Function values at sample points, shape (N,).
    cos_theta : np.ndarray
        Cosine of colatitude angles, shape (N,).
    phi : np.ndarray
        Azimuthal angles in [0, 2*pi), shape (N,).
    L : int
        Angular bandlimit.

    Returns
    -------
    flm : np.ndarray
        Spherical harmonic coefficients, shape (L, 2*L-1).
        flm[l, m + L - 1] = a_lm for m in [-l, l].
    """
    N = len(E)

    # Compute associated Legendre polynomials
    Plm = _compute_plm_recurrence(L, cos_theta)

    # Compute normalization factors
    K = _compute_sh_normalization_factors(L)

    # Precompute exp(i*m*phi) for all m values
    exp_imphi = np.zeros((N, 2*L - 1), dtype=np.complex128)
    for m in range(-(L-1), L):
        exp_imphi[:, m + L - 1] = np.cos(m * phi) + 1j * np.sin(m * phi)

    # Compute coefficients
    flm = np.zeros((L, 2*L - 1), dtype=np.complex128)

    for l in range(L):
        for m in range(-l, l + 1):
            # Compute Y_lm
            if m >= 0:
                Y_lm = K[l, m] * Plm[:, l, m] * exp_imphi[:, m + L - 1]
            else:
                # Y_l^{-|m|} = (-1)^|m| * conj(Y_l^|m|)
                abs_m = -m
                sign = 1.0 if abs_m % 2 == 0 else -1.0
                Y_lm = sign * K[l, abs_m] * Plm[:, l, abs_m] * np.conj(exp_imphi[:, abs_m + L - 1])

            # a_lm = (4*pi/N) * sum(E * conj(Y_lm))
            a_lm = 0.0 + 0.0j
            for i in range(N):
                a_lm += E[i] * np.conj(Y_lm[i])
            flm[l, m + L - 1] = (4 * np.pi / N) * a_lm

    return flm



# =============================================================================
# Ball Harmonic Coefficient Container
# =============================================================================

@dataclass
class BallHarmonicCoefficients:
    """
    Container for ball harmonic coefficients.

    Attributes
    ----------
    flmp : np.ndarray
        Spherical harmonic coefficients for each radial shell, shape (P, L, 2L-1).
    L : int
        Angular bandlimit.
    P : int
        Number of radial shells.
    shell_edges : np.ndarray
        Radial shell boundaries in Å⁻¹, shape (P+1,).
    shell_centers : np.ndarray
        Radial shell centers in Å⁻¹, shape (P,).
    shell_counts : np.ndarray
        Number of reflections in each shell, shape (P,).
    """
    flmp: np.ndarray
    L: int
    P: int
    shell_edges: np.ndarray
    shell_centers: np.ndarray
    shell_counts: np.ndarray


def _compute_radial_shells_np(
    d_min: float,
    d_max: float,
    P: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute uniform radial shell boundaries in reciprocal space.

    Parameters
    ----------
    d_min : float
        High resolution limit in Angstrom.
    d_max : float
        Low resolution limit in Angstrom.
    P : int
        Number of radial shells.

    Returns
    -------
    shell_edges : np.ndarray
        Shell boundaries in Angstrom^-1, shape (P+1,).
    shell_centers : np.ndarray
        Shell centers in Angstrom^-1, shape (P,).
    """
    s_min = 1.0 / d_max  # Low resolution end
    s_max = 1.0 / d_min  # High resolution end

    shell_edges = np.linspace(s_min, s_max, P + 1)
    shell_centers = 0.5 * (shell_edges[:-1] + shell_edges[1:])

    return shell_edges, shell_centers


def _compute_equal_count_shells_np(
    s_mag: np.ndarray,
    P: int,
    d_min: float,
    d_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial shell boundaries such that each shell has equal reflection count.

    This addresses the issue that uniform spacing in s leads to highly imbalanced
    shell counts (low resolution shells have few reflections, high resolution
    shells have many). Equal-count binning ensures each shell contributes equally
    to the spherical harmonic expansion.

    Parameters
    ----------
    s_mag : np.ndarray
        Magnitude of s-vectors (|s| = 1/d), shape (N,).
    P : int
        Number of radial shells.
    d_min : float
        High resolution limit in Angstrom.
    d_max : float
        Low resolution limit in Angstrom.

    Returns
    -------
    shell_edges : np.ndarray
        Shell boundaries in Angstrom^-1, shape (P+1,).
    shell_centers : np.ndarray
        Shell centers in Angstrom^-1, shape (P,).
    """
    s_min = 1.0 / d_max  # Low resolution end
    s_max = 1.0 / d_min  # High resolution end

    # Filter to resolution range
    mask = (s_mag >= s_min) & (s_mag <= s_max)
    s_in_range = s_mag[mask]

    if len(s_in_range) == 0:
        # Fallback to uniform if no reflections in range
        return _compute_radial_shells_np(d_min, d_max, P)

    # Compute quantiles for equal-count binning
    # We want P bins, so we need P+1 edges at quantiles 0, 1/P, 2/P, ..., 1
    quantiles = np.linspace(0, 1, P + 1)
    shell_edges = np.quantile(s_in_range, quantiles)

    # Ensure edges are strictly within bounds
    shell_edges[0] = max(shell_edges[0], s_min)
    shell_edges[-1] = min(shell_edges[-1], s_max)

    # Ensure strictly increasing (can happen with many identical values)
    for i in range(1, len(shell_edges)):
        if shell_edges[i] <= shell_edges[i-1]:
            shell_edges[i] = shell_edges[i-1] + 1e-10

    shell_centers = 0.5 * (shell_edges[:-1] + shell_edges[1:])

    return shell_edges, shell_centers


def get_mw_grid(L: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get McEwen-Wiaux (MW) sampling grid positions.

    Parameters
    ----------
    L : int
        Angular bandlimit.

    Returns
    -------
    thetas : np.ndarray
        Colatitude samples in [0, π], shape (L,).
    phis : np.ndarray
        Azimuth samples in [0, 2π), shape (2L-1,).
    """
    thetas = np.array([(2*t + 1) * np.pi / (2*L) for t in range(L)])
    phis = np.array([2 * np.pi * p / (2*L - 1) for p in range(2*L - 1)])
    return thetas, phis


def splat_to_mw_grid(
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    L: int,
    mean_center: bool = True,
) -> np.ndarray:
    """
    Splat values onto MW sampling grid using bilinear interpolation.

    Parameters
    ----------
    theta : np.ndarray
        Colatitude angles in [0, π], shape (N,).
    phi : np.ndarray
        Azimuthal angles in [0, 2π), shape (N,).
    values : np.ndarray
        Values to splat, shape (N,).
    L : int
        Angular bandlimit.
    mean_center : bool
        If True, subtract mean from grid.

    Returns
    -------
    grid : np.ndarray
        Splatted grid of shape (L, 2L-1).
    """
    n_theta = L
    n_phi = 2 * L - 1

    grid = np.zeros((n_theta, n_phi), dtype=np.float64)
    weights = np.zeros((n_theta, n_phi), dtype=np.float64)

    # MW grid: theta_t = (2t + 1) * π / (2L)
    theta_px = (theta * 2 * L / np.pi - 1) / 2
    phi_px = phi * (2 * L - 1) / (2 * np.pi)

    theta_px = np.clip(theta_px, 0, n_theta - 1 - 1e-6)

    theta_lo = np.floor(theta_px).astype(int)
    phi_lo = np.floor(phi_px).astype(int)

    theta_frac = theta_px - theta_lo
    phi_frac = phi_px - phi_lo

    theta_hi = np.clip(theta_lo + 1, 0, n_theta - 1)
    theta_lo = np.clip(theta_lo, 0, n_theta - 1)
    phi_hi = (phi_lo + 1) % n_phi
    phi_lo = phi_lo % n_phi

    w00 = (1 - theta_frac) * (1 - phi_frac)
    w01 = (1 - theta_frac) * phi_frac
    w10 = theta_frac * (1 - phi_frac)
    w11 = theta_frac * phi_frac

    np.add.at(grid, (theta_lo, phi_lo), w00 * values)
    np.add.at(grid, (theta_lo, phi_hi), w01 * values)
    np.add.at(grid, (theta_hi, phi_lo), w10 * values)
    np.add.at(grid, (theta_hi, phi_hi), w11 * values)

    np.add.at(weights, (theta_lo, phi_lo), w00)
    np.add.at(weights, (theta_lo, phi_hi), w01)
    np.add.at(weights, (theta_hi, phi_lo), w10)
    np.add.at(weights, (theta_hi, phi_hi), w11)

    mask = weights > 0
    grid[mask] /= weights[mask]

    if mean_center:
        grid = grid - grid.mean()

    return grid


def _assign_to_shells_np(
    s_mag: np.ndarray,
    shell_edges: np.ndarray,
) -> np.ndarray:
    """
    Internal NumPy version for ball-specific functions.
    """
    shell_idx = np.digitize(s_mag, shell_edges) - 1
    P = len(shell_edges) - 1
    # Mark out-of-range as -1
    shell_idx[(shell_idx < 0) | (shell_idx >= P)] = -1
    return shell_idx


def splat_evalues_to_ball(
    E_values: np.ndarray,
    s_vectors: np.ndarray,
    L: int,
    P: int,
    d_min: float,
    d_max: float,
    mean_center_shells: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splat E-values onto a 3D ball grid using uniform radial shells.

    Parameters
    ----------
    E_values : np.ndarray
        E² values, shape (N,).
    s_vectors : np.ndarray
        Reciprocal space vectors in Å⁻¹, shape (N, 3).
    L : int
        Angular bandlimit.
    P : int
        Number of radial shells.
    d_min : float
        High resolution limit in Å.
    d_max : float
        Low resolution limit in Å.
    mean_center_shells : bool
        If True, mean-center each radial shell.

    Returns
    -------
    ball_grid : np.ndarray
        3D ball grid of shape (P, L, 2L-1).
    shell_edges : np.ndarray
        Shell boundaries in Å⁻¹.
    shell_centers : np.ndarray
        Shell centers in Å⁻¹.
    shell_counts : np.ndarray
        Number of reflections per shell.
    """
    # Compute uniform radial shells
    shell_edges, shell_centers = _compute_radial_shells_np(d_min, d_max, P)

    # Compute |s| and angles
    s_mag = np.linalg.norm(s_vectors, axis=1)
    s_normed = s_vectors / np.maximum(s_mag[:, np.newaxis], 1e-10)

    # Spherical angles
    theta = np.arccos(np.clip(s_normed[:, 2], -1, 1))
    phi = np.arctan2(s_normed[:, 1], s_normed[:, 0])
    phi = phi % (2 * np.pi)

    # Assign to shells
    shell_idx = _assign_to_shells_np(s_mag, shell_edges)

    # Initialize ball grid
    ball_grid = np.zeros((P, L, 2 * L - 1), dtype=np.float64)
    shell_counts = np.zeros(P, dtype=np.int64)

    # Splat each shell
    for p in range(P):
        mask = shell_idx == p
        count = mask.sum()
        shell_counts[p] = count

        if count == 0:
            continue

        ball_grid[p] = splat_to_mw_grid(
            theta[mask],
            phi[mask],
            E_values[mask],
            L,
            mean_center=mean_center_shells,
        )

    return ball_grid, shell_edges, shell_centers, shell_counts


def compute_ball_harmonic_coefficients(
    ball_grid: np.ndarray,
    L: int,
    shell_edges: np.ndarray,
    shell_centers: np.ndarray,
    shell_counts: np.ndarray,
) -> BallHarmonicCoefficients:
    """
    Compute spherical harmonic coefficients for each radial shell using s2fft.

    Parameters
    ----------
    ball_grid : np.ndarray
        3D ball grid of shape (P, L, 2L-1).
    L : int
        Angular bandlimit.
    shell_edges : np.ndarray
        Shell boundaries.
    shell_centers : np.ndarray
        Shell centers.
    shell_counts : np.ndarray
        Number of reflections per shell.

    Returns
    -------
    coeffs : BallHarmonicCoefficients
        Ball harmonic coefficients (SH coeffs for each shell).
    """
    P = ball_grid.shape[0]

    # Compute SH coefficients for each shell using s2fft
    flmp = np.zeros((P, L, 2*L - 1), dtype=np.complex128)

    for p in range(P):
        if shell_counts[p] > 0:
            # Use s2fft forward transform: grid -> SH coefficients
            flmp[p] = s2fft.forward(
                ball_grid[p],
                L,
                sampling="mw",
                method="jax",
                reality=False,
            )

    return BallHarmonicCoefficients(
        flmp=flmp,
        L=L,
        P=P,
        shell_edges=shell_edges,
        shell_centers=shell_centers,
        shell_counts=shell_counts,
    )


def compute_ball_harmonic_coefficients_analytical(
    E_values: np.ndarray,
    s_vectors: np.ndarray,
    L: int,
    P: int,
    d_min: float,
    d_max: float,
    normalize_shells: bool = True,
    equal_count_shells: bool = True,
) -> BallHarmonicCoefficients:
    """
    Compute ball harmonic coefficients analytically without splatting.

    This method computes SH coefficients directly from the reflection positions
    using the analytical formula:
        a_lm = (4*pi/N) * sum_i(E_i * conj(Y_lm(theta_i, phi_i)))

    This avoids discretization errors from splatting onto a grid and is
    significantly faster when using numba acceleration.

    Parameters
    ----------
    E_values : np.ndarray
        E-values (normalized structure factor amplitudes), shape (N,).
    s_vectors : np.ndarray
        Reciprocal space vectors in Angstrom^-1, shape (N, 3).
    L : int
        Angular bandlimit.
    P : int
        Number of radial shells.
    d_min : float
        High resolution limit in Angstrom.
    d_max : float
        Low resolution limit in Angstrom.
    normalize_shells : bool
        If True, mean-center and variance-normalize E-values in each shell.
        This removes DC bias and makes correlations comparable across shells.
    equal_count_shells : bool
        If True (default), compute shell boundaries such that each shell has
        approximately the same number of reflections. This ensures equal
        contribution from each resolution shell to the spherical harmonic
        expansion. If False, use uniform spacing in s (which leads to
        imbalanced shell counts due to increasing reflection density at
        higher resolution).

    Returns
    -------
    coeffs : BallHarmonicCoefficients
        Ball harmonic coefficients computed analytically.
    """
    # Compute s-vector magnitudes (needed for shell assignment)
    s_mag = np.linalg.norm(s_vectors, axis=1)

    # Compute radial shells
    if equal_count_shells:
        shell_edges, shell_centers = _compute_equal_count_shells_np(s_mag, P, d_min, d_max)
    else:
        shell_edges, shell_centers = _compute_radial_shells_np(d_min, d_max, P)

    # Compute normalized s-vectors for angular coordinates
    s_normed = s_vectors / np.maximum(s_mag[:, np.newaxis], 1e-10)

    # Spherical angles
    theta = np.arccos(np.clip(s_normed[:, 2], -1, 1))
    cos_theta = np.cos(theta)
    phi = np.arctan2(s_normed[:, 1], s_normed[:, 0])
    phi = phi % (2 * np.pi)

    # Assign to shells
    shell_idx = _assign_to_shells_np(s_mag, shell_edges)

    # Initialize coefficient arrays
    flm = np.zeros((P, L, 2*L - 1), dtype=np.complex128)
    shell_counts = np.zeros(P, dtype=np.int64)

    # Compute SH coefficients for each shell using numba-accelerated function
    for p in range(P):
        mask = shell_idx == p
        count = mask.sum()
        shell_counts[p] = count

        if count == 0:
            continue

        E_shell = E_values[mask].copy()
        cos_theta_shell = cos_theta[mask]
        phi_shell = phi[mask]

        if normalize_shells:
            # Mean-center (removes DC bias)
            E_shell = E_shell - E_shell.mean()
            # Variance-normalize (makes correlations comparable across shells)
            E_std = E_shell.std()
            if E_std > 1e-10:
                E_shell = E_shell / E_std

        # Use numba-accelerated analytical SH computation
        flm[p] = _compute_sh_coeffs_analytical_numba(E_shell, cos_theta_shell, phi_shell, L)

    return BallHarmonicCoefficients(
        flmp=flm,
        L=L,
        P=P,
        shell_edges=shell_edges,
        shell_centers=shell_centers,
        shell_counts=shell_counts,
    )


def compute_ball_cross_correlation_coefficients(
    f_coeffs: BallHarmonicCoefficients,
    g_coeffs: BallHarmonicCoefficients,
    radial_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Wigner coefficients for ball cross-correlation.

    The cross-correlation is:
        C(R) = Σ_{p,l,m,n} f*_{p,l,m} g_{p,l,n} D^l_{m,n}(R)
             = Σ_{l,m,n} ξ_{l,m,n} D^l_{m,n}(R)

    where ξ_{l,m,n} = Σ_p w_p f*_{p,l,m} g_{p,l,n}

    Parameters
    ----------
    f_coeffs : BallHarmonicCoefficients
        Ball harmonic coefficients of function f (observed).
    g_coeffs : BallHarmonicCoefficients
        Ball harmonic coefficients of function g (calculated).
    radial_weights : np.ndarray, optional
        Weights for each radial shell, shape (P,).
        Default: uniform weights based on shell counts.

    Returns
    -------
    xi_nlm : np.ndarray
        Wigner coefficients, shape (2N-1, L, 2L-1) where N=L.
    """
    assert f_coeffs.L == g_coeffs.L, "Angular bandlimits must match"
    assert f_coeffs.P == g_coeffs.P, "Radial bandlimits must match"

    L = f_coeffs.L
    P = f_coeffs.P
    N = L

    if radial_weights is None:
        # Weight by number of reflections in each shell (normalized)
        radial_weights = f_coeffs.shell_counts.astype(np.float64)
        radial_weights = np.where(radial_weights > 0, radial_weights, 0)

    # Normalize weights
    weight_sum = radial_weights.sum()
    if weight_sum > 0:
        radial_weights = radial_weights / weight_sum
    else:
        radial_weights = np.ones(P) / P

    # Initialize Wigner coefficients: (2N-1, L, 2L-1) = [n_idx, l, m_idx]
    xi_nlm = np.zeros((2*N - 1, L, 2*L - 1), dtype=np.complex128)

    # f_coeffs.flmp has shape (P, L, 2L-1)
    # Sum over radial index p
    #
    # For cross-correlation C(R) that finds rotation R such that g(R⁻¹x) ≈ f(x):
    #   C(R) = ∫ f(x) g(R⁻¹x) dx = Σ f*_{lm} g_{ln} D^l_{mn}(R)
    #
    # But D^l_{mn}(R) convention in s2ball gives R^T, so we swap f↔g:
    #   ξ[n_idx, l, m_idx] = Σ_p w_p * conj(g[l, m_idx]) * f[l, n_idx]
    #
    for p in range(P):
        w = radial_weights[p]
        f_lm = f_coeffs.flmp[p]  # (L, 2L-1)
        g_lm = g_coeffs.flmp[p]  # (L, 2L-1)

        # Swap f and g to get R instead of R^T
        g_conj = np.conj(g_lm)  # (L, 2L-1)

        # For s2fft/s2ball wigner convention:
        # coeffs[n_idx, l, m_idx] corresponds to D^l_{m,n}
        # where m = m_idx - (L-1), n = n_idx - (N-1)

        # Build the product: for each l, compute outer product over m and n
        for l in range(L):
            # Valid m range: -l to l, i.e., m_idx from L-1-l to L-1+l
            m_start = L - 1 - l
            m_end = L - 1 + l + 1

            # Valid n range: -l to l, i.e., n_idx from N-1-l to N-1+l
            n_start = N - 1 - l
            n_end = N - 1 + l + 1

            # g_conj[l, m_start:m_end] shape: (2l+1,)
            # f_lm[l, n_start:n_end] shape: (2l+1,)
            g_l = g_conj[l, m_start:m_end]  # (2l+1,) - conjugated g for m index
            f_l = f_lm[l, n_start:n_end]    # (2l+1,) - f for n index

            # Outer product: (2l+1, 2l+1) -> [n, m]
            outer = np.outer(f_l, g_l)  # (2l+1, 2l+1) = [n, m]

            xi_nlm[n_start:n_end, l, m_start:m_end] += w * outer

    return xi_nlm


def evaluate_rotation_function(
    xi_nlm: np.ndarray,
    L: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate rotation function from Wigner coefficients using s2ball.

    Parameters
    ----------
    xi_nlm : np.ndarray
        Wigner coefficients, shape (2N-1, L, 2L-1).
    L : int
        Angular bandlimit.

    Returns
    -------
    rotation_function : np.ndarray
        Rotation function, shape (2N-1, L, 2L-1) = (gamma, beta, alpha).
    alphas : np.ndarray
        Alpha angle grid.
    betas : np.ndarray
        Beta angle grid.
    gammas : np.ndarray
        Gamma angle grid.
    """
    N = L

    # Inverse Wigner transform
    rotation_function = wigner_transform.inverse(
        xi_nlm,
        L=L,
        N=N,
        method="jax",
    )

    # MW sampling grid positions
    betas = np.array([(2*t + 1) * np.pi / (2*L) for t in range(L)])
    alphas = np.array([2 * np.pi * p / (2*L - 1) for p in range(2*L - 1)])
    gammas = np.array([2 * np.pi * p / (2*N - 1) for p in range(2*N - 1)])

    return np.asarray(rotation_function), alphas, betas, gammas


def ball_rotation_search(
    E_obs: np.ndarray,
    s_obs: np.ndarray,
    E_calc: np.ndarray,
    s_calc: np.ndarray,
    L: int = 32,
    P: int = 20,
    d_min: float = 4.0,
    d_max: float = 50.0,
    n_peaks: int = 100,
    radial_weights: Optional[np.ndarray] = None,
    refine_subvoxel: bool = True,
    refine_analytical: bool = False,
    analytical_embedding: bool = True,
    equal_count_shells: bool = True,
    return_coefficients: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, tuple, list]:
    """
    Perform ball harmonic rotation function search.

    Main entry point for the ball-based fast rotation function.

    Parameters
    ----------
    E_obs : np.ndarray
        Observed E² values, shape (N_obs,).
    s_obs : np.ndarray
        Observed s-vectors in Å⁻¹, shape (N_obs, 3).
    E_calc : np.ndarray
        Calculated E² values, shape (N_calc,).
    s_calc : np.ndarray
        Calculated s-vectors in Å⁻¹, shape (N_calc, 3).
    L : int
        Angular bandlimit.
    P : int
        Radial bandlimit.
    d_min : float
        High resolution limit in Å.
    d_max : float
        Low resolution limit in Å.
    n_peaks : int
        Number of peaks to extract.
    radial_weights : np.ndarray, optional
        Weights for radial shells.
    refine_subvoxel : bool
        If True, refine peak positions using fast quadratic fitting.
        Ignored if refine_analytical=True.
    refine_analytical : bool
        If True, refine peak positions using analytical Wigner D-matrix
        evaluation. This is more accurate but slower than subvoxel refinement.
        Provides exact sub-grid positions for the band-limited rotation function.
    analytical_embedding : bool
        If True (default), compute spherical harmonic coefficients analytically
        from reflection positions using numba-accelerated computation. This
        avoids discretization errors from splatting and is significantly faster.
        If False, use the original splatting approach.
    equal_count_shells : bool
        If True (default), compute radial shell boundaries such that each shell
        has approximately the same number of reflections. This ensures equal
        contribution from each resolution shell. Only applies when
        analytical_embedding=True.
    return_coefficients : bool
        If True, also return the Wigner coefficients xi_nlm for later use
        (e.g., for rescoring or additional analytical refinement).
    verbose : bool
        Print progress.

    Returns
    -------
    rotation_function : np.ndarray
        Full rotation function, shape (2L-1, L, 2L-1).
    angles_grid : tuple
        (alphas, betas, gammas) angle grids.
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples.
    xi_nlm : np.ndarray, optional
        Wigner coefficients, shape (2N-1, L, 2L-1). Only returned if
        return_coefficients=True.
    """
    import time

    start_time = time.time()

    if verbose:
        print(f"Ball rotation search: L={L}, P={P}")
        print(f"Resolution range: {d_min:.2f} - {d_max:.2f} Å")
        print(f"Embedding method: {'analytical' if analytical_embedding else 'splatting'}")

    if analytical_embedding:
        # Analytical approach: compute SH coefficients directly from reflection positions
        # This avoids discretization errors from splatting and is much faster
        if verbose:
            print("Computing analytical spherical harmonic coefficients...")
            if equal_count_shells:
                print("  Using equal-count shell binning")

        coeffs_obs = compute_ball_harmonic_coefficients_analytical(
            E_obs, s_obs, L, P, d_min, d_max,
            normalize_shells=True, equal_count_shells=equal_count_shells
        )
        coeffs_calc = compute_ball_harmonic_coefficients_analytical(
            E_calc, s_calc, L, P, d_min, d_max,
            normalize_shells=True, equal_count_shells=equal_count_shells
        )

        if verbose:
            print(f"  Coefficients shape: {coeffs_obs.flmp.shape}")
            print(f"  Reflections per shell (obs): min={coeffs_obs.shell_counts.min()}, max={coeffs_obs.shell_counts.max()}")

    else:
        # Original splatting approach
        # Step 1: Splat E-values onto ball grids (uniform radial shells)
        if verbose:
            print("Splatting E-values onto ball grid...")

        ball_obs, shell_edges, shell_centers, shell_counts_obs = splat_evalues_to_ball(
            E_obs, s_obs, L, P, d_min, d_max
        )
        ball_calc, _, _, shell_counts_calc = splat_evalues_to_ball(
            E_calc, s_calc, L, P, d_min, d_max
        )

        if verbose:
            print(f"  Ball grid shape: {ball_obs.shape}")
            print(f"  Shell range: [{shell_edges[0]:.4f}, {shell_edges[-1]:.4f}] Å⁻¹")
            print(f"  Reflections per shell (obs): min={shell_counts_obs.min()}, max={shell_counts_obs.max()}")

        # Step 2: Compute spherical harmonic coefficients for each shell
        if verbose:
            print("Computing spherical harmonic coefficients per shell...")

        coeffs_obs = compute_ball_harmonic_coefficients(
            ball_obs, L, shell_edges, shell_centers, shell_counts_obs
        )
        coeffs_calc = compute_ball_harmonic_coefficients(
            ball_calc, L, shell_edges, shell_centers, shell_counts_calc
        )

        if verbose:
            print(f"  Coefficients shape: {coeffs_obs.flmp.shape}")

    # Step 3: Compute cross-correlation Wigner coefficients
    if verbose:
        print("Computing cross-correlation coefficients...")

    xi_nlm = compute_ball_cross_correlation_coefficients(
        coeffs_obs, coeffs_calc, radial_weights
    )

    if verbose:
        print(f"  Wigner coefficients shape: {xi_nlm.shape}")
        print(f"  Max |ξ|: {np.abs(xi_nlm).max():.6e}")

    # Step 4: Evaluate rotation function
    if verbose:
        print("Evaluating rotation function via inverse Wigner transform...")

    rotation_function, alphas, betas, gammas = evaluate_rotation_function(xi_nlm, L)

    rf_real = np.real(rotation_function)

    if verbose:
        print(f"  Rotation function shape: {rf_real.shape}")
        print(f"  RF range: [{rf_real.min():.4f}, {rf_real.max():.4f}]")
        print(f"  RF mean: {rf_real.mean():.4f}, std: {rf_real.std():.4f}")

    # Step 5: Find peaks
    if verbose:
        print("Finding peaks...")

    peaks = find_rotation_peaks(rf_real, alphas, betas, gammas, n_peaks=n_peaks)

    # Step 6: Optionally refine peaks
    if peaks:
        if refine_analytical:
            # Analytical refinement using Wigner D-matrices (more accurate)
            if verbose:
                print("Refining peaks using analytical Wigner D-matrices...")

            rf_mean = rf_real.mean()
            rf_std = rf_real.std()
            peaks = refine_peaks_analytical(
                peaks, xi_nlm, L, rf_mean=rf_mean, rf_std=rf_std, verbose=verbose
            )
        elif refine_subvoxel:
            # Fast quadratic subvoxel refinement
            if verbose:
                print("Refining peaks to sub-voxel accuracy...")

            peaks = refine_peaks_subvoxel_wrapper(
                peaks, rf_real, alphas, betas, gammas
            )

            if verbose:
                print(f"  Refined {len(peaks)} peaks")

    elapsed = time.time() - start_time
    if verbose:
        print(f"Ball rotation search completed in {elapsed:.2f}s")
        if peaks:
            print(f"Top peak: alpha={np.degrees(peaks[0][0]):.2f}°, "
                  f"beta={np.degrees(peaks[0][1]):.2f}°, "
                  f"gamma={np.degrees(peaks[0][2]):.2f}°, "
                  f"sigma={peaks[0][4]:.2f}")

    if return_coefficients:
        return rf_real, (alphas, betas, gammas), peaks, xi_nlm
    return rf_real, (alphas, betas, gammas), peaks


def refine_peaks_subvoxel_wrapper(
    peaks: list,
    rotation_function: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
) -> list:
    """
    Refine peak positions to sub-voxel accuracy using quadratic fitting.

    Parameters
    ----------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples.
    rotation_function : np.ndarray
        Rotation function grid, shape (n_gamma, n_beta, n_alpha).
    alphas, betas, gammas : np.ndarray
        Angle grids.

    Returns
    -------
    refined_peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples with refined positions.
    """
    import jax.numpy as jnp
    from torchref.alignment.jax_subpixel_peaks import refine_peaks_subvoxel

    if not peaks:
        return peaks

    n_gamma, n_beta, n_alpha = rotation_function.shape

    # Compute angle spacings
    d_alpha = alphas[1] - alphas[0] if len(alphas) > 1 else 2 * np.pi / n_alpha
    d_beta = betas[1] - betas[0] if len(betas) > 1 else np.pi / n_beta
    d_gamma = gammas[1] - gammas[0] if len(gammas) > 1 else 2 * np.pi / n_gamma

    # Convert peaks to grid indices
    peak_indices = []
    for alpha, beta, gamma, score, sigma in peaks:
        # Find nearest grid indices
        a_idx = int(round((alpha - alphas[0]) / d_alpha)) % n_alpha
        b_idx = int(round((beta - betas[0]) / d_beta))
        b_idx = max(0, min(b_idx, n_beta - 1))
        g_idx = int(round((gamma - gammas[0]) / d_gamma)) % n_gamma

        peak_indices.append([g_idx, b_idx, a_idx])  # shape matches rf: (gamma, beta, alpha)

    peak_indices = jnp.array(peak_indices, dtype=jnp.int32)
    grid_jax = jnp.array(rotation_function)

    # Call JAX subvoxel refinement
    refined_coords, refined_values = refine_peaks_subvoxel(grid_jax, peak_indices)

    # Convert back to numpy
    refined_coords = np.array(refined_coords)
    refined_values = np.array(refined_values)

    # Compute mean and std for sigma calculation
    rf_mean = rotation_function.mean()
    rf_std = rotation_function.std()

    # Convert refined grid indices back to angles
    refined_peaks = []
    for i, ((alpha_orig, beta_orig, gamma_orig, score_orig, sigma_orig), refined_val) in enumerate(
        zip(peaks, refined_values)
    ):
        g_refined, b_refined, a_refined = refined_coords[i]

        # Convert to angles (with periodic wrapping for alpha and gamma)
        alpha_new = alphas[0] + a_refined * d_alpha
        alpha_new = alpha_new % (2 * np.pi)

        beta_new = betas[0] + b_refined * d_beta
        beta_new = np.clip(beta_new, 0, np.pi)

        gamma_new = gammas[0] + g_refined * d_gamma
        gamma_new = gamma_new % (2 * np.pi)

        # Compute refined sigma
        sigma_new = (refined_val - rf_mean) / rf_std if rf_std > 1e-10 else 0.0

        refined_peaks.append((alpha_new, beta_new, gamma_new, float(refined_val), float(sigma_new)))

    # Note: We do NOT re-sort by refined score because the quadratic interpolation
    # gives an approximate value, not the true score. Keeping original order
    # preserves the ranking from the grid-based search.
    return refined_peaks


# =============================================================================
# Analytical Peak Refinement using Wigner D-matrices
# =============================================================================

def build_wigner_index_mapping(L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build index mapping from xi_nlm array to Wigner D-matrix indices.

    Pre-computes the mapping between the xi_nlm coefficient array
    (shape: 2N-1, L, 2L-1) and the flattened Wigner D-matrix from spherical.

    Parameters
    ----------
    L : int
        Angular bandlimit.

    Returns
    -------
    xi_indices : np.ndarray
        Array of (n_idx, l, m_idx) indices, shape (n_terms, 3).
    D_indices : np.ndarray
        Corresponding indices into spherical's D array, shape (n_terms,).
    l_values : np.ndarray
        The l value for each term, shape (n_terms,).
    """
    N = L
    xi_indices = []
    D_indices = []
    l_values = []

    for ell in range(L):
        for m in range(-ell, ell + 1):
            m_idx = m + (L - 1)
            for n in range(-ell, ell + 1):
                n_idx = n + (N - 1)
                D_idx = spherical.WignerDindex(ell, m, n)
                xi_indices.append((n_idx, ell, m_idx))
                D_indices.append(D_idx)
                l_values.append(ell)

    return (np.array(xi_indices, dtype=np.int32),
            np.array(D_indices, dtype=np.int32),
            np.array(l_values, dtype=np.int32))


def evaluate_rotation_function_at_angles(
    xi_nlm: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    wigner: spherical.Wigner,
    xi_indices: np.ndarray,
    D_indices: np.ndarray,
    l_values: np.ndarray,
) -> float:
    """
    Evaluate rotation function at arbitrary Euler angles using Wigner D-matrices.

    The s2ball inverse Wigner transform uses the normalization:
        RF(α,β,γ) = Σ_{l,m,n} ξ_{l,m,n} * [(2l+1)/(8π²)] * D^l_{m,n}(α,β,γ)

    This allows exact evaluation of the band-limited rotation function at
    any point, not just grid points.

    Parameters
    ----------
    xi_nlm : np.ndarray
        Wigner coefficients from compute_ball_cross_correlation_coefficients(),
        shape (2N-1, L, 2L-1).
    alpha, beta, gamma : float
        ZYZ Euler angles in radians.
    wigner : spherical.Wigner
        Pre-initialized Wigner calculator.
    xi_indices, D_indices : np.ndarray
        Pre-computed index mappings from build_wigner_index_mapping().
    l_values : np.ndarray
        The l value for each term, shape (n_terms,).

    Returns
    -------
    float
        Rotation function value (real part).
    """
    R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
    D_all = wigner.D(R)

    xi_flat = xi_nlm[xi_indices[:, 0], xi_indices[:, 1], xi_indices[:, 2]]
    D_flat = D_all[D_indices]

    # Apply the s2ball normalization factor (2l+1)/(8π²) for each term
    norm_factors = (2 * l_values + 1) / (8 * np.pi**2)

    return np.sum(xi_flat * D_flat * norm_factors).real


def refine_peaks_analytical(
    peaks: list,
    xi_nlm: np.ndarray,
    L: int,
    rf_mean: float = None,
    rf_std: float = None,
    verbose: bool = False,
) -> list:
    """
    Refine peak positions using analytical Wigner D-matrix evaluation.

    This provides exact sub-grid peak positions by evaluating the rotation
    function directly from its Wigner coefficient expansion, rather than
    using grid interpolation.

    Parameters
    ----------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples from grid search.
    xi_nlm : np.ndarray
        Wigner coefficients from compute_ball_cross_correlation_coefficients(),
        shape (2N-1, L, 2L-1).
    L : int
        Angular bandlimit.
    rf_mean, rf_std : float, optional
        Mean and std of rotation function for sigma calculation.
        If None, sigma values are preserved from input.
    verbose : bool
        Print progress.

    Returns
    -------
    refined_peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples with refined positions.
    """
    from scipy.optimize import minimize

    if not peaks:
        return peaks

    # Pre-compute index mappings (done once)
    if verbose:
        print("  Building Wigner index mappings...")
    xi_indices, D_indices, l_values = build_wigner_index_mapping(L)
    wigner = spherical.Wigner(L - 1)

    # Grid spacing for search bounds
    d_angle = 2 * np.pi / (2 * L - 1)
    d_beta = np.pi / L

    refined_peaks = []

    for i, (alpha, beta, gamma, score, sigma) in enumerate(peaks):
        def neg_rf(angles):
            return -evaluate_rotation_function_at_angles(
                xi_nlm, angles[0], angles[1], angles[2],
                wigner, xi_indices, D_indices, l_values
            )

        # Local search within ~2 grid spacings
        bounds = [
            (alpha - 2*d_angle, alpha + 2*d_angle),
            (max(0.01, beta - 2*d_beta), min(np.pi - 0.01, beta + 2*d_beta)),
            (gamma - 2*d_angle, gamma + 2*d_angle)
        ]

        result = minimize(neg_rf, [alpha, beta, gamma],
                         method='L-BFGS-B', bounds=bounds,
                         options={'ftol': 1e-10, 'gtol': 1e-10})

        alpha_ref = result.x[0] % (2 * np.pi)
        beta_ref = np.clip(result.x[1], 0, np.pi)
        gamma_ref = result.x[2] % (2 * np.pi)
        score_ref = -result.fun

        # Compute sigma if stats provided
        if rf_mean is not None and rf_std is not None and rf_std > 1e-10:
            sigma_ref = (score_ref - rf_mean) / rf_std
        else:
            sigma_ref = sigma  # Keep original

        refined_peaks.append((alpha_ref, beta_ref, gamma_ref, score_ref, sigma_ref))

        if verbose and (i + 1) % 100 == 0:
            print(f"  Refined {i + 1}/{len(peaks)} peaks")

    if verbose:
        print(f"  Refined {len(peaks)} peaks")

    return refined_peaks


def find_rotation_peaks(
    rotation_function: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    gammas: np.ndarray,
    n_peaks: int = 100,
    sigma_cutoff: float = 2.0,
    cluster_radius_deg: float = 5.0,
) -> list:
    """
    Extract and cluster peaks from rotation function.

    Parameters
    ----------
    rotation_function : np.ndarray
        Rotation function, shape (n_gamma, n_beta, n_alpha).
    alphas, betas, gammas : np.ndarray
        Angle grids.
    n_peaks : int
        Maximum number of peaks.
    sigma_cutoff : float
        Minimum sigma above mean.
    cluster_radius_deg : float
        Clustering radius in degrees.

    Returns
    -------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples.
    """
    rf_mean = rotation_function.mean()
    rf_std = rotation_function.std()

    if rf_std < 1e-10:
        return []

    threshold = rf_mean + sigma_cutoff * rf_std

    # Get sorted indices (descending)
    flat_rf = rotation_function.flatten()
    sorted_idx = np.argsort(flat_rf)[::-1]

    n_gamma, n_beta, n_alpha = rotation_function.shape
    cluster_rad = np.radians(cluster_radius_deg)

    peaks = []
    used_angles = []

    for flat_i in sorted_idx:
        if len(peaks) >= n_peaks:
            break

        g_idx, b_idx, a_idx = np.unravel_index(flat_i, rotation_function.shape)
        score = rotation_function[g_idx, b_idx, a_idx]

        if score < threshold:
            break

        alpha = alphas[a_idx]
        beta = betas[b_idx]
        gamma = gammas[g_idx]

        # Check if too close to existing peak
        is_new = True
        for prev_alpha, prev_beta, prev_gamma in used_angles:
            da = min(abs(alpha - prev_alpha), 2*np.pi - abs(alpha - prev_alpha))
            db = abs(beta - prev_beta)
            dg = min(abs(gamma - prev_gamma), 2*np.pi - abs(gamma - prev_gamma))
            if np.sqrt(da**2 + db**2 + dg**2) < cluster_rad:
                is_new = False
                break

        if is_new:
            sigma = (score - rf_mean) / rf_std
            peaks.append((alpha, beta, gamma, score, sigma))
            used_angles.append((alpha, beta, gamma))

    return peaks


def rotation_matrix_from_euler_zyz(
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Create rotation matrix from ZYZ Euler angles.

    R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    R = np.array([
        [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
        [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
        [-sb*cg,            sb*sg,             cb]
    ])

    return R


def check_rotation_recovery(
    peaks: list,
    true_alpha: float,
    true_beta: float,
    true_gamma: float,
    symmetry_matrices: Optional[np.ndarray] = None,
    tolerance_deg: float = 10.0,
) -> Tuple[bool, int, float]:
    """
    Check if true rotation was recovered among top peaks.

    Parameters
    ----------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) peaks.
    true_alpha, true_beta, true_gamma : float
        True rotation angles in radians.
    symmetry_matrices : np.ndarray, optional
        Symmetry operations for equivalence checking.
    tolerance_deg : float
        Angular tolerance in degrees.

    Returns
    -------
    found : bool
        Whether the rotation was found.
    rank : int
        Rank of matching peak (0-indexed), or -1.
    min_error : float
        Minimum angular error in degrees.
    """
    R_true = rotation_matrix_from_euler_zyz(true_alpha, true_beta, true_gamma)
    tolerance_rad = np.radians(tolerance_deg)

    min_error = float('inf')
    best_rank = -1

    for rank, (alpha, beta, gamma, score, sigma) in enumerate(peaks):
        R_peak = rotation_matrix_from_euler_zyz(alpha, beta, gamma)

        error = _rotation_matrix_error(R_peak, R_true)

        if error < min_error:
            min_error = error
            best_rank = rank

        if symmetry_matrices is not None:
            for S in symmetry_matrices:
                S_rot = S[:3, :3] if S.shape[0] > 3 else S
                for R_combined in [S_rot @ R_true, R_true @ S_rot]:
                    err = _rotation_matrix_error(R_peak, R_combined)
                    if err < min_error:
                        min_error = err
                        best_rank = rank

    found = min_error < tolerance_rad
    return found, best_rank, np.degrees(min_error)


def _rotation_matrix_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute angular error between rotation matrices."""
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle = np.arccos((trace - 1) / 2)
    return abs(angle)


@dataclass
class RotationCluster:
    """
    Container for a cluster of rotation peaks.

    Attributes
    ----------
    alpha : float
        Alpha angle of best peak (radians).
    beta : float
        Beta angle of best peak (radians).
    gamma : float
        Gamma angle of best peak (radians).
    best_score : float
        Score of the best peak in cluster.
    best_sigma : float
        Sigma of the best peak.
    size : int
        Number of peaks in cluster.
    sum_score : float
        Sum of all scores in cluster.
    mean_score : float
        Mean score of peaks in cluster.
    sum_sigma : float
        Sum of all sigma values in cluster.
    mean_sigma : float
        Mean sigma of peaks in cluster.
    score_std : float
        Standard deviation of scores in cluster.
    avg_alpha : float
        Score-weighted average alpha (radians).
    avg_beta : float
        Score-weighted average beta (radians).
    avg_gamma : float
        Score-weighted average gamma (radians).
    angular_spread : float
        Average angular distance of peaks from cluster center (degrees).
    """
    alpha: float
    beta: float
    gamma: float
    best_score: float
    best_sigma: float
    size: int
    sum_score: float
    mean_score: float
    sum_sigma: float
    mean_sigma: float
    score_std: float
    avg_alpha: float
    avg_beta: float
    avg_gamma: float
    angular_spread: float

    def to_tuple(self) -> tuple:
        """Return basic tuple (alpha, beta, gamma, score, sigma, size)."""
        return (self.alpha, self.beta, self.gamma, self.best_score, self.best_sigma, self.size)

    def __repr__(self) -> str:
        return (
            f"RotationCluster(α={np.degrees(self.alpha):.1f}°, "
            f"β={np.degrees(self.beta):.1f}°, γ={np.degrees(self.gamma):.1f}°, "
            f"score={self.best_score:.4f}, size={self.size}, "
            f"sum_score={self.sum_score:.4f}, spread={self.angular_spread:.2f}°)"
        )


def _average_quaternions_weighted(quaternions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted average of quaternions.

    Uses the eigenvector method for quaternion averaging.

    Parameters
    ----------
    quaternions : np.ndarray
        Array of quaternions [w, x, y, z], shape (N, 4).
    weights : np.ndarray
        Weights for each quaternion, shape (N,).

    Returns
    -------
    avg_q : np.ndarray
        Averaged quaternion [w, x, y, z], shape (4,).
    """
    # Normalize weights
    weights = weights / weights.sum()

    # Build weighted outer product sum
    M = np.zeros((4, 4))
    for q, w in zip(quaternions, weights):
        M += w * np.outer(q, q)

    # Eigenvector with largest eigenvalue is the average
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg_q = eigenvectors[:, -1]  # Largest eigenvalue

    # Ensure w >= 0
    if avg_q[0] < 0:
        avg_q = -avg_q

    return avg_q


def _quaternion_to_euler_zyz(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to ZYZ Euler angles."""
    w, x, y, z = q

    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return rotation_matrix_to_euler_zyz(R)


def cluster_rotation_peaks(
    peaks: list,
    cluster_radius_deg: float = 5.0,
    symmetry_matrices: Optional[np.ndarray] = None,
    return_details: bool = False,
) -> list:
    """
    Cluster rotation peaks based on rotation matrix distance.

    Uses greedy clustering: iterates through peaks (sorted by score), and
    assigns each peak to an existing cluster if within the angular threshold,
    otherwise creates a new cluster. Returns the best peak from each cluster.

    Parameters
    ----------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples.
        Angles are in radians.
    cluster_radius_deg : float
        Maximum angular distance (in degrees) for peaks to be in the same cluster.
    symmetry_matrices : np.ndarray, optional
        If provided, considers symmetry-equivalent rotations when clustering.
        Shape (N, 3, 3) or (N, 4, 4).
    return_details : bool
        If True, return list of RotationCluster objects with full statistics.
        If False, return simple tuples (alpha, beta, gamma, score, sigma, size).

    Returns
    -------
    clustered_peaks : list
        If return_details=False: List of (alpha, beta, gamma, score, sigma, cluster_size) tuples.
        If return_details=True: List of RotationCluster objects with aggregate statistics.
        Sorted by best score (descending).
    """
    if not peaks:
        return []

    cluster_radius_rad = np.radians(cluster_radius_deg)

    # Sort peaks by score (descending)
    sorted_peaks = sorted(peaks, key=lambda p: p[3], reverse=True)

    # Each cluster stores: (best_peak, rotation_matrix, list_of_all_peaks)
    clusters = []

    for peak in sorted_peaks:
        alpha, beta, gamma = peak[0], peak[1], peak[2]
        R_peak = rotation_matrix_from_euler_zyz(alpha, beta, gamma)

        # Find closest cluster
        best_cluster_idx = -1
        best_distance = float('inf')

        for i, (_, R_cluster, _) in enumerate(clusters):
            # Direct distance
            dist = _rotation_matrix_error(R_peak, R_cluster)

            # Check symmetry-equivalent rotations if provided
            if symmetry_matrices is not None:
                for S in symmetry_matrices:
                    S_rot = S[:3, :3] if S.shape[0] > 3 else S
                    # Check both S @ R_peak and R_peak @ S
                    for R_equiv in [S_rot @ R_peak, R_peak @ S_rot]:
                        d = _rotation_matrix_error(R_equiv, R_cluster)
                        dist = min(dist, d)

            if dist < best_distance:
                best_distance = dist
                best_cluster_idx = i

        if best_distance < cluster_radius_rad and best_cluster_idx >= 0:
            # Add to existing cluster
            best_peak, R_cluster, peak_list = clusters[best_cluster_idx]
            peak_list.append(peak)
            clusters[best_cluster_idx] = (best_peak, R_cluster, peak_list)
        else:
            # Create new cluster
            clusters.append((peak, R_peak, [peak]))

    # Compute cluster statistics
    clustered_peaks = []
    for best_peak, R_center, peak_list in clusters:
        alpha, beta, gamma, score, sigma = best_peak[:5]
        size = len(peak_list)

        # Basic statistics
        scores = np.array([p[3] for p in peak_list])
        sigmas = np.array([p[4] for p in peak_list])

        sum_score = scores.sum()
        mean_score = scores.mean()
        sum_sigma = sigmas.sum()
        mean_sigma = sigmas.mean()
        score_std = scores.std() if size > 1 else 0.0

        # Compute weighted average rotation using quaternions
        if size > 1:
            quaternions = []
            weights = []
            for p in peak_list:
                R_p = rotation_matrix_from_euler_zyz(p[0], p[1], p[2])
                q = rotation_matrix_to_quaternion(R_p)
                quaternions.append(q)
                weights.append(max(p[3], 0.0))  # Use score as weight (non-negative)

            quaternions = np.array(quaternions)
            weights = np.array(weights)

            if weights.sum() > 0:
                avg_q = _average_quaternions_weighted(quaternions, weights)
                avg_alpha, avg_beta, avg_gamma = _quaternion_to_euler_zyz(avg_q)
            else:
                avg_alpha, avg_beta, avg_gamma = alpha, beta, gamma
        else:
            avg_alpha, avg_beta, avg_gamma = alpha, beta, gamma

        # Compute angular spread (average distance from center)
        if size > 1:
            distances = []
            for p in peak_list:
                R_p = rotation_matrix_from_euler_zyz(p[0], p[1], p[2])
                dist = _rotation_matrix_error(R_p, R_center)
                distances.append(dist)
            angular_spread = np.degrees(np.mean(distances))
        else:
            angular_spread = 0.0

        if return_details:
            cluster = RotationCluster(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                best_score=score,
                best_sigma=sigma,
                size=size,
                sum_score=sum_score,
                mean_score=mean_score,
                sum_sigma=sum_sigma,
                mean_sigma=mean_sigma,
                score_std=score_std,
                avg_alpha=avg_alpha,
                avg_beta=avg_beta,
                avg_gamma=avg_gamma,
                angular_spread=angular_spread,
            )
            clustered_peaks.append(cluster)
        else:
            clustered_peaks.append((alpha, beta, gamma, score, sigma, size))

    # Sort by best score (descending)
    if return_details:
        clustered_peaks.sort(key=lambda c: c.best_score, reverse=True)
    else:
        clustered_peaks.sort(key=lambda p: p[3], reverse=True)

    return clustered_peaks


def cluster_rotation_peaks_torch(
    peaks: list,
    cluster_radius_deg: float = 5.0,
    symmetry_matrices: Optional[torch.Tensor] = None,
    return_details: bool = False,
) -> list:
    """
    Torch wrapper for cluster_rotation_peaks.

    Parameters
    ----------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples.
    cluster_radius_deg : float
        Maximum angular distance for clustering.
    symmetry_matrices : torch.Tensor, optional
        Symmetry matrices, shape (N, 3, 3) or (N, 4, 4).
    return_details : bool
        If True, return RotationCluster objects with full statistics.

    Returns
    -------
    clustered_peaks : list
        If return_details=False: List of (alpha, beta, gamma, score, sigma, cluster_size) tuples.
        If return_details=True: List of RotationCluster objects.
    """
    sym_np = None
    if symmetry_matrices is not None:
        sym_np = symmetry_matrices.detach().cpu().numpy()
    return cluster_rotation_peaks(peaks, cluster_radius_deg, sym_np, return_details)


# Torch convenience wrappers
def ball_rotation_search_torch(
    E_obs: torch.Tensor,
    s_obs: torch.Tensor,
    E_calc: torch.Tensor,
    s_calc: torch.Tensor,
    **kwargs,
) -> Tuple[np.ndarray, tuple, list]:
    """Torch wrapper for ball_rotation_search."""
    return ball_rotation_search(
        E_obs.detach().cpu().numpy(),
        s_obs.detach().cpu().numpy(),
        E_calc.detach().cpu().numpy(),
        s_calc.detach().cpu().numpy(),
        **kwargs,
    )


# =============================================================================
# Symmetry Reduction Functions
# =============================================================================

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.

    Parameters
    ----------
    R : np.ndarray
        Rotation matrix, shape (3, 3).

    Returns
    -------
    q : np.ndarray
        Unit quaternion [w, x, y, z], shape (4,).
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    # Normalize
    q = q / np.linalg.norm(q)
    # Ensure w >= 0 for canonical form (q and -q represent same rotation)
    if q[0] < 0:
        q = -q
    return q


def quaternion_to_euler_zyz(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert unit quaternion [w, x, y, z] to ZYZ Euler angles.

    Parameters
    ----------
    q : np.ndarray
        Unit quaternion [w, x, y, z], shape (4,).

    Returns
    -------
    alpha, beta, gamma : float
        ZYZ Euler angles in radians.
    """
    w, x, y, z = q

    # First convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return rotation_matrix_to_euler_zyz(R)


def rotation_matrix_to_euler_zyz(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract ZYZ Euler angles from rotation matrix.

    R = Rz(alpha) @ Ry(beta) @ Rz(gamma)

    Parameters
    ----------
    R : np.ndarray
        Rotation matrix, shape (3, 3).

    Returns
    -------
    alpha, beta, gamma : float
        ZYZ Euler angles in radians, with:
        - alpha in [0, 2π)
        - beta in [0, π]
        - gamma in [0, 2π)
    """
    # beta from R[2,2] = cos(beta)
    cos_beta = np.clip(R[2, 2], -1.0, 1.0)
    beta = np.arccos(cos_beta)

    sin_beta = np.sin(beta)

    if np.abs(sin_beta) > 1e-10:
        # General case: sin(beta) != 0
        # alpha from R[0,2] = cos(alpha)*sin(beta), R[1,2] = sin(alpha)*sin(beta)
        alpha = np.arctan2(R[1, 2], R[0, 2])
        # gamma from R[2,0] = -sin(beta)*cos(gamma), R[2,1] = sin(beta)*sin(gamma)
        gamma = np.arctan2(R[2, 1], -R[2, 0])
    else:
        # Gimbal lock: beta ≈ 0 or beta ≈ π
        # Only (alpha + gamma) or (alpha - gamma) is determined
        gamma = 0.0
        if cos_beta > 0:  # beta ≈ 0
            alpha = np.arctan2(R[1, 0], R[0, 0])
        else:  # beta ≈ π
            alpha = np.arctan2(-R[1, 0], -R[0, 0])

    # Normalize to [0, 2π)
    alpha = alpha % (2 * np.pi)
    gamma = gamma % (2 * np.pi)

    return alpha, beta, gamma


def reduce_rotation_by_symmetry(
    alpha: float,
    beta: float,
    gamma: float,
    symmetry_matrices: np.ndarray,
) -> Tuple[float, float, float, int]:
    """
    Map rotation angles to a canonical representative using crystal symmetry.

    Given a rotation R and symmetry operations {S_i}, finds the canonical
    representative among all symmetry-equivalent rotations {S_i @ R}.
    The canonical form is chosen as the one with the smallest Euler angle
    norm (closest to identity rotation), with all angles positive.

    Parameters
    ----------
    alpha, beta, gamma : float
        ZYZ Euler angles in radians.
    symmetry_matrices : np.ndarray
        Crystallographic symmetry rotation matrices, shape (N, 3, 3).
        These should be the rotation parts of the space group operations.

    Returns
    -------
    alpha_red, beta_red, gamma_red : float
        Reduced ZYZ Euler angles in radians (all positive, smallest norm).
    sym_index : int
        Index of the symmetry operation that gave the canonical form.
    """
    R = rotation_matrix_from_euler_zyz(alpha, beta, gamma)

    best_norm = float('inf')
    best_angles = (alpha % (2 * np.pi), beta, gamma % (2 * np.pi))
    best_sym_idx = 0

    for i, S in enumerate(symmetry_matrices):
        # Extract 3x3 rotation part if needed
        S_rot = S[:3, :3] if S.shape[0] > 3 else S

        # Apply symmetry: S @ R
        R_equiv = S_rot @ R

        # Extract Euler angles (already positive from rotation_matrix_to_euler_zyz)
        a, b, g = rotation_matrix_to_euler_zyz(R_equiv)

        # Compute norm of the angle vector
        norm = np.sqrt(a**2 + b**2 + g**2)

        if norm < best_norm:
            best_norm = norm
            best_angles = (a, b, g)
            best_sym_idx = i

    return (*best_angles, best_sym_idx)


def reduce_peaks_by_symmetry(
    peaks: list,
    symmetry_matrices: np.ndarray,
) -> list:
    """
    Reduce a list of rotation peaks to canonical symmetry representatives.

    Parameters
    ----------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples.
        Angles are in radians.
    symmetry_matrices : np.ndarray
        Crystallographic symmetry rotation matrices, shape (N, 3, 3).

    Returns
    -------
    reduced_peaks : list
        List of (alpha, beta, gamma, score, sigma, sym_index) tuples.
        Angles are the canonical representatives in radians.
    """
    reduced_peaks = []

    for alpha, beta, gamma, score, sigma in peaks:
        alpha_red, beta_red, gamma_red, sym_idx = reduce_rotation_by_symmetry(
            alpha, beta, gamma, symmetry_matrices
        )
        reduced_peaks.append((alpha_red, beta_red, gamma_red, score, sigma, sym_idx))

    return reduced_peaks


def reduce_peaks_by_symmetry_torch(
    peaks: list,
    symmetry_matrices: torch.Tensor,
) -> list:
    """
    Torch wrapper for reduce_peaks_by_symmetry.

    Parameters
    ----------
    peaks : list
        List of (alpha, beta, gamma, score, sigma) tuples.
    symmetry_matrices : torch.Tensor
        Symmetry matrices, shape (N, 3, 3) or (N, 4, 4).

    Returns
    -------
    reduced_peaks : list
        List of (alpha, beta, gamma, score, sigma, sym_index) tuples.
    """
    sym_np = symmetry_matrices.detach().cpu().numpy()
    return reduce_peaks_by_symmetry(peaks, sym_np)
