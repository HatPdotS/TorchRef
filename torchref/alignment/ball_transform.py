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
import warnings

import numpy as np
import torch

# Enable JAX 64-bit precision before importing s2fft/s2ball
import jax
jax.config.update("jax_enable_x64", True)

import s2fft
import s2ball.transform.wigner as wigner_transform

# Import normalization functions from the centralized module
from torchref.math_functions.normalization import (
    compute_radial_shells as _compute_radial_shells_torch,
    assign_to_shells as _assign_to_shells_torch,
    compute_anisotropy_correction as _compute_anisotropy_correction_torch,
    fit_anisotropy_correction as _fit_anisotropy_correction_torch,
    apply_anisotropy_correction as _apply_anisotropy_correction_torch,
    F_squared_to_E_values as _F_squared_to_E_values_torch,
)


# =============================================================================
# NumPy Wrappers for Backward Compatibility
# =============================================================================
# These functions wrap the PyTorch implementations for NumPy inputs.
# They are kept for backward compatibility with existing code.


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def _to_torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(np.asarray(array)).to(device)


def U_to_matrix_np(U: np.ndarray) -> np.ndarray:
    """
    Convert anisotropic parameters from 6-component vector to 3x3 matrix.

    .. deprecated::
        Use `torchref.math_functions.math_torch.U_to_matrix` instead.

    Parameters
    ----------
    U : np.ndarray
        Anisotropic parameters [u11, u22, u33, u12, u13, u23], shape (6,).

    Returns
    -------
    np.ndarray
        Symmetric 3x3 matrix.
    """
    warnings.warn(
        "U_to_matrix_np is deprecated. Use torchref.math_functions.math_torch.U_to_matrix instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    U_matrix = np.array([
        [U[0], U[3], U[4]],
        [U[3], U[1], U[5]],
        [U[4], U[5], U[2]]
    ])
    return U_matrix


def compute_anisotropy_correction(
    s_vectors: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """
    Compute anisotropic correction factor for F² values.

    .. deprecated::
        Use `torchref.math_functions.normalization.compute_anisotropy_correction` instead.
    """
    warnings.warn(
        "compute_anisotropy_correction (NumPy version) is deprecated. "
        "Use torchref.math_functions.normalization.compute_anisotropy_correction instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    s_torch = _to_torch(s_vectors)
    U_torch = _to_torch(U)
    result = _compute_anisotropy_correction_torch(s_torch, U_torch)
    return _to_numpy(result)


def compute_shell_std(
    F2_values: np.ndarray,
    shell_idx: np.ndarray,
    P: int,
) -> float:
    """
    Compute mean standard deviation of F² values within resolution shells.

    .. deprecated::
        Use `torchref.math_functions.normalization.compute_shell_cv` instead.
    """
    warnings.warn(
        "compute_shell_std is deprecated. "
        "Use torchref.math_functions.normalization.compute_shell_cv instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from torchref.math_functions.normalization import compute_shell_cv
    F2_torch = _to_torch(F2_values)
    shell_torch = _to_torch(shell_idx).to(torch.int64)
    return compute_shell_cv(F2_torch, shell_torch, P)


def fit_anisotropy_correction(
    F2_values: np.ndarray,
    s_vectors: np.ndarray,
    P: int = 20,
    d_min: float = 4.0,
    d_max: float = 50.0,
    n_iterations: int = 100,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Fit anisotropy correction parameters to minimize variance within shells.

    .. deprecated::
        Use `torchref.math_functions.normalization.fit_anisotropy_correction` instead.
    """
    warnings.warn(
        "fit_anisotropy_correction (NumPy version) is deprecated. "
        "Use torchref.math_functions.normalization.fit_anisotropy_correction instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    F2_torch = _to_torch(F2_values)
    s_torch = _to_torch(s_vectors)
    U_torch, final_cv = _fit_anisotropy_correction_torch(
        F2_torch, s_torch, n_shells=P, d_min=d_min, d_max=d_max,
        n_iterations=n_iterations, verbose=verbose
    )
    return _to_numpy(U_torch), final_cv


def apply_anisotropy_correction(
    F2_values: np.ndarray,
    s_vectors: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """
    Apply anisotropic correction to F² values.

    .. deprecated::
        Use `torchref.math_functions.normalization.apply_anisotropy_correction` instead.
    """
    warnings.warn(
        "apply_anisotropy_correction (NumPy version) is deprecated. "
        "Use torchref.math_functions.normalization.apply_anisotropy_correction instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    F2_torch = _to_torch(F2_values)
    s_torch = _to_torch(s_vectors)
    U_torch = _to_torch(U)
    result = _apply_anisotropy_correction_torch(F2_torch, s_torch, U_torch)
    return _to_numpy(result)


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
    Internal NumPy version for ball-specific functions.
    """
    s_min = 1.0 / d_max  # Low resolution end
    s_max = 1.0 / d_min  # High resolution end

    shell_edges = np.linspace(s_min, s_max, P + 1)
    shell_centers = 0.5 * (shell_edges[:-1] + shell_edges[1:])

    return shell_edges, shell_centers


def compute_radial_shells(
    d_min: float,
    d_max: float,
    P: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute uniform radial shell boundaries in reciprocal space.

    .. deprecated::
        Use `torchref.math_functions.normalization.compute_radial_shells` instead.

    Shells are spaced uniformly in 1/d (|s|) for even coverage of resolution.

    Parameters
    ----------
    d_min : float
        High resolution limit in Å.
    d_max : float
        Low resolution limit in Å.
    P : int
        Number of radial shells.

    Returns
    -------
    shell_edges : np.ndarray
        Shell boundaries in Å⁻¹, shape (P+1,).
    shell_centers : np.ndarray
        Shell centers in Å⁻¹, shape (P,).
    """
    warnings.warn(
        "compute_radial_shells (NumPy version) is deprecated. "
        "Use torchref.math_functions.normalization.compute_radial_shells instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _compute_radial_shells_np(d_min, d_max, P)


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


def assign_to_shells(
    s_mag: np.ndarray,
    shell_edges: np.ndarray,
) -> np.ndarray:
    """
    Assign reflections to radial shells.

    .. deprecated::
        Use `torchref.math_functions.normalization.assign_to_shells` instead.

    Parameters
    ----------
    s_mag : np.ndarray
        |s| values in Å⁻¹, shape (N,).
    shell_edges : np.ndarray
        Shell boundaries in Å⁻¹, shape (P+1,).

    Returns
    -------
    shell_idx : np.ndarray
        Shell index for each reflection, shape (N,).
        Values 0 to P-1, or -1 for out-of-range.
    """
    warnings.warn(
        "assign_to_shells (NumPy version) is deprecated. "
        "Use torchref.math_functions.normalization.assign_to_shells instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _assign_to_shells_np(s_mag, shell_edges)


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
        If True, refine peak positions to sub-voxel accuracy.
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
    """
    import time

    start_time = time.time()

    if verbose:
        print(f"Ball rotation search: L={L}, P={P}")
        print(f"Resolution range: {d_min:.2f} - {d_max:.2f} Å")

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

    # Step 6: Optionally refine peaks to sub-voxel accuracy
    if refine_subvoxel and peaks:
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


def F2_to_E_values(
    F2_values: np.ndarray,
    s_vectors: np.ndarray,
    P: int = 20,
    d_min: float = 4.0,
    d_max: float = 50.0,
) -> np.ndarray:
    """
    Convert F² values to E-values by normalizing within resolution shells.

    .. deprecated::
        Use `torchref.math_functions.normalization.F_squared_to_E_values` instead.

    Parameters
    ----------
    F2_values : np.ndarray
        F² values, shape (N,).
    s_vectors : np.ndarray
        Reciprocal space vectors in Å⁻¹, shape (N, 3).
    P : int
        Number of resolution shells for normalization.
    d_min : float
        High resolution limit in Å.
    d_max : float
        Low resolution limit in Å.

    Returns
    -------
    E_values : np.ndarray
        Normalized E-values, shape (N,).
    """
    warnings.warn(
        "F2_to_E_values (NumPy version) is deprecated. "
        "Use torchref.math_functions.normalization.F_squared_to_E_values instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    F2_torch = _to_torch(F2_values)
    s_torch = _to_torch(s_vectors)
    E, _, _ = _F_squared_to_E_values_torch(
        F2_torch, s_torch, n_shells=P, d_min=d_min, d_max=d_max
    )
    return _to_numpy(E)


def ball_rotation_search_with_anisotropy(
    F2_obs: np.ndarray,
    s_obs: np.ndarray,
    F2_calc: np.ndarray,
    s_calc: np.ndarray,
    L: int = 32,
    P: int = 20,
    d_min: float = 4.0,
    d_max: float = 50.0,
    n_peaks: int = 100,
    fit_anisotropy: bool = True,
    U_init: Optional[np.ndarray] = None,
    radial_weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, tuple, list, Optional[np.ndarray]]:
    """
    Perform ball harmonic rotation search with optional anisotropy correction.

    This is the recommended entry point when working with F² values directly.
    Anisotropy correction is applied to F² BEFORE converting to E-values.

    Parameters
    ----------
    F2_obs : np.ndarray
        Observed F² values, shape (N_obs,).
    s_obs : np.ndarray
        Observed s-vectors in Å⁻¹, shape (N_obs, 3).
    F2_calc : np.ndarray
        Calculated F² values, shape (N_calc,).
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
    fit_anisotropy : bool
        If True, fit and apply anisotropy correction to F² values.
    U_init : np.ndarray, optional
        Initial U parameters (6,). If provided and fit_anisotropy=True,
        used as starting point. If fit_anisotropy=False, applied directly.
    radial_weights : np.ndarray, optional
        Weights for radial shells.
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
    U_fitted : np.ndarray or None
        Fitted anisotropy parameters (6,), or None if not fitted.
    """
    import time
    start_time = time.time()

    U_fitted = None

    if fit_anisotropy:
        if verbose:
            print("=" * 60)
            print("Step 1: Fitting anisotropy correction on observed F²")
            print("=" * 60)

        U_fitted, final_cv = fit_anisotropy_correction(
            F2_obs, s_obs, P=P, d_min=d_min, d_max=d_max, verbose=verbose
        )

        # Apply correction to both observed and calculated F²
        if verbose:
            print("\nApplying anisotropy correction to F² values...")

        F2_obs_corrected = apply_anisotropy_correction(F2_obs, s_obs, U_fitted)
        F2_calc_corrected = apply_anisotropy_correction(F2_calc, s_calc, U_fitted)

    elif U_init is not None:
        # Apply provided U without fitting
        if verbose:
            print("Applying provided anisotropy correction...")

        F2_obs_corrected = apply_anisotropy_correction(F2_obs, s_obs, U_init)
        F2_calc_corrected = apply_anisotropy_correction(F2_calc, s_calc, U_init)
        U_fitted = U_init

    else:
        # No anisotropy correction
        F2_obs_corrected = F2_obs
        F2_calc_corrected = F2_calc

    # Convert F² to E-values
    if verbose:
        print("\nConverting F² to E-values...")

    E_obs = F2_to_E_values(F2_obs_corrected, s_obs, P=50, d_min=d_min, d_max=d_max)
    E_calc = F2_to_E_values(F2_calc_corrected, s_calc, P=50, d_min=d_min, d_max=d_max)

    if verbose:
        print(f"  E_obs range: [{E_obs.min():.2f}, {E_obs.max():.2f}]")
        print(f"  E_calc range: [{E_calc.min():.2f}, {E_calc.max():.2f}]")

    # Run ball rotation search
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2: Ball harmonic rotation search")
        print("=" * 60)

    rotation_function, angles_grid, peaks = ball_rotation_search(
        E_obs, s_obs,
        E_calc, s_calc,
        L=L, P=P,
        d_min=d_min, d_max=d_max,
        n_peaks=n_peaks,
        radial_weights=radial_weights,
        verbose=verbose,
    )

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTotal time: {elapsed:.2f}s")

    return rotation_function, angles_grid, peaks, U_fitted


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


def ball_rotation_search_with_anisotropy_torch(
    F2_obs: torch.Tensor,
    s_obs: torch.Tensor,
    F2_calc: torch.Tensor,
    s_calc: torch.Tensor,
    **kwargs,
) -> Tuple[np.ndarray, tuple, list, Optional[np.ndarray]]:
    """Torch wrapper for ball_rotation_search_with_anisotropy."""
    return ball_rotation_search_with_anisotropy(
        F2_obs.detach().cpu().numpy(),
        s_obs.detach().cpu().numpy(),
        F2_calc.detach().cpu().numpy(),
        s_calc.detach().cpu().numpy(),
        **kwargs,
    )
