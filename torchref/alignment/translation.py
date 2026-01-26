"""
Fast FFT-based translation search for molecular replacement.

Translation t shifts phase: F(hkl, t) = F(hkl) * exp(2*pi*i * hkl.t)
Correlation: C(t) = IFFT{ conj(F_obs) * F_calc }

This module provides efficient FFT-based translation search that finds the
optimal translation to position a model after rotation has been determined.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TranslationPeak:
    """
    Translation search peak.

    Attributes
    ----------
    translation : np.ndarray
        Fractional coordinates (3,).
    score : float
        Correlation score.
    sigma : float
        Z-score above mean.
    """
    translation: np.ndarray
    score: float
    sigma: float


def fft_translation_search(
    F_obs: np.ndarray,
    F_calc: np.ndarray,
    hkl: np.ndarray,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    n_peaks: int = 10,
    cluster_radius: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[TranslationPeak]]:
    """
    FFT-based translation search (vectorized).

    The translation function is:
        TF(t) = Re{ IFFT{ conj(F_obs) * F_calc } }

    This finds translation t such that F_calc shifted by t best matches F_obs.

    Parameters
    ----------
    F_obs : np.ndarray
        Observed structure factor amplitudes (or complex), shape (N,).
    F_calc : np.ndarray
        Calculated structure factors (complex), shape (N,).
    hkl : np.ndarray
        Miller indices, shape (N, 3).
    grid_shape : tuple, optional
        (Nx, Ny, Nz) grid size for FFT. If None, auto-computed from HKL range.
    n_peaks : int
        Number of peaks to return.
    cluster_radius : float
        Minimum fractional distance between peaks for clustering.

    Returns
    -------
    correlation_map : np.ndarray
        Full translation function, shape grid_shape.
    best_translation : np.ndarray
        Best translation in fractional coordinates, shape (3,).
    peaks : list
        Top peaks as TranslationPeak objects.

    Examples
    --------
    ::

        import numpy as np
        from torchref.alignment.translation import fft_translation_search

        # Known translation test
        hkl = np.array([[1,0,0], [0,1,0], [1,1,0], [0,0,1]])
        F_obs = np.array([1.0, 1.0, 1.0, 1.0])
        F_calc = np.exp(2j * np.pi * hkl @ [0.25, 0.0, 0.0])
        _, best, peaks = fft_translation_search(F_obs, F_calc, hkl)
        print(f'Recovered: {best}')  # Should be ~[0.25, 0, 0]
    """
    # Auto grid shape from HKL range
    if grid_shape is None:
        hkl_abs = np.abs(hkl).astype(int)
        grid_shape = tuple(2 * (hkl_abs[:, i].max() + 1) for i in range(3))

    Nx, Ny, Nz = grid_shape
    product_grid = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

    # Standard translation function: TF(t) = Re{ IFFT{ conj(F_obs) * F_calc } }
    # This finds t such that F_calc(t) = F_calc * exp(2*pi*i*hkl.t) matches F_obs
    product = np.conj(F_obs) * F_calc

    # Place at HKL positions using vectorized add.at for accumulation
    hkl_int = hkl.astype(int)
    h_idx = hkl_int[:, 0] % Nx
    k_idx = hkl_int[:, 1] % Ny
    l_idx = hkl_int[:, 2] % Nz

    np.add.at(product_grid, (h_idx, k_idx, l_idx), product)

    # IFFT gives correlation at all translations
    correlation_map = np.fft.ifftn(product_grid).real

    # Find peaks
    peaks = find_translation_peaks(correlation_map, n_peaks, cluster_radius)
    best = peaks[0].translation if peaks else np.zeros(3)

    return correlation_map, best, peaks


def find_translation_peaks(
    correlation_map: np.ndarray,
    n_peaks: int = 10,
    cluster_radius: float = 0.05,
) -> List[TranslationPeak]:
    """
    Extract and cluster peaks from translation function.

    Parameters
    ----------
    correlation_map : np.ndarray
        Translation function values, shape (Nx, Ny, Nz).
    n_peaks : int
        Maximum number of peaks to return.
    cluster_radius : float
        Minimum fractional distance between peaks (periodic).

    Returns
    -------
    peaks : list
        List of TranslationPeak objects sorted by score.
    """
    Nx, Ny, Nz = correlation_map.shape
    mean_val = correlation_map.mean()
    std_val = correlation_map.std()

    if std_val < 1e-10:
        return []

    flat = correlation_map.flatten()
    sorted_idx = np.argsort(flat)[::-1]

    peaks = []
    used = []

    for idx in sorted_idx:
        if len(peaks) >= n_peaks:
            break

        pos_3d = np.unravel_index(idx, correlation_map.shape)
        trans = np.array([pos_3d[0] / Nx, pos_3d[1] / Ny, pos_3d[2] / Nz])
        score = flat[idx]
        sigma = (score - mean_val) / std_val

        # Check clustering - skip if too close to existing peak
        is_new = True
        for prev in used:
            diff = np.abs(trans - prev)
            diff = np.minimum(diff, 1 - diff)  # Periodic boundary
            if np.linalg.norm(diff) < cluster_radius:
                is_new = False
                break

        if is_new:
            peaks.append(TranslationPeak(trans, score, sigma))
            used.append(trans)

    return peaks


def fft_translation_search_torch(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    hkl: torch.Tensor,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, List[TranslationPeak]]:
    """
    Torch wrapper for fft_translation_search.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes (or complex).
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    hkl : torch.Tensor
        Miller indices.
    **kwargs
        Additional arguments passed to fft_translation_search.

    Returns
    -------
    correlation_map : np.ndarray
        Full translation function.
    best_translation : np.ndarray
        Best translation in fractional coordinates.
    peaks : list
        Top peaks as TranslationPeak objects.
    """
    return fft_translation_search(
        F_obs.detach().cpu().numpy(),
        F_calc.detach().cpu().numpy(),
        hkl.detach().cpu().numpy(),
        **kwargs,
    )


def apply_translation_to_fcalc(
    F_calc: np.ndarray,
    hkl: np.ndarray,
    translation_frac: np.ndarray,
) -> np.ndarray:
    """
    Apply translation phase shift to calculated structure factors.

    F(hkl, t) = F(hkl) * exp(2*pi*i * hkl.t)

    Parameters
    ----------
    F_calc : np.ndarray
        Calculated structure factors (complex), shape (N,).
    hkl : np.ndarray
        Miller indices, shape (N, 3).
    translation_frac : np.ndarray
        Translation in fractional coordinates, shape (3,).

    Returns
    -------
    F_calc_shifted : np.ndarray
        Phase-shifted structure factors, shape (N,).
    """
    phase_shift = 2 * np.pi * (hkl @ translation_frac)
    return F_calc * np.exp(1j * phase_shift)


def apply_translation_to_fcalc_torch(
    F_calc: torch.Tensor,
    hkl: torch.Tensor,
    translation_frac: torch.Tensor,
) -> torch.Tensor:
    """
    Apply translation phase shift to calculated structure factors (PyTorch).

    F(hkl, t) = F(hkl) * exp(2*pi*i * hkl.t)

    Parameters
    ----------
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    hkl : torch.Tensor
        Miller indices.
    translation_frac : torch.Tensor
        Translation in fractional coordinates.

    Returns
    -------
    F_calc_shifted : torch.Tensor
        Phase-shifted structure factors.
    """
    phase_shift = 2 * torch.pi * (hkl.to(translation_frac.dtype) @ translation_frac)
    return F_calc * torch.exp(1j * phase_shift)
