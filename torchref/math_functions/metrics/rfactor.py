"""
R-factor calculation functions.

Functions for computing crystallographic R-factors and related metrics.
"""

import numpy as np
import torch


def get_rfactor_torch(F_obs, F_calc):
    """
    Calculate R-factor between observed and calculated structure factors (PyTorch version).

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factor amplitudes.

    Returns
    -------
    torch.Tensor
        R-factor value.
    """
    F_obs = torch.abs(F_obs)
    F_calc = torch.abs(F_calc)
    return torch.sum(torch.abs(F_obs - F_calc)) / torch.sum(F_obs)


def get_rfactor(F_obs, F_calc):
    """
    Calculate the R-factor between observed and calculated structure factors (NumPy version).

    The R-factor is a measure of agreement between observed and calculated
    structure factor amplitudes, defined as sum(|F_obs - F_calc|) / sum(F_obs).

    Parameters
    ----------
    F_obs : numpy.ndarray
        Observed structure factor amplitudes.
    F_calc : numpy.ndarray
        Calculated structure factor amplitudes.

    Returns
    -------
    float
        R-factor value between 0 and 1.
    """
    F_obs = np.abs(F_obs)
    F_calc = np.abs(F_calc)
    return np.sum(np.abs(F_obs - F_calc)) / np.sum(F_obs)


def rfactor(F_obs: torch.Tensor, F_calc: torch.Tensor) -> float:
    """
    Calculate R-factor between observed and calculated structure factors.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes of shape (N,).
    F_calc : torch.Tensor
        Calculated structure factor amplitudes of shape (N,).

    Returns
    -------
    float
        R-factor value.
    """
    numerator = torch.sum(torch.abs(F_obs - F_calc))
    denominator = torch.sum(torch.abs(F_obs))
    r_factor = (numerator / denominator).item()
    return r_factor


def get_rfactors(
    F_obs: torch.Tensor, F_calc: torch.Tensor, rfree: torch.Tensor
) -> tuple:
    """
    Get R-factors for working and test sets.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes of shape (N,).
    F_calc : torch.Tensor
        Calculated structure factor amplitudes of shape (N,).
    rfree : torch.Tensor
        Boolean mask indicating R-free reflections of shape (N,).
        1 is Working set, 0 is Test set.

    Returns
    -------
    tuple
        (r_work, r_test) where r_work is the R-factor for the working set
        and r_test is the R-factor for the test set.
    """
    rfree = rfree.to(torch.bool)
    r_work = rfactor(F_obs[rfree], F_calc[rfree])
    r_test = rfactor(F_obs[~rfree], F_calc[~rfree])
    return r_work, r_test


def bin_wise_rfactors(
    F_obs: torch.Tensor, F_calc: torch.Tensor, rfree: torch.Tensor, bins: torch.Tensor
) -> tuple:
    """
    Calculate bin-wise R-factors between observed and calculated structure factors.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factors.
    F_calc : torch.Tensor
        Calculated structure factors.
    rfree : torch.Tensor
        R-free mask.
    bins : torch.Tensor
        Bin indices for each reflection.

    Returns
    -------
    r_work_bins : torch.Tensor
        R-factors for working set (per bin).
    r_test_bins : torch.Tensor
        R-factors for test set (per bin).
    """
    r_work_bins = []
    r_test_bins = []
    for b in range(bins.max().item() + 1):
        mask = bins == b
        r_work = rfactor(F_obs[mask & rfree], F_calc[mask & rfree])
        r_test = rfactor(F_obs[mask & ~rfree], F_calc[mask & ~rfree])
        r_work_bins.append(r_work)
        r_test_bins.append(r_test)
    return torch.tensor(r_work_bins), torch.tensor(r_test_bins)


def calc_outliers(F_obs, F_calc, z):
    """
    Identify outlier reflections based on deviation from expected values (PyTorch version).

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factor amplitudes.
    z : float
        Number of standard deviations for outlier threshold.

    Returns
    -------
    torch.Tensor
        Boolean mask where True indicates outlier reflections.
    """
    F_obs = torch.abs(F_obs)
    F_calc = torch.abs(F_calc)
    diff = torch.abs(F_obs - F_calc) / F_obs
    std = torch.std(diff)
    outliers = diff > z * std
    return outliers


def calc_outliers_numpy(F_obs, F_calc, z):
    """
    Identify outlier reflections based on structure factor differences (NumPy version).

    Detects reflections where the normalized difference between observed
    and calculated structure factors exceeds z standard deviations.

    Parameters
    ----------
    F_obs : numpy.ndarray
        Observed structure factor amplitudes.
    F_calc : numpy.ndarray
        Calculated structure factor amplitudes.
    z : float
        Number of standard deviations for outlier threshold.

    Returns
    -------
    numpy.ndarray
        Boolean array where True indicates an outlier reflection.
    """
    F_obs = np.abs(F_obs)
    F_calc = np.abs(F_calc)
    diff = np.abs(F_obs - F_calc) / F_obs * np.mean(F_obs)
    std = np.std(diff)
    outliers = diff > z * std
    return outliers
