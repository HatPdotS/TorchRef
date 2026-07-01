"""
R-factor calculation functions.

Functions for computing crystallographic R-factors and related metrics.
"""

import torch


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


def rfactor_work_free(data, F_calc_amp: torch.Tensor) -> tuple:
    """R-work / R-free over a ReflectionData's canonical work / free subsets.

    The single shared R-factor partition: ``R_work`` on ``data.work`` and
    ``R_free`` on ``data.free`` (the same subset accessors the refinement loss
    uses — validity masks applied, work/test split applied, and any separate
    validation set excluded from both). Both the refinement reporting
    (:meth:`XrayTarget.get_rfactor`) and the scaler's scale-fit diagnostic call
    this, so they cannot disagree on convention.

    Parameters
    ----------
    data : ReflectionData
        Must expose ``.work`` / ``.free`` subset accessors with ``.F`` and
        ``.select(full_array)``.
    F_calc_amp : torch.Tensor
        Full-size, already-scaled calculated **amplitudes** (``|F_calc|``),
        aligned to ``data.hkl``.

    Returns
    -------
    tuple
        ``(R_work, R_free)`` as Python floats.
    """
    work = data.work
    free = data.free
    r_work = rfactor(work.F, work.select(F_calc_amp))
    r_free = rfactor(free.F, free.select(F_calc_amp))
    return r_work, r_free


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
        R-free mask. Must be a boolean tensor: it is used directly in
        ``mask & rfree`` without an internal cast (unlike ``get_rfactors``,
        which applies ``.to(torch.bool)``), so a non-boolean ``rfree``
        silently misbehaves.
    bins : torch.Tensor
        Bin indices for each reflection.

    Returns
    -------
    tuple of torch.Tensor
        ``(r_work_bins, r_test_bins)``, a pair of 1-D tensors holding the
        per-bin R-factors for the working set and the test set respectively.
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
