"""
Crystallographic R-factor calculations.

Note the flag convention used throughout: in a ``rfree`` mask, **1 (True) is the WORK set
and 0 is the test set**. :func:`rfactor_work_free` is the canonical partition and avoids
the question entirely; prefer it over :func:`get_rfactors` for anything reportable.
"""

import torch


def rfactor(F_obs: torch.Tensor, F_calc: torch.Tensor) -> float:
    """
    ``sum|F_obs - F_calc| / sum|F_obs|`` over every reflection passed.

    Applies no masking and no work/free split -- select the subset before calling. Both
    arguments must be **amplitudes**, not complex structure factors, and syncs to a Python
    float.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes of shape (N,).
    F_calc : torch.Tensor
        Calculated structure factor amplitudes of shape (N,).
    """
    numerator = torch.sum(torch.abs(F_obs - F_calc))
    denominator = torch.sum(torch.abs(F_obs))
    r_factor = (numerator / denominator).item()
    return r_factor


def get_rfactors(
    F_obs: torch.Tensor, F_calc: torch.Tensor, rfree: torch.Tensor
) -> tuple:
    """
    R-factors for the working and test sets, split by a flag array.

    Applies no validity mask, so invalid reflections are included -- use
    :func:`rfactor_work_free` for anything reported.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes of shape (N,).
    F_calc : torch.Tensor
        Calculated structure factor amplitudes of shape (N,).
    rfree : torch.Tensor
        Flag array of shape (N,), cast to bool here: **1 is WORK, 0 is test**. If R-free
        comes out below R-work, this array is inverted.

    Returns
    -------
    tuple
        ``(r_work, r_test)`` as Python floats.
    """
    rfree = rfree.to(torch.bool)
    r_work = rfactor(F_obs[rfree], F_calc[rfree])
    r_test = rfactor(F_obs[~rfree], F_calc[~rfree])
    return r_work, r_test


def rfactor_work_free(data, F_calc_amp: torch.Tensor) -> tuple:
    """R-work / R-free over a ReflectionData's canonical work / free subsets.

    The one shared R-factor partition, using the same subset accessors as the refinement
    loss -- validity masks and work/test split applied, any separate validation set excluded
    from both -- so refinement reporting and the scaler diagnostic cannot disagree.

    Parameters
    ----------
    data : ReflectionData
        Must expose ``.work`` / ``.free`` accessors with ``.F`` and ``.select()``.
    F_calc_amp : torch.Tensor
        Full-size, already-**scaled** calculated **amplitudes**, aligned to ``data.hkl``.

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
    Per-bin R-work and R-test, one entry per bin index in ``[0, bins.max()]``.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factor amplitudes.
    rfree : torch.Tensor
        **Must already be boolean** -- used directly in ``mask & rfree`` with no cast, unlike
        :func:`get_rfactors`, so an integer mask silently misbehaves. 1 is WORK.
    bins : torch.Tensor
        Bin index per reflection.

    Returns
    -------
    tuple of torch.Tensor
        ``(r_work_bins, r_test_bins)``. An empty bin yields NaN (0/0).
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
