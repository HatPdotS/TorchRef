"""Per-bin scale estimation.

A single fast primitive for the recurring "find the scalar that best matches
``|F_calc|`` to ``|F_obs|`` in each resolution bin" operation, shared by the
initial-scale calculation and the R-factor reporting correction.
"""

from typing import Optional

import torch


def binwise_scale(
    F_calc: torch.Tensor,
    F_obs: torch.Tensor,
    bins: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    nbins: Optional[int] = None,
    weights: Optional[torch.Tensor] = None,
    min_count: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-bin least-squares scale matching ``|F_calc|`` to ``|F_obs|``.

    Returns the minimiser of ``sum_b w (|F_obs| - c_b |F_calc|)**2``, i.e.
    ``c_b = sum(w |F_obs||F_calc|) / sum(w |F_calc|^2)``, via ``scatter_add``
    (O(N), no Python loop over bins). Multiply ``|F_calc|`` by ``c[bins]``.

    Parameters
    ----------
    F_calc, F_obs : torch.Tensor
        Calculated and observed structure factors, shape ``(N,)``. Complex
        inputs are reduced to amplitudes via ``abs``.
    bins : torch.Tensor
        Per-reflection bin index, shape ``(N,)`` (cast to ``int64`` internally).
    valid : torch.Tensor, optional
        Boolean mask of the reflections used to *fit* the scale (default all); it
        is multiplied into ``weights``, so a non-boolean tensor reweights rather
        than gates. The returned scale applies to every reflection regardless.
    nbins : int, optional
        Number of bins. Defaults to ``bins.max() + 1``.
    weights : torch.Tensor, optional
        Per-reflection weights ``w`` (shape ``(N,)``). Defaults to ones.
    min_count : int, default 1
        Bins with fewer contributing reflections are left at ``c_b = 1``, so
        sparse shells are not corrected on noise.
    eps : float, default 1e-12
        Denominator floor.

    Returns
    -------
    torch.Tensor
        Per-bin scale factors, shape ``(nbins,)``.
    """
    Fc = F_calc.abs() if F_calc.is_complex() else F_calc.abs()
    Fo = F_obs.abs() if F_obs.is_complex() else F_obs.abs()
    Fc = Fc.reshape(-1)
    Fo = Fo.reshape(-1)
    device, dtype = Fc.device, Fc.dtype

    bins = bins.reshape(-1).to(device=device, dtype=torch.int64)
    if nbins is None:
        nbins = int(bins.max().item()) + 1 if bins.numel() else 0

    w = (
        torch.ones_like(Fo)
        if weights is None
        else weights.reshape(-1).to(device=device, dtype=dtype)
    )
    if valid is not None:
        w = w * valid.reshape(-1).to(device=device, dtype=dtype)

    if nbins == 0:
        return torch.ones(0, device=device, dtype=dtype)

    num = torch.zeros(nbins, device=device, dtype=dtype).scatter_add(
        0, bins, w * Fo * Fc
    )
    den = torch.zeros(nbins, device=device, dtype=dtype).scatter_add(
        0, bins, w * Fc * Fc
    )
    count = torch.zeros(nbins, device=device, dtype=dtype).scatter_add(
        0, bins, (w > 0).to(dtype)
    )

    c = torch.where(
        (count >= min_count) & (den > eps),
        num / den.clamp(min=eps),
        torch.ones(nbins, device=device, dtype=dtype),
    )
    return c
