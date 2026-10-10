"""Chebyshev basis in resolution, shared by every fit of a smooth curve in ``|s|``.

The abscissa is ``sin(theta)/lambda`` rather than ``s**2``, because a resolution scale
has its structure at low resolution, where a basis uniform in ``s**2`` has almost no
support. The columns are prefix-nested: ``chebyshev_design(x, k)`` equals
``chebyshev_design(x, n)[:, :k]`` for ``k <= n``. Outside ``[lo, hi]`` every column
saturates, so a curve fitted on one reflection set and evaluated on another returns a
flat extrapolation unless both designs share an explicit ``lo``/``hi``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

__all__ = ["chebyshev_design"]


def chebyshev_design(
    x: torch.Tensor,
    n_coeff: int,
    lo: Optional[Union[float, torch.Tensor]] = None,
    hi: Optional[Union[float, torch.Tensor]] = None,
) -> torch.Tensor:
    """``(N, n_coeff)`` Chebyshev design matrix in ``x``.

    Parameters
    ----------
    x : torch.Tensor
        ``(N,)`` abscissa, normally ``sin(theta)/lambda``.
    n_coeff : int
        Number of Chebyshev terms. ``1`` gives a single constant column, i.e. a
        global scale with no resolution dependence.
    lo, hi : float or torch.Tensor, optional
        Range to map onto ``[-1, 1]``. Both default to ``x``'s own extremes,
        which is right for a single dataset and wrong the moment two fits have
        to be compared -- see the module docstring.

    Returns
    -------
    torch.Tensor
        ``(N, n_coeff)``, column 0 all ones, every entry in ``[-1, 1]``.
    """
    if n_coeff < 1:
        raise ValueError(f"n_coeff must be at least 1, got {n_coeff}")
    lo = x.min() if lo is None else torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi = x.max() if hi is None else torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    u = (2 * (x - lo) / (hi - lo).clamp(min=1e-12) - 1).clamp(-1.0, 1.0)
    cols = [torch.ones_like(u), u]
    for _ in range(2, n_coeff):
        cols.append(2 * u * cols[-1] - cols[-2])       # Chebyshev recurrence
    # The slice is what makes ``n_coeff == 1`` work: the loop does not run and
    # the pre-seeded linear column is dropped.
    return torch.stack(cols[:n_coeff], dim=1)
