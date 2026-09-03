"""Chebyshev basis in resolution, shared by everything that fits a smooth curve in |s|.

Two things in here carry argument rather than convention, and both were settled
by the scaler rework:

* **The abscissa is ``sin(theta)/lambda``, not ``s**2``.** The modulation a
  resolution-dependent scale has to represent is gentle through the bulk of the
  range and has real structure in the first few percent of ``s**2``; a basis
  uniform in ``s**2`` spends nearly all its resolution where nothing happens.
* **The basis is prefix-nested.** ``chebyshev_design(x, k)`` equals
  ``chebyshev_design(x, n)[:, :k]`` for ``k <= n``, so raising the order adds
  detail without redefining the terms already fitted, and a caller can slice
  instead of rebuilding.

``lo``/``hi`` exist for **extrapolation**, and the reason is the clamp rather
than the mapping. An affine remap does not change the space a polynomial basis
spans, so two fits over different ranges recover the same *function* where their
data overlap -- only the coefficients and the conditioning differ. What does
differ is outside the fitted range: ``u`` saturates at the ends, so every column
goes constant and the curve is frozen at its endpoint value.

So a fit over one resolution range, evaluated somewhere else, silently returns a
flat extrapolation. That is the case a shared ``lo``/``hi`` is for -- fitting on
one reflection set and using the curve on another, which is what comparing two
fits, or fitting on a crystal lattice and evaluating on a dense sampling,
actually requires. ``ScalerBase`` does not need it: it builds one design over
all reflections and slices rows.
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
