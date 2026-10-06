"""Resolution-shell machinery for the model-error estimators.

Equal-count shells over ``d*^2``, atomic-free segment sums, and linear interpolation of
per-shell values back to reflections. :mod:`.sigma_a` builds on these, imported under
private aliases.

Plain tensors in and out. Every result lives on the device of its inputs, and float
work happens in the dtype of the inputs, so callers control both by what they pass.
"""

from functools import lru_cache

import torch

from torchref.config import get_int_dtype


@lru_cache(maxsize=8)
def segment_layout(lengths: tuple[int, ...], device_str: str):
    """``(index, mask)`` placing contiguous segments on a padded ``(n_seg, max_len)`` grid.

    Cached: the sigma_A solve reduces ``n_grid * n_stages`` times over one layout.
    ``lengths`` is a tuple so it can be a cache key.
    """
    device = torch.device(device_str)
    L = torch.tensor(lengths, dtype=get_int_dtype(), device=device)
    total = int(L.sum())
    max_len = int(L.max()) if L.numel() else 0
    zero = torch.zeros(1, dtype=get_int_dtype(), device=device)
    starts = torch.cat([zero, L.cumsum(0)[:-1]])
    ar = torch.arange(max_len, device=device).reshape(1, max_len)
    # Clamp keeps the gather in bounds for the padding slots; `mask` zeroes them anyway.
    index = (starts.reshape(-1, 1) + ar).clamp(max=max(total - 1, 0))
    mask = ar < L.reshape(-1, 1)
    return index, mask


def segsum(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Sum ``x`` over contiguous segments, reducing along a padded trailing axis.

    Replaces ``torch.segment_reduce``, which is unimplemented on MPS. Keeps the properties
    that op was chosen for: atomic-free, one fixed reduction order per segment, so the
    result is bit-stable run to run and does not depend on ``scatter_add``'s CUDA atomicAdd
    accumulation order (see ``tests/unit/refinement/test_estimate_beta_determinism.py``).

    Deliberately NOT ``cumsum[end] - cumsum[start]``, the usual contiguous-segment trick:
    that recovers each shell sum by subtracting two running totals of the whole array,
    reintroducing the large-minus-large these estimators are written to avoid.

    ``x`` reduces over its last axis, so a leading batch dimension is handled in one call.
    Segments differ in length by at most one element, so the padding overhead is at most
    ``n_seg`` slots.
    """
    index, mask = segment_layout(tuple(int(v) for v in lengths), str(x.device))
    return (x[..., index] * mask.to(x.dtype)).sum(dim=-1)


def interp_in_dss(
    dss_all: torch.Tensor, bin_dss: torch.Tensor, vals: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of per-bin ``vals`` (at ``bin_dss``) to all reflections by
    their ``d_star_sq``; clamp-to-edge outside the range."""
    n_bins = bin_dss.numel()
    if n_bins == 1:
        return torch.full_like(dss_all, float(vals[0]))
    idx = torch.searchsorted(bin_dss, dss_all).clamp(1, n_bins - 1)
    x0 = bin_dss[idx - 1]
    x1 = bin_dss[idx]
    wlin = ((dss_all - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
    return (1 - wlin) * vals[idx - 1] + wlin * vals[idx]


def equal_count_shells(
    dss: torch.Tensor, *, per_bin: int, min_bins: int, min_per_bin: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Equal-count resolution shells over ``dss``, sorted ascending.

    Parameters
    ----------
    dss : torch.Tensor
        ``d*^2`` of the reflections entering the fit, shape ``(n,)``, in A^-2.
    per_bin : int
        Target reflections per shell.
    min_bins, min_per_bin : int
        Floor on the shell count for sparse sets: at least ``min_bins`` shells as long
        as each still holds ``min_per_bin`` reflections.

    Returns
    -------
    tuple
        ``(order, seg, seg_lengths, n_bins)``. ``order`` sorts ``dss`` ascending (a stable
        sort, so tied values bin identically on every backend); ``seg`` is the shell index
        of each sorted reflection, a non-decreasing ramp; ``seg_lengths`` the count per
        shell.
    """
    n = int(dss.numel())
    order = torch.argsort(dss, stable=True)
    n_by_count = max(1, n // per_bin)
    n_cap = max(1, n // min_per_bin)
    n_bins = max(n_by_count, min(min_bins, n_cap))
    seg = (
        torch.arange(n, device=dss.device) * n_bins
    ) // n  # dtype-ok: bincount input; PyTorch requires int64
    seg_lengths = torch.bincount(seg, minlength=n_bins)
    return order, seg, seg_lengths, n_bins
