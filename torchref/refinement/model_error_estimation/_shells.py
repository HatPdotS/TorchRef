"""Resolution-shell machinery shared by the model-error estimators.

Equal-count shells over ``d*^2``, atomic-free segment sums, linear interpolation of
per-shell values back to reflections, and DerSimonian-Laird shrinkage of noisy per-shell
estimates toward a weighted straight line. :mod:`.sigma_a` and :mod:`.sigma_d` both
build on these; ``estimate_beta`` keeps its own module-level aliases so that its body
resolves the same globals it always did.

Plain tensors in and out. Every result lives on the device of its inputs, and float
work happens in the dtype of the inputs, so callers control both by what they pass.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def segment_layout(lengths: tuple[int, ...], device_str: str):
    """``(index, mask)`` placing contiguous segments on a padded ``(n_seg, max_len)`` grid.

    Cached: the sigma_A solve reduces ``n_grid * n_stages`` times over one layout.
    ``lengths`` is a tuple so it can be a cache key.
    """
    device = torch.device(device_str)
    # dtype-ok: segment lengths for cumsum offsets/gather index; PyTorch requires int64
    L = torch.tensor(lengths, dtype=torch.long, device=device)
    total = int(L.sum())
    max_len = int(L.max()) if L.numel() else 0
    # dtype-ok: zero offset concatenated into gather index; PyTorch requires int64
    zero = torch.zeros(1, dtype=torch.long, device=device)
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


def dl_shrink_to_line(
    y: torch.Tensor,
    var: torch.Tensor,
    x: torch.Tensor,
    *,
    slope_min: float | None = None,
    slope_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Shrink noisy per-shell values toward a weighted straight line in ``x``.

    DerSimonian-Laird shrinkage toward a two-parameter line fitted across all shells,
    which the two parameters determine far better than any one shell is determined::

        fit    y = a + b*x, weights 1/var
        tau^2  = DL between-shell variance about the line, weights 1/var
        w_i    = var_i / (var_i + tau^2)
        y_i   <- (1 - w_i)*y_i + w_i*line_i

    ``tau^2`` is the size of the dataset-specific residual the line does not capture, so
    ``w_i -> 0`` where that residual is real and large and ``w_i -> 1`` where the shell is
    badly determined. One shot, no iteration: the target is a fixed line. Weights are
    ``1/var``, never counts, because count weighting lets high-``var`` shells dominate
    ``Q`` and veto shrinkage entirely.

    Parameters
    ----------
    y, var, x : torch.Tensor
        Per-shell value, its sampling variance and the abscissa (``d*^2``), shape
        ``(k,)``. A shell with non-finite ``y`` or ``var`` (or ``var <= 0``) takes no part
        in the fit and is replaced by the line outright (``w = 1``).
    slope_min, slope_max : float, optional
        Clamp on the fitted slope ``b``; ``a`` is refitted after the clamp so the line
        still passes through the weighted centroid.

    Returns
    -------
    tuple
        ``(y_shrunk, w, tau_sq, a, b)``. With fewer than four usable shells, or when the
        slope is unidentifiable (all shells at one ``x``), the input is returned
        unchanged with ``w = 0``, ``tau_sq = 0`` and NaN line coefficients.
    """
    nan = float("nan")
    usable = torch.isfinite(y) & torch.isfinite(var) & (var > 0)
    k = int(usable.sum())
    # Two fitted parameters need at least two residual degrees of freedom.
    if k < 4:
        return y, torch.zeros_like(y), y.new_zeros(()), nan, nan

    wt = torch.where(usable, 1.0 / var.clamp(min=1e-30), torch.zeros_like(var))
    yz = torch.where(usable, y, torch.zeros_like(y))
    S = wt.sum()
    Sx = (wt * x).sum()
    Sxx = (wt * x * x).sum()
    Sy = (wt * yz).sum()
    Sxy = (wt * x * yz).sum()
    det = S * Sxx - Sx * Sx
    # Relative, not absolute: `det` is a difference of two ~`S**2 * x**2` terms, so on a
    # degenerate input it lands at the cancellation floor, not near zero.
    if float(det.abs()) <= 1e-12 * float((S * Sxx).abs()):
        return y, torch.zeros_like(y), y.new_zeros(()), nan, nan
    b = (S * Sxy - Sx * Sy) / det
    if slope_min is not None:
        b = b.clamp(min=slope_min)
    if slope_max is not None:
        b = b.clamp(max=slope_max)
    a = (wt * (yz - b * x)).sum() / S.clamp(min=1e-30)
    line = a + b * x

    resid = torch.where(usable, yz - line, torch.zeros_like(y))
    Q = (wt * resid * resid).sum()
    dof = float(k - 2)  # two parameters were fitted
    c = (S - (wt * wt).sum() / S.clamp(min=1e-30)).clamp(min=1e-30)
    # Q < k-2 means the scatter about the line is SMALLER than the noise alone predicts,
    # i.e. no evidence of structure the line is missing -> tau^2 = 0 -> take the line.
    tau_sq = ((Q - dof) / c).clamp(min=0.0)
    w = torch.where(usable, var / (var + tau_sq).clamp(min=1e-30), torch.ones_like(var))
    out = (1.0 - w) * torch.where(usable, y, line) + w * line
    return out, w, tau_sq, float(a), float(b)
