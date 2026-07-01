"""
Correlation scoring utilities for reciprocal-space alignment.

Provides resolution-binned (Pearson) correlation between two sets of values,
used for scoring rotation and translation searches.
"""

import torch


def binned_correlation(x, y, bins):
    """Compute the binned correlation between x and y.

    Parameters
    ----------
    x : torch.Tensor
        First set of values, shape (N,) or (B, N). Broadcasting is applied if
        one of x or y has shape (1, N) or (N,).
    y : torch.Tensor
        Second set of values, shape (N,) or (B, N).
    bins : torch.Tensor
        Bin index for each value, shape (N,).

    Returns
    -------
    torch.Tensor
        Binned correlation coefficients, shape (B, num_bins). Inputs are
        always promoted to 2-D internally, so a 1-D input of shape (N,)
        yields output of shape (1, num_bins); the result is never squeezed
        back to 1-D.

    Raises
    ------
    ValueError
        If the first dimensions of ``x`` and ``y`` differ and neither is 1
        (i.e. they cannot be broadcast against each other).
    """
    num_bins = bins.max().item() + 1

    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    max_first_dim = max(x.shape[0], y.shape[0])
    if x.shape[0] != y.shape[0]:
        if x.shape[0] == 1:
            x = x.expand(max_first_dim, -1)
        elif y.shape[0] == 1:
            y = y.expand(max_first_dim, -1)
        else:
            raise ValueError("x and y must have the same first dimension or one of them must be 1 or squeezed.")

    # Batched case: x, y have shape (B, N)
    B, N = x.shape
    bins_expanded = bins.unsqueeze(0).expand(B, -1)  # (B, N)

    # Compute bin counts (same for all batches)
    bin_counts = torch.zeros(num_bins, dtype=x.dtype, device=x.device)
    bin_counts = torch.scatter_add(bin_counts, 0, bins, torch.ones(N, dtype=x.dtype, device=x.device)).clamp(min=1.0)
    bin_counts = bin_counts.unsqueeze(0)  # (1, num_bins) for broadcasting

    # Compute means per bin per batch
    mean_x = torch.zeros(B, num_bins, dtype=x.dtype, device=x.device)
    mean_y = torch.zeros(B, num_bins, dtype=x.dtype, device=x.device)
    mean_x = torch.scatter_add(mean_x, 1, bins_expanded, x) / bin_counts
    mean_y = torch.scatter_add(mean_y, 1, bins_expanded, y) / bin_counts

    # Center the data
    x = x - torch.gather(mean_x, 1, bins_expanded)
    y = y - torch.gather(mean_y, 1, bins_expanded)

    xy = x * y
    xx = x * x
    yy = y * y

    bin_sums_xy = torch.zeros(B, num_bins, dtype=x.dtype, device=x.device)
    bin_sums_xx = torch.zeros(B, num_bins, dtype=x.dtype, device=x.device)
    bin_sums_yy = torch.zeros(B, num_bins, dtype=x.dtype, device=x.device)

    bin_sums_xy = torch.scatter_add(bin_sums_xy, 1, bins_expanded, xy) / bin_counts
    bin_sums_xx = torch.scatter_add(bin_sums_xx, 1, bins_expanded, xx) / bin_counts
    bin_sums_yy = torch.scatter_add(bin_sums_yy, 1, bins_expanded, yy) / bin_counts

    corr = bin_sums_xy / torch.sqrt(bin_sums_xx * bin_sums_yy).clamp(min=1e-6)
    return corr