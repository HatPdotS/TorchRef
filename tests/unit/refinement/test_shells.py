"""Shell helpers shared by sigma_A and sigma_D.

Pinned: the helpers ``sigma_a`` re-imports are the same objects ``_shells`` defines, so a
fit through either module reduces identically; ``equal_count_shells`` reproduces the
shell construction written out in ``estimate_beta``; the line shrinkage takes the line
outright when the scatter is below the noise, passes through with too few shells,
honours a slope clamp, and treats a shell without a finite variance as undetermined.
"""

import math

import pytest
import torch

from torchref.refinement.model_error_estimation import _shells, sigma_a
from torchref.refinement.model_error_estimation._shells import (
    dl_shrink_to_line,
    equal_count_shells,
    interp_in_dss,
    segsum,
)


@pytest.mark.unit
def test_sigma_a_uses_the_shared_helpers():
    assert sigma_a._segsum is _shells.segsum
    assert sigma_a._interp_in_dss is _shells.interp_in_dss
    assert sigma_a._segment_layout is _shells.segment_layout


@pytest.mark.unit
@pytest.mark.parametrize("n,per_bin", [(2000, 140), (37, 140), (9, 140), (5000, 500)])
def test_equal_count_shells_matches_estimate_beta_construction(n, per_bin, any_device):
    """The construction ``estimate_beta`` writes out inline, reproduced independently."""
    g = torch.Generator().manual_seed(3)
    dss = (torch.rand(n, generator=g) * 0.3 + 0.02).to(any_device)
    min_bins, min_per_bin = 5, 40
    order, seg, seg_lengths, n_bins = equal_count_shells(
        dss, per_bin=per_bin, min_bins=min_bins, min_per_bin=min_per_bin
    )
    # Oracle: the four lines of estimate_beta.
    ref_order = torch.argsort(dss, stable=True)
    n_by_count = max(1, n // per_bin)
    n_cap = max(1, n // min_per_bin)
    ref_bins = max(n_by_count, min(min_bins, n_cap))
    ref_seg = (torch.arange(n, device=dss.device) * ref_bins) // n
    assert n_bins == ref_bins
    assert torch.equal(order, ref_order)
    assert torch.equal(seg, ref_seg)
    assert torch.equal(seg_lengths, torch.bincount(ref_seg, minlength=ref_bins))
    assert int(seg_lengths.sum()) == n
    assert int(seg_lengths.max() - seg_lengths.min()) <= 1


@pytest.mark.unit
def test_segsum_and_interp_round_trip(any_device):
    lengths = torch.tensor([3, 2, 4], device=any_device)
    x = torch.arange(9, dtype=torch.float32, device=any_device)
    assert torch.equal(
        segsum(x, lengths), torch.tensor([3.0, 7.0, 26.0], device=any_device)
    )
    bin_dss = torch.tensor([0.1, 0.2, 0.3], device=any_device)
    vals = torch.tensor([1.0, 3.0, 5.0], device=any_device)
    grid = torch.tensor([0.0, 0.15, 0.25, 0.5], device=any_device)
    out = interp_in_dss(grid, bin_dss, vals)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 4.0, 5.0], device=any_device))


@pytest.mark.unit
def test_shrink_passes_through_with_fewer_than_four_shells():
    y = torch.tensor([1.0, 2.0, 3.0])
    var = torch.ones(3)
    x = torch.tensor([0.1, 0.2, 0.3])
    out, w, tau_sq, a, b = dl_shrink_to_line(y, var, x)
    assert torch.equal(out, y)
    assert torch.equal(w, torch.zeros(3))
    assert float(tau_sq) == 0.0 and math.isnan(a) and math.isnan(b)


@pytest.mark.unit
def test_shrink_takes_the_line_when_scatter_is_below_noise():
    x = torch.linspace(0.05, 0.35, 8, dtype=torch.float64)
    line = 2.0 - 3.0 * x
    g = torch.Generator().manual_seed(1)
    var = torch.full((8,), 0.04, dtype=torch.float64)
    y = line + 0.01 * torch.randn(8, generator=g, dtype=torch.float64)
    out, w, tau_sq, a, b = dl_shrink_to_line(y, var, x)
    assert float(tau_sq) == 0.0
    assert torch.allclose(w, torch.ones(8, dtype=torch.float64))
    assert torch.allclose(out, a + b * x)
    assert abs(a - 2.0) < 0.05 and abs(b + 3.0) < 0.3


@pytest.mark.unit
def test_shrink_keeps_a_real_departure():
    x = torch.linspace(0.05, 0.35, 10, dtype=torch.float64)
    y = 1.0 - 2.0 * x
    y[4] += 3.0  # one shell far off the line, far beyond its own variance
    var = torch.full((10,), 1e-4, dtype=torch.float64)
    out, w, tau_sq, _, _ = dl_shrink_to_line(y, var, x)
    assert float(tau_sq) > 0.0
    assert float(w[4]) < 0.01
    assert abs(float(out[4] - y[4])) < 0.05


@pytest.mark.unit
def test_shrink_slope_clamp_is_honoured():
    x = torch.linspace(0.05, 0.35, 8, dtype=torch.float64)
    y = 1.0 + 4.0 * x
    var = torch.full((8,), 0.01, dtype=torch.float64)
    _, _, _, _, b = dl_shrink_to_line(y, var, x, slope_max=0.0)
    assert b == 0.0
    _, _, _, _, b2 = dl_shrink_to_line(-y, var, x, slope_min=0.0)
    assert b2 == 0.0


@pytest.mark.unit
def test_shrink_replaces_undetermined_shells_by_the_line():
    x = torch.linspace(0.05, 0.35, 8, dtype=torch.float64)
    y = 2.0 - 3.0 * x
    var = torch.full((8,), 0.01, dtype=torch.float64)
    y[2] = float("nan")
    var[5] = float("inf")
    out, w, _, a, b = dl_shrink_to_line(y, var, x)
    assert float(w[2]) == 1.0 and float(w[5]) == 1.0
    assert torch.isfinite(out).all()
    assert torch.allclose(out[[2, 5]], a + b * x[[2, 5]])
