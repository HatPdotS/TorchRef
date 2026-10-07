"""Shell helpers used by sigma_A.

Pinned: the helpers ``sigma_a`` re-imports are the same objects ``_shells`` defines, so a
fit through either module reduces identically, and ``equal_count_shells`` matches an
independent construction of the same shells.
"""

import pytest
import torch

from torchref.refinement.model_error_estimation import _shells, sigma_a
from torchref.refinement.model_error_estimation._shells import (
    equal_count_shells,
    interp_in_dss,
    segsum,
)


@pytest.mark.unit
def test_sigma_a_uses_the_shared_helpers():
    assert sigma_a._equal_count_shells is _shells.equal_count_shells
    assert sigma_a._segsum is _shells.segsum
    assert sigma_a._interp_in_dss is _shells.interp_in_dss
    assert sigma_a._segment_layout is _shells.segment_layout


@pytest.mark.unit
@pytest.mark.parametrize("n,per_bin", [(2000, 140), (37, 140), (9, 140), (5000, 500)])
def test_equal_count_shells_matches_an_independent_construction(n, per_bin, any_device):
    """The stable sort, shell count and ramp ``estimate_beta`` bins by, written out."""
    g = torch.Generator().manual_seed(3)
    dss = (torch.rand(n, generator=g) * 0.3 + 0.02).to(any_device)
    min_bins, min_per_bin = 5, 40
    order, seg, seg_lengths, n_bins = equal_count_shells(
        dss, per_bin=per_bin, min_bins=min_bins, min_per_bin=min_per_bin
    )
    # Oracle: the same construction, spelled out.
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
