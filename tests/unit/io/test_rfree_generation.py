"""Unit tests for resolution-stratified R-free flag generation.

Covers the binning contract of ``ReflectionData._generate_rfree_flags`` /
``regenerate_rfree_flags``: each resolution bin holds >= min_per_bin (1000)
reflections and contributes >= min_free_per_bin (50) free reflections, the
flags are binary, generation is reproducible under a seed, and tiny datasets
degrade gracefully to a single clamped bin.
"""

import pytest
import torch

from torchref.io.datasets.reflection_data import ReflectionData


def _synthetic_data(h=12, k=12, lmax=30, seed=0, device="cpu"):
    """A P1 ReflectionData with strictly-positive l (already in the ASU, so no
    Friedel folding changes the count). (2h+1)(2k+1)*lmax reflections.

    Built with placeholder (all-work) flags, then flags are (re)generated via the
    public ``regenerate_rfree_flags`` once the validity masks exist.
    """
    hs = torch.arange(-h, h + 1)
    ks = torch.arange(-k, k + 1)
    ls = torch.arange(1, lmax + 1)
    hkl = torch.stack(
        torch.meshgrid(hs, ks, ls, indexing="ij"), dim=-1
    ).reshape(-1, 3).to(torch.int32)
    n = hkl.shape[0]
    g = torch.Generator().manual_seed(0)
    F = torch.rand(n, generator=g) * 100.0 + 1.0
    F_sigma = F * 0.1
    data = ReflectionData.from_tensors(
        hkl, F, F_sigma, (50.0, 60.0, 70.0, 90.0, 90.0, 90.0), "P 1",
        rfree_flags=torch.ones(n, dtype=torch.bool), device=device,
        verbose=0, friedel_merged=True,
    )
    data.regenerate_rfree_flags(force=True, seed=seed)
    return data


def _per_bin_counts(data, n_bins=10, min_per_bin=1000):
    bin_idx, n = data.get_bins(n_bins=n_bins, min_per_bin=min_per_bin)
    free = (data.rfree_flags == 0)
    sizes, frees = [], []
    for b in range(n):
        m = bin_idx == b
        sizes.append(int(m.sum()))
        frees.append(int((m & free).sum()))
    return n, sizes, frees


@pytest.mark.unit
def test_flags_are_binary():
    data = _synthetic_data()
    vals = set(torch.unique(data.rfree_flags).tolist())
    assert vals.issubset({0, 1})


@pytest.mark.unit
def test_from_tensors_autogenerates_flags():
    """from_tensors(rfree_flags=None) generates flags during construction and
    honors the binning contract (validity mask is seeded before get_bins)."""
    h = k = 12
    lmax = 30
    hs, ks, ls = torch.arange(-h, h + 1), torch.arange(-k, k + 1), torch.arange(1, lmax + 1)
    hkl = torch.stack(
        torch.meshgrid(hs, ks, ls, indexing="ij"), dim=-1
    ).reshape(-1, 3).to(torch.int32)
    n = hkl.shape[0]
    gg = torch.Generator().manual_seed(0)
    F = torch.rand(n, generator=gg) * 100.0 + 1.0
    data = ReflectionData.from_tensors(
        hkl, F, F * 0.1, (50.0, 60.0, 70.0, 90.0, 90.0, 90.0), "P 1",
        rfree_flags=None, device="cpu", verbose=0, friedel_merged=True,
    )
    assert data.rfree_flags is not None
    assert set(torch.unique(data.rfree_flags).tolist()).issubset({0, 1})
    nb, sizes, frees = _per_bin_counts(data, n_bins=10, min_per_bin=1000)
    assert nb >= 2
    assert all(s >= 1000 for s in sizes)
    assert all(f >= 50 for f in frees)


@pytest.mark.unit
def test_min_reflections_and_free_per_bin():
    data = _synthetic_data()  # ~18750 reflections
    n, sizes, frees = _per_bin_counts(data, n_bins=10, min_per_bin=1000)
    assert n >= 2, "expected several resolution bins for this dataset"
    assert all(s >= 1000 for s in sizes), f"bin sizes below 1000: {sizes}"
    assert all(f >= 50 for f in frees), f"free-per-bin below 50: {frees}"


@pytest.mark.unit
def test_default_generation_matches_min_free_floor():
    """The default 2% fraction is below the 50-floor for ~1875-refl bins, so
    each bin should land on exactly the floor (50)."""
    data = _synthetic_data()
    n, sizes, frees = _per_bin_counts(data, n_bins=10, min_per_bin=1000)
    # 0.02 * ~1875 = ~37 < 50 -> floor wins
    assert all(f == 50 for f in frees), frees


@pytest.mark.unit
def test_seed_reproducible():
    data = _synthetic_data(seed=42)
    flags_a = data.rfree_flags.clone()
    data.regenerate_rfree_flags(force=True, seed=42)
    assert torch.equal(flags_a, data.rfree_flags)


@pytest.mark.unit
def test_tiny_dataset_single_clamped_bin():
    data = _synthetic_data(h=4, k=4, lmax=10)  # ~810 reflections < min_per_bin
    n, sizes, frees = _per_bin_counts(data, n_bins=10, min_per_bin=1000)
    assert n == 1
    assert frees[0] == min(sizes[0], 50)
    # at least one work reflection remains for cross-validation
    assert int((data.rfree_flags != 0).sum()) >= 1
