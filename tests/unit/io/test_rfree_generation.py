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


def _duplicated_asu_data(spacegroup, transform, seed=0, device="cpu"):
    """A dataset where every reflection appears twice under indices that
    canonicalize to the same ASU representative.

    ``transform`` maps an (N,3) index array to a symmetry-equivalent one, so the
    loaded object holds 2N rows collapsing onto N canonical indices -- the same
    shape as stacked Bijvoet data, without needing an anomalous MTZ.
    """
    hs, ks, ls = torch.arange(-6, 7), torch.arange(-6, 7), torch.arange(1, 16)
    base = torch.stack(
        torch.meshgrid(hs, ks, ls, indexing="ij"), dim=-1
    ).reshape(-1, 3).to(torch.int32)
    hkl = torch.cat([base, transform(base)], dim=0)
    n = hkl.shape[0]
    g = torch.Generator().manual_seed(seed)
    F = torch.rand(n, generator=g) * 100.0 + 1.0
    return ReflectionData.from_tensors(
        hkl, F, F * 0.1, (50.0, 60.0, 70.0, 90.0, 90.0, 90.0), spacegroup,
        rfree_flags=None, device=device, verbose=0,
    )


def _mixed_flag_groups(data):
    """Canonical ASU indices whose rows disagree about work vs free."""
    groups = {}
    for h, fr in zip(map(tuple, data.hkl.tolist()), data.rfree_flags.tolist()):
        groups.setdefault(h, set()).add(bool(fr))
    return [h for h, v in groups.items() if len(v) > 1]


@pytest.mark.unit
def test_friedel_mates_share_a_flag():
    """(h,k,l) and (-h,-k,-l) collapse to one canonical index and must not be
    split across work/free -- the two differ only by the anomalous signal, so a
    split leaks the held-out reflection into the work set."""
    data = _duplicated_asu_data("P 1", lambda t: -t)
    # Premise: the input really did fold onto shared canonical indices.
    assert bool(data.friedel_flags.any())
    assert bool((data.rfree_flags == 0).any())
    assert _mixed_flag_groups(data) == []


@pytest.mark.unit
def test_symmetry_equivalents_share_a_flag():
    """The grouping is by canonical ASU index, so it covers ordinary symmetry
    mates too, not only Friedel pairs. In P 4, (h,k,l) ~ (-k,h,l)."""
    def rot4(t):
        return torch.stack([-t[:, 1], t[:, 0], t[:, 2]], dim=1)

    data = _duplicated_asu_data("P 4", rot4)
    # Premise: rows really do collapse onto shared canonical indices here, and
    # via rotation rather than Friedel conjugation.
    assert len(torch.unique(data.hkl, dim=0)) < len(data.hkl)
    assert bool((data.rfree_flags == 0).any())
    assert _mixed_flag_groups(data) == []


@pytest.mark.unit
def test_free_set_excludes_masked_reflections():
    """The free quota is spent only on reflections that survive the validity
    masks; a free flag on a masked-out row contributes nothing to R-free."""
    data = _synthetic_data()
    # Mask out a third of the reflections, as sanitize_F / flag_wilson_outliers
    # would for unusable observations, then redraw.
    keep = torch.ones(len(data.hkl), dtype=torch.bool, device=data.device)
    keep[::3] = False
    data.masks["unusable"] = keep
    data.regenerate_rfree_flags(force=True, seed=0)

    valid = data.masks().to(torch.bool)
    assert not bool(valid.all()), "expected some masked-out reflections"
    free = data.rfree_flags == 0
    assert bool(free.any())
    assert not bool((free & ~valid).any())


@pytest.mark.unit
def test_asu_grouping_requires_canonicalized_data():
    """Grouping raw Miller indices would not unite +h with -h, so it must raise
    rather than silently return a useless grouping."""
    data = ReflectionData(verbose=0)
    data.hkl = torch.tensor([[1, 2, 3], [-1, -2, -3]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="canonicalized"):
        data.asu_group_indices()
