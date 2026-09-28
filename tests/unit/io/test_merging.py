"""merge_to_spacegroup: symmetry merging via P1 and its merging statistics.

Pinned behaviour: expansion copies are never counted as independent
observations, row order of the input does not matter, masked or unusable
observations do not contribute, Bijvoet mates stay apart (and share one R-free
flag) only when asked, and the R values are ``None`` rather than zero when no
reflection has two observations.
"""

import math

import pytest
import torch

from torchref.io import ReflectionData, merge_to_spacegroup
from torchref.symmetry import SpaceGroup

CELL = (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
SG = "P 21 21 21"


def _asu_grid(n=6, sg=SG):
    """Unique ASU indices of ``sg`` from a block of Miller indices, minus absences."""
    r = torch.arange(-n, n + 1)
    grid = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    grid = grid[(grid != 0).any(dim=-1)].to(torch.int32)
    group = SpaceGroup(sg)
    canon, *_ = group.canonicalize_hkl(grid, include_friedel=True)
    uniq = torch.unique(canon, dim=0)
    return uniq[~group.is_absent(uniq)]


def _invariant(hkl):
    """A value shared by all P212121 equivalents (they differ only in signs)."""
    h, k, l = (hkl.to(torch.float32).abs().T)
    return 10.0 + h + 2.0 * k + 3.0 * l


def _data(hkl, I, sigma, sg=SG, **kw):
    return ReflectionData.from_tensors(
        hkl,
        I.clamp(min=0).sqrt(),
        sigma / (2.0 * I.clamp(min=1e-3).sqrt()),
        CELL,
        sg,
        verbose=0,
        device="cpu",
        I=I,
        I_sigma=sigma,
        **kw,
    )


def _synthetic(n=6):
    hkl = _asu_grid(n)
    I = _invariant(hkl)
    return _data(hkl, I, torch.ones_like(I))


def _by_hkl(data, values):
    return {tuple(h): float(v) for h, v in zip(data.hkl.tolist(), values)}


@pytest.mark.unit
def test_p1_round_trip_recovers_values():
    d = _synthetic()
    p1 = d.expand_to_p1()
    p1.verbose = 0
    merged, stats = merge_to_spacegroup(p1, SG)

    merged._assert_per_reflection_consistent()
    assert len(merged.hkl) == len(d.hkl)
    orig = _by_hkl(d, d.I)
    for h, v in _by_hkl(merged, merged.I).items():
        assert v == pytest.approx(orig[h], rel=1e-5)
    assert stats.overall.r_merge == pytest.approx(0.0, abs=1e-6)
    # Every P1 copy counts once: at most n_ops * 2 (Friedel) copies per reflection.
    assert 1.0 < stats.overall.multiplicity <= 8.0


@pytest.mark.unit
def test_noise_gives_the_expected_rmeas():
    """For constant I and Gaussian noise, Rmeas -> (sigma / I) * sqrt(2 / pi)."""
    hkl = _asu_grid(8)
    I0, sigma = 100.0, 10.0
    p1 = _data(hkl, torch.full((len(hkl),), I0), torch.full((len(hkl),), sigma))
    p1 = p1.expand_to_p1()
    p1.verbose = 0
    g = torch.Generator().manual_seed(0)
    p1.I = p1.I + sigma * torch.randn(len(p1.I), generator=g)

    _, stats = merge_to_spacegroup(p1, SG)

    expected = sigma / I0 * math.sqrt(2.0 / math.pi)
    assert stats.overall.r_meas == pytest.approx(expected, rel=0.05)
    assert stats.overall.r_merge < stats.overall.r_meas


@pytest.mark.unit
def test_same_group_has_no_agreement_to_measure():
    d = _synthetic()
    merged, stats = merge_to_spacegroup(d, SG)
    assert len(merged.hkl) == len(d.hkl)
    assert stats.overall.multiplicity == 1.0
    assert stats.overall.r_merge is None
    assert stats.overall.cc_sym is None


@pytest.mark.unit
def test_lowering_symmetry_generates_the_missing_reflections():
    d = _synthetic()
    merged, stats = merge_to_spacegroup(d, "P 1 21 1")
    assert len(merged.hkl) > len(d.hkl)
    assert stats.overall.multiplicity == 1.0
    orig = _invariant(merged.hkl)
    torch.testing.assert_close(merged.I, orig)


@pytest.mark.unit
def test_incompatible_cell_is_refused():
    d = _synthetic()
    with pytest.raises(ValueError, match="incompatible"):
        merge_to_spacegroup(d, "P 4 21 2")


@pytest.mark.unit
def test_row_order_does_not_matter():
    p1 = _synthetic().expand_to_p1()
    p1.verbose = 0
    g = torch.Generator().manual_seed(1)
    p1.I = p1.I + torch.randn(len(p1.I), generator=g)
    shuffled = p1.__select__(torch.randperm(len(p1.hkl), generator=g))

    a, sa = merge_to_spacegroup(p1, SG)
    b, sb = merge_to_spacegroup(shuffled, SG)

    assert torch.equal(a.hkl, b.hkl)
    torch.testing.assert_close(a.I, b.I)
    assert sa.overall.r_merge == pytest.approx(sb.overall.r_merge)


@pytest.mark.unit
def test_same_seed_gives_identical_statistics():
    p1 = _synthetic().expand_to_p1()
    p1.verbose = 0
    p1.I = p1.I + torch.randn(len(p1.I), generator=torch.Generator().manual_seed(2))
    assert merge_to_spacegroup(p1, SG, seed=3)[1] == merge_to_spacegroup(p1, SG, seed=3)[1]


@pytest.mark.unit
def test_masked_and_unusable_observations_do_not_contribute():
    p1 = _synthetic().expand_to_p1()
    p1.verbose = 0
    canon, _, _, order = SpaceGroup(SG).canonicalize_hkl(p1.hkl, include_friedel=True)
    per_row = torch.empty_like(canon)
    per_row[order] = canon
    first = per_row[0]
    members = (per_row == first).all(dim=-1)
    # Every copy of one reflection masked; a NaN sigma and a huge outlier
    # elsewhere, the outlier masked too.
    other = torch.nonzero(~members).squeeze(-1)
    p1.I_sigma[other[0]] = float("nan")
    p1.I[other[1]] = 1e6
    keep = ~members
    keep[other[1]] = False
    p1.masks["test"] = keep

    merged, stats = merge_to_spacegroup(p1, SG)

    assert tuple(first.tolist()) not in _by_hkl(merged, merged.I)
    assert float(merged.I.max()) < 1e3
    assert stats.overall.r_merge == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_bijvoet_mates_stay_apart_and_share_rfree():
    hkl = _asu_grid()
    acentric = ~SpaceGroup(SG).is_centric(hkl)
    hkl = hkl[acentric]
    both = torch.cat([hkl, -hkl])
    I = torch.cat([_invariant(hkl), _invariant(hkl) * 1.1])
    rfree = torch.ones(len(both), dtype=torch.bool)
    rfree[: len(hkl) // 10] = False  # free on the "+" member only
    d = _data(both, I, torch.ones_like(I), rfree_flags=rfree, friedel_merged=False)
    assert d.friedel_merged is False

    merged, _ = merge_to_spacegroup(d, SG)
    assert len(merged.hkl) == len(both)
    assert int(merged.friedel_flags.sum()) == len(hkl)
    gid, n = merged.asu_group_indices()
    flags = merged.rfree_flags.to(torch.bool)
    split = ReflectionData._group_any(flags, gid, n) & ReflectionData._group_any(
        ~flags, gid, n
    )
    assert not bool(split.any())
    assert int((~flags).sum()) == 2 * (len(hkl) // 10)

    pooled, stats = merge_to_spacegroup(d, SG, anomalous=False)
    assert len(pooled.hkl) == len(hkl)
    assert stats.overall.r_merge > 0


@pytest.mark.unit
def test_amplitude_only_data_merge_on_f_squared(mtz_dir):
    d = ReflectionData(verbose=0, device="cpu").load_mtz(str(mtz_dir / "3E98.mtz"))
    assert d.I is None
    p1 = d.expand_to_p1()
    p1.verbose = 0
    _, stats = merge_to_spacegroup(p1, d.spacegroup)
    assert stats.on == "F^2"
    assert stats.overall.r_merge == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_from_tensors_keeps_intensities_and_validation_row_aligned():
    """Canonicalization reorders rows; I and validation flags must move with them."""
    hkl = _asu_grid()
    # Off-ASU equivalents in reverse order force a non-trivial permutation.
    raw = torch.flip(hkl * torch.tensor([-1, 1, -1], dtype=hkl.dtype), dims=[0])
    I = _invariant(raw)
    validation = (I.round().to(torch.int64) % 2) == 0
    d = _data(raw, I, torch.ones_like(I), validation_flags=validation)
    torch.testing.assert_close(d.I, _invariant(d.hkl))
    expected = (_invariant(d.hkl).round().to(torch.int64) % 2) == 0
    assert torch.equal(d.validation_flags, expected)
