"""Unit tests for shared R-free assignment across datasets (torchref.io.rfree)."""

import gemmi
import numpy as np
import pytest

rs = pytest.importorskip("reciprocalspaceship")

from torchref.io import rfree  # noqa: E402


@pytest.fixture(scope="module")
def full(mtz_dir):
    path = mtz_dir / "3GR5.mtz"
    if not path.exists():
        pytest.skip("3GR5.mtz not found")
    return rs.read_mtz(str(path))


def _subset(ds, fraction, seed, dmin=None):
    rng = np.random.default_rng(seed)
    keep = rng.random(len(ds)) < fraction
    if dmin is not None:
        keep &= ds.compute_dHKL()["dHKL"].to_numpy() > dmin
    return ds[keep].copy()


def _table(ds, flags):
    return dict(zip(rfree.hkl_keys(rfree.asu_hkl(ds)).tolist(), flags.tolist()))


def test_complete_table_exact_per_shell(full):
    keys, flags = rfree.complete_flag_table(
        full.cell, full.spacegroup, 2.0, 20, shell_size=1000
    )
    assert len(flags) % 1000 == 0
    hkl = rfree._unkey(keys)
    np.testing.assert_array_equal(rfree.hkl_keys(hkl), keys)
    order = np.argsort(1.0 / rs.utils.compute_dHKL(hkl, full.cell) ** 2, kind="stable")
    shells = flags[order].reshape(-1, 1000)
    # every shell holds exactly 5% free (up to ties in resolution at edges)
    assert np.all(np.abs((shells == 0).sum(1) - 50) <= 2)


def test_table_is_prefix_stable(full):
    """Extending to higher resolution never changes existing flags."""
    k1, f1 = rfree.complete_flag_table(full.cell, full.spacegroup, 2.5, 20, seed=3)
    k2, f2 = rfree.complete_flag_table(full.cell, full.spacegroup, 2.0, 20, seed=3)
    values, found = rfree._lookup(k2, f2, k1)
    assert found.all()
    np.testing.assert_array_equal(values, f1)


def test_common_reflections_share_flags(full):
    sets = {
        "dark": _subset(full, 0.9, 1, dmin=2.3),
        "light1": _subset(full, 0.8, 2),
        "light2": _subset(full, 0.7, 3),
    }
    flags, info = rfree.uniform_rfree(sets, free_fraction=0.05, seed=0)
    assert info["n_flags"] == 20
    tables = [_table(ds, flags[n]) for n, ds in sets.items()]
    common = set.intersection(*(set(t) for t in tables))
    assert common
    assert all(len({t[k] for t in tables}) == 1 for k in common)
    for n, ds in sets.items():
        assert len(flags[n]) == len(ds)
        assert abs((flags[n] == 0).mean() - 0.05) < 0.01


def test_symmetry_and_friedel_mates_share_flag(full):
    ds = _subset(full, 1.0, 0)
    # add a copy of every reflection as its Friedel mate
    mate = ds.copy().reset_index()
    mate[["H", "K", "L"]] *= -1
    both = rs.concat([ds, mate.set_index(["H", "K", "L"])])
    flags, _ = rfree.uniform_rfree({"x": both}, seed=0)
    n = len(ds)
    np.testing.assert_array_equal(flags["x"][:n], flags["x"][n:])


def test_deterministic_and_independent_of_coverage(full):
    a = _subset(full, 0.6, 5)
    b = _subset(full, 0.9, 6)
    fa, _ = rfree.uniform_rfree({"a": a}, seed=7)
    fb, _ = rfree.uniform_rfree({"b": b}, seed=7, dmin=1.8)
    ta, tb = _table(a, fa["a"]), _table(b, fb["b"])
    common = set(ta) & set(tb)
    assert all(ta[k] == tb[k] for k in common)
    fa2, _ = rfree.uniform_rfree({"a": a}, seed=7)
    np.testing.assert_array_equal(fa["a"], fa2["a"])


@pytest.mark.parametrize("convention", ["ccp4", "phenix"])
def test_reference_free_set_is_inherited(full, convention):
    ref = _subset(full, 0.8, 11)
    free = np.random.default_rng(3).random(len(ref)) < 0.1
    if convention == "ccp4":
        values = np.where(free, 0, np.random.default_rng(4).integers(1, 10, len(ref)))
    else:
        values = free.astype(int)  # Phenix: 1 = free
    ref["R-free-flags"] = rs.DataSeries(values, index=ref.index, dtype="I")

    target = _subset(full, 0.9, 12)
    flags, info = rfree.uniform_rfree({"t": target}, reference=ref, seed=0)
    assert info["reference"]["column"] == "R-free-flags"
    assert info["reference"]["convention"].startswith(
        "ccp4" if convention == "ccp4" else "binary"
    )
    ref_free = _table(ref, free.astype(int))
    t = _table(target, flags["t"])
    for k in set(ref_free) & set(t):
        assert (t[k] == 0) == bool(ref_free[k])


def test_check_compatible_flags_mismatch(full):
    other = full.copy()
    other.cell = gemmi.UnitCell(
        *(np.array(full.cell.parameters) * [1.05, 1, 1, 1, 1, 1])
    )
    assert rfree.check_compatible({"a": full, "b": full.copy()}) == []
    assert rfree.check_compatible({"a": full, "b": other})


def test_apply_flags_replaces_existing_columns(full):
    ds = full.copy()
    flags = np.zeros(len(ds), dtype=np.int32)
    out = rfree.apply_flags(ds, flags)
    assert list(out.columns).count("FreeR_flag") == 1
    assert "FreeR_flag_orig" not in out.columns
    kept = rfree.apply_flags(ds, flags, keep_old=True)
    assert "FreeR_flag_orig" in kept.columns
    assert set(ds.columns) - {"FreeR_flag"} <= set(out.columns)


def test_scale_columns_amplitude_and_intensity():
    ds = rs.DataSet(
        {
            "H": [1, 2],
            "K": [0, 0],
            "L": [0, 0],
            "FP": [10.0, 20.0],
            "SIGFP": [1.0, 2.0],
            "I": [100.0, 400.0],
            "SIGI": [10.0, 20.0],
            "PHI": [30.0, 40.0],
            "FWT": [5.0, 6.0],
            "PHWT": [0.0, 90.0],
        },
        cell=[50, 50, 50, 90, 90, 90],
        spacegroup=1,
    ).set_index(["H", "K", "L"])
    ds = ds.astype(
        {
            "FP": "F",
            "SIGFP": "Q",
            "I": "J",
            "SIGI": "Q",
            "PHI": "P",
            "FWT": "F",
            "PHWT": "P",
        }
    )
    out, cols = rfree.scale_columns(ds, np.array([2.0, 0.5]))
    assert cols == ["FP", "SIGFP", "I", "SIGI"]
    np.testing.assert_allclose(out.FP, [20, 10])
    np.testing.assert_allclose(out.SIGFP, [2, 1])
    np.testing.assert_allclose(out.I, [400, 100])
    np.testing.assert_allclose(out.SIGI, [40, 5])
    np.testing.assert_allclose(out.PHI, [30, 40])
    np.testing.assert_allclose(out.FWT, [5, 6])  # map coefficients untouched


def _with_flags(ds, free, phenix=False):
    out = ds.copy()
    for c in [c for c in out.columns if c in rfree.FLAG_COLUMN_NAMES]:
        out = out.drop(columns=c)
    values = free.astype(int) if phenix else (~free).astype(int)
    out["R-free-flags" if phenix else "FreeR_flag"] = rs.DataSeries(
        values, index=out.index, dtype="I"
    )
    return out


def test_extension_seed_depends_only_on_reference_partition(full):
    """Extending one free set gives the same new flags whatever its format."""
    d = full.compute_dHKL()["dHKL"].to_numpy()
    low = full[d > 2.6]
    free = np.random.default_rng(0).random(len(low)) < 0.05
    ccp4 = _with_flags(low, free)
    phenix = _with_flags(low, free, phenix=True).sample(frac=1.0, random_state=1)
    target = full.copy()
    f1, i1 = rfree.uniform_rfree({"t": target}, reference=ccp4)
    f2, i2 = rfree.uniform_rfree({"t": target}, reference=phenix)
    assert i1["seed_source"] == "reference free set"
    assert i1["seed"] == i2["seed"]
    np.testing.assert_array_equal(f1["t"] == 0, f2["t"] == 0)
    assert i1["n_generated_beyond_reference"] > 0

    # a different reference partition changes the extension
    other = _with_flags(low, np.roll(free, 1))
    f3, i3 = rfree.uniform_rfree({"t": target}, reference=other)
    assert i3["seed"] != i1["seed"]
    beyond = d <= 2.6
    assert ((f1["t"] == 0) != (f3["t"] == 0))[beyond].any()

    # an explicit seed still wins
    _, i4 = rfree.uniform_rfree({"t": target}, reference=ccp4, seed=5)
    assert i4["seed"] == 5 and i4["seed_source"] == "user"


def test_extension_is_identical_across_targets(full):
    d = full.compute_dHKL()["dHKL"].to_numpy()
    low = full[d > 2.6]
    ref = _with_flags(low, np.random.default_rng(2).random(len(low)) < 0.05)
    a, b = _subset(full, 0.7, 21), _subset(full, 0.7, 22)
    fa, _ = rfree.uniform_rfree({"a": a}, reference=ref)
    fb, _ = rfree.uniform_rfree({"b": b}, reference=ref)
    ta, tb = _table(a, fa["a"] == 0), _table(b, fb["b"] == 0)
    common = set(ta) & set(tb)
    assert all(ta[k] == tb[k] for k in common)


@pytest.fixture
def small(mtz_dir):
    """The first 30 rows of deposited 1DAW."""
    path = mtz_dir / "1DAW.mtz"
    if not path.exists():
        pytest.skip("1DAW.mtz not found")
    return rs.read_mtz(str(path)).iloc[:30].copy()


def test_all_excluded_rows_stay_excluded(small):
    small["FreeR_flag"] = rs.DataSeries(
        np.full(len(small), -1), index=small.index, dtype="I"
    )
    flags, info = rfree.uniform_rfree({"x": small}, seed=0)
    assert info["n_excluded"]["x"] == len(small)
    assert (flags["x"] == -1).all()


def test_conflicting_equivalents_are_inconsistent(small):
    free_row = small[small["FreeR_flag"] == 0].iloc[:1]
    if free_row.empty:
        free_row = small.iloc[:1].copy()
        small.loc[free_row.index, "FreeR_flag"] = 0
        free_row = small.loc[free_row.index]
    dup = free_row.copy()
    dup["FreeR_flag"] = rs.DataSeries([1], index=dup.index, dtype="I")  # work copy
    bad = rs.concat([small, dup])
    report = rfree.compare_free_sets({"good": small, "bad": bad})
    assert report["files"]["bad"]["n_conflicting"] == 1
    assert not report["consistent"]
    # a single inconsistent file is not consistent either
    assert not rfree.compare_free_sets({"bad": bad})["consistent"]
    assert rfree.compare_free_sets({"good": small})["consistent"]


def test_reference_without_free_reflections_is_rejected(small):
    small["FreeR_flag"] = rs.DataSeries(
        np.ones(len(small)), index=small.index, dtype="I"
    )
    with pytest.raises(ValueError, match="no reflection as free"):
        rfree.uniform_rfree({"x": small}, reference=small)
    report = rfree.compare_free_sets({"x": small})
    assert report["files"]["x"]["n_free"] == 0 and not report["consistent"]


def test_hkl_keys_reject_indices_beyond_the_encoding():
    edge = np.array([[1023, -1023, 0], [-1023, 1023, 1023]])
    np.testing.assert_array_equal(rfree._unkey(rfree.hkl_keys(edge)), edge)
    # (0, 1024, 0) and (1, -1024, 0) would share a key
    with pytest.raises(ValueError, match="supported range"):
        rfree.hkl_keys(np.array([[0, 1024, 0]]))
    with pytest.raises(ValueError, match="supported range"):
        rfree.hkl_keys(np.array([[1, -1024, 0]]))


def test_max_free_must_be_positive(small):
    with pytest.raises(ValueError, match="max_free"):
        rfree.uniform_rfree({"x": small}, max_free=0)
