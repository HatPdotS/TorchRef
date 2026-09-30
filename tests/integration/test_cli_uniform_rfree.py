"""End-to-end tests for ``torchref.uniform-rfree``."""

import subprocess
import sys

import gemmi
import numpy as np
import pytest

rs = pytest.importorskip("reciprocalspaceship")

from torchref.io import rfree  # noqa: E402

pytestmark = pytest.mark.integration


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "torchref.cli.uniform_rfree", *map(str, args)],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def inputs(mtz_dir, cif_sf_dir, tmp_path):
    mtz, cif = mtz_dir / "3GR5.mtz", cif_sf_dir / "3GR5-sf.cif"
    if not mtz.exists() or not cif.exists():
        pytest.skip("3GR5 test files not found")
    full = rs.read_mtz(str(mtz))
    d = full.compute_dHKL()["dHKL"].to_numpy()
    rng = np.random.default_rng(0)
    dark = full[(d > 2.3) & (rng.random(len(full)) < 0.9)]
    light = full[rng.random(len(full)) < 0.8].copy()
    # a scale and anisotropy difference the scaler should remove
    L = light.get_hkls()[:, 2].astype(float)
    f = 1.7 * np.exp(-2.0 * (L / L.max()) ** 2)
    light["FP"] = rs.DataSeries(light.FP.to_numpy() * f, index=light.index, dtype="F")
    light["SIGFP"] = rs.DataSeries(light.SIGFP.to_numpy() * f, index=light.index, dtype="Q")
    paths = [tmp_path / "dark.mtz", tmp_path / "light.mtz"]
    dark.write_mtz(str(paths[0]))
    light.write_mtz(str(paths[1]))
    return full, paths + [cif]


def _free_tables(paths):
    tables = []
    for p in paths:
        ds = rfree.read_sf_file(str(p))
        keys = rfree.hkl_keys(rfree.asu_hkl(ds)).tolist()
        tables.append(dict(zip(keys, (ds["FreeR_flag"].to_numpy() == 0).tolist())))
    return tables


def test_uniform_flags_mtz_and_cif(inputs, tmp_path):
    _, paths = inputs
    out = tmp_path / "out"
    res = _run(*paths, "-o", out, "--format", "mtz", "cif")
    assert res.returncode == 0, res.stderr
    written = sorted(out.iterdir())
    assert len(written) == 6
    tables = _free_tables(written)
    common = set.intersection(*(set(t) for t in tables))
    assert common
    assert all(len({t[k] for t in tables}) == 1 for k in common)
    dark = rs.read_mtz(str(out / "dark_rfree.mtz"))
    assert {"FP", "SIGFP", "FreeR_flag"} <= set(dark.columns)


def test_scale_onto_reference(inputs, tmp_path):
    full, paths = inputs
    out = tmp_path / "out"
    res = _run(*paths[:2], "-o", out, "--scale", "--scale-reference", "dark", "--device", "cpu")
    assert res.returncode == 0, res.stderr
    light = rs.read_mtz(str(out / "light_rfree.mtz"))
    ratio = light.FP.to_numpy() / full.loc[light.index].FP.to_numpy()
    assert abs(np.median(ratio) - 1) < 0.02


def test_torchref_reads_flags(inputs, tmp_path):
    from torchref.io.datasets.reflection_data import ReflectionData

    _, paths = inputs
    out = tmp_path / "out"
    assert _run(paths[0], "-o", out, "--fresh").returncode == 0
    data = ReflectionData(device="cpu", verbose=0).load_mtz(str(out / "dark_rfree.mtz"))
    assert not str(data.rfree_source).startswith("Generated")
    free_fraction = 1 - data.rfree_flags.float().mean().item()
    assert abs(free_fraction - 0.05) < 0.01


def test_mismatched_cell_fails(inputs, tmp_path):
    _, paths = inputs
    other = rs.read_mtz(str(paths[1]))
    other.cell = gemmi.UnitCell(*(np.array(other.cell.parameters) * [1.05, 1, 1, 1, 1, 1]))
    bad = tmp_path / "bad.mtz"
    other.write_mtz(str(bad))
    res = _run(paths[0], bad, "-o", tmp_path / "out")
    assert res.returncode == 1
    assert "cell" in res.stderr


def _strip_flags(src, dst):
    ds = rs.read_mtz(str(src))
    ds.drop(columns=[c for c in ds.columns if c in rfree.FLAG_COLUMN_NAMES]).write_mtz(str(dst))


def test_check_mode(inputs, tmp_path):
    _, paths = inputs
    # subsets of one file share its deposited free set
    res = _run(*paths[:2], "--check")
    assert res.returncode == 0, res.stderr
    assert "consistent" in res.stdout
    assert not any(tmp_path.glob("*_rfree.*"))
    fresh = tmp_path / "fresh"
    assert _run(paths[1], "--fresh", "--seed", "5", "-o", fresh).returncode == 0
    res = _run(paths[0], fresh / "light_rfree.mtz", "--check")
    assert res.returncode == 2
    assert "DISAGREE" in res.stdout


def test_auto_inherits_or_generates(inputs, tmp_path):
    full, paths = inputs
    out = tmp_path / "out"
    bare = tmp_path / "bare.mtz"
    _strip_flags(paths[1], bare)
    res = _run(bare, paths[0], "-o", out)  # flags only in the second input
    assert res.returncode == 0, res.stderr
    assert "inherited from dark" in res.stdout
    orig = _free_tables([paths[0]])[0]  # the reference: dark's own free set
    new = _free_tables([out / "bare_rfree.mtz"])[0]
    common = set(new) & set(orig)
    assert common
    assert all(new[k] == orig[k] for k in common)

    none = tmp_path / "none"
    _strip_flags(paths[0], tmp_path / "dark_bare.mtz")
    res = _run(tmp_path / "dark_bare.mtz", bare, "-o", none)
    assert res.returncode == 0, res.stderr
    assert "generating a new free set" in res.stdout


def test_mixed_resolution_and_max_free(inputs, tmp_path):
    _, paths = inputs
    out = tmp_path / "out"
    # dark (reference) stops at 2.3 A, light extends to 2.05 A
    res = _run(*paths[:2], "-o", out, "--max-free", "500", "--fresh")
    assert res.returncode == 0, res.stderr
    assert "--free-fraction" in res.stdout  # reproducibility hint for the cap
    tables = _free_tables([out / "dark_rfree.mtz", out / "light_rfree.mtz"])
    assert sum(tables[1].values()) <= 500
    res = _run(*paths[:2], "-o", tmp_path / "inh")
    assert "reference ends at" in res.stdout
    assert "seed from reference free set" in res.stdout
    light = rs.read_mtz(str(tmp_path / "inh" / "light_rfree.mtz"))
    d = light.compute_dHKL()["dHKL"].to_numpy()
    free = light["FreeR_flag"].to_numpy() == 0
    # extension beyond the reference keeps the reference's free fraction
    assert abs(free[d < 2.3].mean() - free[d >= 2.3].mean()) < 0.02


def test_excluded_flags_survive(inputs, tmp_path):
    from torchref.io.datasets.reflection_data import ReflectionData

    _, paths = inputs
    ds = rs.read_mtz(str(paths[0]))
    flags = ds["FreeR_flag"].to_numpy().astype(int)
    flags[:100] = -1
    ds["FreeR_flag"] = rs.DataSeries(flags, index=ds.index, dtype="I")
    src = tmp_path / "excl.mtz"
    ds.write_mtz(str(src))
    out = tmp_path / "out"
    res = _run(src, paths[1], "-o", out, "--format", "mtz", "cif")
    assert res.returncode == 0, res.stderr
    back = rs.read_mtz(str(out / "excl_rfree.mtz"))
    assert (back["FreeR_flag"].to_numpy()[:100] == -1).all()
    # the light file does not inherit dark's exclusions
    assert (rs.read_mtz(str(out / "light_rfree.mtz"))["FreeR_flag"].to_numpy() >= 0).all()
    for f in ("excl_rfree.mtz", "excl_rfree.cif"):
        data = ReflectionData(device="cpu", verbose=0)
        data.load_mtz(str(out / f)) if f.endswith("mtz") else data.load_cif(str(out / f))
        assert int((~data.masks["flagged_initial"]).sum()) == 100, f


def test_unusable_reference_is_skipped_or_rejected(inputs, tmp_path):
    _, paths = inputs
    ds = rs.read_mtz(str(paths[0]))
    ds["FreeR_flag"] = rs.DataSeries(np.ones(len(ds)), index=ds.index, dtype="I")
    allwork = tmp_path / "allwork.mtz"
    ds.write_mtz(str(allwork))
    bare = tmp_path / "bare.mtz"
    _strip_flags(paths[1], bare)
    # auto: skipped with a warning, new set generated
    res = _run(allwork, bare, "-o", tmp_path / "auto")
    assert res.returncode == 0, res.stderr
    assert "not inheriting from 'allwork'" in res.stderr
    assert "generating a new free set" in res.stdout
    # explicit: clean error, no traceback
    res = _run(allwork, bare, "--reference", "allwork", "-o", tmp_path / "explicit")
    assert res.returncode == 1
    assert "no reflection as free" in res.stderr and "Traceback" not in res.stderr
