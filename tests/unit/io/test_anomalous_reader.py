"""Reader-level anomalous (Bijvoet) ingestion + friedel_merged gating.

Covers:
  * A normal merged MTZ loads with friedel_merged=True and unchanged rows.
  * An MTZ carrying F(+)/F(-) columns is auto-stacked into explicit signed-HKL
    Bijvoet pairs: friedel_merged=False, ~doubled rows, friedel_flags populated,
    Friedel mates share one R-free flag, and centrics are not duplicated.
  * ModelFT.apply_bijvoet round-trips through state_dict (it is a buffer).
"""

import collections

import numpy as np
import pytest
import reciprocalspaceship as rs
import torch

from torchref.base.french_wilson import is_centric_from_hkl
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model.model_ft import ModelFT


@pytest.fixture
def anomalous_two_column_mtz(mtz_dir, tmp_path):
    """1DAW reflections written with F(+)/F(-) (two-column anomalous) columns."""
    ds = rs.read_mtz(str(mtz_dir / "1DAW.mtz"))
    amp = next(c for c in ds.columns if ds.dtypes[c].mtztype == "F")
    sig = next((c for c in ds.columns if ds.dtypes[c].mtztype == "Q"), None)
    cols = [amp] + ([sig] if sig else [])
    keep = cols + [c for c in ds.columns if ds.dtypes[c].mtztype == "I"]
    unst = ds[keep].unstack_anomalous(columns=cols)
    out = tmp_path / "anom_two_column.mtz"
    unst.write_mtz(str(out))
    return str(out), len(ds)


def test_merged_mtz_default(mtz_dir):
    """A plain merged MTZ: friedel_merged=True, no signed-HKL expansion."""
    d = ReflectionData(verbose=0)
    d.load_mtz(str(mtz_dir / "1DAW.mtz"))
    assert d.friedel_merged is True
    assert torch.equal(d.hkl_for_sf(), d.hkl)
    assert not bool(d.friedel_flags.any())


def test_anomalous_mtz_unstacked(anomalous_two_column_mtz):
    path, n_merged = anomalous_two_column_mtz
    d = ReflectionData(verbose=0)
    d.load_mtz(path)

    # Stacked into Bijvoet pairs.
    assert d.friedel_merged is False
    # More rows than the merged ASU, but no more than 2x (centrics not split).
    assert n_merged < len(d.hkl) <= 2 * n_merged
    # The minus mates are flagged and carry the negated signed HKL.
    assert bool(d.friedel_flags.any())
    sf = d.hkl_for_sf()
    assert torch.equal(sf[d.friedel_flags], -d.hkl[d.friedel_flags])
    assert torch.equal(sf[~d.friedel_flags], d.hkl[~d.friedel_flags])


def test_anomalous_opt_out_forces_merged(anomalous_two_column_mtz, mtz_dir):
    """anomalous=False forces a merged load even when (+)/(-) columns are present."""
    path, n_merged = anomalous_two_column_mtz

    merged = ReflectionData(verbose=0)
    merged.load_mtz(path, anomalous=False)
    assert merged.friedel_merged is True
    assert len(merged.hkl) == n_merged  # back to one row per ASU reflection
    assert merged.F is not None and len(merged.F) == len(merged.hkl)
    assert not bool(merged.friedel_flags.any())

    # anomalous=True keeps the Bijvoet pairs (same as the auto default here).
    forced = ReflectionData(verbose=0)
    forced.load_mtz(path, anomalous=True)
    assert forced.friedel_merged is False
    assert len(forced.hkl) > n_merged


def test_centrics_not_duplicated(anomalous_two_column_mtz):
    path, _ = anomalous_two_column_mtz
    d = ReflectionData(verbose=0)
    d.load_mtz(path)
    centric = is_centric_from_hkl(d.hkl, d.spacegroup)
    # Centric reflections obey Friedel's law and must appear exactly once each.
    canon = [tuple(h) for h in d.hkl[centric].tolist()]
    assert len(canon) == len(set(canon))
    # ... and are never flagged as conjugated mates.
    assert not bool((d.friedel_flags & centric).any())


def _mixed_partition_groups(d, include_validation=False):
    """Canonical ASU indices whose rows disagree about which set they belong to.

    Bijvoet mates share a canonical index, so a non-empty result means a pair was
    split across work/free -- the held-out mate has leaked into the work set.
    """
    cols = [d.rfree_flags.tolist()]
    if include_validation:
        cols.append(d.validation_flags.tolist())
    groups = collections.defaultdict(set)
    for h, *state in zip(map(tuple, d.hkl.tolist()), *cols):
        groups[h].add(tuple(bool(s) for s in state))
    return [h for h, v in groups.items() if len(v) > 1]


def test_rfree_shared_across_mates(anomalous_two_column_mtz):
    """Flags read from the file keep Bijvoet mates in the same set."""
    path, _ = anomalous_two_column_mtz
    d = ReflectionData(verbose=0)
    d.load_mtz(path)
    if d.rfree_flags is None:
        pytest.skip("no R-free flags in source")
    assert _mixed_partition_groups(d) == []


def test_generated_rfree_shared_across_mates(anomalous_two_column_mtz):
    """*Generated* flags keep Bijvoet mates together too.

    The read path gets this for free because rs.stack_anomalous duplicates the
    FreeR column; the generated path has to group by canonical ASU index itself.
    """
    path, _ = anomalous_two_column_mtz
    d = ReflectionData(verbose=0)
    d.load_mtz(path)
    # Guard the premise: this fixture really does carry split Bijvoet pairs.
    assert d.friedel_merged is False
    assert bool(d.friedel_flags.any())

    d.regenerate_rfree_flags(force=True, seed=0)
    assert d.rfree_source == "Generated (resolution-binned, ASU-grouped)"
    assert bool((d.rfree_flags == 0).any())  # a free set actually exists
    assert _mixed_partition_groups(d) == []


def test_generated_validation_set_shared_across_mates(anomalous_two_column_mtz):
    """The free/validation carve-out must not re-split Bijvoet pairs."""
    path, _ = anomalous_two_column_mtz
    d = ReflectionData(verbose=0)
    d.load_mtz(path)
    d.regenerate_rfree_flags(force=True, seed=0)
    d.generate_validation_set(val_fraction_of_free=0.5, seed=0)

    assert bool(d.validation_flags.any())
    assert _mixed_partition_groups(d, include_validation=True) == []


def test_write_auto_anomalous_from_merge_state(
    anomalous_two_column_mtz, mtz_dir, pdb_dir, tmp_path
):
    """write_mtz() with anomalous=None picks the layout from friedel_merged."""
    path, _ = anomalous_two_column_mtz

    def written_columns(in_mtz, apply_bijvoet):
        d = ReflectionData(verbose=0)
        d.load_mtz(in_mtz)
        model = ModelFT(
            verbose=0, max_res=2.0, wavelength=1.54, apply_bijvoet=apply_bijvoet
        )
        model.load_pdb(str(pdb_dir / "1DAW.pdb"))
        with torch.no_grad():
            fcalc = model(d.hkl_for_sf())
        out = tmp_path / "auto_out.mtz"
        d.write_mtz(str(out), fcalc=fcalc)  # anomalous=None -> auto
        return set(rs.read_mtz(str(out)).columns)

    # Unmerged input -> anomalous (+/-) columns emitted automatically.
    anom_cols = written_columns(path, apply_bijvoet=True)
    assert {"F-obs(+)", "F-obs(-)", "F-model(+)", "F-model(-)"} <= anom_cols

    # Merged input -> legacy layout, no (+/-) columns.
    merged_cols = written_columns(str(mtz_dir / "1DAW.mtz"), apply_bijvoet=False)
    assert "F-obs(+)" not in merged_cols


def test_from_tensors_friedel_merged(mtz_dir):
    """from_tensors auto-detects merge state and honors an explicit override."""
    from torchref.symmetry import Cell, SpaceGroup

    ds = rs.read_mtz(str(mtz_dir / "1DAW.mtz"))
    cell = Cell(
        [ds.cell.a, ds.cell.b, ds.cell.c, ds.cell.alpha, ds.cell.beta, ds.cell.gamma]
    )
    sg = SpaceGroup(ds.spacegroup.hm)
    amp = next(c for c in ds.columns if ds.dtypes[c].mtztype == "F")
    sig = next(c for c in ds.columns if ds.dtypes[c].mtztype == "Q")

    def tensors(dataset):
        hkl = torch.tensor(
            dataset.reset_index()[["H", "K", "L"]].to_numpy().astype("int32")
        )
        F = torch.tensor(dataset[amp].to_numpy().astype("float32"))
        S = torch.tensor(dataset[sig].to_numpy().astype("float32"))
        R = torch.ones(len(F), dtype=torch.bool)
        return hkl, F, S, R

    # Merged ASU input -> auto-detects merged.
    hkl, F, S, R = tensors(ds)
    rd = ReflectionData.from_tensors(hkl, F, S, cell, sg, rfree_flags=R, verbose=0)
    assert rd.friedel_merged is True

    # Stacked (+h/-h) input -> auto-detects unmerged.
    st = ds[[amp, sig]].stack_anomalous()
    hkl_s, F_s, S_s, R_s = tensors(st)
    rd_s = ReflectionData.from_tensors(
        hkl_s, F_s, S_s, cell, sg, rfree_flags=R_s, verbose=0
    )
    assert rd_s.friedel_merged is False
    assert bool(rd_s.friedel_flags.any())

    # Explicit override wins over auto-detection.
    rd_o = ReflectionData.from_tensors(
        hkl, F, S, cell, sg, rfree_flags=R, verbose=0, friedel_merged=False
    )
    assert rd_o.friedel_merged is False


def test_apply_bijvoet_buffer_roundtrips(pdb_dir):
    # Defaults off (merged behavior); on when requested. The flag is a registered
    # buffer, so it appears in state_dict and is carried by copy().
    assert bool(ModelFT(verbose=0).anomalous_bijvoet) is False

    model = ModelFT(verbose=0, max_res=2.0, wavelength=1.54, apply_bijvoet=True)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    assert bool(model.anomalous_bijvoet) is True
    assert "anomalous_bijvoet" in model.state_dict()
    assert bool(model.copy().anomalous_bijvoet) is True
