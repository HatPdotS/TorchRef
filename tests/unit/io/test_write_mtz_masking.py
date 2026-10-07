"""Map and model columns written by ``write_mtz`` respect ``masks()``.

Pins that FWT/PHWT, DELFWT/PHDELWT and F-model/PH-model are MTZ-missing for every
reflection the refinement excludes -- past the resolution cut or rejected -- while the
observed columns stay complete, in both the per-row and the anomalous layout.
"""

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs
import torch

from torchref.io import mtz
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model.model_ft import ModelFT

CUT = 3.0
MAP_COLS = ["FWT", "PHWT", "DELFWT", "PHDELWT", "F-model", "PH-model"]


def _fcalc(data, pdb_dir):
    model = ModelFT(verbose=0, max_res=CUT)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    with torch.no_grad():
        return data.structure_factors(model, cached=False)


def _d_spacing(out):
    return out.compute_dHKL()["dHKL"].to_numpy()


@pytest.fixture
def cut_data(mtz_dir):
    data = ReflectionData(verbose=0)
    data.load_mtz(str(mtz_dir / "1DAW.mtz"))
    assert data.resolution.min().item() < CUT - 0.5, "data must extend past the cut"
    data.filter_by_resolution(d_min=CUT)
    return data


@pytest.fixture
def excluded_mtz(mtz_dir, tmp_path):
    """1DAW.mtz with FreeR_flag -1 (excluded) on 500 rows."""
    ds = rs.read_mtz(str(mtz_dir / "1DAW.mtz"))
    flags = ds["FreeR_flag"].to_numpy().copy()
    flags[np.random.default_rng(0).choice(len(ds), 500, replace=False)] = -1
    ds["FreeR_flag"] = rs.DataSeries(flags, index=ds.index).astype("I")
    path = tmp_path / "excluded.mtz"
    ds.write_mtz(str(path))
    return path


def _load(path):
    data = ReflectionData(verbose=0)
    data.load_mtz(str(path))
    return data


@pytest.fixture
def cut_anomalous_data(mtz_dir, tmp_path):
    stacked = tmp_path / "anom_in.mtz"
    rs.read_mtz(str(mtz_dir / "1DAW.mtz")).stack_anomalous().write_mtz(str(stacked))
    data = ReflectionData(verbose=0)
    data.load_mtz(str(stacked))
    data.filter_by_resolution(d_min=CUT)
    return data


class TestPerRowLayout:
    def test_map_columns_missing_exactly_where_masked(self, cut_data, pdb_dir, tmp_path):
        out_path = tmp_path / "out.mtz"
        cut_data.write_mtz(str(out_path), fcalc=_fcalc(cut_data, pdb_dir), anomalous=False)
        out = rs.read_mtz(str(out_path))

        mask = cut_data.masks().cpu().numpy()
        assert len(out) == len(mask)
        assert (~mask).sum() > 1000, "the cut must exclude a real number of rows"
        for col in MAP_COLS:
            finite = np.isfinite(out[col].to_numpy("float32"))
            np.testing.assert_array_equal(finite, mask, err_msg=col)

    def test_observed_columns_complete(self, cut_data, pdb_dir, tmp_path):
        out_path = tmp_path / "out.mtz"
        cut_data.write_mtz(str(out_path), fcalc=_fcalc(cut_data, pdb_dir), anomalous=False)
        out = rs.read_mtz(str(out_path))
        assert np.isfinite(out["F-obs"].to_numpy("float32")).all()
        assert _d_spacing(out).min() < CUT - 0.5

    def test_map_resolution_is_the_cut(self, cut_data, pdb_dir, tmp_path):
        out_path = tmp_path / "out.mtz"
        cut_data.write_mtz(str(out_path), fcalc=_fcalc(cut_data, pdb_dir), anomalous=False)
        out = rs.read_mtz(str(out_path))
        finite = np.isfinite(out["FWT"].to_numpy("float32"))
        assert _d_spacing(out)[finite].min() >= CUT - 1e-3


class TestAnomalousLayout:
    def _write(self, data, pdb_dir, tmp_path):
        out_path = tmp_path / "anom_out.mtz"
        data.write_mtz(str(out_path), fcalc=_fcalc(data, pdb_dir), anomalous=True)
        return rs.read_mtz(str(out_path))

    def test_map_columns_stop_at_the_cut(self, cut_anomalous_data, pdb_dir, tmp_path):
        out = self._write(cut_anomalous_data, pdb_dir, tmp_path)
        d = _d_spacing(out)
        inside = d >= CUT - 1e-3
        assert (~inside).sum() > 1000
        for col in MAP_COLS + ["ANOM", "F-model(+)", "F-model(-)"]:
            vals = out[col].to_numpy("float32")
            assert not np.isfinite(vals[~inside]).any(), col
        for col in MAP_COLS:
            assert np.isfinite(out[col].to_numpy("float32")[inside]).mean() > 0.99, col
        assert np.isfinite(out["F-obs"].to_numpy("float32")).all()
        assert np.isfinite(out["F-obs(+)"].to_numpy("float32")[~inside]).any()

    def test_rejected_mate_does_not_enter_the_merged_map(
        self, cut_anomalous_data, pdb_dir, tmp_path
    ):
        data = cut_anomalous_data
        flag = data.friedel_flags.cpu()
        hkl = data.hkl.cpu()
        inside = (data.resolution.cpu() >= CUT).numpy()

        # Reject the (-) mate of one acentric pair inside the cut.
        groups = {}
        for i, (h, f) in enumerate(zip(map(tuple, hkl.tolist()), flag.tolist())):
            groups.setdefault(h, {})[f] = i
        key, pair = next(
            (h, g)
            for h, g in groups.items()
            if len(g) == 2 and inside[g[False]]
        )
        reject = torch.ones(len(hkl), dtype=torch.bool)
        reject[pair[True]] = False
        data.masks["test_reject"] = reject.to(data.device)

        out = self._write(data, pdb_dir, tmp_path)
        row = out.loc[key]
        F_plus = data.F[pair[False]].item()
        assert np.isnan(row["ANOM"])
        assert np.isnan(row["F-model(-)"])
        assert np.isfinite(row["F-model(+)"])
        np.testing.assert_allclose(
            row["FWT"], abs(2 * F_plus - row["F-model"]), rtol=1e-4
        )


class TestExcludedFlags:
    """Reflections the input's flags exclude stay excluded, not free."""

    @pytest.mark.parametrize("anomalous", [False, True])
    def test_excluded_rows_survive_a_round_trip(
        self, excluded_mtz, tmp_path, anomalous
    ):
        source = excluded_mtz
        if anomalous:
            source = tmp_path / "excluded_anomalous.mtz"
            rs.read_mtz(str(excluded_mtz)).stack_anomalous().write_mtz(str(source))
        data = _load(source)
        n_excluded = int((~data.masks["flagged_initial"]).sum())
        assert n_excluded >= 500

        out_path = tmp_path / "out.mtz"
        data.write_mtz(str(out_path), anomalous=anomalous)
        reloaded = _load(out_path)
        assert int((~reloaded.masks["flagged_initial"]).sum()) == n_excluded
        assert reloaded.free.n == data.free.n


def test_write_leaves_a_named_index_out(tmp_path):
    """``mtz.write`` writes H, K, L and the frame's columns, not its index."""
    df = pd.DataFrame(
        {"H": [1, 2], "K": [0, 1], "L": [4, 5], "F-obs": [9.0, 7.0]},
        index=pd.Index([10, 11], name="row"),
    )
    path = tmp_path / "out.mtz"
    mtz.write(df, [50, 60, 70, 90, 90, 90], "P 21 21 21", str(path))
    labels = [c.label for c in gemmi.read_mtz_file(str(path)).columns]
    assert labels == ["H", "K", "L", "F-obs"]
