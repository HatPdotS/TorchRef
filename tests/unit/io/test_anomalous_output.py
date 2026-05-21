"""Tests for anomalous (Bijvoet) refinement support.

Covers the Option B1 changes:
  * ReflectionData carries a signed HKL (hkl_anomalous / hkl_for_sf) alongside
    the canonical ASU index, with friedel_flags bookkeeping.
  * ModelFT produces distinct |F_calc| for Friedel mates when a wavelength and
    anomalous scatterers are present, and identical |F_calc| otherwise.
  * write_mtz(anomalous=True) emits a phenix-style MTZ on the canonical ASU
    (no duplicate indices) with unstacked (+)/(-) columns.
"""

import collections

import numpy as np
import pytest
import torch

import reciprocalspaceship as rs

from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model.model_ft import ModelFT
from torchref.base.french_wilson import is_centric_from_hkl


@pytest.fixture
def stacked_anomalous_mtz(mtz_dir, tmp_path):
    """1DAW reflections stacked to the full (+/-) list, written to a temp MTZ."""
    ds = rs.read_mtz(str(mtz_dir / "1DAW.mtz")).stack_anomalous()
    out = tmp_path / "anom_in.mtz"
    ds.write_mtz(str(out))
    return str(out)


@pytest.fixture
def anomalous_data(stacked_anomalous_mtz):
    data = ReflectionData(verbose=0)
    data.load_mtz(stacked_anomalous_mtz)
    return data


def _group_by_canonical(hkl, flag):
    """Return list of (plus_row, minus_row) for acentric Bijvoet pairs."""
    groups = collections.defaultdict(list)
    for i, (h, f) in enumerate(zip(hkl.tolist(), flag.tolist())):
        groups[tuple(h)].append((i, f))
    pairs = []
    for members in groups.values():
        if len(members) == 2 and members[0][1] != members[1][1]:
            (i0, f0), (i1, f1) = members
            plus, minus = (i0, i1) if not f0 else (i1, i0)
            pairs.append((plus, minus))
    return pairs


class TestDualHklRepresentation:
    def test_fields_populated(self, anomalous_data):
        d = anomalous_data
        assert d.friedel_flags is not None
        assert d.hkl_anomalous is not None
        assert d.hkl_anomalous.shape == d.hkl.shape
        assert d.friedel_flags.dtype == torch.bool

    def test_hkl_for_sf_signs(self, anomalous_data):
        d = anomalous_data
        sf = d.hkl_for_sf()
        flag = d.friedel_flags
        assert torch.equal(sf[~flag], d.hkl[~flag])
        assert torch.equal(sf[flag], -d.hkl[flag])

    def test_centrics_not_flagged(self, anomalous_data):
        d = anomalous_data
        centric = is_centric_from_hkl(d.hkl, d.spacegroup)
        assert not bool((d.friedel_flags & centric).any())

    def test_hkl_for_sf_fallback(self):
        """Without canonicalization bookkeeping, falls back to canonical hkl."""
        d = ReflectionData(verbose=0)
        d.hkl = torch.tensor([[1, 0, 0]], dtype=torch.int32)
        assert torch.equal(d.hkl_for_sf(), d.hkl)


class TestBijvoetStructureFactors:
    def _fcalc(self, data, pdb_dir, wavelength):
        model = ModelFT(verbose=0, max_res=2.0, wavelength=wavelength)
        model.load_pdb(str(pdb_dir / "1DAW.pdb"))
        with torch.no_grad():
            return model(data.hkl_for_sf())

    def test_anomalous_signal_present(self, anomalous_data, pdb_dir):
        fc = torch.abs(self._fcalc(anomalous_data, pdb_dir, wavelength=1.54))
        pairs = _group_by_canonical(anomalous_data.hkl, anomalous_data.friedel_flags)
        assert len(pairs) > 0
        diffs = np.array([abs(fc[p].item() - fc[m].item()) for p, m in pairs])
        # With a wavelength + anomalous scatterers, mates differ.
        assert (diffs > 1e-3).mean() > 0.5

    def test_no_signal_without_wavelength(self, anomalous_data, pdb_dir):
        fc = torch.abs(self._fcalc(anomalous_data, pdb_dir, wavelength=None))
        pairs = _group_by_canonical(anomalous_data.hkl, anomalous_data.friedel_flags)
        diffs = np.array([abs(fc[p].item() - fc[m].item()) for p, m in pairs])
        # Hermitian grid -> equal amplitudes (only float interpolation noise).
        assert diffs.max() < 1e-1


class TestAnomalousMtzOutput:
    def _write(self, data, pdb_dir, tmp_path, wavelength=1.54):
        model = ModelFT(verbose=0, max_res=2.0, wavelength=wavelength)
        model.load_pdb(str(pdb_dir / "1DAW.pdb"))
        with torch.no_grad():
            fcalc = model(data.hkl_for_sf())
        out = tmp_path / "anom_out.mtz"
        data.write_mtz(str(out), fcalc=fcalc, anomalous=True)
        return rs.read_mtz(str(out))

    def test_no_duplicate_asu_index(self, anomalous_data, pdb_dir, tmp_path):
        out = self._write(anomalous_data, pdb_dir, tmp_path)
        assert not out.index.duplicated().any()

    def test_phenix_columns_present(self, anomalous_data, pdb_dir, tmp_path):
        out = self._write(anomalous_data, pdb_dir, tmp_path)
        for col in [
            "F-obs(+)", "F-obs(-)", "F-model(+)", "F-model(-)",
            "PHIF-model(+)", "PHIF-model(-)", "FWT", "PHWT", "DELFWT", "PHDELWT",
        ]:
            assert col in out.columns, f"missing {col}"
        # Friedel dtype inferred for the (+/-) amplitude columns.
        assert out.dtypes["F-obs(+)"].mtztype == "G"

    def test_display_map_is_fft_safe(self, anomalous_data, pdb_dir, tmp_path):
        out = self._write(anomalous_data, pdb_dir, tmp_path)
        for col in ["FWT", "PHWT", "DELFWT", "PHDELWT", "F-model"]:
            assert np.isfinite(out[col].to_numpy("float32")).all()
