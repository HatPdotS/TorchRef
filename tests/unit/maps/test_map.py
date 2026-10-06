"""Tests for torchref.maps.Map and torchref.maps.DifferenceMap."""

import os
import tempfile

import pytest
import torch

from torchref.io import ReflectionData
from torchref.maps import DifferenceMap, Map
from torchref.model.model_ft import ModelFT
from torchref.scaling import Scaler


@pytest.fixture(scope="module")
def model_ft_and_data(sample_structure_pair):
    """Load a ModelFT and ReflectionData from test files."""
    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))

    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    return model, data, sample_structure_pair


@pytest.fixture(scope="module")
def scaler(model_ft_and_data):
    """One fitted Scaler for the shared pair, so each map need not fit its own."""
    model, data, _ = model_ft_and_data
    fitted = Scaler(model, data, verbose=0)
    fitted.initialize()
    fitted.refine_lbfgs(verbose=False)
    return fitted


class TestMap:
    """Tests for the base Map class."""

    def test_init_default(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m = Map(data, model)
        assert m.map_type == "2Fo-Fc"
        assert m.gridsize is None
        assert m.map_data is None

    def test_init_fcalc(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc")
        assert m.map_type == "Fcalc"

    def test_init_invalid_map_type(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        with pytest.raises(ValueError, match="map_type must be one of"):
            Map(data, model, map_type="invalid")

    def test_calculate_2fo_fc(self, model_ft_and_data, scaler):
        model, data, _ = model_ft_and_data
        m = Map(data, model, scaler=scaler)
        result = m.calculate()

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.is_floating_point()
        assert m.map_data is not None
        assert torch.equal(result, m.map_data)

    def test_calculate_fcalc(self, model_ft_and_data, scaler):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc", scaler=scaler)
        result = m.calculate()

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.is_floating_point()

    def test_different_map_types_give_different_results(
        self, model_ft_and_data, scaler
    ):
        model, data, _ = model_ft_and_data
        m1 = Map(data, model, map_type="2Fo-Fc", scaler=scaler)
        m2 = Map(data, model, map_type="Fcalc", scaler=scaler)
        r1 = m1.calculate()
        r2 = m2.calculate()

        assert not torch.allclose(r1, r2)

    def test_explicit_gridsize(self, model_ft_and_data, scaler):
        model, data, _ = model_ft_and_data
        gridsize = (32, 36, 40)
        m = Map(data, model, gridsize=gridsize, map_type="Fcalc", scaler=scaler)
        result = m.calculate()

        assert result.shape == gridsize

    def test_auto_gridsize(self, model_ft_and_data, scaler):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc", scaler=scaler)
        result = m.calculate()

        for dim in result.shape:
            assert dim > 0

    def test_write_ccp4(self, model_ft_and_data, scaler):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc", scaler=scaler)

        with tempfile.NamedTemporaryFile(suffix=".ccp4", delete=False) as f:
            filepath = f.name

        try:
            ret = m.write(filepath)
            assert ret == 1
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)

    def test_write_auto_calculates(self, model_ft_and_data, scaler):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc", scaler=scaler)
        assert m.map_data is None

        with tempfile.NamedTemporaryFile(suffix=".ccp4", delete=False) as f:
            filepath = f.name

        try:
            m.write(filepath)
            assert m.map_data is not None
        finally:
            os.unlink(filepath)


class TestMapScale:
    """F_calc enters every map on the observed amplitudes' scale."""

    def test_2fo_fc_correlates_with_the_model_map_on_raw_data(self, model_ft_and_data):
        """Without a scaler, Map fits one: raw 1DAW amplitudes sit far below the
        model's absolute scale, and an unscaled 2Fo-Fc map is an inverted Fcalc
        map."""
        model, data, _ = model_ft_and_data
        two_fo_fc = Map(data, model).calculate()
        fcalc = Map(data, model, map_type="Fcalc").calculate()
        cc = torch.corrcoef(torch.stack([two_fo_fc.flatten(), fcalc.flatten()]))
        assert cc[0, 1] > 0.5


class TestDifferenceMap:
    """Tests for the DifferenceMap class."""

    def test_difference_map_computes(self, model_ft_and_data):
        model, _, pair = model_ft_and_data

        # Load two copies of the same dataset
        data_ref = ReflectionData()
        data_ref.load_mtz(str(pair["reflections"]))
        data_pert = ReflectionData()
        data_pert.load_mtz(str(pair["reflections"]))

        dm = DifferenceMap(data_pert, data_ref, model)
        result = dm.calculate()

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.is_floating_point()

    def test_difference_map_write(self, model_ft_and_data):
        model, _, pair = model_ft_and_data

        data_ref = ReflectionData()
        data_ref.load_mtz(str(pair["reflections"]))
        data_pert = ReflectionData()
        data_pert.load_mtz(str(pair["reflections"]))

        dm = DifferenceMap(data_pert, data_ref, model)

        with tempfile.NamedTemporaryFile(suffix=".ccp4", delete=False) as f:
            filepath = f.name

        try:
            ret = dm.write(filepath)
            assert ret == 1
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)


def _merged_and_anomalous(data):
    """The acentric reflections of ``data``, once merged and once as Bijvoet
    pairs F*(1 +/- eps) whose mean is the merged F. Both keep every row, so the
    outlier masks recomputed on construction cannot make them differ."""
    keep = data.masks() & ~data.centric
    hkl, F, sigF = data.hkl[keep], data.F[keep], data.F_sigma[keep]
    common = dict(cell=data.cell, spacegroup=data.spacegroup, verbose=0)
    merged = ReflectionData.from_tensors(hkl, F, sigF, **common)
    anom = ReflectionData.from_tensors(
        torch.cat([hkl, -hkl]),
        torch.cat([F * 1.2, F * 0.8]),
        torch.cat([sigF, sigF]),
        friedel_merged=False,
        **common,
    )
    for d in (merged, anom):
        d.masks.clear()
        d.masks["all"] = torch.ones(len(d.hkl), dtype=torch.bool)
    return merged, anom


class TestAnomalousInput:
    """Bijvoet pairs enter a map once, at their mean amplitude."""

    def test_bijvoet_helpers(self, model_ft_and_data):
        _, data, _ = model_ft_and_data
        merged, anom = _merged_and_anomalous(data)
        assert anom.friedel_merged is False

        rows = anom.bijvoet_representatives()
        assert len(rows) == len(merged.hkl)
        mean = anom.bijvoet_mean(anom.F)
        by_hkl = dict(zip(map(tuple, merged.hkl.tolist()), merged.F.tolist()))
        for h, f in zip(anom.hkl[rows].tolist(), mean[rows].tolist()):
            assert f == pytest.approx(by_hkl[tuple(h)], rel=1e-5)

        # Merged data pass through untouched.
        assert torch.equal(merged.bijvoet_mean(merged.F), merged.F)
        assert torch.equal(
            merged.bijvoet_representatives(),
            torch.arange(len(merged.hkl), device=merged.device),
        )

    def test_map_from_anomalous_data_equals_merged(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        merged, anom = _merged_and_anomalous(data)
        grid = Map(merged, model, map_type="2Fo-Fc")._determine_gridsize()

        # Unfitted scalers are the identity: a scale fitted to each row set would
        # differ slightly, and the comparison is about the Bijvoet averaging.
        expected = Map(
            merged,
            model,
            gridsize=grid,
            map_type="2Fo-Fc",
            scaler=Scaler(model, merged, verbose=0),
        ).calculate()
        result = Map(
            anom,
            model,
            gridsize=grid,
            map_type="2Fo-Fc",
            scaler=Scaler(model, anom, verbose=0),
        ).calculate()

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_difference_map_from_anomalous_data_equals_merged(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        merged, anom = _merged_and_anomalous(data)
        merged_pert, anom_pert = _merged_and_anomalous(data)
        merged_pert.F = merged_pert.F * 1.1
        anom_pert.F = anom_pert.F * 1.1

        expected = DifferenceMap(merged_pert, merged, model).calculate()
        result = DifferenceMap(anom_pert, anom, model).calculate()

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
