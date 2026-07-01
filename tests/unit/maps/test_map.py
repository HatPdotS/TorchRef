"""Tests for torchref.maps.Map and torchref.maps.DifferenceMap."""

import os
import tempfile

import pytest
import torch

from torchref.io import ReflectionData
from torchref.maps import DifferenceMap, Map
from torchref.model.model_ft import ModelFT


@pytest.fixture(scope="module")
def model_ft_and_data(sample_structure_pair):
    """Load a ModelFT and ReflectionData from test files."""
    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))

    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    return model, data, sample_structure_pair


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

    def test_calculate_2fo_fc(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m = Map(data, model)
        result = m.calculate()

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.is_floating_point()
        assert m.map_data is not None
        assert torch.equal(result, m.map_data)

    def test_calculate_fcalc(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc")
        result = m.calculate()

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.is_floating_point()

    def test_different_map_types_give_different_results(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m1 = Map(data, model, map_type="2Fo-Fc")
        m2 = Map(data, model, map_type="Fcalc")
        r1 = m1.calculate()
        r2 = m2.calculate()

        assert not torch.allclose(r1, r2)

    def test_explicit_gridsize(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        gridsize = (32, 36, 40)
        m = Map(data, model, gridsize=gridsize, map_type="Fcalc")
        result = m.calculate()

        assert result.shape == gridsize

    def test_auto_gridsize(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc")
        result = m.calculate()

        for dim in result.shape:
            assert dim > 0

    def test_write_ccp4(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc")

        with tempfile.NamedTemporaryFile(suffix=".ccp4", delete=False) as f:
            filepath = f.name

        try:
            ret = m.write(filepath)
            assert ret == 1
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)

    def test_write_auto_calculates(self, model_ft_and_data):
        model, data, _ = model_ft_and_data
        m = Map(data, model, map_type="Fcalc")
        assert m.map_data is None

        with tempfile.NamedTemporaryFile(suffix=".ccp4", delete=False) as f:
            filepath = f.name

        try:
            m.write(filepath)
            assert m.map_data is not None
        finally:
            os.unlink(filepath)


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
