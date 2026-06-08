"""
Tests for the top-level object-creation readers (read_mtz / read_cif / read_pdb).
"""
import pytest


@pytest.mark.unit
class TestReadMtz:
    def test_read_mtz_returns_reflectiondata(self, mtz_dir):
        from torchref import read_mtz
        from torchref.io import ReflectionData

        data = read_mtz(str(mtz_dir / "1DAW.mtz"), verbose=0)
        assert isinstance(data, ReflectionData)
        assert data.hkl.shape[0] > 0


@pytest.mark.unit
class TestReadPdb:
    def test_read_pdb_defaults_to_modelft(self, pdb_dir):
        from torchref import read_pdb
        from torchref.model import ModelFT

        model = read_pdb(str(pdb_dir / "1DAW.pdb"))
        assert isinstance(model, ModelFT)
        assert len(model.pdb) > 0

    def test_read_pdb_model_class_override(self, pdb_dir):
        from torchref import read_pdb
        from torchref.model import Model, ModelFT

        model = read_pdb(str(pdb_dir / "1DAW.pdb"), model_class=Model)
        assert isinstance(model, Model)
        assert not isinstance(model, ModelFT)


@pytest.mark.unit
class TestReadCif:
    def test_read_cif_model(self, cif_dir):
        from torchref import read_cif
        from torchref.model import ModelFT

        obj = read_cif(str(cif_dir / "1DAW.cif"), verbose=0)
        assert isinstance(obj, ModelFT)
        assert len(obj.pdb) > 0

    def test_read_cif_reflections(self, cif_sf_dir):
        from torchref import read_cif
        from torchref.io import ReflectionData

        obj = read_cif(str(cif_sf_dir / "1DAW-sf.cif"), verbose=0)
        assert isinstance(obj, ReflectionData)
        assert obj.hkl.shape[0] > 0
