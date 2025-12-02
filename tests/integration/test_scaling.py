"""
Integration tests for the scaling module.

Tests scaler functionality with real data.
"""

import pytest
import torch
from pathlib import Path


class TestScalerInitialization:
    """Tests for Scaler initialization patterns."""

    @pytest.mark.integration
    def test_empty_scaler_creation(self):
        """Test creating an empty scaler for state_dict loading."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler()
        
        assert scaler._model is None
        assert scaler._data is None
        assert scaler.nbins == 20

    @pytest.mark.integration
    def test_scaler_with_model_and_data(self, sample_structure_pair):
        """Test creating a scaler with model and data."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        assert scaler._model is not None
        assert scaler._data is not None
        assert scaler.nbins == 10
        assert scaler.s is not None
        assert scaler.bins is not None


class TestScalerOperations:
    """Tests for Scaler operations."""

    @pytest.mark.integration
    def test_scaler_freeze_unfreeze(self, sample_structure_pair):
        """Test freezing and unfreezing the scaler."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        assert not scaler.frozen
        
        scaler.freeze()
        assert scaler.frozen
        
        scaler.unfreeze()
        assert not scaler.frozen

    @pytest.mark.integration
    def test_scaler_set_model_and_data(self, sample_structure_pair):
        """Test setting model and data after empty init."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        # Create empty scaler
        scaler = Scaler()
        
        # Load model and data
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Set them on the scaler
        scaler.set_model_and_data(model, data)
        
        assert scaler._model is not None
        assert scaler._data is not None


class TestScalerAnisotropy:
    """Tests for anisotropy correction."""

    @pytest.mark.integration
    def test_setup_anisotropy_correction(self, sample_structure_pair):
        """Test setting up anisotropy correction parameters."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        
        assert hasattr(scaler, 'U')
        assert scaler.U.shape == (6,)
        
    @pytest.mark.integration
    def test_anisotropy_correction_computation(self, sample_structure_pair):
        """Test computing anisotropy correction."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        
        correction = scaler.anisotropy_correction()
        
        assert correction is not None
        assert correction.shape[0] == data.hkl.shape[0]
        assert torch.all(torch.isfinite(correction))


class TestScalerBuffers:
    """Tests for scaler buffer operations."""

    @pytest.mark.integration
    def test_scaler_has_scattering_vectors(self, sample_structure_pair):
        """Test that scaler computes scattering vectors."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        # s should be the scattering vectors
        assert scaler.s is not None
        assert scaler.s.shape[0] == data.hkl.shape[0]
        assert scaler.s.shape[1] == 3

    @pytest.mark.integration
    def test_scaler_has_bins(self, sample_structure_pair):
        """Test that scaler has resolution bins."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        assert scaler.bins is not None
        assert scaler.bins.shape[0] == data.hkl.shape[0]
        # Bins should be integers from 0 to nbins-1
        assert scaler.bins.min() >= 0
        assert scaler.bins.max() < 10


class TestScalerDeviceHandling:
    """Tests for scaler device handling."""

    @pytest.mark.integration
    def test_scaler_default_device(self, sample_structure_pair):
        """Test that scaler defaults to CPU."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        assert scaler.device.type == 'cpu'
        assert scaler.s.device.type == 'cpu'
        assert scaler.bins.device.type == 'cpu'
