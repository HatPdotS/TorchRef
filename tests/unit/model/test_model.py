"""
Unit tests for torchref.model.model

Tests the Model class for atomic structure representation.
Note: Unit tests use mock data, not real file I/O.
"""

import pytest
import torch
import torch.nn as nn


class TestModelInitialization:
    """Tests for Model class initialization."""

    @pytest.mark.unit
    def test_model_empty_initialization(self):
        """Test Model can be initialized without files."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert model.initialized == False
        assert model.pdb is None
        assert model.xyz is None
        assert model.b is None

    @pytest.mark.unit
    def test_model_is_nn_module(self):
        """Model should be a nn.Module."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert isinstance(model, nn.Module)

    @pytest.mark.unit
    def test_model_default_dtype(self):
        """Test default dtype is float32."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert model.dtype_float == torch.float32

    @pytest.mark.unit
    def test_model_custom_dtype(self):
        """Test custom dtype specification."""
        from torchref.model.model import Model
        
        model = Model(dtype_float=torch.float64)
        
        assert model.dtype_float == torch.float64

    @pytest.mark.unit
    def test_model_strip_h_default(self):
        """Test strip_H defaults to True."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert model.strip_H == True

    @pytest.mark.unit
    def test_model_bool_uninitialized(self):
        """Uninitialized model should be falsy."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert bool(model) == False


class TestModelDeviceHandling:
    """Tests for device handling in Model."""

    @pytest.mark.unit
    def test_model_default_device(self):
        """Test default device is CPU."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert model.device == torch.device('cpu')

    @pytest.mark.unit
    def test_model_custom_device(self):
        """Test custom device specification."""
        from torchref.model.model import Model
        
        model = Model(device=torch.device('cpu'))
        
        assert model.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_model_gpu_device(self, gpu_device):
        """Test GPU device specification."""
        from torchref.model.model import Model
        
        model = Model(device=gpu_device)
        
        assert model.device.type == 'cuda'


class TestModelGetSelectionMask:
    """Tests for Model.get_selection_mask() method."""

    @pytest.mark.unit
    def test_get_selection_mask_uninitialized_raises(self):
        """Test that get_selection_mask() raises RuntimeError on uninitialized model."""
        from torchref.model.model import Model
        
        model = Model()
        
        with pytest.raises(RuntimeError, match="uninitialized"):
            model.get_selection_mask("chain A")
