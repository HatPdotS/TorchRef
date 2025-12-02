"""
Unit tests for torchref.model.parameter_wrappers

Tests MixedTensor and other parameter wrapper classes.
"""

import pytest
import torch
import torch.nn as nn


class TestMixedTensorInitialization:
    """Tests for MixedTensor initialization."""

    @pytest.mark.unit
    def test_mixed_tensor_empty_init(self):
        """Test MixedTensor can be initialized empty."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        mixed = MixedTensor()
        
        assert mixed.refinable_mask is None

    @pytest.mark.unit
    def test_mixed_tensor_full_init(self, random_coordinates):
        """Test MixedTensor with initial values."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        values = random_coordinates(n_atoms=10)
        mixed = MixedTensor(values)
        
        assert mixed.refinable_mask is not None
        # All should be refinable by default
        assert torch.all(mixed.refinable_mask)

    @pytest.mark.unit
    def test_mixed_tensor_with_mask(self, random_coordinates):
        """Test MixedTensor with custom refinable mask."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        values = random_coordinates(n_atoms=10)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True  # Only first 5 refinable
        
        mixed = MixedTensor(values, refinable_mask=mask)
        
        # Check mask is stored correctly
        assert mixed.refinable_mask.sum() == 5

    @pytest.mark.unit
    def test_mixed_tensor_is_nn_module(self):
        """MixedTensor should be a nn.Module."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        mixed = MixedTensor()
        
        assert isinstance(mixed, nn.Module)


class TestMixedTensorOperations:
    """Tests for MixedTensor operations."""

    @pytest.mark.unit
    def test_mixed_tensor_call_returns_full(self, random_coordinates):
        """Calling MixedTensor should return full tensor."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        values = random_coordinates(n_atoms=10)
        mixed = MixedTensor(values)
        
        result = mixed()
        
        assert result.shape == values.shape
        assert torch.allclose(result, values)

    @pytest.mark.unit
    def test_mixed_tensor_gradient_flow(self, random_coordinates):
        """Test gradient flow through MixedTensor."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        values = random_coordinates(n_atoms=10)
        mixed = MixedTensor(values, requires_grad=True)
        
        result = mixed()
        loss = result.sum()
        loss.backward()
        
        # Should have gradients on refinable params
        assert mixed.refinable_params.grad is not None

    @pytest.mark.unit
    def test_mixed_tensor_partial_refinement(self, random_coordinates):
        """Test gradient only flows to refinable elements."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        values = random_coordinates(n_atoms=10)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True  # Only first 5 refinable
        
        mixed = MixedTensor(values, refinable_mask=mask, requires_grad=True)
        
        result = mixed()
        loss = result.sum()
        loss.backward()
        
        # Gradient should only affect refinable params
        assert mixed.refinable_params.grad is not None
        # Number of refinable params should match mask
        assert mixed.refinable_params.numel() == 5 * 3  # 5 atoms * 3 coords


class TestMixedTensorDeviceHandling:
    """Tests for device handling in MixedTensor."""

    @pytest.mark.unit
    def test_mixed_tensor_cpu(self, random_coordinates):
        """Test MixedTensor on CPU."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        values = random_coordinates(n_atoms=10)
        mixed = MixedTensor(values, device=torch.device('cpu'))
        
        result = mixed()
        assert result.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_mixed_tensor_gpu(self, random_coordinates, gpu_device):
        """Test MixedTensor on GPU."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        values = random_coordinates(n_atoms=10)
        mixed = MixedTensor(values, device=gpu_device)
        
        result = mixed()
        assert result.device.type == 'cuda'


class TestOccupancyTensor:
    """Tests for OccupancyTensor (constrained occupancy handling)."""

    @pytest.mark.unit
    def test_occupancy_tensor_import(self):
        """Test OccupancyTensor can be imported."""
        from torchref.model.parameter_wrappers import OccupancyTensor
        
        assert OccupancyTensor is not None

    @pytest.mark.unit
    def test_occupancy_tensor_basic_init(self, random_occupancies):
        """Test basic OccupancyTensor initialization."""
        from torchref.model.parameter_wrappers import OccupancyTensor
        
        occ = random_occupancies(n_atoms=10)
        
        occ_tensor = OccupancyTensor(initial_values=occ)
        
        assert occ_tensor is not None


class TestPositiveMixedTensor:
    """Tests for PositiveMixedTensor (B-factors must be positive)."""

    @pytest.mark.unit
    def test_positive_mixed_tensor_import(self):
        """Test PositiveMixedTensor can be imported."""
        from torchref.model.parameter_wrappers import PositiveMixedTensor
        
        assert PositiveMixedTensor is not None

    @pytest.mark.unit
    def test_positive_mixed_tensor_stays_positive(self, random_b_factors):
        """B-factors should always be positive."""
        from torchref.model.parameter_wrappers import PositiveMixedTensor
        
        b_factors = random_b_factors(n_atoms=10)
        
        pos_tensor = PositiveMixedTensor(b_factors)
        result = pos_tensor()
        
        assert torch.all(result > 0)
