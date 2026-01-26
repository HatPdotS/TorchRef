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


class TestMixedTensorSet:
    """Tests for MixedTensor.set() method."""

    @pytest.mark.unit
    def test_set_1d_tensor_basic(self):
        """Test set() on 1D tensor with basic mask."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        t = MixedTensor(torch.arange(10, dtype=torch.float32), name='test')
        mask = torch.tensor([False, False, True, True, True, False, False, False, False, False])
        new_values = torch.tensor([100.0, 200.0, 300.0])
        
        t.set(new_values, mask)
        result = t()
        
        assert result[2] == 100.0
        assert result[3] == 200.0
        assert result[4] == 300.0
        # Unchanged values
        assert result[0] == 0.0
        assert result[1] == 1.0

    @pytest.mark.unit
    def test_set_2d_tensor_xyz(self, random_coordinates):
        """Test set() on 2D tensor (xyz-like coordinates)."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        xyz = MixedTensor(torch.randn(5, 3), name='xyz')
        mask = torch.tensor([True, False, True, False, False])
        new_coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        xyz.set(new_coords, mask)
        result = xyz()
        
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(result[2], torch.tensor([4.0, 5.0, 6.0]))

    @pytest.mark.unit
    def test_set_updates_refinable_params(self):
        """Test that set() correctly updates refinable_params."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        refinable_mask = torch.tensor([True, True, False, False, False])
        t = MixedTensor(torch.arange(5, dtype=torch.float32), refinable_mask=refinable_mask, name='test')
        
        # Original refinable params should be [0, 1]
        assert t.refinable_params[0].item() == 0.0
        assert t.refinable_params[1].item() == 1.0
        
        # Update the first (refinable) element
        mask = torch.tensor([True, False, False, False, False])
        t.set(torch.tensor([99.0]), mask)
        
        # After set(), refinable_params should reflect the update
        assert t.refinable_params[0].item() == 99.0
        assert t.refinable_params[1].item() == 1.0

    @pytest.mark.unit
    def test_set_wrong_mask_shape_raises(self):
        """Test that set() raises ValueError for wrong mask shape."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        t = MixedTensor(torch.arange(10, dtype=torch.float32), name='test')
        
        with pytest.raises(ValueError, match="Mask shape"):
            wrong_mask = torch.tensor([True, False, True])  # Wrong size
            t.set(torch.tensor([1.0, 2.0]), wrong_mask)

    @pytest.mark.unit
    def test_set_wrong_values_shape_raises(self):
        """Test that set() raises ValueError for wrong values shape."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        t = MixedTensor(torch.arange(10, dtype=torch.float32), name='test')
        mask = torch.tensor([True, True, False, False, False, False, False, False, False, False])
        
        with pytest.raises(ValueError, match="Values shape"):
            wrong_values = torch.tensor([1.0, 2.0, 3.0])  # 3 values for 2 selected
            t.set(wrong_values, mask)

    @pytest.mark.unit
    def test_set_2d_mask_raises(self):
        """Test that set() raises ValueError for 2D mask."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        t = MixedTensor(torch.arange(10, dtype=torch.float32), name='test')
        
        with pytest.raises(ValueError, match="Mask"):
            wrong_mask = torch.tensor([[True, False]])  # 2D mask
            t.set(torch.tensor([1.0]), wrong_mask)

    @pytest.mark.unit
    def test_set_preserves_requires_grad(self):
        """Test that set() preserves the requires_grad attribute."""
        from torchref.model.parameter_wrappers import MixedTensor
        
        t = MixedTensor(torch.arange(5, dtype=torch.float32), requires_grad=True, name='test')
        assert t.refinable_params.requires_grad is True
        
        mask = torch.tensor([True, False, False, False, False])
        t.set(torch.tensor([99.0]), mask)
        
        # requires_grad should still be True after set()
        assert t.refinable_params.requires_grad is True


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
    def test_positive_mixed_tensor_stays_positive(self, random_adp):
        """ADPs should always be positive."""
        from torchref.model.parameter_wrappers import PositiveMixedTensor

        adp_values = random_adp(n_atoms=10)

        pos_tensor = PositiveMixedTensor(adp_values)
        result = pos_tensor()
        
        assert torch.all(result > 0)

    @pytest.mark.unit
    def test_positive_mixed_tensor_set_basic(self):
        """Test set() on PositiveMixedTensor."""
        from torchref.model.parameter_wrappers import PositiveMixedTensor
        
        # Initial B-factors
        initial_b = torch.tensor([20.0, 25.0, 30.0, 35.0, 40.0])
        t = PositiveMixedTensor(initial_b, name='b_factors')
        
        # Set first two B-factors to 50.0
        mask = torch.tensor([True, True, False, False, False])
        new_values = torch.tensor([50.0, 50.0])
        t.set(new_values, mask)
        
        result = t()
        # Check that values are close (not exact due to log-space transformation)
        assert torch.allclose(result[:2], torch.tensor([50.0, 50.0]), rtol=0.01)
        # Other values should be unchanged (approximately)
        assert result[2] > 25.0 and result[2] < 35.0

    @pytest.mark.unit
    def test_positive_mixed_tensor_set_non_positive_raises(self):
        """Test that set() raises ValueError for non-positive values."""
        from torchref.model.parameter_wrappers import PositiveMixedTensor
        
        t = PositiveMixedTensor(torch.tensor([20.0, 25.0, 30.0]), name='b_factors')
        
        with pytest.raises(ValueError, match="positive"):
            mask = torch.tensor([True, False, False])
            t.set(torch.tensor([0.0]), mask)  # Zero is not positive

    @pytest.mark.unit
    def test_positive_mixed_tensor_set_updates_refinable_params(self):
        """Test that set() correctly updates refinable_params in PositiveMixedTensor."""
        from torchref.model.parameter_wrappers import PositiveMixedTensor
        
        refinable_mask = torch.tensor([True, True, False, False, False])
        t = PositiveMixedTensor(
            torch.tensor([20.0, 25.0, 30.0, 35.0, 40.0]),
            refinable_mask=refinable_mask, 
            name='b_factors'
        )
        
        # Update the first (refinable) element
        mask = torch.tensor([True, False, False, False, False])
        t.set(torch.tensor([100.0]), mask)
        
        # After set(), output should be close to 100.0
        result = t()
        assert torch.allclose(result[0], torch.tensor(100.0), rtol=0.01)
