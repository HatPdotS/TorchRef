"""
Unit tests for torchref.refinement.loss_weighting

Tests loss weighting schemes for crystallographic refinement.
"""

import pytest
import torch
import torch.nn as nn


class TestFixedWeighting:
    """Tests for FixedWeighting class."""

    @pytest.mark.unit
    def test_fixed_weighting_initialization(self):
        """Test FixedWeighting can be initialized with default weights."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        
        assert weighting.target_weights is not None
        assert 'xray' in weighting.target_weights
        assert 'restraints' in weighting.target_weights
        assert 'adp' in weighting.target_weights

    @pytest.mark.unit
    def test_fixed_weighting_custom_weights(self):
        """Test FixedWeighting with custom weights."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        custom = {'xray': 2.0, 'restraints': 0.5}
        weighting = FixedWeighting(target_weights=custom)
        
        assert torch.isclose(weighting.target_weights['xray'], torch.tensor(2.0))
        assert torch.isclose(weighting.target_weights['restraints'], torch.tensor(0.5))

    @pytest.mark.unit
    def test_fixed_weighting_forward_all(self):
        """Test forward pass returns weights for 'all' phase."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3})
        
        # Mock refinement object (not used in FixedWeighting)
        mock_ref = None
        
        weights = weighting.forward(mock_ref, phase='all')
        
        assert 'xray' in weights
        assert 'restraints' in weights
        assert 'adp' in weights
        assert weights['xray'] > 0
        assert weights['restraints'] > 0
        assert weights['adp'] > 0

    @pytest.mark.unit
    def test_fixed_weighting_xyz_phase(self):
        """Test that xyz phase disables ADP loss."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3})
        
        weights = weighting.forward(None, phase='xyz')
        
        # ADP should be zero during xyz phase
        assert torch.isclose(weights['adp'], torch.tensor(0.0))
        # Other weights should be non-zero
        assert weights['xray'] > 0
        assert weights['restraints'] > 0

    @pytest.mark.unit
    def test_fixed_weighting_b_phase(self):
        """Test that b phase disables geometry restraints."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3})
        
        weights = weighting.forward(None, phase='b')
        
        # Restraints should be zero during b phase
        assert torch.isclose(weights['restraints'], torch.tensor(0.0))
        # Other weights should be non-zero
        assert weights['xray'] > 0
        assert weights['adp'] > 0


class TestResolutionDependentWeighting:
    """Tests for ResolutionDependentWeighting class."""

    @pytest.mark.unit
    def test_resolution_weighting_initialization(self):
        """Test ResolutionDependentWeighting initialization."""
        from torchref.refinement.loss_weighting import ResolutionDependentWeighting
        
        weighting = ResolutionDependentWeighting()
        
        assert weighting.resolution_bins is not None
        assert len(weighting.resolution_bins) > 0

    @pytest.mark.unit
    def test_resolution_scale_high_res(self):
        """High resolution should have lower restraint weight."""
        from torchref.refinement.loss_weighting import ResolutionDependentWeighting
        
        weighting = ResolutionDependentWeighting()
        
        xray_scale, restraint_scale = weighting.get_resolution_scale(1.2)
        
        # At high resolution, restraints should be lower
        assert restraint_scale < 1.0

    @pytest.mark.unit
    def test_resolution_scale_low_res(self):
        """Low resolution should have higher restraint weight."""
        from torchref.refinement.loss_weighting import ResolutionDependentWeighting
        
        weighting = ResolutionDependentWeighting()
        
        xray_scale, restraint_scale = weighting.get_resolution_scale(3.5)
        
        # At low resolution, restraints should be higher
        assert restraint_scale >= 1.0


class TestLossWeightingModule:
    """Tests for base LossWeightingModule class."""

    @pytest.mark.unit
    def test_target_weights_device_movement(self):
        """Test that weights move with module to different devices."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3})
        
        # Check initial device (CPU)
        for key, weight in weighting.target_weights.items():
            assert weight.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_target_weights_gpu_movement(self, gpu_device):
        """Test weight movement to GPU."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 1.0, 'restraints': 1.0, 'adp': 0.3})
        weighting = weighting.to(gpu_device)
        
        for key, weight in weighting.target_weights.items():
            assert weight.device.type == 'cuda'
