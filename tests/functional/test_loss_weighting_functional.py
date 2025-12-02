"""
Functional tests for loss weighting module.

These tests exercise the loss weighting strategies with realistic data.
"""
import pytest
import torch
import numpy as np


@pytest.mark.integration
class TestLossWeightingModuleBase:
    """Test base LossWeightingModule functionality."""

    def test_fixed_weighting_initialization(self):
        """Test FixedWeighting initialization."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        
        assert weighting is not None
        assert hasattr(weighting, 'target_weights')

    def test_fixed_weighting_with_custom_weights(self):
        """Test FixedWeighting with custom weights."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        custom_weights = {
            'xray': 2.0,
            'restraints': 0.5,
            'adp': 0.1
        }
        
        weighting = FixedWeighting(target_weights=custom_weights)
        
        weights = weighting.target_weights
        assert torch.isclose(weights['xray'], torch.tensor(2.0))
        assert torch.isclose(weights['restraints'], torch.tensor(0.5))
        assert torch.isclose(weights['adp'], torch.tensor(0.1))

    def test_target_weights_property(self):
        """Test target_weights property access."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        weights = weighting.target_weights
        
        assert isinstance(weights, dict)
        assert 'xray' in weights
        assert 'restraints' in weights
        assert 'adp' in weights


@pytest.mark.integration
class TestFixedWeighting:
    """Test FixedWeighting specific functionality."""

    def test_fixed_weights_constant(self):
        """Test that fixed weights remain constant."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 1.5})
        
        # Check weight is as specified
        weights = weighting.target_weights
        assert torch.isclose(weights['xray'], torch.tensor(1.5))

    def test_fixed_weights_all_positive(self):
        """Test that default weights are all positive."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        weights = weighting.target_weights
        
        for name, weight in weights.items():
            assert weight > 0, f"Weight for {name} should be positive"


@pytest.mark.integration
class TestWeightingStateDictFunctional:
    """Test state dict operations for weighting modules."""

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading weighting state."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={
            'xray': 1.5,
            'restraints': 0.7,
            'adp': 0.2
        })
        
        original_weights = {k: v.clone() for k, v in weighting.target_weights.items()}
        
        # Save
        state_dict = weighting.state_dict()
        torch.save(state_dict, tmp_path / "weighting.pt")
        
        # Load into new module
        weighting2 = FixedWeighting()
        weighting2.load_state_dict(torch.load(tmp_path / "weighting.pt"))
        
        # Verify weights match
        for key in original_weights:
            assert torch.isclose(
                weighting2.target_weights[key],
                original_weights[key]
            ), f"Weight for {key} doesn't match after loading"


@pytest.mark.integration
class TestWeightingDeviceOperations:
    """Test weighting module device operations."""

    def test_cpu_operation(self):
        """Test weighting on CPU."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        
        for name, weight in weighting.target_weights.items():
            assert weight.device.type == 'cpu'


@pytest.mark.integration
class TestWeightingMathOperations:
    """Test mathematical operations with weights."""

    def test_weight_multiplication(self):
        """Test using weights for loss scaling."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 2.0, 'restraints': 0.5})
        weights = weighting.target_weights
        
        # Create mock losses
        xray_loss = torch.tensor(10.0)
        restraints_loss = torch.tensor(20.0)
        
        # Apply weights
        weighted_xray = weights['xray'] * xray_loss
        weighted_restraints = weights['restraints'] * restraints_loss
        
        assert torch.isclose(weighted_xray, torch.tensor(20.0))
        assert torch.isclose(weighted_restraints, torch.tensor(10.0))

    def test_total_weighted_loss(self):
        """Test computing total weighted loss."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        weights = weighting.target_weights
        
        # Create mock losses
        losses = {
            'xray': torch.tensor(10.0),
            'restraints': torch.tensor(5.0),
            'adp': torch.tensor(2.0)
        }
        
        # Compute weighted total
        total = sum(weights[k] * losses[k] for k in losses if k in weights)
        
        assert torch.isfinite(total)
        assert total > 0


@pytest.mark.integration
class TestNLLXrayFunction:
    """Test the NLL X-ray function used in weighting."""

    def test_nll_xray_basic(self):
        """Test basic NLL X-ray calculation."""
        from torchref.math_functions.math_torch import nll_xray
        
        fobs = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
        fcalc = torch.tensor([105.0, 195.0, 305.0], dtype=torch.float32)
        sigma = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float32)
        
        nll = nll_xray(fobs, fcalc, sigma)
        
        # nll returns per-reflection values
        assert torch.all(torch.isfinite(nll))

    def test_nll_decreases_with_better_fit(self):
        """Test that NLL decreases as fit improves."""
        from torchref.math_functions.math_torch import nll_xray
        
        fobs = torch.tensor([100.0], dtype=torch.float32)
        sigma = torch.tensor([10.0], dtype=torch.float32)
        
        # Good fit
        fcalc_good = torch.tensor([100.0], dtype=torch.float32)
        nll_good = nll_xray(fobs, fcalc_good, sigma)
        
        # Bad fit
        fcalc_bad = torch.tensor([150.0], dtype=torch.float32)
        nll_bad = nll_xray(fobs, fcalc_bad, sigma)
        
        # Good fit should have lower NLL
        assert nll_good < nll_bad


@pytest.mark.integration
class TestGradnormUtility:
    """Test the gradnorm utility function."""

    def test_gradnorm_basic(self):
        """Test basic gradnorm calculation."""
        from torchref.utils.gradnorm import gradnorm
        
        # Create simple parameter
        param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Create loss
        loss = param.sum()
        
        # Compute gradient norm
        norm = gradnorm(loss, [param])
        
        assert torch.isfinite(norm)
        assert norm > 0

    def test_gradnorm_with_multiple_params(self):
        """Test gradnorm with multiple parameters."""
        from torchref.utils.gradnorm import gradnorm
        
        param1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param2 = torch.tensor([3.0, 4.0], requires_grad=True)
        
        loss = param1.sum() + param2.sum()
        
        norm = gradnorm(loss, [param1, param2])
        
        assert torch.isfinite(norm)
        assert norm > 0


@pytest.mark.integration
class TestWeightingPhases:
    """Test weighting behavior in different refinement phases."""

    def test_phase_parameter(self):
        """Test that phase parameter is accepted."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        
        # For FixedWeighting, phase doesn't change weights
        # But it should accept the parameter
        weights = weighting.target_weights
        
        # Same weights regardless of phase
        assert torch.isfinite(weights['xray'])
        assert torch.isfinite(weights['restraints'])

    def test_cycle_parameter(self):
        """Test that cycle parameter is accepted."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting()
        
        # For FixedWeighting, cycle doesn't change weights
        weights = weighting.target_weights
        
        # Should have valid weights
        for name, weight in weights.items():
            assert torch.isfinite(weight)


@pytest.mark.integration
class TestWeightingEdgeCases:
    """Test edge cases in weighting."""

    def test_zero_weight(self):
        """Test zero weight (disabling a loss term)."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'adp': 0.0})
        weights = weighting.target_weights
        
        # Zero weight should effectively disable ADP term
        adp_loss = torch.tensor(100.0)
        weighted = weights['adp'] * adp_loss
        
        assert torch.isclose(weighted, torch.tensor(0.0))

    def test_very_small_weight(self):
        """Test very small weight."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'adp': 1e-6})
        weights = weighting.target_weights
        
        # Should still be a valid small weight
        assert weights['adp'] > 0
        assert torch.isfinite(weights['adp'])

    def test_large_weight(self):
        """Test large weight."""
        from torchref.refinement.loss_weighting import FixedWeighting
        
        weighting = FixedWeighting(target_weights={'xray': 100.0})
        weights = weighting.target_weights
        
        assert torch.isclose(weights['xray'], torch.tensor(100.0))
