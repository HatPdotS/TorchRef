"""
Functional tests for loss weighting module.

These tests exercise the loss weighting strategies with realistic data.
Updated to use the new component_weighting and LossState architecture.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock


@pytest.mark.integration
class TestManualWeightingFunctional:
    """Test ManualWeighting functionality."""

    def test_manual_weighting_initialization(self):
        """Test ManualWeighting initialization."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weighting = ManualWeighting(mock_ref, weights={'xray': 1.0})

        assert weighting is not None

    def test_manual_weighting_with_custom_weights(self):
        """Test ManualWeighting with custom weights."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        custom_weights = {
            'xray': 2.0,
            'geometry': 0.5,
            'adp': 0.1
        }

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weighting = ManualWeighting(mock_ref, weights=custom_weights)

        weights = weighting.forward()
        assert torch.isclose(weights['xray'], torch.tensor(2.0))
        assert torch.isclose(weights['geometry'], torch.tensor(0.5))
        assert torch.isclose(weights['adp'], torch.tensor(0.1))


@pytest.mark.integration
class TestLossStateWeightingFunctional:
    """Test LossState weighting functionality."""

    def test_loss_state_add_and_get_weights(self):
        """Test adding and getting weights from LossState."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.set_weight('xray', 1.5)
        state.set_weight('geometry', 0.7)

        assert state.get_weight('xray') == 1.5
        assert state.get_weight('geometry') == 0.7

    def test_loss_state_hierarchical_weights(self):
        """Test hierarchical weights in LossState."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.set_weight('geometry', 2.0)
        state.set_weight('geometry/bond', 3.0)

        # Effective weight should be product: 2.0 * 3.0 = 6.0
        effective = state.get_effective_weight('geometry/bond')
        assert effective == 6.0


@pytest.mark.integration
class TestWeightingMathOperations:
    """Test mathematical operations with weights."""

    def test_weight_multiplication(self):
        """Test using weights for loss scaling."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weighting = ManualWeighting(mock_ref, weights={'xray': 2.0, 'geometry': 0.5})
        weights = weighting.forward()

        # Create mock losses
        xray_loss = torch.tensor(10.0)
        geometry_loss = torch.tensor(20.0)

        # Apply weights
        weighted_xray = weights['xray'] * xray_loss
        weighted_geometry = weights['geometry'] * geometry_loss

        assert torch.isclose(weighted_xray, torch.tensor(20.0))
        assert torch.isclose(weighted_geometry, torch.tensor(10.0))

    def test_total_weighted_loss_from_state(self):
        """Test computing total weighted loss from LossState via aggregate."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(10.0))
        state.register_target('geometry', lambda: torch.tensor(5.0))
        state.register_target('adp', lambda: torch.tensor(2.0))

        state.set_weight('xray', 1.0)
        state.set_weight('geometry', 0.5)
        state.set_weight('adp', 0.25)

        total = state.aggregate()

        # Expected: 10*1.0 + 5*0.5 + 2*0.25 = 10 + 2.5 + 0.5 = 13.0
        assert torch.isclose(total, torch.tensor(13.0))


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
class TestWeightingEdgeCases:
    """Test edge cases in weighting."""

    def test_zero_weight(self):
        """Test zero weight (disabling a loss term)."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('adp', lambda: torch.tensor(100.0))
        state.set_weight('adp', 0.0)

        # Zero weight should effectively disable ADP term
        total = state.aggregate()
        assert torch.isclose(total, torch.tensor(0.0))

    def test_very_small_weight(self):
        """Test very small weight."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weighting = ManualWeighting(mock_ref, weights={'adp': 1e-6})
        weights = weighting.forward()

        # Should still be a valid small weight
        assert weights['adp'] > 0
        assert torch.isfinite(weights['adp'])

    def test_large_weight(self):
        """Test large weight."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weighting = ManualWeighting(mock_ref, weights={'xray': 100.0})
        weights = weighting.forward()

        assert torch.isclose(weights['xray'], torch.tensor(100.0))


@pytest.mark.integration
class TestLossAggregatorFunctional:
    """Test LossAggregator functionality."""

    def test_aggregator_basic(self):
        """Test basic aggregator functionality (LossState.aggregate)."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(2.0))
        state.register_target('bond', lambda: torch.tensor(1.0))
        state.set_weight('xray', 1.0)
        state.set_weight('bond', 0.5)

        total = state.aggregate()

        # Expected: 2.0 * 1.0 + 1.0 * 0.5 = 2.5
        expected = torch.tensor(2.5)
        assert torch.isclose(total, expected)

    def test_loss_state_caches_losses(self):
        """Test that LossState caches computed losses."""
        from torchref.refinement.loss_state import LossState

        call_count = [0]
        def counting_target():
            call_count[0] += 1
            return torch.tensor(2.0)

        state = LossState()
        state.register_target('xray', counting_target)
        state.set_weight('xray', 1.0)

        # First aggregation computes the loss
        total1 = state.aggregate()
        assert call_count[0] == 1

        # Get cached loss doesn't recompute
        cached = state.get_loss('xray')
        assert cached is not None
        assert torch.isclose(cached, torch.tensor(2.0))


