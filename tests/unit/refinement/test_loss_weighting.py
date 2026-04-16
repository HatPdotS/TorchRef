"""
Unit tests for torchref.refinement.weighting.component_weighting

Tests the component weighting system for loss aggregation.
Note: Many component weighting tests require a full Refinement object,
so they are placed in functional/integration tests. These unit tests
cover the basic interfaces without requiring real refinement objects.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock


class TestWeightingSchemeBase:
    """Tests for WeightingScheme base class."""

    @pytest.mark.unit
    def test_weighting_scheme_is_nn_module(self):
        """Test WeightingScheme is a nn.Module."""
        from torchref.refinement.weighting.component_weighting import WeightingScheme

        # WeightingScheme is abstract, so we can't instantiate it directly
        # But we can check the inheritance
        assert issubclass(WeightingScheme, nn.Module)

    @pytest.mark.unit
    def test_weighting_scheme_has_forward(self):
        """Test WeightingScheme has forward method."""
        from torchref.refinement.weighting.component_weighting import WeightingScheme

        assert hasattr(WeightingScheme, 'forward')


class TestManualWeighting:
    """Tests for ManualWeighting class."""

    @pytest.mark.unit
    def test_manual_weighting_initialization(self):
        """Test ManualWeighting can be initialized with weights."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        weights = {'xray': 1.0, 'geometry': 0.5}
        weighting = ManualWeighting(weights=weights, device=torch.device('cpu'))

        assert weighting is not None

    @pytest.mark.unit
    def test_manual_weighting_forward(self):
        """Test ManualWeighting forward returns weights."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting
        from torchref.refinement.loss_state import LossState

        weights = {'xray': 1.0, 'geometry': 0.5}
        weighting = ManualWeighting(weights=weights, device=torch.device('cpu'))

        state = LossState()
        result = weighting.forward(state)

        assert 'xray' in result
        assert 'geometry' in result
        # Values are now floats (not tensors)
        assert isinstance(result['xray'], float)
        assert isinstance(result['geometry'], float)


class TestLossStateIntegration:
    """Tests for LossState integration with weighting."""

    @pytest.mark.unit
    def test_manual_weighting_forward_sets_weights(self):
        """Test ManualWeighting.forward returns weights that can be set on state."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting
        from torchref.refinement.loss_state import LossState

        weights = {'xray': 1.0, 'bond': 0.5}
        weighting = ManualWeighting(weights=weights, device=torch.device('cpu'))

        state = LossState()
        computed_weights = weighting.forward(state)
        state.set_weights(computed_weights)

        assert 'xray' in state.weights
        assert 'bond' in state.weights
        assert state.weights['xray'] == 1.0
        assert state.weights['bond'] == 0.5

    @pytest.mark.unit
    def test_hierarchical_weights_multiply(self):
        """Test that hierarchical weights multiply in get_effective_weight."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        # Set group weight
        state.set_weight('geometry', 2.0)
        # Set component weight
        state.set_weight('geometry/bond', 3.0)

        # Effective weight should multiply: 2.0 * 3.0 = 6.0
        effective = state.get_effective_weight('geometry/bond')
        assert effective == 6.0


class TestTotalLossFromState:
    """Tests for computing total loss from LossState."""

    @pytest.mark.unit
    def test_total_weighted_loss(self):
        """Test computing total weighted loss from state via aggregate."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(2.0))
        state.register_target('bond', lambda: torch.tensor(1.0))
        state.set_weight('xray', 1.0)
        state.set_weight('bond', 0.5)

        total = state.aggregate(log_values=False)

        # Expected: 2.0 * 1.0 + 1.0 * 0.5 = 2.5
        expected = torch.tensor(2.5)
        assert torch.isclose(total, expected)



