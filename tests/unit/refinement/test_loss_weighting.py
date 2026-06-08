"""
Unit tests for LossState weight handling.

Covers the retained ``LossState`` weight API (``set_weight`` /
``get_effective_weight`` / ``aggregate``). The standalone weighting
schemes were removed; refinement now aggregates at uniform weight by
default, with explicit per-target/group multipliers set via the
``LossState`` weight dict.
"""

import pytest
import torch


class TestLossStateWeights:
    """Tests for the LossState weight dict (hierarchical multipliers)."""

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
