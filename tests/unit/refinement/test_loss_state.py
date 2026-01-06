"""
Unit tests for torchref.refinement.loss_state

Tests the LossState with hierarchical naming and lazy evaluation.
"""

import pytest
import torch


class TestLossStateBasic:
    """Tests for basic LossState functionality."""

    @pytest.mark.unit
    def test_loss_state_creation(self):
        """Test LossState can be created with default values."""
        from torchref.refinement.loss_state import LossState

        state = LossState()

        assert state.device == torch.device('cpu')
        assert state.targets == {}
        assert state.weights == {}
        assert state.history == []

    @pytest.mark.unit
    def test_loss_state_with_device(self):
        """Test LossState creation with specific device."""
        from torchref.refinement.loss_state import LossState

        device = torch.device('cpu')
        state = LossState(device=device)

        assert state.device == device


class TestTargetRegistration:
    """Tests for target registration."""

    @pytest.mark.unit
    def test_register_target(self):
        """Test registering a single target."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        target_fn = lambda: torch.tensor(1.0)

        result = state.register_target('geometry/bond', target_fn)

        assert 'geometry/bond' in state.targets
        assert result is state  # Method chaining

    @pytest.mark.unit
    def test_register_multiple_targets(self):
        """Test registering multiple targets."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        targets = {
            'xray/work': lambda: torch.tensor(1.0),
            'geometry/bond': lambda: torch.tensor(0.5),
            'adp/simu': lambda: torch.tensor(0.3),
        }

        state.register_targets(targets)

        assert len(state.targets) == 3
        assert 'xray/work' in state.targets
        assert 'geometry/bond' in state.targets
        assert 'adp/simu' in state.targets


class TestWeightManagement:
    """Tests for weight management."""

    @pytest.mark.unit
    def test_set_weight(self):
        """Test setting a weight."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        result = state.set_weight('geometry', 0.5)

        assert state.weights['geometry'] == 0.5
        assert result is state  # Method chaining

    @pytest.mark.unit
    def test_set_multiple_weights(self):
        """Test setting multiple weights."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.set_weights({
            'xray': 1.0,
            'geometry': 0.5,
            'geometry/bond': 2.0,
        })

        assert state.weights['xray'] == 1.0
        assert state.weights['geometry'] == 0.5
        assert state.weights['geometry/bond'] == 2.0

    @pytest.mark.unit
    def test_get_weight_default(self):
        """Test getting a weight with default value."""
        from torchref.refinement.loss_state import LossState

        state = LossState()

        # Missing weight returns default
        weight = state.get_weight('nonexistent', default=1.0)
        assert weight == 1.0

    @pytest.mark.unit
    def test_get_effective_weight_simple(self):
        """Test getting effective weight for a simple name."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.set_weight('xray', 2.0)

        effective = state.get_effective_weight('xray')
        assert effective == 2.0

    @pytest.mark.unit
    def test_get_effective_weight_hierarchical(self):
        """Test getting effective weight with hierarchy."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.set_weight('geometry', 0.5)
        state.set_weight('geometry/bond', 2.0)

        # geometry/bond -> geometry (0.5) * geometry/bond (2.0) = 1.0
        effective = state.get_effective_weight('geometry/bond')
        assert effective == 1.0

    @pytest.mark.unit
    def test_get_effective_weight_missing_intermediate(self):
        """Test effective weight when intermediate is missing (defaults to 1.0)."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.set_weight('geometry/bond', 2.0)
        # 'geometry' not set, defaults to 1.0

        effective = state.get_effective_weight('geometry/bond')
        assert effective == 2.0  # 1.0 * 2.0


class TestAggregation:
    """Tests for loss aggregation."""

    @pytest.mark.unit
    def test_aggregate_simple(self):
        """Test simple aggregation."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(2.0))
        state.register_target('geometry', lambda: torch.tensor(1.0))
        state.set_weight('xray', 1.0)
        state.set_weight('geometry', 0.5)

        total = state.aggregate(log_values=False)

        # 2.0 * 1.0 + 1.0 * 0.5 = 2.5
        assert torch.isclose(total, torch.tensor(2.5))

    @pytest.mark.unit
    def test_aggregate_hierarchical(self):
        """Test aggregation with hierarchical weights."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('geometry/bond', lambda: torch.tensor(1.0))
        state.register_target('geometry/angle', lambda: torch.tensor(2.0))
        state.set_weight('geometry', 0.5)
        state.set_weight('geometry/bond', 2.0)
        state.set_weight('geometry/angle', 1.0)

        total = state.aggregate(log_values=False)

        # geometry/bond: 1.0 * 0.5 * 2.0 = 1.0
        # geometry/angle: 2.0 * 0.5 * 1.0 = 1.0
        # total = 2.0
        assert torch.isclose(total, torch.tensor(2.0))

    @pytest.mark.unit
    def test_aggregate_default_weights(self):
        """Test aggregation with default weights (1.0)."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(2.0))
        state.register_target('geometry', lambda: torch.tensor(1.0))
        # No weights set - all default to 1.0

        total = state.aggregate(log_values=False)

        # 2.0 * 1.0 + 1.0 * 1.0 = 3.0
        assert torch.isclose(total, torch.tensor(3.0))

    @pytest.mark.unit
    def test_aggregate_caches_losses(self):
        """Test that aggregate caches computed losses."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(2.0))

        state.aggregate(log_values=False)

        loss = state.get_loss('xray')
        assert torch.isclose(loss, torch.tensor(2.0))


class TestHistoryLogging:
    """Tests for history logging."""

    @pytest.mark.unit
    def test_log_value(self):
        """Test logging a value."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.log('test_key', 1.5)

        assert len(state.history) == 1
        assert state.history[0]['test_key'] == 1.5

    @pytest.mark.unit
    def test_log_tensor(self):
        """Test logging a tensor (converted to float)."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.log('test_key', torch.tensor(1.5))

        assert state.history[0]['test_key'] == 1.5
        assert isinstance(state.history[0]['test_key'], float)

    @pytest.mark.unit
    def test_new_entry(self):
        """Test creating new history entries."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.log('key1', 1.0)
        state.new_entry()
        state.log('key2', 2.0)

        assert len(state.history) == 2
        assert 'key1' in state.history[0]
        assert 'key2' in state.history[1]

    @pytest.mark.unit
    def test_get_history(self):
        """Test getting history for a key."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.log('total', 1.0)
        state.new_entry()
        state.log('total', 2.0)
        state.new_entry()
        state.log('total', 3.0)

        totals = state.get_history('total')
        assert totals == [1.0, 2.0, 3.0]

    @pytest.mark.unit
    def test_aggregate_logs_values(self):
        """Test that aggregate logs losses, weights, and total."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(2.0))
        state.set_weight('xray', 1.5)

        state.aggregate(log_values=True)

        assert len(state.history) == 1
        entry = state.history[0]
        assert 'loss/xray' in entry
        assert 'weight/xray' in entry
        assert 'weighted/xray' in entry
        assert 'total' in entry


class TestBreakdown:
    """Tests for breakdown and analysis."""

    @pytest.mark.unit
    def test_get_breakdown(self):
        """Test getting breakdown by group."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('geometry/bond', lambda: torch.tensor(1.0))
        state.register_target('geometry/angle', lambda: torch.tensor(2.0))
        state.register_target('adp/simu', lambda: torch.tensor(0.5))
        state.set_weight('geometry', 0.5)

        state.aggregate(log_values=False)
        breakdown = state.get_breakdown()

        assert 'geometry' in breakdown
        assert 'adp' in breakdown
        assert 'bond' in breakdown['geometry']
        assert 'angle' in breakdown['geometry']
        assert 'simu' in breakdown['adp']

    @pytest.mark.unit
    def test_get_group_totals(self):
        """Test getting totals per group."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('geometry/bond', lambda: torch.tensor(1.0))
        state.register_target('geometry/angle', lambda: torch.tensor(1.0))
        state.register_target('xray/work', lambda: torch.tensor(2.0))
        state.set_weight('geometry', 0.5)
        state.set_weight('xray', 1.0)

        state.aggregate(log_values=False)
        totals = state.get_group_totals()

        # geometry: (1.0 + 1.0) * 0.5 = 1.0
        # xray: 2.0 * 1.0 = 2.0
        assert abs(totals['geometry'] - 1.0) < 1e-6
        assert abs(totals['xray'] - 2.0) < 1e-6


class TestUtility:
    """Tests for utility methods."""

    @pytest.mark.unit
    def test_clear(self):
        """Test clearing cached losses."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(2.0))
        state.aggregate(log_values=False)

        assert state.get_loss('xray') is not None

        state.clear()

        assert state.get_loss('xray') is None
        assert 'xray' in state.targets  # Targets not cleared

    @pytest.mark.unit
    def test_clear_history(self):
        """Test clearing history."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.log('key', 1.0)

        assert len(state.history) == 1

        state.clear_history()

        assert len(state.history) == 0

    @pytest.mark.unit
    def test_repr(self):
        """Test string representation."""
        from torchref.refinement.loss_state import LossState

        state = LossState()
        state.register_target('xray', lambda: torch.tensor(1.0))
        state.set_weight('xray', 1.0)

        repr_str = repr(state)

        assert 'LossState' in repr_str
        assert 'targets=1' in repr_str
        assert 'weights=1' in repr_str


class TestFactory:
    """Tests for factory function."""

    @pytest.mark.unit
    def test_create_loss_state(self):
        """Test create_loss_state factory."""
        from torchref.refinement.loss_state import create_loss_state

        targets = {'xray': lambda: torch.tensor(1.0)}
        weights = {'xray': 2.0}

        state = create_loss_state(
            device=torch.device('cpu'),
            targets=targets,
            weights=weights,
        )

        assert 'xray' in state.targets
        assert state.weights['xray'] == 2.0
