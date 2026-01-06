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
    def test_weighting_scheme_has_apply_to_state(self):
        """Test WeightingScheme has apply_to_state method."""
        from torchref.refinement.weighting.component_weighting import WeightingScheme

        assert hasattr(WeightingScheme, 'apply_to_state')


class TestManualWeighting:
    """Tests for ManualWeighting class."""

    @pytest.mark.unit
    def test_manual_weighting_initialization(self):
        """Test ManualWeighting can be initialized with weights."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        # Create with mock refinement providing device
        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weights = {'xray': 1.0, 'geometry': 0.5}
        weighting = ManualWeighting(refinement=mock_ref, weights=weights)

        assert weighting is not None

    @pytest.mark.unit
    def test_manual_weighting_forward(self):
        """Test ManualWeighting forward returns weights."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weights = {'xray': 1.0, 'geometry': 0.5}
        weighting = ManualWeighting(refinement=mock_ref, weights=weights)

        result = weighting.forward()

        assert 'xray' in result
        assert 'geometry' in result
        # Values should be tensors
        assert isinstance(result['xray'], torch.Tensor)
        assert isinstance(result['geometry'], torch.Tensor)


class TestLossStateIntegration:
    """Tests for LossState integration with weighting."""

    @pytest.mark.unit
    def test_manual_weighting_apply_to_state(self):
        """Test ManualWeighting.apply_to_state sets weights in state."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting
        from torchref.refinement.loss_state import LossState

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weights = {'xray': 1.0, 'bond': 0.5}
        weighting = ManualWeighting(refinement=mock_ref, weights=weights)

        state = LossState()
        result = weighting.apply_to_state(state)

        assert result is state
        assert 'xray' in state.weights
        assert 'bond' in state.weights

    @pytest.mark.unit
    def test_weighting_multiplies_existing_weights(self):
        """Test that apply_to_state multiplies existing weights."""
        from torchref.refinement.weighting.component_weighting import ManualWeighting
        from torchref.refinement.loss_state import LossState

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        weights = {'xray': 2.0}
        weighting = ManualWeighting(refinement=mock_ref, weights=weights)

        state = LossState()
        state.set_weight('xray', 3.0)
        weighting.apply_to_state(state)

        # Should multiply: 3.0 * 2.0 = 6.0
        assert state.weights['xray'] == 6.0


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


class TestPolicyComponentWeighting:
    """Tests for PolicyComponentWeighting class."""

    @pytest.mark.unit
    def test_policy_weighting_import(self):
        """Test PolicyComponentWeighting can be imported."""
        from torchref.refinement.weighting import (
            PolicyComponentWeighting,
            StepState,
            StepRecord,
            TrajectoryData,
            trajectory_to_dict,
            COMPONENTS,
            COMPONENT_TO_LOSS_STATE,
        )

        assert PolicyComponentWeighting is not None
        assert len(COMPONENTS) == 10  # 10 component types

    @pytest.mark.unit
    def test_policy_network_architecture(self):
        """Test policy network has correct architecture."""
        from torchref.refinement.weighting.policy_weighting import PolicyComponentWeighting

        # Create mock refinement
        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        mock_ref.verbose = 0

        policy = PolicyComponentWeighting(mock_ref, policy_path=None, sample=False)

        # Check network exists
        assert hasattr(policy, 'policy')
        assert isinstance(policy.policy, nn.Module)

        # Test forward pass shape
        dummy_input = torch.randn(1, 25)
        log_weights, log_sigmas = policy.policy(dummy_input)

        assert log_weights.shape == (1, 10)  # 10 components
        assert log_sigmas.shape == (1, 10)

    @pytest.mark.unit
    def test_policy_training_vs_eval_mode(self):
        """Test policy can switch between training and evaluation modes."""
        from torchref.refinement.weighting.policy_weighting import PolicyComponentWeighting

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        mock_ref.verbose = 0

        # Create in eval mode (default)
        policy = PolicyComponentWeighting(mock_ref, sample=False)
        assert policy.sample is False

        # Switch to training mode
        policy.set_training_mode(sample=True, temperature=0.5)
        assert policy.sample is True
        assert policy.temperature == 0.5

        # Switch back to eval mode
        policy.set_eval_mode()
        assert policy.sample is False

    @pytest.mark.unit
    def test_step_state_dataclass(self):
        """Test StepState dataclass fields."""
        from torchref.refinement.weighting import StepState

        state = StepState(
            step=0,
            rwork=0.25,
            rfree=0.28,
            rfree_gap=0.03,
            delta_rwork=-0.01,
            delta_rfree=-0.02,
            xray_loss=100.0,
            xray_loss_test=110.0,
            geometry_loss=50.0,
            adp_loss=10.0,
            bond_rmsd=0.015,
            angle_rmsd=1.5,
            mean_b=25.0,
            component_losses={'bond': 10.0, 'angle': 20.0},
            component_weights={'bond': 1.0, 'angle': 0.5},
        )

        assert state.step == 0
        assert state.rfree == 0.28
        assert state.component_losses['bond'] == 10.0

    @pytest.mark.unit
    def test_trajectory_data_dataclass(self):
        """Test TrajectoryData dataclass."""
        from torchref.refinement.weighting import TrajectoryData, StepRecord, StepState

        trajectory = TrajectoryData(
            pdb_id='3GR5',
            structure_path='/path/to/3GR5.pdb',
            sf_path='/path/to/3GR5.mtz',
            initial_rfree=0.35,
            final_rfree=0.28,
        )

        assert trajectory.pdb_id == '3GR5'
        assert trajectory.initial_rfree == 0.35
        assert len(trajectory.steps) == 0  # Empty initially

    @pytest.mark.unit
    def test_trajectory_to_dict(self):
        """Test trajectory_to_dict serialization."""
        from torchref.refinement.weighting import (
            TrajectoryData, StepRecord, StepState, trajectory_to_dict
        )

        state = StepState(
            step=0, rwork=0.25, rfree=0.28, rfree_gap=0.03,
            delta_rwork=0.0, delta_rfree=0.0,
            xray_loss=100.0, xray_loss_test=110.0,
            geometry_loss=50.0, adp_loss=10.0,
            bond_rmsd=0.015, angle_rmsd=1.5, mean_b=25.0,
        )
        record = StepRecord(
            state=state,
            log_weights={'xray': 0.0, 'bond': 1.0},
            weights={'xray': 1.0, 'bond': 2.7},
            reward=0.02,
        )
        trajectory = TrajectoryData(
            pdb_id='3GR5',
            structure_path='/path/to/3GR5.pdb',
            sf_path='/path/to/3GR5.mtz',
            steps=[record],
            initial_rfree=0.35,
            final_rfree=0.28,
        )

        result = trajectory_to_dict(trajectory)

        assert result['pdb_id'] == '3GR5'
        assert result['initial_rfree'] == 0.35
        assert len(result['steps']) == 1
        assert result['steps'][0]['reward'] == 0.02

    @pytest.mark.unit
    def test_component_to_loss_state_mapping(self):
        """Test mapping from policy components to LossState names."""
        from torchref.refinement.weighting import COMPONENT_TO_LOSS_STATE, LOSS_STATE_TO_COMPONENT

        # Check key mappings
        assert COMPONENT_TO_LOSS_STATE['bond'] == 'geometry/bond'
        assert COMPONENT_TO_LOSS_STATE['simu'] == 'adp/simu'
        assert COMPONENT_TO_LOSS_STATE['xray'] == 'xray'

        # Check reverse mapping
        assert LOSS_STATE_TO_COMPONENT['geometry/bond'] == 'bond'
        assert LOSS_STATE_TO_COMPONENT['adp/simu'] == 'simu'

    @pytest.mark.unit
    def test_policy_recording_lifecycle(self):
        """Test trajectory recording start/stop lifecycle."""
        from torchref.refinement.weighting.policy_weighting import PolicyComponentWeighting

        mock_ref = Mock()
        mock_ref.device = torch.device('cpu')
        mock_ref.verbose = 0
        mock_ref.get_rfactor = Mock(return_value=(torch.tensor(0.25), torch.tensor(0.28)))

        policy = PolicyComponentWeighting(mock_ref, sample=True)

        # Initially not recording
        assert policy._recording is False
        assert policy.trajectory is None

        # Start recording
        policy.start_recording(
            pdb_id='3GR5',
            structure_path='/path/to/pdb',
            sf_path='/path/to/mtz',
            seed=42,
        )

        assert policy._recording is True
        assert policy.trajectory is not None
        assert policy.trajectory.pdb_id == '3GR5'
        assert policy.trajectory.initial_rfree == pytest.approx(0.28, rel=1e-5)

        # Stop recording
        trajectory = policy.stop_recording()

        assert policy._recording is False
        assert policy.trajectory is None
        assert trajectory is not None
        assert trajectory.final_rfree == pytest.approx(0.28, rel=1e-5)
