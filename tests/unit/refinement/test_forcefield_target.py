"""
Unit and integration tests for ForceFieldTarget.

Tests the force field energy target using TorchMD-Net neural network potentials.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import importlib


# Path to test files
TEST_FILES_DIR = Path(__file__).parent.parent.parent / "files"

_has_torchmdnet = importlib.util.find_spec("torchmdnet") is not None
TEST_CHECKPOINT = TEST_FILES_DIR / "torchmdnet_test.ckpt"
TEST_PDB_WITH_H = TEST_FILES_DIR / "pdb" / "1AK5_with_H.pdb"


class MockAtomicModel(nn.Module):
    """Mock atomic model for testing that is an nn.Module."""

    def __init__(self, n_atoms=5, has_hydrogens=True):
        super().__init__()
        # Create atomic numbers
        if has_hydrogens:
            self.Z = torch.tensor([6, 1, 1, 1, 1])  # C + 4H
        else:
            self.Z = torch.tensor([6, 6, 7, 8, 16])  # No H

        # Create coordinates as a parameter for gradient testing
        self._coords = nn.Parameter(torch.randn(len(self.Z), 3))

    def xyz(self):
        """Return current coordinates."""
        return self._coords

    def forward(self, *args, **kwargs):
        return self._coords


class TestForceFieldTargetImport:
    """Tests for ForceFieldTarget import and basic initialization."""

    @pytest.mark.unit
    def test_forcefield_target_importable(self):
        """ForceFieldTarget should be importable from targets module."""
        from torchref.refinement.targets import ForceFieldTarget

        assert ForceFieldTarget is not None

    @pytest.mark.unit
    def test_forcefield_target_is_model_target(self):
        """ForceFieldTarget should inherit from ModelTarget."""
        from torchref.refinement.targets import ForceFieldTarget, ModelTarget

        assert issubclass(ForceFieldTarget, ModelTarget)

    @pytest.mark.unit
    def test_forcefield_target_has_name(self):
        """ForceFieldTarget should have name attribute."""
        from torchref.refinement.targets import ForceFieldTarget

        assert ForceFieldTarget.name == "forcefield"

    @pytest.mark.unit
    def test_forcefield_target_empty_init(self):
        """ForceFieldTarget should allow empty initialization."""
        from torchref.refinement.targets import ForceFieldTarget

        target = ForceFieldTarget()

        assert target.model is None
        assert target._model_path is None
        assert target._normalize_by_atoms is True

    @pytest.mark.unit
    def test_forcefield_target_with_params(self):
        """ForceFieldTarget should accept configuration parameters."""
        from torchref.refinement.targets import ForceFieldTarget

        target = ForceFieldTarget(
            model_path="/path/to/model.ckpt",
            cutoff=6.0,
            normalize_by_atoms=False,
            verbose=1,
        )

        assert target._model_path == "/path/to/model.ckpt"
        assert target._cutoff.item() == 6.0
        assert target._normalize_by_atoms is False
        assert target.verbose == 1

    @pytest.mark.unit
    def test_forcefield_target_is_nn_module(self):
        """ForceFieldTarget should be a nn.Module."""
        from torchref.refinement.targets import ForceFieldTarget
        import torch.nn as nn

        target = ForceFieldTarget()

        assert isinstance(target, nn.Module)

    @pytest.mark.unit
    def test_cutoff_is_buffer(self):
        """Cutoff should be registered as a buffer for state_dict compatibility."""
        from torchref.refinement.targets import ForceFieldTarget

        target = ForceFieldTarget(cutoff=7.5)

        assert "_cutoff" in dict(target.named_buffers())
        assert target._cutoff.item() == 7.5


class TestForceFieldTargetErrorHandling:
    """Tests for error handling in ForceFieldTarget."""

    @pytest.mark.unit
    def test_forward_without_model_path_raises(self):
        """forward() should raise ValueError if model_path is None."""
        from torchref.refinement.targets import ForceFieldTarget

        # Create a mock model using MockAtomicModel
        mock_model = MockAtomicModel(has_hydrogens=True)

        target = ForceFieldTarget(model=mock_model, model_path=None)

        with pytest.raises(ValueError, match="model_path is required"):
            target.forward()

    @pytest.mark.unit
    def test_forward_without_torchmdnet_raises(self):
        """forward() should raise ImportError if torchmd-net is not installed."""
        from torchref.refinement.targets import ForceFieldTarget

        mock_model = MockAtomicModel(has_hydrogens=True)

        target = ForceFieldTarget(model=mock_model, model_path="/fake/path.ckpt")

        # Mock the import to fail
        with patch.dict('sys.modules', {'torchmdnet': None, 'torchmdnet.models': None, 'torchmdnet.models.model': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'torchmdnet'")):
                # Reset the cached potential
                target._nn_potential = None
                with pytest.raises(ImportError, match="torchmd-net"):
                    target._ensure_nn_potential()


class TestForceFieldTargetWithMock:
    """Tests using mocked TorchMD-Net model."""

    @pytest.mark.unit
    def test_forward_returns_tensor(self):
        """forward() should return a scalar tensor."""
        from torchref.refinement.targets import ForceFieldTarget

        # Create mock atomic model
        mock_atomic_model = MockAtomicModel(has_hydrogens=True)

        mock_nn_potential = Mock()
        mock_nn_potential.return_value = (torch.tensor([[1.5]]), torch.tensor([]))

        target = ForceFieldTarget(model=mock_atomic_model, model_path="/fake/path.ckpt")
        target._nn_potential = mock_nn_potential  # Inject mock
        target._validated = True  # Skip validation

        result = target.forward()

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar

    @pytest.mark.unit
    def test_forward_normalizes_by_atoms(self):
        """forward() should normalize energy by number of atoms when enabled."""
        from torchref.refinement.targets import ForceFieldTarget

        n_atoms = 5
        raw_energy = 10.0

        mock_atomic_model = MockAtomicModel(has_hydrogens=True)

        mock_nn_potential = Mock()
        mock_nn_potential.return_value = (torch.tensor([[raw_energy]]), torch.tensor([]))

        target = ForceFieldTarget(
            model=mock_atomic_model,
            model_path="/fake/path.ckpt",
            normalize_by_atoms=True,
        )
        target._nn_potential = mock_nn_potential
        target._validated = True

        result = target.forward()

        expected = raw_energy / n_atoms
        assert torch.isclose(result, torch.tensor(expected))

    @pytest.mark.unit
    def test_forward_no_normalization(self):
        """forward() should not normalize when normalize_by_atoms=False."""
        from torchref.refinement.targets import ForceFieldTarget

        n_atoms = 5
        raw_energy = 10.0

        mock_atomic_model = MockAtomicModel(has_hydrogens=True)

        mock_nn_potential = Mock()
        mock_nn_potential.return_value = (torch.tensor([[raw_energy]]), torch.tensor([]))

        target = ForceFieldTarget(
            model=mock_atomic_model,
            model_path="/fake/path.ckpt",
            normalize_by_atoms=False,
        )
        target._nn_potential = mock_nn_potential
        target._validated = True

        result = target.forward()

        assert torch.isclose(result, torch.tensor(raw_energy))

    @pytest.mark.unit
    def test_stats_returns_dict(self):
        """stats() should return a dictionary with expected keys."""
        from torchref.refinement.targets import ForceFieldTarget

        mock_atomic_model = MockAtomicModel(has_hydrogens=True)

        mock_nn_potential = Mock()
        mock_nn_potential.return_value = (torch.tensor([[1.5]]), torch.tensor([]))

        target = ForceFieldTarget(
            model=mock_atomic_model,
            model_path="/test/path.ckpt",
        )
        target._nn_potential = mock_nn_potential
        target._validated = True

        stats = target.stats()

        assert isinstance(stats, dict)
        assert "loss" in stats
        assert "n_atoms" in stats
        assert "model_path" in stats


@pytest.mark.skipif(
    not _has_torchmdnet or not TEST_CHECKPOINT.exists(),
    reason="torchmd-net not installed or test checkpoint not found"
)
class TestForceFieldTargetIntegration:
    """Integration tests with real TorchMD-Net model."""

    @pytest.mark.integration
    def test_load_real_checkpoint(self):
        """Test loading a real TorchMD-Net checkpoint."""
        from torchmdnet.models.model import load_model

        model = load_model(str(TEST_CHECKPOINT))

        assert model is not None

    @pytest.mark.integration
    def test_forward_with_real_model(self):
        """Test forward pass with real TorchMD-Net model."""
        from torchref.refinement.targets import ForceFieldTarget

        # Create a minimal mock atomic model using MockAtomicModel
        mock_atomic_model = MockAtomicModel(has_hydrogens=True)

        target = ForceFieldTarget(
            model=mock_atomic_model,
            model_path=str(TEST_CHECKPOINT),
        )

        result = target.forward()

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0
        assert torch.isfinite(result)


@pytest.mark.skipif(
    not _has_torchmdnet or not TEST_CHECKPOINT.exists() or not TEST_PDB_WITH_H.exists(),
    reason="torchmd-net not installed, test checkpoint or PDB with hydrogens not found"
)
class TestForceFieldTargetWithRealModel:
    """Integration tests with real Model and TorchMD-Net checkpoint."""

    @pytest.mark.integration
    def test_forward_with_real_pdb(self):
        """Test forward pass with real PDB file containing hydrogens."""
        from torchref.model import Model
        from torchref.refinement.targets import ForceFieldTarget

        # Load model WITH hydrogens
        model = Model(strip_H=False)
        model.load_pdb(str(TEST_PDB_WITH_H))

        # Check we have hydrogens
        n_hydrogens = (model.Z == 1).sum().item()
        assert n_hydrogens > 0, "Model should contain hydrogen atoms"

        # Create force field target
        target = ForceFieldTarget(
            model=model,
            model_path=str(TEST_CHECKPOINT),
            normalize_by_atoms=True,
        )

        # Compute energy
        energy = target.forward()

        assert isinstance(energy, torch.Tensor)
        assert energy.dim() == 0
        assert torch.isfinite(energy)

        print(f"Energy per atom: {energy.item():.4f}")
        print(f"Number of atoms: {len(model.Z)}")
        print(f"Number of hydrogens: {n_hydrogens}")

    @pytest.mark.integration
    def test_gradient_flow(self):
        """Test that gradients flow through the target."""
        from torchref.model import Model
        from torchref.refinement.targets import ForceFieldTarget

        # Load model WITH hydrogens
        model = Model(strip_H=False)
        model.load_pdb(str(TEST_PDB_WITH_H))

        # Create target
        target = ForceFieldTarget(
            model=model,
            model_path=str(TEST_CHECKPOINT),
        )

        # Compute energy and check gradients
        energy = target.forward()
        energy.backward()

        # Check that some parameter has gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Gradients should flow through the model"

    @pytest.mark.integration
    def test_stats_with_real_model(self):
        """Test stats() with real model."""
        from torchref.model import Model
        from torchref.refinement.targets import ForceFieldTarget

        model = Model(strip_H=False)
        model.load_pdb(str(TEST_PDB_WITH_H))

        target = ForceFieldTarget(
            model=model,
            model_path=str(TEST_CHECKPOINT),
        )

        stats = target.stats()

        assert "loss" in stats
        assert stats["n_atoms"].value == len(model.Z)
        assert "model_path" in stats


class TestForceFieldTargetHydrogenValidation:
    """Tests for hydrogen atom validation."""

    @pytest.mark.unit
    def test_warns_on_missing_hydrogens(self):
        """Should warn when model lacks hydrogen atoms."""
        from torchref.refinement.targets import ForceFieldTarget
        import warnings

        # Mock model without hydrogens
        mock_model = MockAtomicModel(has_hydrogens=False)

        mock_nn_potential = Mock()
        mock_nn_potential.return_value = (torch.tensor([[1.0]]), torch.tensor([]))

        target = ForceFieldTarget(
            model=mock_model,
            model_path="/fake/path.ckpt",
            verbose=1,  # Enable warnings
        )
        target._nn_potential = mock_nn_potential

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            target.forward()

            # Check that warning was raised
            hydrogen_warnings = [x for x in w if "hydrogen" in str(x.message).lower()]
            assert len(hydrogen_warnings) > 0

    @pytest.mark.unit
    def test_no_warning_with_hydrogens(self):
        """Should not warn when model has hydrogen atoms."""
        from torchref.refinement.targets import ForceFieldTarget
        import warnings

        # Mock model with hydrogens
        mock_model = MockAtomicModel(has_hydrogens=True)

        mock_nn_potential = Mock()
        mock_nn_potential.return_value = (torch.tensor([[1.0]]), torch.tensor([]))

        target = ForceFieldTarget(
            model=mock_model,
            model_path="/fake/path.ckpt",
            verbose=1,
        )
        target._nn_potential = mock_nn_potential

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            target.forward()

            hydrogen_warnings = [x for x in w if "hydrogen" in str(x.message).lower()]
            assert len(hydrogen_warnings) == 0
