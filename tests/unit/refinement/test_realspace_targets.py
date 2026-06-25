"""
Unit and integration tests for real-space targets.

Tests RealSpaceCorrelationTarget and RealSpaceDifferenceTarget.
"""

import pytest
import torch
import numpy as np


# =============================================================================
# Unit Tests (synthetic data, fast)
# =============================================================================


class TestRealSpaceTargetNames:
    """Test that target names are set correctly for LossState integration."""

    @pytest.mark.unit
    def test_correlation_target_name(self):
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        target = RealSpaceCorrelationTarget()
        assert target.name == "realspace/correlation"

    @pytest.mark.unit
    def test_difference_target_name(self):
        from torchref.experimental.targets import RealSpaceDifferenceTarget
        target = RealSpaceDifferenceTarget()
        assert target.name == "realspace/difference"


class TestRealSpaceTargetImports:
    """Test that all classes are importable from the package."""

    @pytest.mark.unit
    def test_import_from_targets(self):
        from torchref.experimental.targets import (
            RealSpaceTarget,
            RealSpaceCorrelationTarget,
            RealSpaceDifferenceTarget,
        )

        assert RealSpaceTarget is not None
        assert RealSpaceCorrelationTarget is not None
        assert RealSpaceDifferenceTarget is not None

    @pytest.mark.unit
    def test_base_inherits_data_target(self):
        from torchref.experimental.targets import RealSpaceTarget
        from torchref.refinement.targets import DataTarget

        assert issubclass(RealSpaceTarget, DataTarget)

    @pytest.mark.unit
    def test_correlation_inherits_realspace(self):
        from torchref.experimental.targets import (
            RealSpaceCorrelationTarget,
            RealSpaceTarget,
        )

        assert issubclass(RealSpaceCorrelationTarget, RealSpaceTarget)

    @pytest.mark.unit
    def test_difference_inherits_realspace(self):
        from torchref.experimental.targets import (
            RealSpaceDifferenceTarget,
            RealSpaceTarget,
        )

        assert issubclass(RealSpaceDifferenceTarget, RealSpaceTarget)


class TestRealSpaceTargetValidation:
    """Test parameter validation."""

    @pytest.mark.unit
    def test_invalid_map_type_raises(self):
        from torchref.experimental.targets import RealSpaceTarget
        with pytest.raises(ValueError, match="map_type"):
            RealSpaceTarget(map_type="invalid")

    @pytest.mark.unit
    def test_valid_map_types(self):
        from torchref.experimental.targets import RealSpaceTarget
        t1 = RealSpaceTarget(map_type="2mFo-DFc")
        assert t1.map_type == "2mFo-DFc"

        t2 = RealSpaceTarget(map_type="Fo-Fc")
        assert t2.map_type == "Fo-Fc"

    @pytest.mark.unit
    def test_correlation_hardcodes_map_type(self):
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        target = RealSpaceCorrelationTarget()
        assert target.map_type == "2mFo-DFc"

    @pytest.mark.unit
    def test_difference_hardcodes_map_type(self):
        from torchref.experimental.targets import RealSpaceDifferenceTarget
        target = RealSpaceDifferenceTarget()
        assert target.map_type == "Fo-Fc"


class TestRSCCComputation:
    """Test RSCC computation logic with synthetic maps."""

    @pytest.mark.unit
    def test_rscc_identical_maps(self):
        """RSCC should be 1.0 for identical maps."""
        map_a = torch.randn(10, 10, 10)
        map_b = map_a.clone()

        a_flat = map_a.flatten()
        b_flat = map_b.flatten()

        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()

        eps = 1e-8
        cov = (a_centered * b_centered).mean()
        std_a = torch.sqrt((a_centered**2).mean() + eps)
        std_b = torch.sqrt((b_centered**2).mean() + eps)

        rscc = cov / (std_a * std_b)
        assert torch.isclose(rscc, torch.tensor(1.0), atol=1e-5)

    @pytest.mark.unit
    def test_rscc_uncorrelated_maps(self):
        """RSCC should be ~0 for uncorrelated maps."""
        torch.manual_seed(42)
        map_a = torch.randn(20, 20, 20)
        torch.manual_seed(123)
        map_b = torch.randn(20, 20, 20)

        a_flat = map_a.flatten()
        b_flat = map_b.flatten()

        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()

        eps = 1e-8
        cov = (a_centered * b_centered).mean()
        std_a = torch.sqrt((a_centered**2).mean() + eps)
        std_b = torch.sqrt((b_centered**2).mean() + eps)

        rscc = cov / (std_a * std_b)
        # For large uncorrelated maps, RSCC should be near 0
        assert abs(rscc.item()) < 0.1

    @pytest.mark.unit
    def test_rscc_negatively_correlated(self):
        """RSCC should be -1 for perfectly anti-correlated maps."""
        map_a = torch.randn(10, 10, 10)
        map_b = -map_a

        a_flat = map_a.flatten()
        b_flat = map_b.flatten()

        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()

        eps = 1e-8
        cov = (a_centered * b_centered).mean()
        std_a = torch.sqrt((a_centered**2).mean() + eps)
        std_b = torch.sqrt((b_centered**2).mean() + eps)

        rscc = cov / (std_a * std_b)
        assert torch.isclose(rscc, torch.tensor(-1.0), atol=1e-5)

    @pytest.mark.unit
    def test_correlation_loss_range(self):
        """1 - RSCC should be in [0, 2] for normal maps."""
        torch.manual_seed(42)
        map_a = torch.randn(10, 10, 10)
        map_b = map_a + 0.5 * torch.randn(10, 10, 10)

        a_flat = map_a.flatten()
        b_flat = map_b.flatten()

        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()

        eps = 1e-8
        cov = (a_centered * b_centered).mean()
        std_a = torch.sqrt((a_centered**2).mean() + eps)
        std_b = torch.sqrt((b_centered**2).mean() + eps)

        rscc = cov / (std_a * std_b)
        loss = 1.0 - rscc
        assert 0.0 <= loss.item() <= 2.0


class TestDifferenceMapComputation:
    """Test difference map loss computation logic."""

    @pytest.mark.unit
    def test_zero_difference_map(self):
        """Mean squared difference should be 0 for zero map."""
        diff_map = torch.zeros(10, 10, 10)
        loss = (diff_map**2).mean()
        assert loss.item() == 0.0

    @pytest.mark.unit
    def test_nonzero_difference_positive_loss(self):
        """Non-zero difference map should give positive loss."""
        torch.manual_seed(42)
        diff_map = torch.randn(10, 10, 10)
        loss = (diff_map**2).mean()
        assert loss.item() > 0.0

    @pytest.mark.unit
    def test_difference_loss_with_mask(self):
        """Mask should correctly select voxels."""
        torch.manual_seed(42)
        diff_map = torch.randn(10, 10, 10)
        mask = torch.zeros(10, 10, 10, dtype=torch.bool)
        mask[3:7, 3:7, 3:7] = True  # Central region

        masked_loss = (diff_map[mask] ** 2).mean()
        full_loss = (diff_map.flatten() ** 2).mean()

        # Masked and full losses should differ (different regions)
        assert masked_loss.item() != full_loss.item()


class TestMolecularMask:
    """Test molecular mask properties."""

    @pytest.mark.unit
    def test_mask_is_boolean(self):
        """Molecular mask should be a boolean tensor."""
        # Simulate what _build_molecular_mask does: invert a solvent mask
        solvent_mask = torch.rand(10, 10, 10) > 0.5
        molecular_mask = ~solvent_mask
        assert molecular_mask.dtype == torch.bool

    @pytest.mark.unit
    def test_mask_complement(self):
        """Molecular mask should be exact complement of solvent mask."""
        solvent_mask = torch.rand(10, 10, 10) > 0.5
        molecular_mask = ~solvent_mask
        # Together they should cover everything
        assert (solvent_mask | molecular_mask).all()
        # They should not overlap
        assert not (solvent_mask & molecular_mask).any()


# =============================================================================
# Integration Tests (real data, slower)
# =============================================================================


class TestRealSpaceTargetsIntegration:
    """Integration tests using real PDB/MTZ data."""

    @pytest.fixture
    def model_data_pair(self, sample_structure_pair):
        """Load real ModelFT and ReflectionData."""
        from torchref.model.model_ft import ModelFT
        from torchref.io import ReflectionData

        model = ModelFT(max_res=2.5, verbose=0)
        model.load_cif(str(sample_structure_pair["model"]))

        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))

        return model, data

    @pytest.mark.integration
    def test_correlation_target_forward(self, model_data_pair):
        """Correlation target forward pass should produce finite value."""
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        model, data = model_data_pair
        target = RealSpaceCorrelationTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        loss = target.forward()
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0  # 1 - RSCC >= 0 for positive correlation
        assert loss.item() < 2.0   # 1 - RSCC < 2

    @pytest.mark.integration
    def test_difference_target_forward(self, model_data_pair):
        """Difference target forward pass should produce finite positive value."""
        from torchref.experimental.targets import RealSpaceDifferenceTarget
        model, data = model_data_pair
        target = RealSpaceDifferenceTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        loss = target.forward()
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0  # Mean squared is always >= 0

    @pytest.mark.integration
    def test_correlation_target_gradient_flow(self, model_data_pair):
        """Gradients should flow through model parameters."""
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        model, data = model_data_pair
        target = RealSpaceCorrelationTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        # Zero existing gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        loss = target.forward()
        loss.backward()

        # At least some model parameters should have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to model parameters"

    @pytest.mark.integration
    def test_difference_target_gradient_flow(self, model_data_pair):
        """Gradients should flow through model parameters for difference target."""
        from torchref.experimental.targets import RealSpaceDifferenceTarget
        model, data = model_data_pair
        target = RealSpaceDifferenceTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        # Zero existing gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        loss = target.forward()
        loss.backward()

        # At least some model parameters should have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to model parameters"

    @pytest.mark.integration
    def test_correlation_stats(self, model_data_pair):
        """Stats should return expected keys."""
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        model, data = model_data_pair
        target = RealSpaceCorrelationTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        stats = target.stats()
        assert "loss" in stats
        assert "rscc" in stats
        assert "n_voxels" in stats

    @pytest.mark.integration
    def test_difference_stats(self, model_data_pair):
        """Stats should return expected keys."""
        from torchref.experimental.targets import RealSpaceDifferenceTarget
        model, data = model_data_pair
        target = RealSpaceDifferenceTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        stats = target.stats()
        assert "loss" in stats
        assert "rms_diff" in stats
        assert "mean_abs_diff" in stats
        assert "max_pos_peak" in stats
        assert "max_neg_peak" in stats
        assert "n_voxels" in stats

    @pytest.mark.integration
    def test_mask_shape_matches_grid(self, model_data_pair):
        """Molecular mask shape should match the density grid."""
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        model, data = model_data_pair
        target = RealSpaceCorrelationTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        # Trigger mask build
        mask = target._get_molecular_mask()
        gridsize = target._get_gridsize()

        assert mask.shape == tuple(gridsize)
        assert mask.dtype == torch.bool

    @pytest.mark.integration
    def test_update_mask(self, model_data_pair):
        """update_mask() should recompute the molecular mask."""
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        model, data = model_data_pair
        target = RealSpaceCorrelationTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        # Build initial mask
        mask1 = target._get_molecular_mask()
        n_protein_1 = mask1.sum().item()

        # Update mask (should be same since model hasn't changed)
        target.update_mask()
        mask2 = target._get_molecular_mask()
        n_protein_2 = mask2.sum().item()

        assert n_protein_1 == n_protein_2

    @pytest.mark.integration
    def test_no_mask_mode(self, model_data_pair):
        """Target should work without molecular mask."""
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        model, data = model_data_pair
        target = RealSpaceCorrelationTarget(
            data=data, model=model, mask_solvent=False, verbose=0,
        )

        loss = target.forward()
        assert torch.isfinite(loss)

    @pytest.mark.integration
    def test_register_with_loss_state(self, model_data_pair):
        """Target should integrate with LossState via register_target."""
        from torchref.experimental.targets import RealSpaceCorrelationTarget
        from torchref.refinement.loss_state import LossState

        model, data = model_data_pair
        target = RealSpaceCorrelationTarget(
            data=data, model=model, mask_solvent=True, verbose=0,
        )

        state = LossState()
        state.register_target(target.name, target)

        # Verify target was registered
        assert "realspace/correlation" in state.targets
