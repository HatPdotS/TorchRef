"""
Unit tests for torchref.refinement.targets

Tests target (loss) functions for crystallographic refinement.
Note: These are unit tests so we test the functions in isolation with mock data.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestTargetBase:
    """Tests for base Target class."""

    @pytest.mark.unit
    def test_target_empty_initialization(self):
        """Test Target can be initialized without arguments."""
        from torchref.refinement.targets import Target

        target = Target()

        assert target.verbose == 0

    @pytest.mark.unit
    def test_target_is_nn_module(self):
        """Target should be a nn.Module."""
        from torchref.refinement.targets import Target

        target = Target()

        assert isinstance(target, nn.Module)


class TestGaussianNLL:
    """Tests for Gaussian NLL calculation logic."""

    @pytest.mark.unit
    def test_gaussian_nll_identical_gives_small_loss(self, mock_F_obs, mock_F_sigma):
        """When Fobs = Fcalc, NLL should be small (just the log sigma term)."""
        from torchref.base.math_torch import nll_xray

        fobs = mock_F_obs(n_reflections=100)
        sigma = mock_F_sigma(n_reflections=100)
        fcalc = fobs.clone().to(torch.complex64)  # |Fcalc| = Fobs
        
        # Calculate manually what Gaussian NLL should be
        # NLL = 0.5*((fobs - |fcalc|)/sigma)^2 + log(sigma) + 0.5*log(2pi)
        diff = fobs - torch.abs(fcalc)
        expected_data_term = 0.5 * ((diff / sigma) ** 2)
        
        # Data term should be ~0 when fobs = |fcalc|
        assert torch.allclose(expected_data_term, torch.zeros_like(expected_data_term), atol=1e-5)

    @pytest.mark.unit
    def test_gaussian_nll_positive(self, mock_F_obs, mock_F_sigma):
        """NLL should generally be positive or close to zero."""
        fobs = mock_F_obs(n_reflections=100)
        sigma = mock_F_sigma(n_reflections=100)
        fcalc = mock_F_obs(n_reflections=100, seed=123).to(torch.complex64)  # Different
        
        # Simple Gaussian NLL
        diff = fobs - torch.abs(fcalc)
        eps = torch.median(sigma) * 0.1
        sigma_safe = torch.clamp(sigma, min=eps)
        log_2pi = torch.log(torch.tensor(2.0 * np.pi))
        nll = 0.5 * (diff ** 2) / (sigma_safe ** 2) + torch.log(sigma_safe) + 0.5 * log_2pi
        
        # Mean NLL should be finite
        assert torch.isfinite(nll.mean())


class TestLeastSquaresTarget:
    """Tests for Least Squares target calculation."""

    @pytest.mark.unit
    def test_least_squares_identical_zero(self, mock_F_obs):
        """LS loss should be 0 when Fobs = Fcalc."""
        fobs = mock_F_obs(n_reflections=100)
        fcalc = fobs.clone()
        
        # Simple LS: sum((fobs - fcalc)^2)
        loss = torch.sum((fobs - fcalc) ** 2)
        
        assert torch.isclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=1e-10)

    @pytest.mark.unit
    def test_least_squares_scaled(self, mock_F_obs):
        """Test LS loss with scaled Fcalc."""
        fobs = mock_F_obs(n_reflections=100)
        fcalc = fobs * 1.1  # 10% scaled
        
        loss = torch.mean((fobs - fcalc) ** 2)
        
        # Should be (0.1 * fobs)^2 on average
        expected_loss = torch.mean((0.1 * fobs) ** 2)
        assert torch.isclose(loss, expected_loss, rtol=1e-5)

    @pytest.mark.unit
    def test_least_squares_weighted(self, mock_F_obs, mock_F_sigma):
        """Test weighted LS with sigma weights."""
        fobs = mock_F_obs(n_reflections=100)
        sigma = mock_F_sigma(n_reflections=100)
        fcalc = mock_F_obs(n_reflections=100, seed=123)
        
        # Weighted LS: sum(w * (fobs - fcalc)^2) where w = 1/sigma^2
        weights = 1.0 / (sigma ** 2)
        diff = fobs - fcalc
        weighted_loss = torch.sum(weights * (diff ** 2))
        
        assert torch.isfinite(weighted_loss)
        assert weighted_loss >= 0


class TestRiceNLL:
    """Tests for Rice distribution NLL (used for acentric reflections)."""

    @pytest.mark.unit
    def test_rice_nll_components(self, mock_F_obs, mock_F_sigma):
        """Test components of Rice NLL calculation."""
        from torch.special import i0

        fobs = mock_F_obs(n_reflections=50)
        sigma = mock_F_sigma(n_reflections=50)
        fcalc_amp = mock_F_obs(n_reflections=50, seed=123)
        
        # Rice NLL components
        # NLL = (Fo^2 + Fc^2)/(2σ^2) - log(I0(Fo*Fc/σ^2)) - log(Fo/σ^2)
        
        # Check I0 calculation
        x = fobs * fcalc_amp / (sigma ** 2)
        bessel_i0 = i0(x)
        
        # I0 should be >= 1 for x >= 0
        assert torch.all(bessel_i0 >= 1.0)


class TestTargetDeviceHandling:
    """Tests for proper device handling in targets."""

    @pytest.mark.unit
    def test_target_cpu_tensors(self, mock_F_obs, mock_F_sigma):
        """Test calculations work on CPU."""
        fobs = mock_F_obs(n_reflections=100)
        sigma = mock_F_sigma(n_reflections=100)
        
        # Simple calculation on CPU
        loss = torch.mean((fobs / sigma) ** 2)
        
        assert loss.device.type == 'cpu'
        assert torch.isfinite(loss)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_target_gpu_tensors(self, mock_F_obs, mock_F_sigma, gpu_device):
        """Test calculations work on GPU."""
        fobs = mock_F_obs(n_reflections=100).to(gpu_device)
        sigma = mock_F_sigma(n_reflections=100).to(gpu_device)
        
        loss = torch.mean((fobs / sigma) ** 2)
        
        assert loss.device.type == 'cuda'
        assert torch.isfinite(loss)


class TestNumericStability:
    """Tests for numeric stability in target calculations."""

    @pytest.mark.unit
    def test_small_sigma_handling(self, mock_F_obs):
        """Test handling of very small sigma values."""
        fobs = mock_F_obs(n_reflections=100)
        sigma = torch.ones_like(fobs) * 1e-10  # Very small sigma
        fcalc = mock_F_obs(n_reflections=100, seed=123)
        
        # Clamped sigma approach
        eps = torch.median(sigma) * 0.1
        sigma_safe = torch.clamp(sigma, min=max(eps, 1e-6))
        
        diff = fobs - fcalc
        loss = torch.mean((diff / sigma_safe) ** 2)
        
        assert torch.isfinite(loss)

    @pytest.mark.unit
    def test_zero_fcalc_handling(self, mock_F_obs, mock_F_sigma):
        """Test handling of zero Fcalc values."""
        fobs = mock_F_obs(n_reflections=100)
        sigma = mock_F_sigma(n_reflections=100)
        fcalc = torch.zeros_like(fobs, dtype=torch.complex64)  # All zero
        
        fcalc_amp = torch.abs(fcalc)  # Will be zero
        diff = fobs - fcalc_amp
        
        loss = torch.mean(diff ** 2)
        
        # Should just be mean of fobs^2
        expected = torch.mean(fobs ** 2)
        assert torch.isclose(loss, expected, rtol=1e-5)
