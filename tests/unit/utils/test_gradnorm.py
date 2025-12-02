"""
Unit tests for torchref.utils.gradnorm

Tests gradient norm calculation utilities.
"""

import pytest
import torch
import torch.nn as nn


class TestGradNorm:
    """Tests for gradient norm calculation."""

    @pytest.mark.unit
    def test_gradnorm_basic(self):
        """Test basic gradient norm calculation."""
        from torchref.utils.gradnorm import gradnorm
        
        # Simple linear model
        model = nn.Linear(10, 1, bias=False)
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        # Forward pass
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        
        # Calculate gradient norm
        grad_norm = gradnorm(loss, model.parameters())
        
        assert isinstance(grad_norm, torch.Tensor)
        assert grad_norm.ndim == 0  # Scalar
        assert grad_norm >= 0  # Non-negative

    @pytest.mark.unit
    def test_gradnorm_zero_gradient(self):
        """Gradient norm should handle zero gradients."""
        from torchref.utils.gradnorm import gradnorm
        
        model = nn.Linear(10, 1, bias=False)
        
        # Create a loss that depends on the model but has zero gradient
        x = torch.randn(3, 10)
        pred = model(x)
        loss = (pred * 0.0).sum()  # Zero gradient
        # DON'T call backward before gradnorm - it calls backward internally
        
        grad_norm = gradnorm(loss, model.parameters())
        
        # Should be 0 (zero gradients)
        assert torch.isclose(grad_norm, torch.tensor(0.0, dtype=grad_norm.dtype), atol=1e-10)

    @pytest.mark.unit
    def test_gradnorm_multiple_params(self):
        """Test gradient norm with multiple parameter groups."""
        from torchref.utils.gradnorm import gradnorm
        
        # Model with multiple layers
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        x = torch.randn(3, 10)
        y = torch.randn(3, 1)
        
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        
        grad_norm = gradnorm(loss, model.parameters())
        
        assert isinstance(grad_norm, torch.Tensor)
        assert grad_norm >= 0
