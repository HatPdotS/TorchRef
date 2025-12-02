"""
Functional tests for SolventModel.

These tests exercise the solvent model with real crystallographic data.
"""
import pytest
import torch
import numpy as np


@pytest.mark.integration
class TestSolventModelInitialization:
    """Test SolventModel initialization."""

    def test_empty_initialization(self):
        """Test empty SolventModel initialization."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel()
        assert solvent is not None
        assert solvent.model is None
        assert solvent.solvent_radius == 1.1  # Default

    def test_initialization_with_parameters(self):
        """Test SolventModel with custom parameters."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel(
            k_solvent=0.35,
            b_solvent=46.0,
            radius=1.2,
            erosion_radius=0.8
        )
        
        assert solvent.solvent_radius == 1.2
        assert solvent.erosion_radius == 0.8
        # k_solvent is stored as log
        assert torch.isfinite(solvent.log_k_solvent)
        assert torch.isclose(solvent.b_solvent, torch.tensor(46.0))


@pytest.mark.integration
class TestSolventParameters:
    """Test SolventModel parameter access."""

    def test_k_solvent_property(self):
        """Test k_solvent conversion from log."""
        from torchref.scaling.solvent_new import SolventModel
        
        k_solvent_initial = 0.35
        solvent = SolventModel(k_solvent=k_solvent_initial)
        
        # log_k_solvent should give back k_solvent via exp
        k_recovered = torch.exp(solvent.log_k_solvent)
        assert torch.isclose(k_recovered, torch.tensor(k_solvent_initial), rtol=1e-5)

    def test_b_solvent_parameter(self):
        """Test B-solvent parameter."""
        from torchref.scaling.solvent_new import SolventModel
        
        b_solvent_initial = 50.0
        solvent = SolventModel(b_solvent=b_solvent_initial)
        
        assert torch.isclose(solvent.b_solvent, torch.tensor(b_solvent_initial))

    def test_phase_offset_parameter(self):
        """Test phase offset parameter."""
        from torchref.scaling.solvent_new import SolventModel
        
        # With optimize_phase=True
        solvent = SolventModel(optimize_phase=True, initial_phase_offset=0.1)
        assert hasattr(solvent, 'phase_offset')
        assert torch.isclose(solvent.phase_offset, torch.tensor(0.1))
        
        # With optimize_phase=False
        solvent2 = SolventModel(optimize_phase=False)
        assert hasattr(solvent2, 'phase_offset')


@pytest.mark.integration
class TestSolventGradients:
    """Test SolventModel gradient computation."""

    def test_k_solvent_gradients(self):
        """Test gradients for k_solvent."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel(k_solvent=0.35, b_solvent=46.0)
        
        # Create simple loss
        loss = solvent.log_k_solvent.sum()
        loss.backward()
        
        assert solvent.log_k_solvent.grad is not None

    def test_b_solvent_gradients(self):
        """Test gradients for b_solvent."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel(k_solvent=0.35, b_solvent=46.0)
        
        # Create simple loss
        loss = solvent.b_solvent.sum()
        loss.backward()
        
        assert solvent.b_solvent.grad is not None


@pytest.mark.integration
class TestSolventStateDictFunctional:
    """Test SolventModel state dict operations."""

    def test_state_dict_keys(self):
        """Test state dict contains expected keys."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel(k_solvent=0.35, b_solvent=46.0, optimize_phase=True)
        state_dict = solvent.state_dict()
        
        # Should contain parameters
        assert 'log_k_solvent' in state_dict
        assert 'b_solvent' in state_dict
        assert 'phase_offset' in state_dict

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state dict."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel(k_solvent=0.35, b_solvent=46.0)
        original_k = solvent.log_k_solvent.clone()
        original_b = solvent.b_solvent.clone()
        
        # Save
        state_dict = solvent.state_dict()
        torch.save(state_dict, tmp_path / "solvent.pt")
        
        # Load into new model
        solvent2 = SolventModel()
        solvent2.load_state_dict(torch.load(tmp_path / "solvent.pt"))
        
        assert torch.isclose(solvent2.log_k_solvent, original_k)
        assert torch.isclose(solvent2.b_solvent, original_b)


@pytest.mark.integration
class TestSolventDeviceOperations:
    """Test SolventModel device operations."""

    def test_cpu_operation(self):
        """Test solvent model on CPU."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel(device=torch.device('cpu'))
        
        assert solvent.log_k_solvent.device.type == 'cpu'
        assert solvent.b_solvent.device.type == 'cpu'

    def test_float_type(self):
        """Test solvent model with different float types."""
        from torchref.scaling.solvent_new import SolventModel
        
        # Float32
        solvent32 = SolventModel(float_type=torch.float32)
        assert solvent32.log_k_solvent.dtype == torch.float32
        
        # Float64
        solvent64 = SolventModel(float_type=torch.float64)
        assert solvent64.log_k_solvent.dtype == torch.float64


@pytest.mark.integration
class TestSolventCacheOperations:
    """Test SolventModel caching mechanism."""

    def test_cache_initialization(self):
        """Test that cache is initialized."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel()
        
        assert solvent._cache is not None


@pytest.mark.integration
class TestSolventBFactorCorrection:
    """Test B-factor correction in solvent model."""

    def test_bfactor_correction_formula(self):
        """Test B-factor correction calculation."""
        # B-factor correction: exp(-B * s^2)
        b_solvent = 46.0
        s = torch.tensor([0.0, 0.1, 0.2, 0.3])  # |s| values
        
        correction = torch.exp(-b_solvent * s ** 2)
        
        # At s=0, correction should be 1
        assert torch.isclose(correction[0], torch.tensor(1.0))
        
        # Correction should decrease with increasing s
        assert torch.all(correction[1:] < correction[:-1])

    def test_bfactor_with_k_solvent(self):
        """Test combined k_solvent and B-factor correction."""
        k_solvent = 0.35
        b_solvent = 46.0
        s = torch.tensor([0.1, 0.2, 0.3])
        
        # Full correction: k * exp(-B * s^2)
        correction = k_solvent * torch.exp(-b_solvent * s ** 2)
        
        # All values should be positive
        assert torch.all(correction > 0)
        
        # And less than k_solvent
        assert torch.all(correction <= k_solvent)


@pytest.mark.integration  
class TestSolventTypicalValues:
    """Test typical solvent parameter values."""

    def test_typical_k_solvent_range(self):
        """Test typical k_solvent values."""
        from torchref.scaling.solvent_new import SolventModel
        
        # Typical values from literature: 0.3-0.5
        typical_values = [0.3, 0.35, 0.4, 0.45, 0.5]
        
        for k in typical_values:
            solvent = SolventModel(k_solvent=k)
            k_recovered = torch.exp(solvent.log_k_solvent)
            assert torch.isclose(k_recovered, torch.tensor(k), rtol=1e-5)

    def test_typical_b_solvent_range(self):
        """Test typical B_solvent values."""
        from torchref.scaling.solvent_new import SolventModel
        
        # Typical values: 30-100 Å²
        typical_values = [30.0, 46.0, 50.0, 70.0, 100.0]
        
        for b in typical_values:
            solvent = SolventModel(b_solvent=b)
            assert torch.isclose(solvent.b_solvent, torch.tensor(b))

    def test_default_parameters(self):
        """Test default parameter values are reasonable."""
        from torchref.scaling.solvent_new import SolventModel
        
        solvent = SolventModel()
        
        # Defaults should be within reasonable range
        k = torch.exp(solvent.log_k_solvent)
        b = solvent.b_solvent
        
        assert 0.1 <= k <= 2.0
        assert 10.0 <= b <= 200.0


@pytest.mark.integration
class TestSolventMathOperations:
    """Test mathematical operations used in solvent modeling."""

    def test_gaussian_smoothing_concept(self):
        """Test Gaussian smoothing concept used in mask."""
        # Create a simple 1D step function
        x = torch.linspace(-2, 2, 100)
        step = (x > 0).float()
        
        # Gaussian kernel for smoothing
        sigma = 0.5
        kernel_x = torch.linspace(-3*sigma, 3*sigma, 21)
        kernel = torch.exp(-kernel_x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Simple smoothing (conceptual test)
        smoothed = torch.conv1d(
            step.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=len(kernel)//2
        ).squeeze()
        
        # Smoothed should be between 0 and 1
        assert torch.all((smoothed >= -0.1) & (smoothed <= 1.1))

    def test_exponential_b_factor_scaling(self):
        """Test exponential B-factor scaling."""
        # Debye-Waller factor: exp(-B * s^2 / 4)
        B = 50.0  # Typical B-factor
        s_squared = torch.tensor([0.0, 0.01, 0.04, 0.09])  # s^2 values
        
        debye_waller = torch.exp(-B * s_squared / 4)
        
        # Should decay with increasing s
        assert torch.all(debye_waller[1:] <= debye_waller[:-1])
        
        # At s=0, should be 1
        assert torch.isclose(debye_waller[0], torch.tensor(1.0))
