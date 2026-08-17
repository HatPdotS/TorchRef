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
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel()
        assert solvent is not None
        assert solvent.model is None
        assert solvent.solvent_radius == 1.1  # Default

    def test_initialization_with_parameters(self):
        """Test SolventModel with custom parameters."""
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel(
            k_solvent=0.35,
            d_half=3.6,
            n_exp=5.0,
            radius=1.2,
            erosion_radius=0.8
        )

        assert solvent.solvent_radius == 1.2
        assert solvent.erosion_radius == 0.8
        # every falloff parameter is stored as its log
        assert torch.isfinite(solvent.log_k_solvent)
        assert torch.isclose(solvent.n_exp(), torch.tensor(5.0))
        assert torch.isclose(
            solvent.ss_half(), torch.tensor(1.0 / (4 * 3.6 ** 2)), rtol=1e-5
        )


@pytest.mark.integration
class TestSolventParameters:
    """Test SolventModel parameter access."""

    def test_k_solvent_property(self):
        """Test k_solvent conversion from log."""
        from torchref.scaling.solvent import SolventModel
        
        k_solvent_initial = 0.35
        solvent = SolventModel(k_solvent=k_solvent_initial)
        
        # log_k_solvent should give back k_solvent via exp
        k_recovered = torch.exp(solvent.log_k_solvent)
        assert torch.isclose(k_recovered, torch.tensor(k_solvent_initial), rtol=1e-5)

    def test_falloff_parameters(self):
        """The falloff half-point and exponent round-trip through their logs."""
        from torchref.scaling.solvent import SolventModel

        solvent = SolventModel(d_half=4.0, n_exp=3.0)

        assert torch.isclose(
            solvent.ss_half(), torch.tensor(1.0 / 64.0), rtol=1e-5
        )
        assert torch.isclose(solvent.n_exp(), torch.tensor(3.0), rtol=1e-5)

    def test_phase_offset_parameter(self):
        """Test phase offset parameter."""
        from torchref.scaling.solvent import SolventModel
        
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
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel(k_solvent=0.35)
        
        # Create simple loss
        loss = solvent.log_k_solvent.sum()
        loss.backward()
        
        assert solvent.log_k_solvent.grad is not None

    def test_falloff_gradients(self):
        """Test gradients for the falloff parameters."""
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel(k_solvent=0.35)
        
        # Create simple loss
        ss = torch.linspace(1e-4, 0.25, 64, device=solvent.device)
        loss = solvent.damping(ss).sum()
        loss.backward()

        assert solvent.log_ss_half.grad is not None
        assert solvent.log_n_exp.grad is not None


@pytest.mark.integration
class TestSolventStateDictFunctional:
    """Test SolventModel state dict operations."""

    def test_state_dict_keys(self):
        """Test state dict contains expected keys."""
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel(k_solvent=0.35, optimize_phase=True)
        state_dict = solvent.state_dict()
        
        # Should contain parameters
        assert 'log_k_solvent' in state_dict
        assert 'log_ss_half' in state_dict
        assert 'log_n_exp' in state_dict
        assert 'phase_offset' in state_dict

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state dict."""
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel(k_solvent=0.35)
        original_k = solvent.log_k_solvent.clone()
        original_ss = solvent.log_ss_half.clone()
        
        # Save
        state_dict = solvent.state_dict()
        torch.save(state_dict, tmp_path / "solvent.pt")
        
        # Load into new model
        solvent2 = SolventModel()
        solvent2.load_state_dict(torch.load(tmp_path / "solvent.pt"))
        
        assert torch.isclose(solvent2.log_k_solvent, original_k)
        assert torch.isclose(solvent2.log_ss_half, original_ss)


@pytest.mark.integration
class TestSolventDeviceOperations:
    """Test SolventModel device operations."""

    def test_cpu_operation(self):
        """Test solvent model on CPU."""
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel(device=torch.device('cpu'))
        
        assert solvent.log_k_solvent.device.type == 'cpu'
        assert solvent.log_ss_half.device.type == 'cpu'

    def test_float_type(self):
        """Test solvent model with different float types."""
        from torchref.scaling.solvent import SolventModel
        
        # Float32
        solvent32 = SolventModel(float_type=torch.float32)
        assert solvent32.log_k_solvent.dtype == torch.float32
        
        # Float64 — pinned to CPU: MPS has no float64, so the default device
        # cannot hold a float64 tensor on Apple silicon.
        solvent64 = SolventModel(float_type=torch.float64, device=torch.device("cpu"))
        assert solvent64.log_k_solvent.dtype == torch.float64


@pytest.mark.integration
class TestSolventCacheOperations:
    """Test SolventModel caching mechanism."""

    def test_cache_initialization(self):
        """Test that cache is initialized."""
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel()
        
        assert solvent._cache is not None


@pytest.mark.integration
class TestSolventBFactorCorrection:
    """Test B-factor correction in solvent model."""

    def test_exponent_one_is_exactly_a_debye_waller_factor(self):
        """The nesting gate: ``n = 1`` must reproduce ``exp(-B ss)`` to machine
        precision, with ``B = ln2 / ss_half``. That is what makes the fitted form a
        strict generalisation of the shipped exponential rather than a replacement, so
        it cannot do worse than it at the optimum."""
        from torchref.scaling.solvent import SolventModel, _LN2

        for b_target in (20.0, 46.0, 90.0):
            ss_half = _LN2 / b_target
            d_half = 1.0 / (2 * ss_half ** 0.5)
            solvent = SolventModel(d_half=d_half, n_exp=1.0, device=torch.device("cpu"))

            ss = torch.linspace(0.0, 0.25, 128, dtype=solvent.float_type)
            assert torch.allclose(
                solvent.damping(ss), torch.exp(-b_target * ss), atol=1e-5
            ), f"n=1 does not reproduce exp(-{b_target} ss)"

    def test_damping_is_monotone_and_starts_at_one(self):
        from torchref.scaling.solvent import SolventModel

        solvent = SolventModel(device=torch.device("cpu"))
        ss = torch.linspace(0.0, 0.3, 64, dtype=solvent.float_type)
        d = solvent.damping(ss)

        assert torch.isclose(d[0], torch.tensor(1.0, dtype=d.dtype), atol=1e-6)
        assert torch.all(d.diff() <= 1e-7)
        assert torch.all((d >= 0.0) & (d <= 1.0))

    def test_falloff_parameters_are_bounded(self):
        """The unbounded 3-parameter fit reaches degenerate slow power laws on some
        structures; the clamps are what keep the fitted curve a switch."""
        from torchref.scaling.solvent import (
            N_EXP_BOUNDS, SS_HALF_BOUNDS, SolventModel,
        )

        solvent = SolventModel(device=torch.device("cpu"))
        with torch.no_grad():
            solvent.log_ss_half.fill_(10.0)
            solvent.log_n_exp.fill_(-10.0)
        assert torch.isclose(
            solvent.ss_half(), torch.tensor(SS_HALF_BOUNDS[1], dtype=solvent.float_type)
        )
        assert torch.isclose(
            solvent.n_exp(), torch.tensor(N_EXP_BOUNDS[0], dtype=solvent.float_type)
        )

        with torch.no_grad():
            solvent.log_ss_half.fill_(-10.0)
            solvent.log_n_exp.fill_(10.0)
        assert torch.isclose(
            solvent.ss_half(), torch.tensor(SS_HALF_BOUNDS[0], dtype=solvent.float_type)
        )
        assert torch.isclose(
            solvent.n_exp(), torch.tensor(N_EXP_BOUNDS[1], dtype=solvent.float_type)
        )

    def test_b_solvent_equivalent_recovers_a_true_debye_waller(self):
        """The reported ``B_SOL`` is back-fitted, so at ``n = 1`` -- where the curve IS
        an exponential -- it must recover that exponential's own B."""
        from torchref.scaling.solvent import SolventModel, _LN2

        b_target = 46.0
        d_half = 1.0 / (2 * (_LN2 / b_target) ** 0.5)
        solvent = SolventModel(d_half=d_half, n_exp=1.0, device=torch.device("cpu"))

        ss = torch.linspace(1e-4, 0.25, 512, dtype=solvent.float_type)
        assert abs(solvent.b_solvent_equivalent(ss) - b_target) < 0.5

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

    @pytest.mark.parametrize("k", [0.3, 0.35, 0.4, 0.45, 0.5])
    def test_typical_k_solvent_range(self, k):
        """Test typical k_solvent values."""
        from torchref.scaling.solvent import SolventModel

        solvent = SolventModel(k_solvent=k)
        k_recovered = torch.exp(solvent.log_k_solvent)
        assert torch.isclose(k_recovered, torch.tensor(k), rtol=1e-5)

    @pytest.mark.parametrize("d_half", [2.5, 3.6, 4.5, 6.0, 10.0])
    def test_typical_d_half_range(self, d_half):
        """Every value inside the bounds must survive the clamp untouched."""
        from torchref.scaling.solvent import SolventModel

        solvent = SolventModel(d_half=d_half)
        assert torch.isclose(
            solvent.ss_half(), torch.tensor(1.0 / (4 * d_half ** 2)), rtol=1e-4
        )

    def test_default_parameters(self):
        """Test default parameter values are reasonable."""
        from torchref.scaling.solvent import SolventModel
        
        solvent = SolventModel()
        
        # Defaults should be within reasonable range
        k = torch.exp(solvent.log_k_solvent)
        d_half = 1.0 / (2 * solvent.ss_half().sqrt())

        assert 0.1 <= k <= 2.0
        assert 2.0 <= d_half <= 12.0
        assert 1.0 <= solvent.n_exp() <= 20.0


@pytest.mark.integration
class TestSolventMathOperations:
    """Test mathematical operations used in solvent modeling."""

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
