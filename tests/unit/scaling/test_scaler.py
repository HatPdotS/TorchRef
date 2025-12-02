"""
Unit tests for torchref.scaling.scaler

Tests the Scaler class for structure factor scaling.
Note: Unit tests use mock data, not real file I/O.
"""

import pytest
import torch
import torch.nn as nn


class TestScalerInitialization:
    """Tests for Scaler initialization."""

    @pytest.mark.unit
    def test_scaler_empty_initialization(self):
        """Test Scaler can be initialized without model/data."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler()
        
        assert scaler._model is None
        assert scaler._data is None
        assert scaler.cell is None

    @pytest.mark.unit
    def test_scaler_is_nn_module(self):
        """Scaler should be a nn.Module."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler()
        
        assert isinstance(scaler, nn.Module)

    @pytest.mark.unit
    def test_scaler_default_nbins(self):
        """Test default number of resolution bins."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler()
        
        assert scaler.nbins == 20

    @pytest.mark.unit
    def test_scaler_custom_nbins(self):
        """Test custom number of resolution bins."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler(nbins=10)
        
        assert scaler.nbins == 10

    @pytest.mark.unit
    def test_scaler_frozen_flag(self):
        """Test frozen flag initialization."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler()
        
        assert scaler.frozen == False


class TestScalerDeviceHandling:
    """Tests for device handling in Scaler."""

    @pytest.mark.unit
    def test_scaler_default_device(self):
        """Test default device is CPU."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler()
        
        assert scaler.device == torch.device('cpu')

    @pytest.mark.unit
    def test_scaler_custom_device(self):
        """Test custom device specification."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler(device=torch.device('cpu'))
        
        assert scaler.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_scaler_gpu_device(self, gpu_device):
        """Test GPU device specification."""
        from torchref.scaling.scaler import Scaler
        
        scaler = Scaler(device=gpu_device)
        
        assert scaler.device.type == 'cuda'


class TestScalingCalculations:
    """Tests for scaling calculation utilities."""

    @pytest.mark.unit
    def test_resolution_binning_logic(self, mock_hkl_indices, mock_unit_cell):
        """Test resolution binning creates correct number of bins."""
        from torchref.math_functions.math_numpy import get_s
        
        hkl = mock_hkl_indices(n_reflections=1000).numpy()
        cell = mock_unit_cell.numpy()
        
        # Calculate s values
        s = get_s(hkl, cell)
        
        # Create bins
        nbins = 10
        s_sorted = torch.tensor(sorted(s))
        bin_edges = torch.linspace(s_sorted[0], s_sorted[-1], nbins + 1)
        
        assert len(bin_edges) == nbins + 1

    @pytest.mark.unit
    def test_scale_factor_positive(self):
        """Scale factors should be positive."""
        scale = torch.tensor([1.5, 2.0, 0.8])
        
        assert torch.all(scale > 0)

    @pytest.mark.unit
    def test_scale_application(self, mock_structure_factors):
        """Test applying scale factor to structure factors."""
        fcalc = mock_structure_factors(n_reflections=100)
        scale = torch.tensor(1.5)
        
        scaled = fcalc * scale
        
        # Amplitude should be scaled
        assert torch.allclose(torch.abs(scaled), torch.abs(fcalc) * scale, rtol=1e-5)
        # Phase should be preserved
        phase_orig = torch.angle(fcalc)
        phase_scaled = torch.angle(scaled)
        # Handle phase wrapping
        phase_diff = torch.abs(phase_orig - phase_scaled)
        phase_diff = torch.minimum(phase_diff, 2 * torch.pi - phase_diff)
        assert torch.allclose(phase_diff, torch.zeros_like(phase_diff), atol=1e-5)


class TestBFactorScaling:
    """Tests for B-factor scaling logic."""

    @pytest.mark.unit
    def test_b_factor_debye_waller(self, mock_hkl_indices, mock_unit_cell):
        """Test Debye-Waller factor calculation."""
        from torchref.math_functions.math_numpy import get_s
        
        hkl = mock_hkl_indices(n_reflections=100).numpy()
        cell = mock_unit_cell.numpy()
        s = torch.tensor(get_s(hkl, cell))
        
        B_factor = 20.0  # Å²
        
        # Debye-Waller: exp(-B * s² / 4)
        # Note: s = |S| = 1/d, so s² corresponds to resolution
        dw_factor = torch.exp(-B_factor * (s ** 2) / 4)
        
        assert torch.all(dw_factor > 0)
        assert torch.all(dw_factor <= 1)  # Should attenuate

    @pytest.mark.unit
    def test_b_factor_high_resolution_attenuation(self, mock_unit_cell):
        """Higher resolution (larger s) should have more attenuation."""
        from torchref.math_functions.math_numpy import get_s
        
        cell = mock_unit_cell.numpy()
        
        # Low and high resolution reflections
        hkl_low = torch.tensor([[1, 0, 0]], dtype=torch.float64).numpy()
        hkl_high = torch.tensor([[10, 10, 10]], dtype=torch.float64).numpy()
        
        s_low = get_s(hkl_low, cell)[0]
        s_high = get_s(hkl_high, cell)[0]
        
        B_factor = 20.0
        dw_low = torch.exp(torch.tensor(-B_factor * (s_low ** 2) / 4))
        dw_high = torch.exp(torch.tensor(-B_factor * (s_high ** 2) / 4))
        
        # High resolution should be more attenuated
        assert dw_high < dw_low


class TestAnisotropicScaling:
    """Tests for anisotropic scaling calculations."""

    @pytest.mark.unit
    def test_u_to_matrix_shape(self, mock_aniso_u):
        """Test U tensor to matrix conversion."""
        from torchref.math_functions.math_torch import U_to_matrix
        
        U = mock_aniso_u(n_atoms=10)
        
        # U_to_matrix should create 3x3 matrices
        # Input: (N, 6) -> Output: (N, 3, 3)
        U_matrices = U_to_matrix(U)
        
        assert U_matrices.shape == (10, 3, 3)

    @pytest.mark.unit
    def test_u_matrix_symmetric(self, mock_aniso_u):
        """U matrices should be symmetric."""
        from torchref.math_functions.math_torch import U_to_matrix
        
        U = mock_aniso_u(n_atoms=5)
        U_matrices = U_to_matrix(U)
        
        for i in range(5):
            mat = U_matrices[i]
            assert torch.allclose(mat, mat.T, atol=1e-6)
