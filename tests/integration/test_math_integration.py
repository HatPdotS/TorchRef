"""
Integration tests for math functions.

Tests mathematical operations with real data scenarios.
"""

import pytest
import torch
import numpy as np


class TestScatteringVectors:
    """Tests for scattering vector calculations."""

    @pytest.mark.integration
    def test_get_scattering_vectors(self, sample_structure_pair):
        """Test computing scattering vectors from HKL and cell."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.math_functions.math_torch import get_scattering_vectors
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        s = get_scattering_vectors(data.hkl, data.cell)
        
        assert s is not None
        assert s.shape[0] == data.hkl.shape[0]
        assert s.shape[1] == 3
        assert torch.all(torch.isfinite(s))

    @pytest.mark.integration
    def test_scattering_vectors_batch(self):
        """Test scattering vectors with batch HKL indices."""
        from torchref.math_functions.math_torch import get_scattering_vectors
        
        # Create test HKL indices and cubic cell
        hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32)
        cell = torch.tensor([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=torch.float32)
        
        s = get_scattering_vectors(hkl, cell)
        
        assert s.shape == (4, 3)
        # For cubic cell, (1,0,0) should have s along a* direction
        assert torch.allclose(s[0, 1:], torch.zeros(2), atol=1e-6)


class TestRfactorCalculations:
    """Tests for R-factor calculations."""

    @pytest.mark.integration
    def test_get_rfactors(self):
        """Test R-factor computation with synthetic data."""
        from torchref.math_functions.math_torch import get_rfactors
        
        # Create synthetic data with known R-factor
        fobs = torch.tensor([100.0, 200.0, 150.0, 300.0], dtype=torch.float32)
        fcalc = torch.tensor([110.0, 190.0, 160.0, 280.0], dtype=torch.float32)  # ~10% error
        rfree_mask = torch.tensor([True, True, False, False])  # Work set mask
        
        r_work, r_free = get_rfactors(fobs, fcalc, rfree_mask)
        
        assert r_work is not None
        assert r_free is not None
        assert 0 <= r_work <= 1
        assert 0 <= r_free <= 1

    @pytest.mark.integration
    def test_bin_wise_rfactors(self):
        """Test bin-wise R-factor computation."""
        from torchref.math_functions.math_torch import bin_wise_rfactors
        
        # Create synthetic data
        n_refl = 100
        fobs = torch.rand(n_refl, dtype=torch.float32) * 100 + 10
        fcalc = fobs * (1 + torch.randn(n_refl) * 0.1)  # ~10% noise
        rfree_mask = torch.rand(n_refl) > 0.1  # 90% work set
        bins = torch.randint(0, 5, (n_refl,))
        
        r_work_bins, r_free_bins = bin_wise_rfactors(fobs, fcalc, rfree_mask, bins)
        
        assert r_work_bins is not None
        assert r_free_bins is not None


class TestNLLFunctions:
    """Tests for negative log-likelihood functions."""

    @pytest.mark.integration
    def test_nll_xray(self):
        """Test Gaussian NLL for X-ray data."""
        from torchref.math_functions.math_torch import nll_xray
        
        fobs = torch.tensor([100.0, 200.0, 150.0], dtype=torch.float32)
        fcalc = torch.tensor([105.0, 195.0, 155.0], dtype=torch.float32)
        sigma = torch.tensor([10.0, 15.0, 12.0], dtype=torch.float32)
        
        nll = nll_xray(fobs, fcalc, sigma)
        
        assert torch.isfinite(nll)
        assert nll > 0  # NLL should be positive

    @pytest.mark.integration
    def test_nll_xray_lognormal(self):
        """Test lognormal NLL for X-ray data."""
        from torchref.math_functions.math_torch import nll_xray_lognormal
        
        fobs = torch.tensor([100.0, 200.0, 150.0], dtype=torch.float32)
        fcalc = torch.tensor([105.0, 195.0, 155.0], dtype=torch.float32)
        sigma = torch.tensor([10.0, 15.0, 12.0], dtype=torch.float32)
        
        nll = nll_xray_lognormal(fobs, fcalc, sigma)
        
        assert torch.isfinite(nll)


class TestMatrixOperations:
    """Tests for matrix operations used in refinement."""

    @pytest.mark.integration
    def test_U_to_matrix(self):
        """Test converting 6-component U tensor to 3x3 matrix."""
        from torchref.math_functions.math_torch import U_to_matrix
        
        # 6 components: U11, U22, U33, U12, U13, U23
        u_params = torch.tensor([0.1, 0.1, 0.1, 0.0, 0.0, 0.0], dtype=torch.float32)
        
        U_matrix = U_to_matrix(u_params)
        
        assert U_matrix.shape == (3, 3)
        # Should be symmetric
        assert torch.allclose(U_matrix, U_matrix.T)
        # Diagonal should match input
        assert torch.allclose(U_matrix.diag(), u_params[:3])


class TestAtomExpansion:
    """Tests for atom expansion by symmetry."""

    @pytest.mark.integration
    def test_symmetry_matrices_for_expansion(self, sample_cif_file):
        """Test that symmetry provides matrices for expansion."""
        from torchref.model.model import Model
        from torchref.symmetry.symmetry import Symmetry
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz()
        sym = Symmetry(model.spacegroup)
        
        # Symmetry should have matrices that can be applied
        assert sym.matrices is not None
        assert sym.matrices.shape[0] >= 1


class TestResolutionCalculations:
    """Tests for resolution-related calculations."""

    @pytest.mark.integration
    def test_get_resolution_from_hkl(self, sample_structure_pair):
        """Test calculating resolution from HKL indices and cell."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # ReflectionData should have resolution values
        if hasattr(data, 'd') and data.d is not None:
            assert data.d.shape[0] == data.hkl.shape[0]
            assert torch.all(data.d > 0)
            # Check resolution range is reasonable
            d_min = data.d.min().item()
            d_max = data.d.max().item()
            assert d_min > 0.5  # Typical protein data doesn't go below 0.5 Å
            assert d_max < 500  # Should be less than 500 Å


class TestFrenchWilson:
    """Tests for French-Wilson algorithm."""

    @pytest.mark.integration
    def test_french_wilson_application(self, sample_mtz_file):
        """Test French-Wilson conversion is applied during MTZ loading."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # F values should all be non-negative after French-Wilson
        valid_F = data.F[~torch.isnan(data.F)]
        if len(valid_F) > 0:
            assert torch.all(valid_F >= 0)


class TestGradientFunctions:
    """Tests for gradient-related functions."""

    @pytest.mark.integration
    def test_scattering_vectors_computation(self):
        """Test that scattering vectors can be computed."""
        from torchref.math_functions.math_torch import get_scattering_vectors
        
        # Create test HKL and cell
        hkl = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        cell = torch.tensor([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=torch.float32)
        
        s = get_scattering_vectors(hkl, cell)
        
        assert s is not None
        assert s.shape == (2, 3)
        assert torch.all(torch.isfinite(s))
