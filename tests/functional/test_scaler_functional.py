"""
Functional tests for the Scaler module.

Tests scaler operations with real model and reflection data.
"""

import pytest
import torch
import numpy as np


class TestScalerCreationFunctional:
    """Functional tests for scaler creation with real data."""

    @pytest.mark.integration
    def test_scaler_full_initialization(self, sample_structure_pair):
        """Test full scaler initialization with model and data."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=20, verbose=0)
        
        # Check all components are initialized
        assert scaler.model is not None
        assert scaler._data is not None
        assert scaler.s is not None
        assert scaler.bins is not None
        assert scaler.nbins == 20
        assert scaler.cell is not None

    @pytest.mark.integration
    @pytest.mark.parametrize("nbins", [5, 10, 15, 20])
    def test_scaler_with_different_nbins(self, sample_structure_pair, nbins):
        """Test scaler with different bin counts."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler

        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))

        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))

        scaler = Scaler(model=model, data=data, nbins=nbins, verbose=0)
        assert scaler.nbins == nbins
        # Verify bin values are in valid range
        assert scaler.bins.min() >= 0
        assert scaler.bins.max() < nbins


class TestScatteringVectorsFunctional:
    """Functional tests for scattering vector calculations."""

    @pytest.mark.integration
    def test_scattering_vectors_shape(self, sample_structure_pair):
        """Test scattering vectors have correct shape."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        # s should have shape (n_reflections, 3)
        n_refl = data.hkl.shape[0]
        assert scaler.s.shape == (n_refl, 3)

    @pytest.mark.integration
    def test_scattering_vectors_magnitude(self, sample_structure_pair):
        """Test scattering vector magnitudes are reasonable."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        # Calculate |s| = sin(theta)/lambda = 1/(2d)
        s_mag = torch.norm(scaler.s, dim=1)
        
        # For typical protein data:
        # Low resolution (d=100Å): |s| ~ 0.005
        # High resolution (d=1Å): |s| ~ 0.5
        assert torch.all(s_mag >= 0)
        assert torch.all(s_mag < 1.0)  # No data beyond 0.5Å resolution typically


class TestAnisotropyCorrectionFunctional:
    """Functional tests for anisotropy correction."""

    @pytest.mark.integration
    def test_anisotropy_setup_and_compute(self, sample_structure_pair):
        """Test setting up and computing anisotropy correction."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        
        # U parameters should exist
        assert hasattr(scaler, 'U')
        assert scaler.U.shape == (6,)  # U11, U22, U33, U12, U13, U23
        
        # Compute correction
        correction = scaler.anisotropy_correction()
        
        # Correction should be positive (exponential)
        assert correction.shape[0] == data.hkl.shape[0]
        assert torch.all(correction > 0)
        assert torch.all(torch.isfinite(correction))

    @pytest.mark.integration
    def test_anisotropy_correction_near_unity(self, sample_structure_pair):
        """Test anisotropy correction starts near unity with small U."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        
        # With small random U values, correction should be close to 1
        correction = scaler.anisotropy_correction()
        
        # Most values should be between 0.5 and 2.0 for small U
        mean_correction = correction.mean().item()
        assert 0.5 < mean_correction < 2.0


class TestBinwiseBfactorFunctional:
    """Functional tests for bin-wise B-factor correction."""

    @pytest.mark.integration
    def test_setup_binwise_bfactor(self, sample_structure_pair):
        """Test setting up bin-wise B-factor parameters."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_bin_wise_bfactor()
        
        assert hasattr(scaler, 'bin_wise_bfactor')
        assert scaler.bin_wise_bfactor.shape == (10,)
        # Initially should be zeros
        assert torch.allclose(scaler.bin_wise_bfactor, torch.zeros(10))

    @pytest.mark.integration
    def test_binwise_bfactor_correction(self, sample_structure_pair):
        """Test computing bin-wise B-factor correction."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_bin_wise_bfactor()
        
        # Set some non-zero B-factors
        scaler.bin_wise_bfactor.data = torch.linspace(0, 20, 10)
        
        correction = scaler.bin_wise_bfactor_correction()
        
        # Correction should have same length as reflections
        assert correction.shape[0] == data.hkl.shape[0]
        # Should be positive (exponential)
        assert torch.all(correction > 0)
        assert torch.all(torch.isfinite(correction))


class TestScalerStateDictFunctional:
    """Functional tests for scaler state dict operations."""

    @pytest.mark.integration
    def test_save_and_load_state_dict(self, sample_structure_pair, tmp_path):
        """Test saving and loading scaler state."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Create scaler with some setup
        scaler1 = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler1.setup_anisotropy_correction()
        scaler1.setup_bin_wise_bfactor()
        
        # Modify parameters
        scaler1.U.data = torch.randn(6)
        scaler1.bin_wise_bfactor.data = torch.randn(10)
        
        # Save state
        state_path = tmp_path / "scaler_state.pt"
        torch.save(scaler1.state_dict(), state_path)
        
        # Create new scaler and load state
        scaler2 = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler2.setup_anisotropy_correction()
        scaler2.setup_bin_wise_bfactor()
        scaler2.load_state_dict(torch.load(state_path, weights_only=False))
        
        # Parameters should match
        assert torch.allclose(scaler1.U, scaler2.U)
        assert torch.allclose(scaler1.bin_wise_bfactor, scaler2.bin_wise_bfactor)


class TestScalerHKLPropertyFunctional:
    """Functional tests for scaler HKL property."""

    @pytest.mark.integration
    def test_hkl_property(self, sample_structure_pair):
        """Test that HKL property returns correct indices."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        
        # HKL from scaler should match data
        hkl = scaler.hkl
        assert hkl is not None
        assert hkl.shape == data.hkl.shape
        assert torch.allclose(hkl, data.hkl)


class TestScalerDeviceOperationsFunctional:
    """Functional tests for scaler device operations."""

    @pytest.mark.integration
    def test_scaler_cpu_operation(self, sample_structure_pair):
        """Test scaler works on CPU."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0, device=torch.device('cpu'))
        scaler.setup_anisotropy_correction()
        
        assert scaler.device.type == 'cpu'
        assert scaler.s.device.type == 'cpu'
        assert scaler.U.device.type == 'cpu'

    @pytest.mark.integration
    def test_scaler_cpu_method(self, sample_structure_pair):
        """Test scaler.cpu() method."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        scaler.cpu()
        
        # All tensors should be on CPU
        for param in scaler.parameters():
            assert param.device.type == 'cpu'


class TestScalerUMatrixFunctional:
    """Functional tests for U matrix (anisotropic B-factor) operations."""

    @pytest.mark.integration
    def test_u_to_matrix_conversion(self, sample_structure_pair):
        """Test conversion from U parameters to 3x3 matrix."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        from torchref.base.math_torch import U_to_matrix
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        
        # Convert U vector to matrix
        U_matrix = U_to_matrix(scaler.U)
        
        # Should be 3x3
        assert U_matrix.shape == (3, 3)
        # Should be symmetric
        assert torch.allclose(U_matrix, U_matrix.T, atol=1e-6)


class TestScalerGradientsFunctional:
    """Functional tests for gradient flow through scaler."""

    @pytest.mark.integration
    def test_anisotropy_gradients(self, sample_structure_pair):
        """Test gradients flow through anisotropy correction."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        
        # Compute correction and loss
        correction = scaler.anisotropy_correction()
        loss = correction.sum()
        loss.backward()
        
        # U should have gradients
        assert scaler.U.grad is not None
        assert torch.all(torch.isfinite(scaler.U.grad))

    @pytest.mark.integration
    def test_binwise_bfactor_gradients(self, sample_structure_pair):
        """Test gradients flow through bin-wise B-factor correction."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_bin_wise_bfactor()
        
        # Compute correction and loss
        correction = scaler.bin_wise_bfactor_correction()
        loss = correction.sum()
        loss.backward()
        
        # bin_wise_bfactor should have gradients
        assert scaler.bin_wise_bfactor.grad is not None
        assert torch.all(torch.isfinite(scaler.bin_wise_bfactor.grad))


class TestScalerMultipleStructuresFunctional:
    """Functional tests with multiple structures."""

    @pytest.mark.integration
    def test_scaler_with_different_structures(self, all_test_structures):
        """Test scaler works with different crystal structures."""
        from torchref.scaling.scaler import Scaler
        
        tested = 0
        for struct in all_test_structures:
            pdb_id = struct["pdb_id"]
            model = struct["model"]
            data = struct["data"]
            
            scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
            scaler.setup_anisotropy_correction()
            
            # Verify scaler is set up correctly
            assert scaler.s is not None
            assert scaler.bins is not None
            assert scaler.U is not None
            
            correction = scaler.anisotropy_correction()
            assert torch.all(torch.isfinite(correction))
            
            tested += 1
            if tested >= 3:  # Test first 3 structures
                break
        
        assert tested >= 1, "No test structures with both CIF and MTZ found"
