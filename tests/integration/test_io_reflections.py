"""
Integration tests for reflection data loading.

Tests loading MTZ and SF-CIF files.
"""

import pytest
import torch
from pathlib import Path


class TestMTZLoading:
    """Tests for loading MTZ reflection files."""

    @pytest.mark.integration
    def test_load_mtz_file(self, sample_mtz_file):
        """Test loading a real MTZ file."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Should have reflections loaded
        assert hasattr(data, 'hkl')
        assert hasattr(data, 'F')
        assert data.hkl is not None

    @pytest.mark.integration
    def test_mtz_reflection_counts(self, sample_mtz_file):
        """Test that MTZ has consistent reflection counts."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        n_refl = data.hkl.shape[0]
        assert n_refl > 0
        
        # F should match hkl count
        if data.F is not None:
            assert data.F.shape[0] == n_refl

    @pytest.mark.integration
    def test_mtz_hkl_indices(self, sample_mtz_file):
        """Test HKL indices are valid integers."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # HKL should have 3 columns
        assert data.hkl.shape[1] == 3
        
        # Should contain integer-like values
        hkl_rounded = torch.round(data.hkl)
        assert torch.allclose(data.hkl, hkl_rounded)

    @pytest.mark.integration
    def test_mtz_cell_parameters(self, sample_mtz_file):
        """Test that MTZ has valid cell parameters."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        if hasattr(data, 'cell') and data.cell is not None:
            assert len(data.cell) == 6
            assert all(c > 0 for c in data.cell[:3].tolist())

    @pytest.mark.integration
    def test_mtz_spacegroup(self, sample_mtz_file):
        """Test that MTZ has a valid spacegroup."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        assert data.spacegroup is not None
        assert isinstance(data.spacegroup, str)

    @pytest.mark.integration
    def test_mtz_sigma_values(self, sample_mtz_file):
        """Test that sigma values are loaded."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        if hasattr(data, 'F_sigma') and data.F_sigma is not None:
            assert data.F_sigma.shape[0] == data.F.shape[0]
            # Check that non-NaN sigma values are positive
            valid_sigma = data.F_sigma[~torch.isnan(data.F_sigma)]
            if len(valid_sigma) > 0:
                assert torch.all(valid_sigma > 0)

    @pytest.mark.integration
    def test_mtz_rfree_flags(self, sample_mtz_file):
        """Test that R-free flags are loaded or generated."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Should have rfree_flags (loaded or generated)
        if hasattr(data, 'rfree_flags') and data.rfree_flags is not None:
            assert data.rfree_flags.shape[0] == data.hkl.shape[0]


class TestSFCIFLoading:
    """Tests for loading structure factor CIF files."""

    @pytest.mark.integration
    def test_load_sf_cif(self, sample_structure_factor_cif):
        """Test loading a structure factor CIF file."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_cif(str(sample_structure_factor_cif))
        
        assert data.hkl is not None


class TestReflectionDataProperties:
    """Tests for computed properties of reflection data."""

    @pytest.mark.integration
    def test_resolution_calculation(self, sample_mtz_file):
        """Test resolution can be calculated from loaded data."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Should have resolution attribute
        if hasattr(data, 'resolution') and data.resolution is not None:
            assert torch.all(data.resolution > 0)
            assert torch.all(torch.isfinite(data.resolution))

    @pytest.mark.integration
    def test_wilson_b_factor(self, sample_mtz_file):
        """Test Wilson B-factor is calculated."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Wilson B should be calculated during loading
        if hasattr(data, 'wilson_b') and data.wilson_b is not None:
            assert data.wilson_b > 0

    @pytest.mark.integration
    def test_data_device_movement(self, sample_mtz_file, cpu_device):
        """Test moving reflection data to different devices."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Move to device
        data = data.to(cpu_device)
        
        # Tensors should be on correct device
        if data.hkl is not None:
            assert data.hkl.device == cpu_device
        if data.F is not None:
            assert data.F.device == cpu_device


class TestMatchingDataPairs:
    """Tests using matching model and reflection data."""

    @pytest.mark.integration
    def test_load_structure_pair(self, sample_structure_pair):
        """Test loading matching model and reflection data."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Both should load successfully
        n_atoms = model.xyz().shape[0]
        assert n_atoms > 0
        assert data.hkl is not None

    @pytest.mark.integration
    def test_cell_consistency(self, sample_structure_pair):
        """Test that model and data have consistent cell parameters."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Cell parameters should be similar (may have small differences)
        if hasattr(data, 'cell') and data.cell is not None:
            model_cell = torch.tensor(model.cell)
            data_cell = torch.tensor(data.cell)
            
            # Allow 1% tolerance for cell parameters
            assert torch.allclose(model_cell, data_cell, rtol=0.01, atol=0.1)
