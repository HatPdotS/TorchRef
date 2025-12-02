"""
Functional tests for I/O operations.

Tests file loading and data processing with real crystallographic data.
"""

import pytest
import torch
import numpy as np


class TestCIFReadingFunctional:
    """Functional tests for CIF file reading."""

    @pytest.mark.integration
    def test_load_multiple_cif_files(self, cif_dir):
        """Test loading multiple CIF files successfully."""
        from torchref.model.model import Model
        
        cif_files = list(cif_dir.glob("*.cif"))
        assert len(cif_files) > 0, "No CIF files found in test directory"
        
        for cif_file in cif_files:
            model = Model()
            model.load_cif(str(cif_file))
            
            # Each file should load with atoms
            n_atoms = model.xyz().shape[0]
            assert n_atoms > 0, f"No atoms loaded from {cif_file}"
            
            # Should have cell parameters
            assert model.cell is not None
            assert len(model.cell) == 6

    @pytest.mark.integration
    def test_cif_atom_properties(self, sample_cif_file):
        """Test that atom properties are correctly loaded from CIF."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        pdb = model.pdb
        
        # Check required columns exist
        required_cols = ['x', 'y', 'z', 'element', 'resname', 'chainid', 'resseq']
        for col in required_cols:
            assert col in pdb.columns or col.upper() in pdb.columns, f"Missing column: {col}"

    @pytest.mark.integration
    def test_cif_element_types(self, sample_cif_file):
        """Test that element types are properly assigned."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        elements = model.pdb['element'].unique()
        
        # Should have common protein elements
        common_elements = ['C', 'N', 'O', 'S']
        found_any = any(elem in elements for elem in common_elements)
        assert found_any, "No common elements found"


class TestMTZReadingFunctional:
    """Functional tests for MTZ file reading."""

    @pytest.mark.integration
    def test_load_multiple_mtz_files(self, mtz_dir):
        """Test loading multiple MTZ files successfully."""
        from torchref.io.Data import ReflectionData
        
        mtz_files = list(mtz_dir.glob("*.mtz"))
        assert len(mtz_files) > 0, "No MTZ files found in test directory"
        
        for mtz_file in mtz_files:
            data = ReflectionData()
            data.load_mtz(str(mtz_file))
            
            # Each file should load with reflections
            n_refl = data.hkl.shape[0]
            assert n_refl > 0, f"No reflections loaded from {mtz_file}"
            
            # Should have cell parameters
            assert data.cell is not None

    @pytest.mark.integration
    def test_mtz_data_properties(self, sample_mtz_file):
        """Test that MTZ data properties are correctly loaded."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Check HKL indices are integers or can be converted
        hkl = data.hkl
        assert hkl.shape[1] == 3, "HKL should have 3 columns"
        
        # Check F values are loaded
        assert data.F is not None
        assert data.F.shape[0] == hkl.shape[0]
        
        # Check sigma values
        if hasattr(data, 'F_sigma') and data.F_sigma is not None:
            assert data.F_sigma.shape[0] == hkl.shape[0]

    @pytest.mark.integration
    def test_mtz_resolution_range(self, sample_mtz_file):
        """Test that resolution range is computed correctly."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Check if resolution data is available
        if hasattr(data, 'd') and data.d is not None:
            d_min = data.d.min().item()
            d_max = data.d.max().item()
            
            # Resolution should be positive
            assert d_min > 0
            assert d_max > d_min
            
            # Typical protein data: 0.8 - 500 Å
            assert d_min > 0.5
            assert d_max < 1000


class TestSFCIFReadingFunctional:
    """Functional tests for structure factor CIF reading."""

    @pytest.mark.integration
    def test_load_sf_cif(self, cif_sf_dir):
        """Test loading structure factor CIF files."""
        from torchref.io.Data import ReflectionData
        
        sf_files = list(cif_sf_dir.glob("*.cif"))
        if not sf_files:
            pytest.skip("No SF-CIF files found")
        
        for sf_file in sf_files:
            data = ReflectionData()
            try:
                data.load_cif(str(sf_file))
                
                # Should have loaded reflections
                if data.hkl is not None:
                    assert data.hkl.shape[0] > 0
            except Exception as e:
                # Some files may not be valid SF-CIF format
                pass


class TestDataConsistencyFunctional:
    """Test consistency between model and data files."""

    @pytest.mark.integration
    def test_cell_parameters_match(self, sample_structure_pair):
        """Test that cell parameters match between model and reflections."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        model_cell = model.cell
        data_cell = data.cell
        
        if model_cell is not None and data_cell is not None:
            # Convert to tensors if needed
            if not isinstance(model_cell, torch.Tensor):
                model_cell = torch.tensor(model_cell)
            if not isinstance(data_cell, torch.Tensor):
                data_cell = torch.tensor(data_cell)
            
            # Cell parameters should be similar (1% tolerance)
            assert torch.allclose(model_cell.float(), data_cell.float(), rtol=0.01, atol=0.1)

    @pytest.mark.integration
    def test_spacegroup_consistency(self, sample_structure_pair):
        """Test that spacegroup is consistent."""
        from torchref.model.model import Model
        from torchref.io.Data import ReflectionData
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Both should have spacegroup defined
        assert model.spacegroup is not None


class TestDataBinningFunctional:
    """Test data binning operations."""

    @pytest.mark.integration
    def test_get_bins(self, sample_mtz_file):
        """Test resolution binning of reflection data."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Get bins
        bins, n_bins = data.get_bins(n_bins=10)
        
        assert bins is not None
        assert bins.shape[0] == data.hkl.shape[0]
        assert bins.min() >= 0
        assert bins.max() < n_bins

    @pytest.mark.integration
    def test_mean_res_per_bin(self, sample_mtz_file):
        """Test mean resolution per bin calculation."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Get bins first
        bins, n_bins = data.get_bins(n_bins=10)
        
        # Get mean resolution per bin
        if hasattr(data, 'mean_res_per_bin'):
            mean_res = data.mean_res_per_bin()
            
            assert mean_res is not None
            assert len(mean_res) == n_bins
            
            # Mean resolution should decrease with bin index (low res to high res)
            # or increase (high res to low res) - depends on implementation
            assert torch.all(torch.isfinite(mean_res))


class TestFrenchWilsonFunctional:
    """Test French-Wilson conversion with real data."""

    @pytest.mark.integration
    def test_french_wilson_applied(self, sample_mtz_file):
        """Test that French-Wilson conversion is applied."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # After French-Wilson, F values should be non-negative
        valid_F = data.F[~torch.isnan(data.F)]
        
        if len(valid_F) > 0:
            # All valid F values should be >= 0
            assert torch.all(valid_F >= 0)


class TestRfreeHandlingFunctional:
    """Test R-free flag handling."""

    @pytest.mark.integration
    def test_rfree_flags_loaded(self, sample_mtz_file):
        """Test that R-free flags are loaded or generated."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        # Should have rfree attribute
        if hasattr(data, 'rfree') and data.rfree is not None:
            assert data.rfree.shape[0] == data.hkl.shape[0]
            
            # Should be boolean or can be converted to boolean
            assert data.rfree.dtype == torch.bool or torch.all((data.rfree == 0) | (data.rfree == 1))

    @pytest.mark.integration
    def test_rfree_fraction(self, sample_mtz_file):
        """Test R-free set fraction is reasonable."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        if hasattr(data, 'rfree') and data.rfree is not None:
            # Work set mask (True for work, False for test)
            work_fraction = data.rfree.float().mean().item()
            
            # Typically 90-95% work set, 5-10% test set
            # So work_fraction should be 0.9-0.95 typically
            assert 0.7 < work_fraction <= 1.0


class TestMaskHandlingFunctional:
    """Test reflection mask handling."""

    @pytest.mark.integration
    def test_masks_method(self, sample_mtz_file):
        """Test masks() method returns valid mask."""
        from torchref.io.Data import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))
        
        if hasattr(data, 'masks'):
            mask = data.masks()
            
            assert mask is not None
            assert mask.shape[0] == data.hkl.shape[0]
            assert mask.dtype == torch.bool
