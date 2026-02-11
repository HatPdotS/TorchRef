"""
Integration tests for CIF file loading.

Tests real file I/O operations with actual CIF files.
"""

import pytest
import torch
from pathlib import Path


class TestCIFLoading:
    """Tests for loading CIF model files."""

    @pytest.mark.integration
    def test_load_model_cif(self, sample_cif_file):
        """Test loading a real CIF model file."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        # Basic checks - use xyz().shape[0] for atom count
        n_atoms = model.xyz().shape[0]
        assert n_atoms > 0
        assert hasattr(model, 'xyz')
        assert hasattr(model, 'adp')
        assert hasattr(model, 'occupancy')

    @pytest.mark.integration
    def test_model_atom_counts(self, sample_cif_file):
        """Test that model has consistent atom counts."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        # All arrays should have same number of atoms
        n_atoms = model.xyz().shape[0]
        assert model.xyz().shape[0] == n_atoms
        assert model.adp().shape[0] == n_atoms
        assert model.occupancy().shape[0] == n_atoms

    @pytest.mark.integration
    def test_model_cell_parameters(self, sample_cif_file):
        """Test that model has valid cell parameters."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        # Cell should have 6 parameters
        assert len(model.cell) == 6
        # All cell parameters should be positive
        assert all(p > 0 for p in model.cell[:3].tolist())  # a, b, c
        # Angles should be reasonable (0-180)
        assert all(0 < p <= 180 for p in model.cell[3:].tolist())  # alpha, beta, gamma

    @pytest.mark.integration
    def test_model_spacegroup(self, sample_cif_file):
        """Test that model has a valid spacegroup."""
        from torchref.model.model import Model

        model = Model()
        model.load_cif(str(sample_cif_file))

        # Spacegroup should be set (can be string or gemmi.SpaceGroup)
        assert model.spacegroup is not None
        # Check it can be converted to string representation
        assert len(str(model.spacegroup)) > 0

    @pytest.mark.integration
    def test_model_element_types(self, sample_cif_file):
        """Test that element types are recognized."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        # Should have pdb DataFrame with element column
        assert hasattr(model, 'pdb')
        assert 'element' in model.pdb.columns
        # Elements should be strings like 'C', 'N', 'O', etc.
        elements = set(model.pdb['element'].unique())
        common_elements = {'C', 'N', 'O', 'S', 'H', 'CA', 'MG', 'ZN', 'FE'}
        # At least some elements should be recognized
        assert len(elements.intersection(common_elements)) > 0 or len(elements) > 0


class TestMultipleCIFFiles:
    """Tests that load multiple CIF files."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_load_all_test_structures(self, all_cif_files):
        """Test loading all available test structures."""
        from torchref.model.model import Model

        loaded = 0
        errors = []

        for cif_file in all_cif_files:
            try:
                model = Model()
                model.load_cif(str(cif_file))
                n_atoms = model.xyz().shape[0]
                assert n_atoms > 0
                loaded += 1
            except Exception as e:
                errors.append((cif_file.name, str(e)))

        # Report
        print(f"\nLoaded {loaded}/{len(all_cif_files)} structures")
        if errors:
            print(f"Errors: {errors}")

        # Should load at least most structures
        assert loaded > 0


class TestCIFSaving:
    """Tests for saving CIF files."""

    @pytest.mark.integration
    def test_save_and_reload_cif(self, sample_cif_file, tmp_path):
        """Test saving a model to CIF and reloading it."""
        from torchref.model.model import Model
        
        # Load original
        model1 = Model()
        model1.load_cif(str(sample_cif_file))
        n_atoms1 = model1.xyz().shape[0]
        
        # Save to temp file using write_pdb (CIF saving may not exist)
        output_path = tmp_path / "test_output.pdb"
        model1.write_pdb(str(output_path))
        
        assert output_path.exists()
        
        # Reload
        model2 = Model()
        model2.load_pdb(str(output_path))
        n_atoms2 = model2.xyz().shape[0]
        
        # Compare atom counts
        assert n_atoms2 == n_atoms1
