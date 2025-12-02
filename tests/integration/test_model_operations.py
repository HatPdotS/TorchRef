"""
Integration tests for the model module.

Tests model loading, manipulation, and calculations with real data.
"""

import pytest
import torch
from pathlib import Path


class TestModelCoordinates:
    """Tests for model coordinate operations."""

    @pytest.mark.integration
    def test_model_xyz_shape(self, sample_cif_file):
        """Test that xyz() returns correct shape."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz()
        
        assert xyz is not None
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3

    @pytest.mark.integration
    def test_model_xyz_dtype(self, sample_cif_file):
        """Test xyz coordinate dtype."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz()
        
        # Should be float32 or float64
        assert xyz.dtype in [torch.float32, torch.float64]

    @pytest.mark.integration
    def test_model_xyz_values_finite(self, sample_cif_file):
        """Test that all coordinate values are finite."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz()
        
        assert torch.all(torch.isfinite(xyz))


class TestModelBfactors:
    """Tests for B-factor operations."""

    @pytest.mark.integration
    def test_model_bfactors_shape(self, sample_cif_file):
        """Test B-factor tensor shape."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        b = model.b()
        n_atoms = model.xyz().shape[0]
        
        assert b is not None
        assert b.shape[0] == n_atoms

    @pytest.mark.integration
    def test_model_bfactors_positive(self, sample_cif_file):
        """Test that B-factors are positive."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        b = model.b()
        
        # B-factors should be positive (physically meaningful)
        assert torch.all(b >= 0)


class TestModelOccupancy:
    """Tests for occupancy operations."""

    @pytest.mark.integration
    def test_model_occupancy_shape(self, sample_cif_file):
        """Test occupancy tensor shape."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        occ = model.occupancy()
        n_atoms = model.xyz().shape[0]
        
        assert occ is not None
        assert occ.shape[0] == n_atoms

    @pytest.mark.integration
    def test_model_occupancy_range(self, sample_cif_file):
        """Test that occupancies are in valid range [0, 1]."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        occ = model.occupancy()
        
        assert torch.all(occ >= 0)
        assert torch.all(occ <= 1)


class TestModelCell:
    """Tests for unit cell operations."""

    @pytest.mark.integration
    def test_model_cell_parameters(self, sample_cif_file):
        """Test that cell parameters are loaded correctly."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        cell = model.cell
        
        assert cell is not None
        # Cell should have 6 parameters: a, b, c, alpha, beta, gamma
        assert len(cell) == 6

    @pytest.mark.integration
    def test_model_cell_values_positive(self, sample_cif_file):
        """Test that cell lengths are positive."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        cell = model.cell
        
        # First 3 values are lengths
        if isinstance(cell, torch.Tensor):
            lengths = cell[:3]
            angles = cell[3:]
        else:
            lengths = cell[:3]
            angles = cell[3:]
        
        assert all(l > 0 for l in lengths)
        # Angles should be in reasonable range (typically 60-120 degrees)
        assert all(30 <= a <= 150 for a in angles)


class TestModelSpacegroup:
    """Tests for spacegroup operations."""

    @pytest.mark.integration
    def test_model_spacegroup_loaded(self, sample_cif_file):
        """Test that spacegroup is loaded."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        sg = model.spacegroup
        
        assert sg is not None

    @pytest.mark.integration
    def test_model_spacegroup_type(self, sample_cif_file):
        """Test spacegroup type."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        sg = model.spacegroup
        
        # Could be string or number
        assert isinstance(sg, (str, int))


class TestModelPDB:
    """Tests for PDB DataFrame operations."""

    @pytest.mark.integration
    def test_model_pdb_dataframe(self, sample_cif_file):
        """Test that PDB DataFrame is loaded."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        pdb = model.pdb
        
        assert pdb is not None
        # Should be a pandas DataFrame
        import pandas as pd
        assert isinstance(pdb, pd.DataFrame)

    @pytest.mark.integration
    def test_model_pdb_has_element(self, sample_cif_file):
        """Test that PDB DataFrame has element column."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        pdb = model.pdb
        
        assert 'element' in pdb.columns

    @pytest.mark.integration
    def test_model_pdb_has_coordinates(self, sample_cif_file):
        """Test that PDB DataFrame has coordinate columns."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        pdb = model.pdb
        
        # Should have x, y, z columns
        assert 'x' in pdb.columns or 'X' in pdb.columns
        assert 'y' in pdb.columns or 'Y' in pdb.columns
        assert 'z' in pdb.columns or 'Z' in pdb.columns


class TestModelDevice:
    """Tests for model device operations."""

    @pytest.mark.integration
    def test_model_default_device(self, sample_cif_file):
        """Test that model uses default CPU device."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz()
        
        assert xyz.device.type == 'cpu'

    @pytest.mark.integration
    def test_model_explicit_cpu_device(self, sample_cif_file, cpu_device):
        """Test model with explicit CPU device."""
        from torchref.model.model import Model
        
        model = Model(device=cpu_device)
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz()
        
        assert xyz.device == cpu_device


class TestModelFileSaving:
    """Tests for model file saving operations."""

    @pytest.mark.integration
    def test_model_write_pdb(self, sample_cif_file, tmp_path):
        """Test writing model to PDB file."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        output_path = tmp_path / "output.pdb"
        model.write_pdb(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.integration
    def test_model_roundtrip_pdb(self, sample_cif_file, tmp_path):
        """Test saving and reloading model via PDB."""
        from torchref.model.model import Model
        
        model1 = Model()
        model1.load_cif(str(sample_cif_file))
        n_atoms1 = model1.xyz().shape[0]
        
        output_path = tmp_path / "output.pdb"
        model1.write_pdb(str(output_path))
        
        model2 = Model()
        model2.load_pdb(str(output_path))
        n_atoms2 = model2.xyz().shape[0]
        
        assert n_atoms2 == n_atoms1


class TestModelMultipleStructures:
    """Tests for loading multiple structures."""

    @pytest.mark.integration
    def test_load_different_cif_files(self, cif_dir):
        """Test loading different CIF files."""
        from torchref.model.model import Model
        
        cif_files = list(cif_dir.glob("*.cif"))[:3]  # Load first 3
        
        for cif_file in cif_files:
            model = Model()
            model.load_cif(str(cif_file))
            
            assert model.xyz().shape[0] > 0
            assert model.cell is not None
            assert model.spacegroup is not None


class TestModelGradients:
    """Tests for gradient operations through the model."""

    @pytest.mark.integration
    def test_xyz_gradient_computation(self, sample_cif_file):
        """Test that gradients can be computed for coordinates."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        # Get coordinates
        xyz = model.xyz()
        
        # Make sure requires_grad and retain_grad for non-leaf
        if not xyz.requires_grad:
            xyz = xyz.requires_grad_(True)
        xyz.retain_grad()
        
        # Simple operation
        loss = xyz.pow(2).sum()
        loss.backward()
        
        assert xyz.grad is not None
        assert xyz.grad.shape == xyz.shape
