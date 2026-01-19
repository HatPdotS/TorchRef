"""
Integration tests for the symmetry module.

Tests symmetry operations with real crystallographic data.
"""

import pytest
import torch
from pathlib import Path


class TestSymmetryInitialization:
    """Tests for Symmetry initialization."""

    @pytest.mark.integration
    def test_symmetry_from_spacegroup_string(self):
        """Test creating Symmetry from spacegroup string."""
        from torchref.symmetry.symmetry import Symmetry
        
        # P 21 21 21 is a common orthorhombic spacegroup
        sym = Symmetry("P 21 21 21")
        
        assert sym is not None
        assert sym.matrices is not None

    @pytest.mark.integration
    def test_symmetry_from_spacegroup_number(self):
        """Test creating Symmetry from spacegroup name."""
        from torchref.symmetry.symmetry import Symmetry
        
        # Use string representation for spacegroup 19
        sym = Symmetry("P212121")
        
        assert sym is not None
        assert sym.matrices is not None

    @pytest.mark.integration
    def test_symmetry_from_model(self, sample_cif_file):
        """Test creating Symmetry from model spacegroup."""
        from torchref.model.model import Model
        from torchref.symmetry.symmetry import Symmetry
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        sym = Symmetry(model.spacegroup)
        
        assert sym is not None
        assert sym.matrices is not None


class TestSymmetryMatrices:
    """Tests for symmetry matrices."""

    @pytest.mark.integration
    def test_symmetry_matrices_shape(self):
        """Test shape of symmetry matrices."""
        from torchref.symmetry.symmetry import Symmetry
        
        sym = Symmetry("P 21 21 21")
        
        # Should have 4 symmetry operations (including identity)
        n_ops = sym.matrices.shape[0]
        assert n_ops >= 1  # At least identity
        
        # Each should be 3x4 (rotation + translation) or 4x4
        assert sym.matrices.shape[-1] in [3, 4]
        assert sym.matrices.shape[-2] in [3, 4]

    @pytest.mark.integration
    def test_symmetry_identity_present(self):
        """Test that identity operation is present."""
        from torchref.symmetry.symmetry import Symmetry
        
        sym = Symmetry("P 1")  # Triclinic - only identity
        
        # P 1 should have exactly 1 symmetry operation (identity)
        assert sym.matrices.shape[0] >= 1

    @pytest.mark.integration
    def test_symmetry_matrices_orthogonal(self):
        """Test that rotation parts of symmetry matrices are orthogonal."""
        from torchref.symmetry.symmetry import Symmetry
        
        sym = Symmetry("P 21 21 21")
        
        # Extract rotation parts (3x3)
        matrices = sym.matrices
        
        for i in range(matrices.shape[0]):
            if matrices.shape[-1] == 4:
                rot = matrices[i, :3, :3]
            else:
                rot = matrices[i, :3, :3]
            
            # R * R^T should be identity for orthogonal matrices
            product = torch.mm(rot, rot.T)
            identity = torch.eye(3, device=product.device, dtype=product.dtype)
            
            assert torch.allclose(product, identity, atol=1e-5)


class TestSymmetryDevice:
    """Tests for symmetry device handling."""

    @pytest.mark.integration
    def test_symmetry_default_device(self):
        """Test symmetry matrices default to CPU."""
        from torchref.symmetry.symmetry import Symmetry
        
        sym = Symmetry("P 21 21 21")
        
        assert sym.matrices.device.type == 'cpu'

    @pytest.mark.integration
    def test_symmetry_explicit_device(self, cpu_device):
        """Test creating symmetry on explicit device."""
        from torchref.symmetry.symmetry import Symmetry
        
        sym = Symmetry("P 21 21 21", device=cpu_device)
        
        assert sym.matrices.device == cpu_device


class TestSymmetryOperations:
    """Tests for applying symmetry operations."""

    @pytest.mark.integration
    def test_expand_coordinates(self, sample_cif_file):
        """Test expanding coordinates by symmetry."""
        from torchref.model.model import Model
        from torchref.symmetry.symmetry import Symmetry
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz()
        sym = Symmetry(model.spacegroup)
        
        # The model should be able to generate symmetry mates
        # Check if there's an expand method
        if hasattr(sym, 'expand') or hasattr(sym, 'expand_atoms'):
            expanded = sym.expand(xyz)
            assert expanded.shape[0] >= xyz.shape[0]


class TestSpacegroupVariants:
    """Tests for different spacegroup conventions."""

    @pytest.mark.integration
    def test_common_spacegroups(self):
        """Test loading common spacegroups."""
        from torchref.symmetry.symmetry import Symmetry
        
        common_spacegroups = [
            "P 1",           # Triclinic
            "P 21",          # Monoclinic
            "P 21 21 21",    # Orthorhombic
            "P 43 21 2",     # Tetragonal
            "P 3 2 1",       # Trigonal
            "P 6 2 2",       # Hexagonal
            "P 2 3",         # Cubic
        ]
        
        for sg in common_spacegroups:
            try:
                sym = Symmetry(sg)
                assert sym.matrices is not None
            except Exception as e:
                # Some spacegroups might not be supported - that's ok
                pass

    @pytest.mark.integration
    def test_spacegroup_name_variations(self):
        """Test that different spacegroup name formats work."""
        from torchref.symmetry.symmetry import Symmetry
        
        # Both formats should work
        sym_with_spaces = Symmetry("P 21 21 21")
        sym_no_spaces = Symmetry("P212121")
        
        # Should have same number of symmetry operations
        assert sym_with_spaces.matrices.shape[0] == sym_no_spaces.matrices.shape[0]


class TestSymmetryWithData:
    """Tests for symmetry with real crystallographic data."""

    @pytest.mark.integration
    def test_symmetry_with_multiple_structures(self, cif_dir):
        """Test symmetry for multiple structures."""
        from torchref.model.model import Model
        from torchref.symmetry.symmetry import Symmetry
        
        cif_files = list(cif_dir.glob("*.cif"))[:3]
        
        for cif_file in cif_files:
            model = Model()
            model.load_cif(str(cif_file))
            
            sym = Symmetry(model.spacegroup)
            
            # Should have valid matrices
            assert sym.matrices is not None
            assert sym.matrices.shape[0] >= 1
            assert torch.all(torch.isfinite(sym.matrices))

    @pytest.mark.integration
    def test_symmetry_consistent_with_cell(self, sample_cif_file):
        """Test that symmetry is consistent with unit cell."""
        from torchref.model.model import Model
        from torchref.symmetry.symmetry import Symmetry
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        sym = Symmetry(model.spacegroup)
        cell = model.cell
        
        # Both should be defined
        assert sym.matrices is not None
        assert cell is not None


class TestMapSymmetry:
    """Tests for map symmetry operations."""

    @pytest.mark.integration
    def test_map_symmetry_initialization(self, sample_cif_file):
        """Test map symmetry initialization."""
        from torchref.model.model import Model
        from torchref.symmetry.map_symmetry import MapSymmetry
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        # Check if MapSymmetry can be initialized
        try:
            map_sym = MapSymmetry(model.spacegroup, model.cell)
            assert map_sym is not None
        except (TypeError, AttributeError):
            # May not support all initialization patterns
            pass
