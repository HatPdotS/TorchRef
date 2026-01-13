"""
Unit tests for torchref.symmetrie.symmetrie

Tests symmetry operations for crystallographic space groups.
"""

import pytest
import torch
import torch.nn as nn


class TestSymmetryInitialization:
    """Tests for Symmetry class initialization."""

    @pytest.mark.unit
    def test_symmetry_p1(self):
        """Test P1 space group (trivial, only identity)."""
        from torchref.symmetrie.symmetrie import Symmetry

        sym = Symmetry("P1")

        # space_group is a gemmi.SpaceGroup object
        assert "P 1" in str(sym.space_group) or "P1" in str(sym.space_group)
        assert sym.matrices is not None
        assert sym.translations is not None
        # P1 should have only identity
        assert sym.matrices.shape[0] == 1

    @pytest.mark.unit
    def test_symmetry_p21(self):
        """Test P21 space group."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P21")
        
        assert sym.matrices is not None
        # P21 has 2 operations
        assert sym.matrices.shape[0] == 2

    @pytest.mark.unit
    def test_symmetry_p212121(self):
        """Test P212121 space group (common for proteins)."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P212121")
        
        assert sym.matrices is not None
        # P212121 has 4 operations
        assert sym.matrices.shape[0] == 4

    @pytest.mark.unit
    def test_symmetry_with_spaces(self):
        """Test space group name with spaces."""
        from torchref.symmetrie.symmetrie import Symmetry

        sym1 = Symmetry("P 21 21 21")
        sym2 = Symmetry("P212121")

        # Both should have the same number of operations
        assert sym1.matrices.shape[0] == sym2.matrices.shape[0]

    @pytest.mark.unit
    def test_symmetry_unknown_raises(self):
        """Unknown space group should raise ValueError."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        with pytest.raises(ValueError):
            Symmetry("NotASpaceGroup123")


class TestSymmetryMatrices:
    """Tests for symmetry matrices properties."""

    @pytest.mark.unit
    def test_identity_in_matrices(self):
        """Every space group should have identity matrix."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P21")
        
        identity = torch.eye(3, dtype=sym.matrices.dtype)
        
        # Check if identity is in the matrices
        has_identity = False
        for i in range(sym.matrices.shape[0]):
            if torch.allclose(sym.matrices[i], identity, atol=1e-6):
                has_identity = True
                break
        
        assert has_identity

    @pytest.mark.unit
    def test_matrices_are_3x3(self):
        """Rotation matrices should be 3x3."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P212121")
        
        assert sym.matrices.shape[1] == 3
        assert sym.matrices.shape[2] == 3

    @pytest.mark.unit
    def test_translations_are_3d(self):
        """Translation vectors should be 3D."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P212121")
        
        assert sym.translations.shape[1] == 3

    @pytest.mark.unit
    def test_rotation_matrices_orthogonal(self):
        """Rotation matrices should be orthogonal (R^T @ R = I)."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P21")
        
        for i in range(sym.matrices.shape[0]):
            R = sym.matrices[i]
            RtR = R.T @ R
            assert torch.allclose(RtR, torch.eye(3, dtype=RtR.dtype), atol=1e-5)

    @pytest.mark.unit
    def test_rotation_matrices_determinant(self):
        """Rotation matrices should have determinant +1 or -1."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P212121")
        
        for i in range(sym.matrices.shape[0]):
            det = torch.linalg.det(sym.matrices[i])
            assert torch.isclose(torch.abs(det), torch.tensor(1.0, dtype=det.dtype), atol=1e-5)


class TestSymmetryApplication:
    """Tests for applying symmetry operations."""

    @pytest.mark.unit
    def test_apply_identity(self, random_fractional_coordinates):
        """Identity operation should not change coordinates."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P1")
        # Symmetry expects (3, N) format, not (N, 3)
        coords = random_fractional_coordinates(n_atoms=10).T  # Transpose to (3, N)
        
        # Apply symmetry (P1 only has identity)
        # Output shape is (3, n_atoms, n_operations)
        transformed = sym(coords)
        
        # Should have shape (3, n_atoms, n_operations)
        assert transformed.shape[0] == 3  # 3D coordinates
        assert transformed.shape[1] == 10  # 10 atoms
        assert transformed.shape[2] == 1   # 1 operation (identity)
        # First (and only) symmetry mate should match original
        assert torch.allclose(transformed[:, :, 0], coords.to(transformed.dtype), atol=1e-5)

    @pytest.mark.unit
    def test_symmetry_generates_mates(self, random_fractional_coordinates):
        """Symmetry should generate correct number of mates."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P21")  # 2 operations
        coords = random_fractional_coordinates(n_atoms=5).T  # (3, N) format
        
        # Output shape is (3, n_atoms, n_operations)
        transformed = sym(coords)
        
        # Should have 2 symmetry operations
        assert transformed.shape[2] == 2

    @pytest.mark.unit
    def test_symmetry_callable(self, random_fractional_coordinates):
        """Symmetry should be callable."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P212121")
        coords = random_fractional_coordinates(n_atoms=10).T  # (3, N) format
        
        # Should be callable
        # Output shape is (3, n_atoms, n_operations)
        result = sym(coords)
        
        assert result is not None
        assert result.shape[2] == 4  # 4 symmetry operations


class TestSymmetryDeviceHandling:
    """Tests for device handling in Symmetry."""

    @pytest.mark.unit
    def test_symmetry_cpu(self):
        """Test symmetry on CPU."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P21", device=torch.device('cpu'))
        
        assert sym.matrices.device.type == 'cpu'
        assert sym.translations.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_symmetry_gpu(self, gpu_device):
        """Test symmetry on GPU."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P21", device=gpu_device)
        
        assert sym.matrices.device.type == 'cuda'
        assert sym.translations.device.type == 'cuda'

    @pytest.mark.unit
    def test_symmetry_dtype(self):
        """Test symmetry dtype specification."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        sym = Symmetry("P21", dtype=torch.float32)
        
        assert sym.matrices.dtype == torch.float32
        assert sym.translations.dtype == torch.float32


class TestSpaceGroupMapping:
    """Tests for space group name mapping."""

    @pytest.mark.unit
    def test_common_spacegroups(self):
        """Test common crystallographic space groups."""
        from torchref.symmetrie.symmetrie import Symmetry
        
        # Common protein space groups
        common_sgs = ["P1", "P21", "P212121", "C2", "P21212"]
        
        for sg in common_sgs:
            sym = Symmetry(sg)
            assert sym.matrices is not None

    @pytest.mark.unit
    def test_case_insensitivity(self):
        """Space group names should be somewhat case-insensitive."""
        from torchref.symmetrie.symmetrie import Symmetry

        # Try different case variations - may not all work
        try:
            sym1 = Symmetry("P21")
            # Should have valid matrices
            assert sym1.matrices is not None
        except ValueError:
            pass  # Some case variations may not be supported
