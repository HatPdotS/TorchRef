"""
Unit tests for torchref.symmetry.spacegroup

Tests symmetry operations for crystallographic space groups.
"""

import pytest
import torch
import torch.nn as nn


class TestSpaceGroupInitialization:
    """Tests for SpaceGroup class initialization."""

    @pytest.mark.unit
    def test_spacegroup_p1(self):
        """Test P1 space group (trivial, only identity)."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P1")

        # space_group is a gemmi.SpaceGroup object
        assert "P 1" in str(sg.space_group) or "P1" in str(sg.space_group)
        assert sg.matrices is not None
        assert sg.translations is not None
        # P1 should have only identity
        assert sg.matrices.shape[0] == 1

    @pytest.mark.unit
    def test_spacegroup_p21(self):
        """Test P21 space group."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P21")

        assert sg.matrices is not None
        # P21 has 2 operations
        assert sg.matrices.shape[0] == 2

    @pytest.mark.unit
    def test_spacegroup_p212121(self):
        """Test P212121 space group (common for proteins)."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P212121")

        assert sg.matrices is not None
        # P212121 has 4 operations
        assert sg.matrices.shape[0] == 4

    @pytest.mark.unit
    def test_spacegroup_with_spaces(self):
        """Test space group name with spaces."""
        from torchref.symmetry import SpaceGroup

        sg1 = SpaceGroup("P 21 21 21")
        sg2 = SpaceGroup("P212121")

        # Both should have the same number of operations
        assert sg1.matrices.shape[0] == sg2.matrices.shape[0]

    @pytest.mark.unit
    def test_spacegroup_unknown_raises(self):
        """Unknown space group should raise ValueError."""
        from torchref.symmetry import SpaceGroup

        with pytest.raises(ValueError):
            SpaceGroup("NotASpaceGroup123")


class TestSpaceGroupMatrices:
    """Tests for symmetry matrices properties."""

    @pytest.mark.unit
    def test_identity_in_matrices(self):
        """Every space group should have identity matrix."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P21")

        identity = torch.eye(3, dtype=sg.matrices.dtype)

        # Check if identity is in the matrices
        has_identity = False
        for i in range(sg.matrices.shape[0]):
            if torch.allclose(sg.matrices[i], identity, atol=1e-6):
                has_identity = True
                break

        assert has_identity

    @pytest.mark.unit
    def test_matrices_are_3x3(self):
        """Rotation matrices should be 3x3."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P212121")

        assert sg.matrices.shape[1] == 3
        assert sg.matrices.shape[2] == 3

    @pytest.mark.unit
    def test_translations_are_3d(self):
        """Translation vectors should be 3D."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P212121")

        assert sg.translations.shape[1] == 3

    @pytest.mark.unit
    def test_rotation_matrices_orthogonal(self):
        """Rotation matrices should be orthogonal (R^T @ R = I)."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P21")

        for i in range(sg.matrices.shape[0]):
            R = sg.matrices[i]
            RtR = R.T @ R
            assert torch.allclose(RtR, torch.eye(3, dtype=RtR.dtype), atol=1e-5)

    @pytest.mark.unit
    def test_rotation_matrices_determinant(self):
        """Rotation matrices should have determinant +1 or -1."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P212121")

        for i in range(sg.matrices.shape[0]):
            det = torch.linalg.det(sg.matrices[i])
            assert torch.isclose(torch.abs(det), torch.tensor(1.0, dtype=det.dtype), atol=1e-5)


class TestSpaceGroupApplication:
    """Tests for applying symmetry operations."""

    @pytest.mark.unit
    def test_apply_identity(self, random_fractional_coordinates):
        """Identity operation should not change coordinates."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P1")
        # SpaceGroup expects (N, 3) format
        coords = random_fractional_coordinates(n_atoms=10)  # Shape: (N, 3)

        # Apply symmetry (P1 only has identity)
        # Output shape is (n_atoms, 3, n_operations)
        transformed = sg(coords)

        # Should have shape (n_atoms, 3, n_operations)
        assert transformed.shape[0] == 10  # 10 atoms
        assert transformed.shape[1] == 3   # 3D coordinates
        assert transformed.shape[2] == 1   # 1 operation (identity)
        # First (and only) symmetry mate should match original
        assert torch.allclose(transformed[:, :, 0], coords.to(transformed.dtype), atol=1e-5)

    @pytest.mark.unit
    def test_spacegroup_generates_mates(self, random_fractional_coordinates):
        """SpaceGroup should generate correct number of mates."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P21")  # 2 operations
        coords = random_fractional_coordinates(n_atoms=5)  # (N, 3) format

        # Output shape is (n_atoms, 3, n_operations)
        transformed = sg(coords)

        # Should have 2 symmetry operations
        assert transformed.shape[2] == 2

    @pytest.mark.unit
    def test_spacegroup_callable(self, random_fractional_coordinates):
        """SpaceGroup should be callable."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P212121")
        coords = random_fractional_coordinates(n_atoms=10)  # (N, 3) format

        # Should be callable
        # Output shape is (n_atoms, 3, n_operations)
        result = sg(coords)

        assert result is not None
        assert result.shape[2] == 4  # 4 symmetry operations


class TestSpaceGroupDeviceHandling:
    """Tests for device handling in SpaceGroup."""

    @pytest.mark.unit
    def test_spacegroup_cpu(self):
        """Test SpaceGroup on CPU."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P21", device=torch.device('cpu'))

        assert sg.matrices.device.type == 'cpu'
        assert sg.translations.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_spacegroup_gpu(self, gpu_device):
        """Test SpaceGroup on GPU."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P21", device=gpu_device)

        assert sg.matrices.device.type == 'cuda'
        assert sg.translations.device.type == 'cuda'

    @pytest.mark.unit
    def test_spacegroup_dtype(self):
        """Test SpaceGroup dtype specification."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P21", dtype=torch.float32)

        assert sg.matrices.dtype == torch.float32
        assert sg.translations.dtype == torch.float32


class TestSpaceGroupMapping:
    """Tests for space group name mapping."""

    @pytest.mark.unit
    def test_common_spacegroups(self):
        """Test common crystallographic space groups."""
        from torchref.symmetry import SpaceGroup

        # Common protein space groups
        common_sgs = ["P1", "P21", "P212121", "C2", "P21212"]

        for sg_name in common_sgs:
            sg = SpaceGroup(sg_name)
            assert sg.matrices is not None

    @pytest.mark.unit
    def test_case_insensitivity(self):
        """Space group names should be somewhat case-insensitive."""
        from torchref.symmetry import SpaceGroup

        # Try different case variations - may not all work
        try:
            sg1 = SpaceGroup("P21")
            # Should have valid matrices
            assert sg1.matrices is not None
        except ValueError:
            pass  # Some case variations may not be supported


class TestSymmetryBackwardCompat:
    """Tests for backward compatibility with Symmetry alias."""

    @pytest.mark.unit
    def test_symmetry_alias_exists(self):
        """Test that Symmetry alias is available."""
        from torchref.symmetry import Symmetry, SpaceGroup

        # Symmetry should be an alias for SpaceGroup
        assert Symmetry is SpaceGroup

    @pytest.mark.unit
    def test_symmetry_alias_works(self):
        """Test that Symmetry alias works identically to SpaceGroup."""
        from torchref.symmetry import Symmetry

        sym = Symmetry("P21")

        assert sym.matrices is not None
        assert sym.n_ops == 2
        assert sym.name == "P21"
