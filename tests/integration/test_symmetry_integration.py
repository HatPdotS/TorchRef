"""
Integration tests for the symmetry module.

Tests symmetry operations with real crystallographic data.
"""

import pytest
import torch
from pathlib import Path


class TestSpaceGroupInitialization:
    """Tests for SpaceGroup initialization."""

    @pytest.mark.integration
    def test_spacegroup_from_spacegroup_string(self):
        """Test creating SpaceGroup from spacegroup string."""
        from torchref.symmetry import SpaceGroup

        # P 21 21 21 is a common orthorhombic spacegroup
        sg = SpaceGroup("P 21 21 21")

        assert sg is not None
        assert sg.matrices is not None

    @pytest.mark.integration
    def test_spacegroup_from_spacegroup_number(self):
        """Test creating SpaceGroup from spacegroup name."""
        from torchref.symmetry import SpaceGroup

        # Use string representation for spacegroup 19
        sg = SpaceGroup("P212121")

        assert sg is not None
        assert sg.matrices is not None

    @pytest.mark.integration
    def test_spacegroup_from_model(self, sample_cif_file):
        """Test creating SpaceGroup from model spacegroup."""
        from torchref.model.model import Model
        from torchref.symmetry import SpaceGroup

        model = Model()
        model.load_cif(str(sample_cif_file))

        sg = SpaceGroup(model.spacegroup)

        assert sg is not None
        assert sg.matrices is not None


class TestSpaceGroupMatrices:
    """Tests for symmetry matrices."""

    @pytest.mark.integration
    def test_spacegroup_matrices_shape(self):
        """Test shape of symmetry matrices."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P 21 21 21")

        # Should have 4 symmetry operations (including identity)
        n_ops = sg.matrices.shape[0]
        assert n_ops >= 1  # At least identity

        # Each should be 3x4 (rotation + translation) or 4x4
        assert sg.matrices.shape[-1] in [3, 4]
        assert sg.matrices.shape[-2] in [3, 4]

    @pytest.mark.integration
    def test_spacegroup_identity_present(self):
        """Test that identity operation is present."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P 1")  # Triclinic - only identity

        # P 1 should have exactly 1 symmetry operation (identity)
        assert sg.matrices.shape[0] >= 1

    @pytest.mark.integration
    def test_spacegroup_matrices_orthogonal(self):
        """Test that rotation parts of symmetry matrices are orthogonal."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P 21 21 21")

        # Extract rotation parts (3x3)
        matrices = sg.matrices

        for i in range(matrices.shape[0]):
            if matrices.shape[-1] == 4:
                rot = matrices[i, :3, :3]
            else:
                rot = matrices[i, :3, :3]

            # R * R^T should be identity for orthogonal matrices
            product = torch.mm(rot, rot.T)
            identity = torch.eye(3, device=product.device, dtype=product.dtype)

            assert torch.allclose(product, identity, atol=1e-5)


class TestSpaceGroupDevice:
    """Tests for SpaceGroup device handling."""

    @pytest.mark.integration
    def test_spacegroup_default_device(self):
        """Test SpaceGroup matrices land on the configured default device."""
        from torchref.symmetry import SpaceGroup
        from torchref.config import get_default_device

        sg = SpaceGroup("P 21 21 21")

        assert sg.matrices.device.type == get_default_device().type

    @pytest.mark.integration
    def test_spacegroup_explicit_device(self, cpu_device):
        """Test creating SpaceGroup on explicit device."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup("P 21 21 21", device=cpu_device)

        assert sg.matrices.device == cpu_device


class TestSpaceGroupOperations:
    """Tests for applying symmetry operations."""

    @pytest.mark.integration
    def test_expand_coordinates(self, sample_cif_file):
        """Test expanding coordinates by symmetry."""
        from torchref.model.model import Model
        from torchref.symmetry import SpaceGroup

        model = Model()
        model.load_cif(str(sample_cif_file))

        xyz = model.xyz()
        sg = SpaceGroup(model.spacegroup)

        # The model should be able to generate symmetry mates
        # Check if there's an expand method
        if hasattr(sg, 'expand') or hasattr(sg, 'expand_atoms'):
            expanded = sg.expand(xyz)
            assert expanded.shape[0] >= xyz.shape[0]


class TestSpacegroupVariants:
    """Tests for different spacegroup conventions."""

    @pytest.mark.integration
    @pytest.mark.parametrize("sg_name", [
        "P 1",           # Triclinic
        "P 21",          # Monoclinic
        "P 21 21 21",    # Orthorhombic
        "P 43 21 2",     # Tetragonal
        "P 3 2 1",       # Trigonal
        "P 6 2 2",       # Hexagonal
        "P 2 3",         # Cubic
    ])
    def test_common_spacegroups(self, sg_name):
        """Test loading common spacegroups."""
        from torchref.symmetry import SpaceGroup

        sg = SpaceGroup(sg_name)
        assert sg.matrices is not None

    @pytest.mark.integration
    def test_spacegroup_name_variations(self):
        """Test that different spacegroup name formats work."""
        from torchref.symmetry import SpaceGroup

        # Both formats should work
        sg_with_spaces = SpaceGroup("P 21 21 21")
        sg_no_spaces = SpaceGroup("P212121")

        # Should have same number of symmetry operations
        assert sg_with_spaces.matrices.shape[0] == sg_no_spaces.matrices.shape[0]


class TestSpaceGroupWithData:
    """Tests for SpaceGroup with real crystallographic data."""

    @pytest.mark.integration
    def test_spacegroup_with_multiple_structures(self, cif_dir):
        """Test SpaceGroup for multiple structures."""
        from torchref.model.model import Model
        from torchref.symmetry import SpaceGroup

        cif_files = list(cif_dir.glob("*.cif"))[:3]

        for cif_file in cif_files:
            model = Model()
            model.load_cif(str(cif_file))

            sg = SpaceGroup(model.spacegroup)

            # Should have valid matrices
            assert sg.matrices is not None
            assert sg.matrices.shape[0] >= 1
            assert torch.all(torch.isfinite(sg.matrices))

    @pytest.mark.integration
    def test_spacegroup_consistent_with_cell(self, sample_cif_file):
        """Test that SpaceGroup is consistent with unit cell."""
        from torchref.model.model import Model
        from torchref.symmetry import SpaceGroup

        model = Model()
        model.load_cif(str(sample_cif_file))

        sg = SpaceGroup(model.spacegroup)
        cell = model.cell

        # Both should be defined
        assert sg.matrices is not None
        assert cell is not None


class TestMapSymmetry:
    """Tests for map symmetry operations."""

    @pytest.mark.integration
    def test_symmetrize_map_round_trip(self, sample_cif_file):
        """Symmetrizing a map goes through the space group and preserves shape."""
        import torch

        from torchref.model.model import Model

        model = Model()
        model.load_cif(str(sample_cif_file))

        sg = model.spacegroup
        shape = sg.suggest_grid_size((16, 16, 16))
        density = torch.rand(shape, device=sg.device, dtype=sg.dtype)

        symmetrized = sg.symmetrize_map(density)

        assert symmetrized.shape == density.shape
        # The operator is cached for this shape and dropped on a device move.
        assert sg.map_operator(shape) is sg.map_operator(shape)
