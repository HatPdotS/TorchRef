"""
Unit tests for torchref.topology.restraints

Tests the Restraints class for geometry restraints.
Note: Unit tests use mock data, not real file I/O.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestRestraintsInitialization:
    """Tests for Restraints initialization."""

    @pytest.mark.unit
    def test_restraints_empty_init(self):
        """Test Restraints can be initialized empty."""
        from torchref.topology.restraints import Restraints
        
        restraints = Restraints()

        assert restraints.topology is None

    @pytest.mark.unit
    def test_restraints_is_nn_module(self):
        """Restraints should be a nn.Module."""
        from torchref.topology.restraints import Restraints
        
        restraints = Restraints()
        
        assert isinstance(restraints, nn.Module)

    @pytest.mark.unit
    def test_restraints_verbose_setting(self):
        """Test verbosity setting."""
        from torchref.topology.restraints import Restraints
        
        restraints = Restraints(verbose=2)
        
        assert restraints.verbose == 2


class TestBondRestraintCalculations:
    """Tests for bond distance restraint calculations."""

    @pytest.mark.unit
    def test_bond_distance_calculation(self, random_coordinates):
        """Test bond distance calculation between two atoms."""
        coords = random_coordinates(n_atoms=10)
        
        # Distance between first two atoms
        atom1 = coords[0]
        atom2 = coords[1]
        
        distance = torch.sqrt(torch.sum((atom1 - atom2) ** 2))
        
        assert distance >= 0
        assert torch.isfinite(distance)

    @pytest.mark.unit
    def test_bond_restraint_loss(self, random_coordinates):
        """Test bond restraint loss calculation."""
        coords = random_coordinates(n_atoms=10)
        
        # Mock bond restraint: atoms 0-1 should be 1.5 Å apart
        target_distance = 1.5
        sigma = 0.02
        
        atom1 = coords[0]
        atom2 = coords[1]
        actual_distance = torch.sqrt(torch.sum((atom1 - atom2) ** 2))
        
        # Simple harmonic restraint: (d - d0)^2 / sigma^2
        loss = ((actual_distance - target_distance) / sigma) ** 2
        
        assert torch.isfinite(loss)
        assert loss >= 0


class TestAngleRestraintCalculations:
    """Tests for angle restraint calculations."""

    @pytest.mark.unit
    def test_angle_calculation(self, random_coordinates):
        """Test angle calculation between three atoms."""
        coords = random_coordinates(n_atoms=10)
        
        # Angle at atom 1 between atoms 0, 1, 2
        v1 = coords[0] - coords[1]  # Vector 1->0
        v2 = coords[2] - coords[1]  # Vector 1->2
        
        # Angle via dot product
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle_rad = torch.acos(cos_angle)
        angle_deg = angle_rad * 180 / torch.pi
        
        assert angle_deg >= 0
        assert angle_deg <= 180
        assert torch.isfinite(angle_deg)

    @pytest.mark.unit
    def test_angle_restraint_loss(self, random_coordinates):
        """Test angle restraint loss calculation."""
        coords = random_coordinates(n_atoms=10)
        
        # Mock angle restraint: angle at atom 1 should be 120°
        target_angle = 120.0  # degrees
        sigma = 2.0  # degrees
        
        v1 = coords[0] - coords[1]
        v2 = coords[2] - coords[1]
        
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        actual_angle = torch.acos(cos_angle) * 180 / torch.pi
        
        loss = ((actual_angle - target_angle) / sigma) ** 2
        
        assert torch.isfinite(loss)
        assert loss >= 0


class TestTorsionRestraintCalculations:
    """Tests for torsion angle restraint calculations."""

    @pytest.mark.unit
    def test_torsion_calculation(self):
        """``Restraints.torsions`` gives gemmi's (IUPAC) dihedral for four atoms."""
        import gemmi

        from torchref.config import get_float_dtype, get_int_dtype
        from torchref.topology.restraints import Restraints

        rows = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.5, 0.0], [3.5, 1.5, 0.5]]
        coords = torch.tensor(rows, dtype=get_float_dtype())
        idx = torch.tensor([[0, 1, 2, 3]], dtype=get_int_dtype())
        expected = np.degrees(
            gemmi.calculate_dihedral(*(gemmi.Position(*row) for row in rows))
        )

        torsion = Restraints().torsions(idx, coords)

        torch.testing.assert_close(torsion, coords.new_tensor([expected]))


class TestRestraintDeviceHandling:
    """Tests for device handling in restraint calculations."""

    @pytest.mark.unit
    def test_bond_distance_cpu(self, random_coordinates):
        """Test bond distance on CPU."""
        coords = random_coordinates(n_atoms=10)
        
        distance = torch.sqrt(torch.sum((coords[0] - coords[1]) ** 2))
        
        assert distance.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_bond_distance_gpu(self, random_coordinates, gpu_device):
        """Test bond distance on GPU."""
        coords = random_coordinates(n_atoms=10).to(gpu_device)
        
        distance = torch.sqrt(torch.sum((coords[0] - coords[1]) ** 2))
        
        assert distance.device.type == gpu_device.type


class TestRestraintNumericStability:
    """Tests for numeric stability in restraint calculations."""

    @pytest.mark.unit
    def test_angle_near_zero(self):
        """Test angle calculation for nearly collinear atoms."""
        # Nearly collinear atoms (angle ~0)
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.001, 0.0]  # Tiny deviation from collinear
        ], dtype=torch.float64)
        
        v1 = coords[0] - coords[1]
        v2 = coords[2] - coords[1]
        
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle) * 180 / torch.pi
        
        assert torch.isfinite(angle)
        # Should be close to 180 (collinear)
        assert angle > 170

    @pytest.mark.unit
    def test_angle_near_180(self):
        """Test angle calculation for atoms at ~180 degrees."""
        # Atoms at nearly 180 degrees
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.001, 0.0]  # Nearly back along original direction
        ], dtype=torch.float64)
        
        v1 = coords[0] - coords[1]
        v2 = coords[2] - coords[1]
        
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle) * 180 / torch.pi
        
        assert torch.isfinite(angle)


@pytest.mark.unit
def test_link_definitions_are_read_silently(capsys):
    """Reading the bundled link library writes nothing to stdout."""
    from torchref.topology.monomer.cif import read_link_definitions

    read_link_definitions.cache_clear()
    link_dict, link_list = read_link_definitions()

    assert "TRANS" in link_dict and link_list is not None
    assert capsys.readouterr().out == ""


_BOND_FREE = """\
CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
HETATM    1  O   HOH A   1       1.000   1.000   1.000  1.00 20.00           O
HETATM    2  O   HOH A   2       5.000   1.000   1.000  1.00 20.00           O
HETATM    3 NA    NA A   3       9.000   1.000   1.000  1.00 20.00          NA
END
"""


@pytest.fixture
def bond_free(tmp_path):
    """Restraints over two waters and an ion, which carry no geometry restraints."""
    from torchref.model.model import Model

    path = tmp_path / "waters.pdb"
    path.write_text(_BOND_FREE)
    model = Model(verbose=0, hydrogens="strip")
    model.load_pdb(str(path))
    return model.restraints, model.xyz().detach().double()


@pytest.mark.unit
def test_bond_free_model_gives_empty_deviations(bond_free):
    """An edge type with no restraints yields empty deviations, not a KeyError."""
    restraints, xyz = bond_free
    for deviations_of in (
        restraints.bond_deviations,
        restraints.angle_deviations,
        restraints.torsion_deviations_with_sigmas,
    ):
        deviations, sigmas = deviations_of(xyz)
        assert deviations.shape == (0,) and sigmas.shape == (0,)
        assert deviations.dtype == xyz.dtype


@pytest.mark.unit
def test_bond_free_empty_results_keep_the_input_dtype(bond_free):
    """Empty ADP differences and bond lengths come back in the caller's dtype."""
    restraints, xyz = bond_free
    adp = torch.full((xyz.shape[0],), 20.0, dtype=torch.float64)
    differences = restraints.adp_b_differences(adp)
    assert differences.shape == (0,) and differences.dtype == torch.float64
    assert restraints.bond_lengths(None, xyz).dtype == torch.float64
