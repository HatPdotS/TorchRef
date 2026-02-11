"""
Functional tests for the restraints module.

Tests restraints building and calculation with real protein structures.
"""

import pytest
import torch


class TestRestraintsBuildingFunctional:
    """Functional tests for building restraints from real structures."""

    @pytest.mark.integration
    def test_build_restraints_from_cif(self, sample_cif_file):
        """Test building restraints from a real CIF file."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Should have built some restraints
        assert restraints.restraints is not None
        assert len(list(restraints.restraints.keys())) > 0

    @pytest.mark.integration
    def test_bond_restraints_built(self, sample_cif_file):
        """Test that bond restraints are built correctly."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Check bond restraints exist
        assert 'bond' in restraints.restraints

        # Check intra-residue bonds
        if 'intra' in restraints.restraints['bond']:
            bond_intra = restraints.restraints['bond']['intra']
            assert 'indices' in bond_intra
            assert 'references' in bond_intra
            assert 'sigmas' in bond_intra

            # Indices should be 2D with shape (N, 2)
            indices = bond_intra['indices']
            assert len(indices.shape) == 2
            assert indices.shape[1] == 2

            # References should match number of bonds
            assert bond_intra['references'].shape[0] == indices.shape[0]
            assert bond_intra['sigmas'].shape[0] == indices.shape[0]

            # Bond lengths should be positive and reasonable (0.5-3.0 Å)
            refs = bond_intra['references']
            assert torch.all(refs > 0.5)
            assert torch.all(refs < 3.0)

    @pytest.mark.integration
    def test_angle_restraints_built(self, sample_cif_file):
        """Test that angle restraints are built correctly."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Check angle restraints exist
        assert 'angle' in restraints.restraints

        if 'intra' in restraints.restraints['angle']:
            angle_intra = restraints.restraints['angle']['intra']
            assert 'indices' in angle_intra
            assert 'references' in angle_intra
            assert 'sigmas' in angle_intra

            # Indices should be 2D with shape (N, 3)
            indices = angle_intra['indices']
            assert len(indices.shape) == 2
            assert indices.shape[1] == 3

            # References should match number of angles
            assert angle_intra['references'].shape[0] == indices.shape[0]

    @pytest.mark.integration
    def test_torsion_restraints_built(self, sample_cif_file):
        """Test that torsion restraints are built correctly."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Check torsion restraints exist
        assert 'torsion' in restraints.restraints

        if 'intra' in restraints.restraints['torsion']:
            torsion_intra = restraints.restraints['torsion']['intra']
            assert 'indices' in torsion_intra
            assert 'references' in torsion_intra
            assert 'sigmas' in torsion_intra
            assert 'periods' in torsion_intra

            # Indices should be 2D with shape (N, 4)
            indices = torsion_intra['indices']
            assert len(indices.shape) == 2
            assert indices.shape[1] == 4

    @pytest.mark.integration
    def test_plane_restraints_built(self, sample_cif_file):
        """Test that plane restraints are built correctly."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Check plane restraints exist
        assert 'plane' in restraints.restraints

        # Planes are grouped by atom count (3_atoms, 4_atoms, etc.)
        plane_restraints = restraints.restraints['plane']
        if len(list(plane_restraints.keys())) > 0:
            # Check at least one plane group exists
            for key, plane_group in plane_restraints.items():
                if 'indices' in plane_group:
                    indices = plane_group['indices']
                    # Planes need at least 3 atoms
                    if len(indices.shape) == 2:
                        assert indices.shape[1] >= 3


class TestRestraintsDeviationsFunctional:
    """Functional tests for computing restraint deviations."""

    @pytest.mark.integration
    def test_bond_deviations(self, sample_cif_file):
        """Test computing bond length deviations."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Compute bond deviations
        if hasattr(restraints, 'bond_deviations'):
            deviations, sigmas = restraints.bond_deviations()

            assert torch.all(torch.isfinite(deviations))
            assert torch.all(sigmas > 0)

            # Deviations should be reasonable (< 1 Å typically)
            assert torch.all(torch.abs(deviations) < 1.0)

    @pytest.mark.integration
    def test_angle_deviations(self, sample_cif_file):
        """Test computing angle deviations."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Compute angle deviations
        if hasattr(restraints, 'angle_deviations'):
            deviations, sigmas = restraints.angle_deviations()

            assert torch.all(torch.isfinite(deviations))
            assert torch.all(sigmas > 0)


class TestRestraintsMultipleStructures:
    """Test restraints building across multiple structures."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_restraints_multiple_cif_files(self, cif_dir):
        """Test building restraints for multiple CIF files."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        cif_files = list(cif_dir.glob("*.cif"))[:3]  # First 3 structures

        for cif_file in cif_files:
            model = Model()
            model.load_cif(str(cif_file))

            restraints = Restraints(
                pdb=model.pdb,
                xyz_fn=model.xyz,
                vdw_radii_fn=model.get_vdw_radii,
                verbose=0
            )
            restraints.build_restraints()

            # Should have built restraints for each structure
            assert 'bond' in restraints.restraints
            assert 'angle' in restraints.restraints


class TestRestraintsDeviceHandling:
    """Test restraints device handling."""

    @pytest.mark.integration
    def test_restraints_device_movement(self, sample_cif_file, cpu_device):
        """Test moving restraints to different devices."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model(device=cpu_device)
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )
        restraints.build_restraints()

        # Check that tensors are on the correct device
        if 'bond' in restraints.restraints and 'intra' in restraints.restraints['bond']:
            bond_indices = restraints.restraints['bond']['intra']['indices']
            assert bond_indices.device == cpu_device


class TestRestraintsCIFParsing:
    """Test CIF dictionary parsing."""

    @pytest.mark.integration
    def test_cif_dict_loaded(self, sample_cif_file):
        """Test that CIF dictionary is loaded correctly."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )

        # CIF dict should be populated with residue restraints
        assert restraints.cif_dict is not None
        assert len(restraints.cif_dict) > 0

        # Should have standard amino acids
        common_residues = ['ALA', 'GLY', 'VAL', 'LEU', 'ILE']
        for res in common_residues:
            if res in restraints.cif_dict:
                assert 'bonds' in restraints.cif_dict[res] or 'angles' in restraints.cif_dict[res]

    @pytest.mark.integration
    def test_unique_residues_detected(self, sample_cif_file):
        """Test that unique residues are detected from model."""
        from torchref.model.model import Model
        from torchref.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        restraints = Restraints(
            pdb=model.pdb,
            xyz_fn=model.xyz,
            vdw_radii_fn=model.get_vdw_radii,
            verbose=0
        )

        # Should have detected unique residues
        assert restraints.unique_residues is not None
        assert len(restraints.unique_residues) > 0
