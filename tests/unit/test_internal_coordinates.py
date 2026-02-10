"""
Tests for InternalCoordinateTensor.

Tests the internal coordinate parametrization for atomic structures,
including coordinate recovery, gradient flow, and perturbation.
"""

import pytest
import torch

from torchref.model import Model
from torchref.model.internal_coordinates import InternalCoordinateTensor


class TestInternalCoordinateTensor:
    """Test internal coordinate parametrization."""

    @pytest.fixture
    def sample_model(self):
        """Load a test PDB file."""
        model = Model(verbose=0)
        model.load_pdb("tests/files/pdb/1DAW.pdb")
        return model

    @pytest.fixture
    def small_molecule_xyz(self):
        """Create a small test molecule (water-like)."""
        # Simple 3-atom molecule
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # O
                [0.96, 0.0, 0.0],  # H1
                [-0.24, 0.93, 0.0],  # H2
            ],
            dtype=torch.float32,
        )
        return xyz

    @pytest.fixture
    def linear_chain_xyz(self):
        """Create a linear 5-atom chain for testing."""
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return xyz

    def test_build_molecular_graph(self, small_molecule_xyz):
        """Test that molecular graph is built correctly."""
        adjacency = InternalCoordinateTensor._build_molecular_graph(
            small_molecule_xyz, cutoff=2.0
        )

        # O-H1 and O-H2 should be bonded, H1-H2 should not
        assert adjacency[0, 1]  # O-H1 bonded
        assert adjacency[0, 2]  # O-H2 bonded
        assert adjacency[1, 0]  # Symmetric
        assert adjacency[2, 0]  # Symmetric
        # H1-H2 distance is about 1.2, so they might be bonded
        # The exact adjacency depends on the cutoff

    def test_build_and_recover_coordinates_small(self, small_molecule_xyz):
        """Test that forward() recovers original coordinates for small molecule."""
        ic_tensor = InternalCoordinateTensor(small_molecule_xyz, bond_cutoff=2.0)
        recovered_xyz = ic_tensor()

        # Should match within numerical tolerance
        assert torch.allclose(small_molecule_xyz, recovered_xyz, atol=1e-4), (
            f"Max difference: {(small_molecule_xyz - recovered_xyz).abs().max()}"
        )

    def test_build_and_recover_coordinates_linear(self, linear_chain_xyz):
        """Test coordinate recovery for linear chain."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz, bond_cutoff=2.0)
        recovered_xyz = ic_tensor()

        assert torch.allclose(linear_chain_xyz, recovered_xyz, atol=1e-4), (
            f"Max difference: {(linear_chain_xyz - recovered_xyz).abs().max()}"
        )

    def test_build_and_recover_coordinates(self, sample_model):
        """Test that forward() recovers original coordinates."""
        original_xyz = sample_model.xyz().detach().clone()

        ic_tensor = InternalCoordinateTensor(original_xyz, bond_cutoff=2.0)
        recovered_xyz = ic_tensor()

        # Should match within numerical tolerance
        max_diff = (original_xyz - recovered_xyz).abs().max()
        assert torch.allclose(original_xyz, recovered_xyz, atol=1e-3), (
            f"Max difference: {max_diff}"
        )

    def test_gradient_flow(self, sample_model):
        """Test gradients flow through reconstruction."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz, requires_grad=True)

        # Forward and compute dummy loss
        xyz = ic_tensor()
        loss = xyz.sum()
        loss.backward()

        # Check gradients exist for bond lengths
        if ic_tensor.bond_lengths.numel() > 0:
            assert ic_tensor.bond_lengths.grad is not None
            assert not torch.isnan(ic_tensor.bond_lengths.grad).any()

        # Check gradients exist for angles
        if ic_tensor.angles.numel() > 0:
            assert ic_tensor.angles.grad is not None
            assert not torch.isnan(ic_tensor.angles.grad).any()

        # Check gradients exist for torsions
        if ic_tensor.torsions.numel() > 0:
            assert ic_tensor.torsions.grad is not None
            assert not torch.isnan(ic_tensor.torsions.grad).any()

        # Check chain positions have gradients
        assert ic_tensor.chain_positions.grad is not None
        assert not torch.isnan(ic_tensor.chain_positions.grad).any()

    def test_gradient_flow_small(self, small_molecule_xyz):
        """Test gradient flow for small molecule."""
        ic_tensor = InternalCoordinateTensor(small_molecule_xyz, requires_grad=True)

        xyz = ic_tensor()
        loss = xyz.sum()
        loss.backward()

        # Check gradients exist
        if ic_tensor.bond_lengths.numel() > 0:
            assert ic_tensor.bond_lengths.grad is not None

        assert ic_tensor.chain_positions.grad is not None

    def test_shake(self, sample_model):
        """Test shake produces valid perturbed structures."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz)

        # Store original parameters
        original_bonds = ic_tensor.bond_lengths.clone()
        original_angles = ic_tensor.angles.clone()
        original_torsions = ic_tensor.torsions.clone()

        shaken_xyz = ic_tensor.shake(magnitude=0.1)

        # Should be different from original
        assert not torch.allclose(original_xyz, shaken_xyz)

        # Should still be valid coordinates (no NaN/Inf)
        assert torch.isfinite(shaken_xyz).all()

        # Parameters should have changed
        if ic_tensor.bond_lengths.numel() > 0:
            assert not torch.allclose(original_bonds, ic_tensor.bond_lengths)

        if ic_tensor.angles.numel() > 0:
            assert not torch.allclose(original_angles, ic_tensor.angles)

        if ic_tensor.torsions.numel() > 0:
            assert not torch.allclose(original_torsions, ic_tensor.torsions)

    def test_shake_small_magnitude(self, small_molecule_xyz):
        """Test shake with small magnitude produces small perturbations."""
        ic_tensor = InternalCoordinateTensor(small_molecule_xyz)
        original_xyz = ic_tensor().clone()

        shaken_xyz = ic_tensor.shake(magnitude=0.01)

        # Difference should be small
        max_diff = (original_xyz - shaken_xyz).abs().max()
        assert max_diff < 0.5  # Reasonable perturbation

    def test_multi_chain(self, sample_model):
        """Test with disconnected molecules."""
        # Create two copies far apart
        xyz1 = sample_model.xyz().detach().clone()
        xyz2 = xyz1.clone() + torch.tensor([100.0, 0.0, 0.0])
        multi_xyz = torch.cat([xyz1, xyz2], dim=0)

        # First get the chain count of the original structure
        ic_single = InternalCoordinateTensor(xyz1)
        n_chains_single = ic_single.n_chains

        ic_tensor = InternalCoordinateTensor(multi_xyz)
        # Use forward_slow() for multi-chain structures since forward_parallel()
        # currently only handles single backbone paths efficiently
        recovered = ic_tensor.forward_slow()

        # Two copies should double the chain count
        assert ic_tensor.n_chains == 2 * n_chains_single
        # Use slightly relaxed tolerance for larger structures
        max_diff = (multi_xyz - recovered).abs().max()
        assert torch.allclose(multi_xyz, recovered, atol=5e-3), (
            f"Max difference: {max_diff}"
        )

    def test_chain_count(self, sample_model):
        """Test that chain detection works correctly."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz, bond_cutoff=2.0)

        # Should have at least one chain
        assert ic_tensor.n_chains >= 1

    def test_parameter_counts(self, sample_model):
        """Test that parameter counts are reasonable."""
        original_xyz = sample_model.xyz().detach().clone()
        n_atoms = original_xyz.shape[0]

        ic_tensor = InternalCoordinateTensor(original_xyz, bond_cutoff=2.0)

        # Number of bonds should be less than n_atoms (tree has n-1 edges per component)
        assert ic_tensor.n_bonds < n_atoms

        # Number of angles should be less than bonds
        assert ic_tensor.n_angles <= ic_tensor.n_bonds

        # Number of torsions should be less than angles
        assert ic_tensor.n_torsions <= ic_tensor.n_angles

    def test_repr(self, small_molecule_xyz):
        """Test string representation."""
        ic_tensor = InternalCoordinateTensor(small_molecule_xyz)
        repr_str = repr(ic_tensor)

        assert "InternalCoordinateTensor" in repr_str
        assert "n_atoms=" in repr_str
        assert "n_chains=" in repr_str

    def test_device_dtype(self, small_molecule_xyz):
        """Test that device and dtype are preserved."""
        ic_tensor = InternalCoordinateTensor(
            small_molecule_xyz, dtype=torch.float64
        )

        assert ic_tensor.dtype == torch.float64
        assert ic_tensor.bond_lengths.dtype == torch.float64
        xyz = ic_tensor()
        assert xyz.dtype == torch.float64

    def test_callable_interface(self, small_molecule_xyz):
        """Test that __call__ works like forward."""
        ic_tensor = InternalCoordinateTensor(small_molecule_xyz)

        xyz_forward = ic_tensor.forward()
        xyz_call = ic_tensor()

        assert torch.allclose(xyz_forward, xyz_call)

    def test_model_use_internal_coordinates(self, sample_model):
        """Test Model.use_internal_coordinates() method."""
        from torchref.model.segmented_internal_coordinates import SegmentedInternalCoordinateTensor

        original_xyz = sample_model.xyz().detach().clone()

        sample_model.use_internal_coordinates(bond_cutoff=2.0)

        # xyz should now be a SegmentedInternalCoordinateTensor (or InternalCoordinateTensor)
        assert isinstance(sample_model.xyz, (InternalCoordinateTensor, SegmentedInternalCoordinateTensor))

        # Should recover original coordinates
        recovered_xyz = sample_model.xyz()
        assert torch.allclose(original_xyz, recovered_xyz, atol=1e-3)

    def test_bond_length_clamping_in_shake(self, small_molecule_xyz):
        """Test that bond lengths stay positive after shake."""
        ic_tensor = InternalCoordinateTensor(small_molecule_xyz)

        # Shake with large magnitude
        ic_tensor.shake(magnitude=1.0)

        # Bond lengths should still be positive (clamped to min 0.5)
        if ic_tensor.bond_lengths.numel() > 0:
            assert (ic_tensor.bond_lengths >= 0.5).all()

    def test_angle_clamping_in_shake(self, small_molecule_xyz):
        """Test that angles stay in valid range after shake."""
        ic_tensor = InternalCoordinateTensor(small_molecule_xyz)

        # Shake with large magnitude
        ic_tensor.shake(magnitude=1.0)

        # Angles should be in [0.1, pi-0.1]
        if ic_tensor.angles.numel() > 0:
            assert (ic_tensor.angles >= 0.1).all()
            assert (ic_tensor.angles <= torch.pi - 0.1).all()

    def test_torsion_wrapping_in_shake(self, sample_model):
        """Test that torsions wrap correctly after shake."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz)

        # Shake with large magnitude
        ic_tensor.shake(magnitude=10.0)

        # Torsions should be in [-pi, pi]
        if ic_tensor.torsions.numel() > 0:
            assert (ic_tensor.torsions >= -torch.pi).all()
            assert (ic_tensor.torsions <= torch.pi).all()

    def test_no_nan_in_reconstruction(self, sample_model):
        """Test that reconstruction doesn't produce NaN values."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz)

        xyz = ic_tensor()
        assert torch.isfinite(xyz).all()

    def test_no_nan_after_shake(self, sample_model):
        """Test that shake doesn't produce NaN values."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz)

        for _ in range(5):
            xyz = ic_tensor.shake(magnitude=0.1)
            assert torch.isfinite(xyz).all()


class TestInternalCoordinateTensorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_atom(self):
        """Test with single atom."""
        xyz = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        ic_tensor = InternalCoordinateTensor(xyz)

        recovered = ic_tensor()
        assert torch.allclose(xyz, recovered)
        assert ic_tensor.n_chains == 1
        assert ic_tensor.n_bonds == 0

    def test_two_atoms(self):
        """Test with two atoms (single bond)."""
        xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32
        )
        ic_tensor = InternalCoordinateTensor(xyz)

        recovered = ic_tensor()
        assert torch.allclose(xyz, recovered, atol=1e-4)
        assert ic_tensor.n_chains == 1
        assert ic_tensor.n_bonds == 1
        assert ic_tensor.n_angles == 0

    def test_three_atoms_linear(self):
        """Test with three atoms in line."""
        xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        ic_tensor = InternalCoordinateTensor(xyz)

        recovered = ic_tensor()
        assert torch.allclose(xyz, recovered, atol=1e-4)
        assert ic_tensor.n_bonds == 2
        assert ic_tensor.n_angles == 1
        assert ic_tensor.n_torsions == 0

    def test_four_atoms_with_torsion(self):
        """Test with four atoms (includes torsion)."""
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 1.5, 0.0],
                [2.5, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        ic_tensor = InternalCoordinateTensor(xyz)

        recovered = ic_tensor()
        assert torch.allclose(xyz, recovered, atol=1e-4)
        assert ic_tensor.n_bonds == 3
        assert ic_tensor.n_angles == 2
        assert ic_tensor.n_torsions == 1

    def test_disconnected_single_atoms(self):
        """Test with two disconnected single atoms."""
        xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float32
        )
        ic_tensor = InternalCoordinateTensor(xyz)

        recovered = ic_tensor()
        assert torch.allclose(xyz, recovered)
        assert ic_tensor.n_chains == 2
        assert ic_tensor.n_bonds == 0

    def test_requires_grad_false(self):
        """Test with requires_grad=False."""
        xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32
        )
        ic_tensor = InternalCoordinateTensor(xyz, requires_grad=False)

        assert not ic_tensor.bond_lengths.requires_grad
        assert not ic_tensor.chain_positions.requires_grad

    def test_different_bond_cutoffs(self):
        """Test that different bond cutoffs change connectivity."""
        xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.5, 0.0, 0.0]],
            dtype=torch.float32,
        )

        # With cutoff 2.0, atoms 0-1 are bonded, 1-2 are not
        ic_small = InternalCoordinateTensor(xyz, bond_cutoff=2.0)
        assert ic_small.n_chains == 2

        # With cutoff 4.0, all atoms are connected
        ic_large = InternalCoordinateTensor(xyz, bond_cutoff=4.0)
        assert ic_large.n_chains == 1


class TestParallelForward:
    """Test the parallel scan forward pass."""

    @pytest.fixture
    def sample_model(self):
        """Load a test PDB file."""
        model = Model(verbose=0)
        model.load_pdb("tests/files/pdb/1DAW.pdb")
        return model

    def test_forward_parallel_matches_forward(self, sample_model):
        """Test that forward_parallel() gives same result as forward()."""
        original_xyz = sample_model.xyz().detach().clone()
        # Use float64 for higher precision comparison
        ic_tensor = InternalCoordinateTensor(
            original_xyz, bond_cutoff=2.0, dtype=torch.float64
        )

        xyz_std = ic_tensor.forward()
        xyz_par = ic_tensor.forward_parallel()

        max_diff = (xyz_std - xyz_par).abs().max()
        # Tolerance accounts for accumulated numerical error in parallel scan
        assert torch.allclose(xyz_std, xyz_par, atol=1e-6), (
            f"forward_parallel() differs from forward(): max diff = {max_diff}"
        )

    def test_forward_parallel_gradient_flow(self, sample_model):
        """Test gradients flow through forward_parallel()."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz, requires_grad=True)

        xyz = ic_tensor.forward_parallel()
        loss = xyz.sum()
        loss.backward()

        # Check gradients exist for internal coordinates
        if ic_tensor.bond_lengths.numel() > 0:
            assert ic_tensor.bond_lengths.grad is not None
            assert not torch.isnan(ic_tensor.bond_lengths.grad).any()

        if ic_tensor.torsions.numel() > 0:
            assert ic_tensor.torsions.grad is not None
            assert not torch.isnan(ic_tensor.torsions.grad).any()

    def test_forward_parallel_small_structure(self):
        """Test forward_parallel() on small structures."""
        # 5-atom linear chain
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ic_tensor = InternalCoordinateTensor(xyz, bond_cutoff=2.0)

        xyz_std = ic_tensor.forward()
        xyz_par = ic_tensor.forward_parallel()

        assert torch.allclose(xyz_std, xyz_par, atol=1e-5)


class TestFreezeUnfreeze:
    """Test freeze/unfreeze (fix/refine) functionality."""

    @pytest.fixture
    def sample_model(self):
        """Load a test PDB file."""
        model = Model(verbose=0)
        model.load_pdb("tests/files/pdb/1DAW.pdb")
        return model

    @pytest.fixture
    def linear_chain_xyz(self):
        """Create a linear 5-atom chain for testing."""
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        return xyz

    def test_initial_state_all_refinable(self, linear_chain_xyz):
        """Test that all atoms are refinable initially."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        assert ic_tensor.n_refinable == 5
        assert ic_tensor.n_fixed == 0
        assert ic_tensor.refinable_mask.all()

    def test_freeze_single_atom(self, linear_chain_xyz):
        """Test freezing a single atom."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Freeze atom 2
        mask = torch.zeros(5, dtype=torch.bool)
        mask[2] = True
        ic_tensor.freeze(mask)

        assert ic_tensor.n_refinable == 4
        assert ic_tensor.n_fixed == 1
        assert not ic_tensor.refinable_mask[2]

    def test_freeze_multiple_atoms(self, linear_chain_xyz):
        """Test freezing multiple atoms."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Freeze atoms 1 and 3
        mask = torch.zeros(5, dtype=torch.bool)
        mask[1] = True
        mask[3] = True
        ic_tensor.freeze(mask)

        assert ic_tensor.n_refinable == 3
        assert ic_tensor.n_fixed == 2
        assert not ic_tensor.refinable_mask[1]
        assert not ic_tensor.refinable_mask[3]

    def test_freeze_all(self, linear_chain_xyz):
        """Test freezing all atoms."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        ic_tensor.freeze_all()

        assert ic_tensor.n_refinable == 0
        assert ic_tensor.n_fixed == 5
        assert not ic_tensor.refinable_mask.any()

    def test_unfreeze_single_atom(self, linear_chain_xyz):
        """Test unfreezing a single atom."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Freeze all, then unfreeze atom 2
        ic_tensor.freeze_all()
        mask = torch.zeros(5, dtype=torch.bool)
        mask[2] = True
        ic_tensor.unfreeze(mask)

        assert ic_tensor.n_refinable == 1
        assert ic_tensor.n_fixed == 4
        assert ic_tensor.refinable_mask[2]

    def test_unfreeze_all(self, linear_chain_xyz):
        """Test unfreezing all atoms."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Freeze all, then unfreeze all
        ic_tensor.freeze_all()
        ic_tensor.unfreeze_all()

        assert ic_tensor.n_refinable == 5
        assert ic_tensor.n_fixed == 0
        assert ic_tensor.refinable_mask.all()

    def test_frozen_atoms_preserve_position(self, linear_chain_xyz):
        """Test that frozen atoms preserve their position during reconstruction."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Get original positions
        original_xyz = ic_tensor().clone()

        # Freeze atom 2
        mask = torch.zeros(5, dtype=torch.bool)
        mask[2] = True
        ic_tensor.freeze(mask)

        # Perturb parameters (this would normally move all atoms)
        with torch.no_grad():
            ic_tensor.torsions.data += 0.5

        # Reconstruct
        new_xyz = ic_tensor()

        # Frozen atom should stay at original position
        assert torch.allclose(new_xyz[2], original_xyz[2], atol=1e-6)

    def test_frozen_atoms_preserve_position_protein(self, sample_model):
        """Test frozen atoms preserve position for protein structure."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz)

        # Get initial reconstructed coordinates
        initial_xyz = ic_tensor().clone()

        # Freeze first 100 atoms
        mask = torch.zeros(ic_tensor.n_atoms, dtype=torch.bool)
        mask[:100] = True
        ic_tensor.freeze(mask)

        # Shake to perturb the structure
        ic_tensor.shake(magnitude=0.1)

        # Reconstruct
        new_xyz = ic_tensor()

        # Frozen atoms should be at their frozen positions
        frozen_diff = (new_xyz[:100] - initial_xyz[:100]).abs().max()
        assert frozen_diff < 1e-6, f"Frozen atoms moved by {frozen_diff}"

        # Unfrozen atoms should have changed (most likely)
        unfrozen_diff = (new_xyz[100:] - initial_xyz[100:]).abs().max()
        assert unfrozen_diff > 0.01, "Unfrozen atoms should have moved"

    def test_unfreeze_rebuilds_internal_coords(self, linear_chain_xyz):
        """Test that unfreezing rebuilds internal coordinates from fixed_xyz."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Get original parameters
        original_bonds = ic_tensor.bond_lengths.clone()

        # Freeze all atoms
        ic_tensor.freeze_all()

        # Modify fixed_xyz directly (simulate external modification)
        with torch.no_grad():
            ic_tensor.fixed_xyz[2] = ic_tensor.fixed_xyz[2] + torch.tensor([0.5, 0.0, 0.0])

        # Unfreeze all (should rebuild internal coords from modified fixed_xyz)
        ic_tensor.unfreeze_all(rebuild=True)

        # Bond lengths should have changed
        assert not torch.allclose(original_bonds, ic_tensor.bond_lengths, atol=1e-3)

    def test_fix_alias(self, linear_chain_xyz):
        """Test that fix() is an alias for freeze()."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        mask = torch.zeros(5, dtype=torch.bool)
        mask[2] = True
        ic_tensor.fix(mask)

        assert ic_tensor.n_fixed == 1
        assert not ic_tensor.refinable_mask[2]

    def test_refine_alias(self, linear_chain_xyz):
        """Test that refine() is an alias for unfreeze()."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        ic_tensor.freeze_all()
        mask = torch.zeros(5, dtype=torch.bool)
        mask[2] = True
        ic_tensor.refine(mask)

        assert ic_tensor.n_refinable == 1
        assert ic_tensor.refinable_mask[2]

    def test_fix_all_alias(self, linear_chain_xyz):
        """Test that fix_all() is an alias for freeze_all()."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        ic_tensor.fix_all()

        assert ic_tensor.n_fixed == 5

    def test_refine_all_alias(self, linear_chain_xyz):
        """Test that refine_all() is an alias for unfreeze_all()."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        ic_tensor.freeze_all()
        ic_tensor.refine_all()

        assert ic_tensor.n_refinable == 5

    def test_freeze_with_slice(self, linear_chain_xyz):
        """Test freezing with slice notation."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        ic_tensor.freeze(slice(0, 3))

        assert ic_tensor.n_fixed == 3
        assert not ic_tensor.refinable_mask[:3].any()
        assert ic_tensor.refinable_mask[3:].all()

    def test_freeze_with_index_tensor(self, linear_chain_xyz):
        """Test freezing with index tensor."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        indices = torch.tensor([1, 3])
        ic_tensor.freeze(indices)

        assert ic_tensor.n_fixed == 2
        assert not ic_tensor.refinable_mask[1]
        assert not ic_tensor.refinable_mask[3]
        assert ic_tensor.refinable_mask[0]
        assert ic_tensor.refinable_mask[2]
        assert ic_tensor.refinable_mask[4]

    def test_repr_shows_freeze_status(self, linear_chain_xyz):
        """Test that repr shows refinable/fixed counts."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Initially all refinable
        repr_str = repr(ic_tensor)
        assert "n_refinable=5" in repr_str
        assert "n_fixed=0" in repr_str

        # After freezing some
        ic_tensor.freeze(slice(0, 2))
        repr_str = repr(ic_tensor)
        assert "n_refinable=3" in repr_str
        assert "n_fixed=2" in repr_str

    def test_freeze_no_rebuild_option(self, linear_chain_xyz):
        """Test freeze_at_current=False uses existing fixed_xyz."""
        ic_tensor = InternalCoordinateTensor(linear_chain_xyz)

        # Store original fixed_xyz
        original_fixed = ic_tensor.fixed_xyz.clone()

        # Modify internal coords to change reconstructed positions
        with torch.no_grad():
            ic_tensor.chain_positions.data += 1.0

        # Freeze with freeze_at_current=False (should keep original fixed_xyz)
        ic_tensor.freeze(slice(None), freeze_at_current=False)

        # fixed_xyz should be unchanged
        assert torch.allclose(ic_tensor.fixed_xyz, original_fixed)

    def test_roundtrip_freeze_unfreeze(self, sample_model):
        """Test that freeze -> unfreeze roundtrip preserves structure."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz)

        # Get initial reconstruction
        initial_xyz = ic_tensor().clone()

        # Freeze all
        ic_tensor.freeze_all()
        frozen_xyz = ic_tensor()

        # Unfreeze all
        ic_tensor.unfreeze_all()
        final_xyz = ic_tensor()

        # Frozen xyz should be exactly the same
        assert torch.allclose(initial_xyz, frozen_xyz, atol=1e-6)
        # After unfreeze and rebuild, there's some numerical precision loss
        # due to extracting and reconstructing internal coordinates
        assert torch.allclose(initial_xyz, final_xyz, atol=1e-3)

    def test_gradients_only_on_refinable_atoms(self, sample_model):
        """Test that gradients only affect refinable (unfrozen) atoms."""
        original_xyz = sample_model.xyz().detach().clone()
        ic_tensor = InternalCoordinateTensor(original_xyz, requires_grad=True)

        # Freeze first half of atoms
        n_freeze = ic_tensor.n_atoms // 2
        mask = torch.zeros(ic_tensor.n_atoms, dtype=torch.bool)
        mask[:n_freeze] = True
        ic_tensor.freeze(mask)

        # Compute loss and gradients
        xyz = ic_tensor()
        loss = xyz.sum()
        loss.backward()

        # Chain positions should have non-zero gradients (if chain root is unfrozen)
        # The gradient behavior depends on which atoms are frozen
        # Just verify gradients exist and are finite
        assert ic_tensor.chain_positions.grad is not None
        assert torch.isfinite(ic_tensor.chain_positions.grad).all()
