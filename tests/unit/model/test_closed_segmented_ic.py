"""
Tests for ClosedSegmentedInternalCoordinateTensor.

Tests cover:
- Construction from a real PDB structure
- Coordinate recovery (forward should reproduce initial xyz)
- Gradient flow through all parameters including junction closure
- Shake produces valid perturbed structures
- Newton solver re-solves after shake
- Caching (CachedForwardMixin integration)
- Fix/freeze/refine interface
- Model.use_internal_coordinates() integration
"""

import math
from pathlib import Path

import pytest
import torch

from torchref.model.model import Model
from torchref.model.closed_segmented_internal_coordinates import (
    ClosedSegmentedInternalCoordinateTensor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def pdb_3gr5():
    """Path to 3GR5 PDB file (small test structure)."""
    path = Path(__file__).parents[2] / "files" / "pdb" / "3GR5.pdb"
    if not path.exists():
        pytest.skip("3GR5.pdb not found in test data")
    return str(path)


@pytest.fixture(scope="module")
def model_3gr5(pdb_3gr5):
    """Loaded Model from 3GR5."""
    model = Model(verbose=0)
    model.load_pdb(pdb_3gr5)
    return model


@pytest.fixture
def ic_tensor(model_3gr5):
    """ClosedSegmentedInternalCoordinateTensor from 3GR5."""
    model = model_3gr5
    current_xyz = model.xyz().detach()
    return ClosedSegmentedInternalCoordinateTensor(
        current_xyz,
        pdb=model.pdb,
        n_aa_per_segment=18,
        junction_size=3,
        bond_cutoff=2.0,
        prefer_loops=True,
        requires_grad=True,
        dtype=current_xyz.dtype,
        device=current_xyz.device,
    )


@pytest.fixture
def original_xyz(model_3gr5):
    """Original Cartesian coordinates."""
    return model_3gr5.xyz().detach().clone()


# =============================================================================
# Construction tests
# =============================================================================


class TestConstruction:
    def test_basic_attributes(self, ic_tensor, model_3gr5):
        """Check that basic attributes are set correctly."""
        assert ic_tensor.n_atoms == model_3gr5.xyz().shape[0]
        assert ic_tensor.n_segments > 0
        assert ic_tensor.junction_size == 3
        assert ic_tensor.n_aa_per_segment == 18

    def test_has_junctions(self, ic_tensor):
        """Structure should have at least one junction."""
        assert ic_tensor.n_junctions > 0

    def test_parameters_exist(self, ic_tensor):
        """All expected parameters should exist."""
        assert ic_tensor.bond_lengths is not None
        assert ic_tensor.angles is not None
        assert ic_tensor.torsions is not None
        assert ic_tensor.segment_positions is not None
        assert ic_tensor.segment_orientations is not None

    def test_parameters_require_grad(self, ic_tensor):
        """Parameters should require gradients."""
        assert ic_tensor.bond_lengths.requires_grad
        assert ic_tensor.angles.requires_grad
        assert ic_tensor.torsions.requires_grad
        assert ic_tensor.segment_positions.requires_grad
        assert ic_tensor.segment_orientations.requires_grad

    def test_junction_buffers_exist(self, ic_tensor):
        """Junction-related buffers should exist."""
        assert hasattr(ic_tensor, "junction_bond_lengths")
        assert hasattr(ic_tensor, "junction_nerf_angles")
        assert hasattr(ic_tensor, "junction_omega")
        assert hasattr(ic_tensor, "junction_psi_prev")
        assert hasattr(ic_tensor, "junction_post_bond_length")
        assert hasattr(ic_tensor, "junction_post_bond_angle")

    def test_junction_solver_exists(self, ic_tensor):
        """Junction solver module should be properly initialized."""
        assert hasattr(ic_tensor, "junction_solver")
        assert ic_tensor.junction_solver.n_junctions == ic_tensor.n_junctions

    def test_repr(self, ic_tensor):
        """Repr should contain key info."""
        r = repr(ic_tensor)
        assert "ClosedSegmentedInternalCoordinateTensor" in r
        assert "n_junctions=" in r
        assert "n_segments=" in r

    def test_no_overlap_segments_junctions(self, ic_tensor):
        """No atom should be in both a segment and a junction."""
        seg_atoms = set()
        for seg in ic_tensor._planned_segments:
            seg_reskeys = set(seg)
            seg_atoms.update(seg_reskeys)

        junc_atoms = set()
        for junc in ic_tensor._planned_junctions:
            junc_reskeys = set(junc)
            junc_atoms.update(junc_reskeys)

        overlap = seg_atoms & junc_atoms
        assert len(overlap) == 0, f"Overlap residues: {overlap}"


# =============================================================================
# Coordinate recovery tests
# =============================================================================


class TestCoordinateRecovery:
    def test_forward_reproduces_xyz(self, ic_tensor, original_xyz):
        """forward() should reproduce original coordinates within tolerance."""
        recovered_xyz = ic_tensor()
        max_diff = (original_xyz - recovered_xyz).abs().max().item()
        assert max_diff < 0.01, f"Max coordinate difference: {max_diff:.6f} A"

    def test_non_junction_atoms_recovery(self, ic_tensor, original_xyz):
        """Non-junction atoms should recover with higher precision."""
        recovered_xyz = ic_tensor()
        non_junction_mask = ~ic_tensor.is_junction_atom
        non_junc_diff = (
            original_xyz[non_junction_mask] - recovered_xyz[non_junction_mask]
        ).abs().max().item()
        assert non_junc_diff < 0.005, \
            f"Non-junction max diff: {non_junc_diff:.6f} A"

    def test_junction_backbone_recovery(self, ic_tensor, original_xyz):
        """Junction backbone atoms should recover within Newton tolerance."""
        recovered_xyz = ic_tensor()
        junc_bb_mask = ic_tensor.is_junction_backbone
        if junc_bb_mask.any():
            junc_bb_diff = (
                original_xyz[junc_bb_mask] - recovered_xyz[junc_bb_mask]
            ).abs().max().item()
            assert junc_bb_diff < 0.01, \
                f"Junction backbone max diff: {junc_bb_diff:.6f} A"

    def test_deterministic(self, ic_tensor):
        """Two forward calls should produce identical results."""
        xyz1 = ic_tensor()
        xyz2 = ic_tensor()
        assert torch.allclose(xyz1, xyz2, atol=1e-10), \
            f"Non-deterministic: max diff = {(xyz1-xyz2).abs().max():.2e}"


# =============================================================================
# Gradient tests
# =============================================================================


class TestGradientFlow:
    def test_backward_no_nan(self, ic_tensor):
        """Backward pass should produce finite gradients for all parameters."""
        xyz = ic_tensor()
        loss = xyz.sum()
        loss.backward()

        params = {
            "bond_lengths": ic_tensor.bond_lengths,
            "angles": ic_tensor.angles,
            "torsions": ic_tensor.torsions,
            "segment_positions": ic_tensor.segment_positions,
            "segment_orientations": ic_tensor.segment_orientations,
        }
        for name, param in params.items():
            if param.numel() > 0:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), \
                    f"Non-finite gradient for {name}"

    def test_torsion_grad_nonzero(self, ic_tensor):
        """Torsion gradients should be non-zero (the main DOFs)."""
        ic_tensor.zero_grad()
        xyz = ic_tensor()
        loss = xyz.sum()
        loss.backward()

        if ic_tensor.torsions.numel() > 0:
            grad_norm = ic_tensor.torsions.grad.abs().max().item()
            assert grad_norm > 1e-10, f"Torsion gradient too small: {grad_norm}"

    def test_segment_position_grad_nonzero(self, ic_tensor):
        """Segment position gradients should be non-zero."""
        ic_tensor.zero_grad()
        xyz = ic_tensor()
        loss = xyz.sum()
        loss.backward()

        grad_norm = ic_tensor.segment_positions.grad.abs().max().item()
        assert grad_norm > 0, "Segment position gradient is zero"


# =============================================================================
# Shake and re-solve tests
# =============================================================================


class TestShake:
    def test_shake_changes_coordinates(self, ic_tensor, original_xyz):
        """Shake should produce different coordinates."""
        shaken_xyz = ic_tensor.shake(magnitude=0.1)
        assert not torch.allclose(original_xyz, shaken_xyz, atol=1e-3), \
            "Shake did not change coordinates"

    def test_shake_produces_finite(self, ic_tensor):
        """Shaken coordinates should be finite."""
        shaken_xyz = ic_tensor.shake(magnitude=0.1)
        assert torch.isfinite(shaken_xyz).all(), "Shake produced non-finite coords"

    def test_shake_preserves_atom_count(self, ic_tensor):
        """Shake should preserve atom count."""
        shaken_xyz = ic_tensor.shake(magnitude=0.1)
        assert shaken_xyz.shape[0] == ic_tensor.n_atoms

    def test_shake_junction_resolves(self, ic_tensor):
        """After shake, junction solver should re-solve for closure."""
        _ = ic_tensor.shake(magnitude=0.05)
        # Forward pass after shake should work without error
        xyz = ic_tensor()
        assert torch.isfinite(xyz).all(), "Forward after shake produced non-finite coords"

    def test_bond_lengths_positive_after_shake(self, ic_tensor):
        """Bond lengths should remain positive after shake."""
        _ = ic_tensor.shake(magnitude=1.0)
        assert (ic_tensor.bond_lengths.data >= 0.5).all(), \
            "Bond lengths went below 0.5 A"

    def test_angles_valid_after_shake(self, ic_tensor):
        """Angles should remain in valid range after shake."""
        _ = ic_tensor.shake(magnitude=1.0)
        if ic_tensor.angles.numel() > 0:
            assert (ic_tensor.angles.data >= 0.1).all(), "Angle below 0.1 rad"
            assert (ic_tensor.angles.data <= math.pi - 0.1).all(), \
                "Angle above pi - 0.1 rad"


# =============================================================================
# Caching tests
# =============================================================================


class TestCaching:
    def test_second_call_uses_cache(self, ic_tensor):
        """CachedForwardMixin should cache forward() results."""
        # First call populates cache
        xyz1 = ic_tensor()
        # Second call (no param changes) should return cached result
        xyz2 = ic_tensor()
        # Results should be identical (same object or exact match)
        assert torch.allclose(xyz1, xyz2, atol=1e-10)

    def test_param_change_invalidates_cache(self, ic_tensor):
        """Changing a parameter via optimizer step should invalidate the cache."""
        xyz1 = ic_tensor().detach().clone()

        # Use an optimizer step — this increments param._version
        # (.data.add_() does NOT increment _version, so it won't invalidate)
        optimizer = torch.optim.SGD([ic_tensor.torsions], lr=1.0)
        ic_tensor.torsions.grad = torch.zeros_like(ic_tensor.torsions)
        ic_tensor.torsions.grad[0] = -0.5  # Will add 0.5 via SGD step
        optimizer.step()

        xyz2 = ic_tensor()
        assert not torch.allclose(xyz1, xyz2, atol=1e-3), \
            "Cache not invalidated after optimizer step"


# =============================================================================
# Fix / freeze / refine tests
# =============================================================================


class TestFixFreeze:
    def test_fix_all(self, ic_tensor):
        """fix_all should mark all atoms as fixed."""
        ic_tensor.fix_all()
        assert ic_tensor.n_fixed == ic_tensor.n_atoms
        assert ic_tensor.n_refinable == 0

    def test_refine_all(self, ic_tensor):
        """refine_all should mark all atoms as refinable."""
        ic_tensor.fix_all()
        ic_tensor.refine_all()
        assert ic_tensor.n_refinable == ic_tensor.n_atoms
        assert ic_tensor.n_fixed == 0

    def test_fix_selection(self, ic_tensor):
        """Fix a subset of atoms."""
        mask = torch.zeros(ic_tensor.n_atoms, dtype=torch.bool)
        mask[:10] = True
        ic_tensor.refine_all()
        ic_tensor.fix(mask)
        assert ic_tensor.n_fixed == 10

    def test_frozen_atoms_use_fixed_xyz(self, ic_tensor, original_xyz):
        """Frozen atoms should use fixed_xyz, not reconstructed."""
        ic_tensor.refine_all()
        mask = torch.zeros(ic_tensor.n_atoms, dtype=torch.bool)
        mask[:10] = True
        ic_tensor.fix(mask, freeze_at_current=True)

        # Perturb parameters
        with torch.no_grad():
            ic_tensor.segment_positions.data += 0.1

        xyz = ic_tensor()
        # Fixed atoms should match fixed_xyz
        assert torch.allclose(
            xyz[mask], ic_tensor.fixed_xyz[mask], atol=1e-5
        )


# =============================================================================
# Model integration tests
# =============================================================================


class TestModelIntegration:
    def test_use_internal_coordinates(self, pdb_3gr5):
        """Model.use_internal_coordinates should create the IC tensor."""
        model = Model(verbose=0)
        model.load_pdb(pdb_3gr5)
        original_xyz = model.xyz().detach().clone()

        model.use_internal_coordinates(n_aa_per_segment=18)

        assert isinstance(model.xyz, ClosedSegmentedInternalCoordinateTensor)
        recovered_xyz = model.xyz()
        max_diff = (original_xyz - recovered_xyz).abs().max().item()
        assert max_diff < 0.01, f"Model integration: max diff = {max_diff:.6f} A"

    def test_model_backward(self, pdb_3gr5):
        """Backward pass through model with IC parametrization."""
        model = Model(verbose=0)
        model.load_pdb(pdb_3gr5)
        model.use_internal_coordinates(n_aa_per_segment=18)

        xyz = model.xyz()
        loss = xyz.sum()
        loss.backward()

        # Check at least torsion gradients exist
        assert model.xyz.torsions.grad is not None
        assert torch.isfinite(model.xyz.torsions.grad).all()

    def test_model_without_prefer_loops(self, pdb_3gr5):
        """Model with prefer_loops=False should still work."""
        model = Model(verbose=0)
        model.load_pdb(pdb_3gr5)
        model.use_internal_coordinates(
            n_aa_per_segment=18, prefer_loops=False
        )
        xyz = model.xyz()
        assert torch.isfinite(xyz).all()


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    def test_small_segment_size(self, model_3gr5):
        """Small n_aa_per_segment should create more junctions."""
        current_xyz = model_3gr5.xyz().detach()
        ic_small = ClosedSegmentedInternalCoordinateTensor(
            current_xyz,
            pdb=model_3gr5.pdb,
            n_aa_per_segment=10,
            junction_size=3,
            bond_cutoff=2.0,
            prefer_loops=False,
            requires_grad=True,
        )
        ic_large = ClosedSegmentedInternalCoordinateTensor(
            current_xyz,
            pdb=model_3gr5.pdb,
            n_aa_per_segment=30,
            junction_size=3,
            bond_cutoff=2.0,
            prefer_loops=False,
            requires_grad=True,
        )
        assert ic_small.n_junctions >= ic_large.n_junctions

    def test_very_large_segment(self, model_3gr5):
        """Very large n_aa_per_segment should result in no junctions."""
        current_xyz = model_3gr5.xyz().detach()
        ic = ClosedSegmentedInternalCoordinateTensor(
            current_xyz,
            pdb=model_3gr5.pdb,
            n_aa_per_segment=1000,
            junction_size=3,
            bond_cutoff=2.0,
            prefer_loops=False,
            requires_grad=True,
        )
        assert ic.n_junctions == 0
        # Should still work as forward pass
        xyz = ic()
        assert torch.isfinite(xyz).all()

    def test_no_grad(self, model_3gr5):
        """IC tensor with requires_grad=False should work."""
        current_xyz = model_3gr5.xyz().detach()
        ic = ClosedSegmentedInternalCoordinateTensor(
            current_xyz,
            pdb=model_3gr5.pdb,
            n_aa_per_segment=18,
            junction_size=3,
            bond_cutoff=2.0,
            prefer_loops=False,
            requires_grad=False,
        )
        xyz = ic()
        assert torch.isfinite(xyz).all()
        assert not ic.torsions.requires_grad


# =============================================================================
# Closure gap diagnostic tests
# =============================================================================


class TestClosureGap:
    def test_max_closure_gap_property(self, ic_tensor):
        """max_closure_gap should return a finite, reasonable value."""
        gap = ic_tensor.max_closure_gap
        assert isinstance(gap, float)
        assert math.isfinite(gap)
        # C-N peptide bond is ~1.33 A; this measures the actual gap
        # which should be close to the peptide bond length
        assert gap < 2.0, f"Closure gap too large: {gap}"
