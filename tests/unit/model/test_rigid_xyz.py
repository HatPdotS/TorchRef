"""Unit tests for RigidXYZTensor and the Model.use_rigid_xyz swap."""

import pytest
import torch

from torchref.base.alignment.rotation import rotation_matrix_euler_xyz
from torchref.model import ModelFT, RigidXYZTensor
from torchref.model.parameter_wrappers import MixedTensor


@pytest.fixture
def fresh_modelft(pdb_dir):
    """Function-scoped: ``use_rigid_xyz`` mutates the model in place."""
    m = ModelFT(verbose=0)
    m.load_pdb(str(pdb_dir / "1DAW.pdb"))
    return m


class TestRigidXYZTensor:
    @pytest.mark.unit
    def test_chain_grouping(self, fresh_modelft):
        n_chains_pdb = fresh_modelft.pdb["chainid"].nunique()
        m = fresh_modelft.use_rigid_xyz()
        assert m is fresh_modelft
        assert isinstance(m.xyz, RigidXYZTensor)
        assert m.xyz.n_chains == n_chains_pdb

    @pytest.mark.unit
    def test_identity_reconstruction(self, fresh_modelft):
        fresh_modelft.use_rigid_xyz()
        with torch.no_grad():
            diff = (fresh_modelft.xyz() - fresh_modelft.xyz.original_xyz).abs().max().item()
        assert diff < 1e-4

    @pytest.mark.unit
    def test_known_transform(self, fresh_modelft):
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz
        device = xyz.device
        dtype = xyz.dtype

        # Pick the first chain (index 0); apply a transform only to that chain.
        # euler_angles is in Angstrom-scaled units, so scale going in and read
        # the physical angle back through rotation_radians.
        ang_vec = torch.tensor([0.0, 0.05, 0.0], dtype=dtype, device=device)
        trans_vec = torch.tensor([0.2, -0.3, 0.5], dtype=dtype, device=device)
        with torch.no_grad():
            xyz.euler_angles.zero_()
            xyz.translations.zero_()
            xyz.euler_angles[0] = ang_vec * xyz.angle_scale[0]
            xyz.translations[0] = trans_vec

        center = xyz.chain_centers[0]
        R = rotation_matrix_euler_xyz(xyz.rotation_radians[0])
        atom_chain = xyz.chain_indices
        chain0_mobile = (atom_chain == 0) & xyz.mobile_mask
        non_mobile = ~xyz.mobile_mask

        expected_mobile = (
            (xyz.original_xyz[chain0_mobile] - center) @ R.T + center + trans_vec
        )

        with torch.no_grad():
            out = xyz()
            diff_mobile = (out[chain0_mobile] - expected_mobile).abs().max().item()
            diff_fixed = (
                (out[non_mobile] - xyz.original_xyz[non_mobile]).abs().max().item()
                if non_mobile.any() else 0.0
            )
        assert diff_mobile < 1e-4
        assert diff_fixed < 1e-6

    @pytest.mark.unit
    def test_xyz_parameters_lists_rigid_leaves(self, fresh_modelft):
        fresh_modelft.use_rigid_xyz()
        leaves = fresh_modelft.xyz.parameters()
        assert len(leaves) == 2
        assert leaves[0] is fresh_modelft.xyz.euler_angles
        assert leaves[1] is fresh_modelft.xyz.translations
        n_chains = fresh_modelft.xyz.n_chains
        assert leaves[0].shape == (n_chains, 3)
        assert leaves[1].shape == (n_chains, 3)

    @pytest.mark.unit
    def test_mass_weighted_centroid_matches_explicit(self, fresh_modelft):
        """``chain_centers`` should equal the mass-weighted COM of mobile atoms.

        ``Model.use_rigid_xyz`` threads atomic Z through as the centroid
        weight. For a chain dominated by C/N/O atoms, Z-weighted and
        uniform centroids differ by a small but nonzero amount — verify
        the implementation matches an explicit Z-weighted computation.
        """
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz
        Z = fresh_modelft.Z.to(dtype=xyz.dtype, device=xyz.device)

        # Expected mass-weighted center for chain 0 over mobile atoms.
        sel = (xyz.chain_indices == 0) & xyz.mobile_mask
        w = Z[sel]
        coords = xyz.original_xyz[sel]
        expected = (coords * w.unsqueeze(1)).sum(dim=0) / w.sum().clamp(min=1e-12)
        got = xyz.chain_centers[0]
        diff = (got - expected).abs().max().item()
        assert diff < 1e-4

        # And the uniform centroid should DIFFER (would be a bug to match).
        uniform = coords.mean(dim=0)
        assert (got - uniform).abs().max().item() > 1e-5

    @pytest.mark.unit
    def test_bake_preserves_forward_and_zeros_params(self, fresh_modelft):
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz
        device = xyz.device
        dtype = xyz.dtype

        # Apply a non-trivial rigid transform to the first chain.
        ang = torch.tensor([0.03, 0.07, -0.04], dtype=dtype, device=device)
        trans = torch.tensor([0.4, -0.2, 0.6], dtype=dtype, device=device)
        with torch.no_grad():
            xyz.euler_angles.zero_()
            xyz.translations.zero_()
            xyz.euler_angles[0] = ang * xyz.angle_scale[0]
            xyz.translations[0] = trans
            before = xyz().detach().clone()
            scale_before = xyz.angle_scale.clone()

        xyz.bake()

        # After bake: forward result is unchanged, params are zero,
        # original_xyz now equals the transformed pose.
        with torch.no_grad():
            after = xyz()
            diff_forward = (after - before).abs().max().item()
            diff_original = (xyz.original_xyz - before).abs().max().item()
        assert diff_forward < 1e-4
        assert diff_original < 1e-4
        assert torch.all(xyz.euler_angles == 0).item()
        assert torch.all(xyz.translations == 0).item()
        # Rigid motion preserves the radius of gyration, so the scale is
        # unchanged; if it drifted, later angles would mean something else.
        assert torch.allclose(xyz.angle_scale, scale_before, rtol=1e-6)

        # Chain centers should have moved by the translation on chain 0.
        # Centroid is mass-weighted (atomic Z) over MOBILE atoms; reconstruct
        # the expected center the same way bake() does it internally.
        mobile_chain0 = (xyz.chain_indices == 0) & xyz.mobile_mask
        w = xyz.atom_weights[mobile_chain0]
        expected_center0 = (
            (before[mobile_chain0] * w.unsqueeze(1)).sum(dim=0)
            / w.sum().clamp(min=1e-12)
        )
        diff_center = (xyz.chain_centers[0] - expected_center0).abs().max().item()
        assert diff_center < 1e-4

    @pytest.mark.unit
    def test_angle_scale_matches_explicit_radius_of_gyration(self, fresh_modelft):
        """``angle_scale`` is the per-chain RMS radius about the rotation centre,
        over mobile atoms, floored at 1 A."""
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz

        for c in range(xyz.n_chains):
            sel = (xyz.chain_indices == c) & xyz.mobile_mask
            d = xyz.original_xyz[sel] - xyz.chain_centers[c]
            expected = d.pow(2).sum(dim=1).mean().sqrt()
            assert torch.allclose(xyz.angle_scale[c], expected, rtol=1e-5)
        assert torch.all(xyz.angle_scale >= 1.0), "scale must be floored at 1 A"

    @pytest.mark.unit
    def test_angle_scale_tracks_non_rigid_update_fixed_values(self, fresh_modelft):
        """``update_fixed_values`` recomputes the scale, not just the centres.

        It accepts any coordinates, not only the rigid re-pose ``bake()`` supplies,
        and a rigid re-pose leaves the radius of gyration unchanged.
        """
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz
        before = xyz.angle_scale.clone()

        # A pure dilation is not a rigid motion, so the radius doubles.
        with torch.no_grad():
            centers = xyz.chain_centers[xyz.chain_indices]
            inflated = centers + 2.0 * (xyz.original_xyz - centers)
        xyz.update_fixed_values(inflated)

        assert torch.allclose(xyz.angle_scale, 2.0 * before, rtol=1e-4)

    @pytest.mark.unit
    def test_unit_angle_scale_reproduces_unscaled_behaviour(self, fresh_modelft):
        """``angle_scale`` of ones gives the unscaled parametrization exactly."""
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz
        dtype, device = xyz.dtype, xyz.device
        ang = torch.tensor([0.02, -0.03, 0.04], dtype=dtype, device=device)

        with torch.no_grad():
            xyz.angle_scale.fill_(1.0)
            xyz.euler_angles.zero_()
            xyz.euler_angles[0] = ang
        xyz.reset_forward_cache()

        center = xyz.chain_centers[0]
        R = rotation_matrix_euler_xyz(ang)
        sel = (xyz.chain_indices == 0) & xyz.mobile_mask
        expected = (xyz.original_xyz[sel] - center) @ R.T + center
        with torch.no_grad():
            assert torch.allclose(xyz()[sel], expected, atol=1e-4)

    @pytest.mark.unit
    def test_rotation_radians_is_the_descaled_angle(self, fresh_modelft):
        """``rotation_radians`` descales the angle, and ``copy()`` carries both."""
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz
        with torch.no_grad():
            xyz.euler_angles.copy_(torch.randn_like(xyz.euler_angles) * 0.1)

        assert torch.allclose(
            xyz.rotation_radians, xyz.euler_angles / xyz.angle_scale.unsqueeze(1)
        )
        clone = xyz.copy()
        assert torch.allclose(clone.angle_scale, xyz.angle_scale)
        assert torch.allclose(clone.euler_angles, xyz.euler_angles)
        with torch.no_grad():
            assert torch.allclose(clone(), xyz(), atol=1e-5)

        # An overridden scale must be carried, not re-derived from geometry:
        # the angles are copied raw, so a re-derived scale changes the pose.
        with torch.no_grad():
            xyz.angle_scale.fill_(1.0)
        xyz.reset_forward_cache()
        clone2 = xyz.copy()
        assert torch.allclose(clone2.angle_scale, torch.ones_like(xyz.angle_scale))
        with torch.no_grad():
            assert torch.allclose(clone2(), xyz(), atol=1e-5)

    @pytest.mark.unit
    def test_restore_commit_bakes_transform(self, fresh_modelft):
        fresh_modelft.use_rigid_xyz()
        xyz = fresh_modelft.xyz
        device = xyz.device
        dtype = xyz.dtype

        ang = torch.tensor([[0.0, 0.05, 0.0]], dtype=dtype, device=device)
        trans = torch.tensor([[0.2, -0.3, 0.5]], dtype=dtype, device=device)
        with torch.no_grad():
            xyz.euler_angles.copy_(ang)
            xyz.translations.copy_(trans)
            expected = xyz().detach().clone()

        fresh_modelft.restore_xyz_from_rigid(commit=True)
        assert isinstance(fresh_modelft.xyz, MixedTensor)
        with torch.no_grad():
            diff = (fresh_modelft.xyz() - expected).abs().max().item()
        assert diff < 1e-4

    @pytest.mark.unit
    def test_restore_no_commit_returns_original(self, fresh_modelft):
        original_container = fresh_modelft.xyz
        with torch.no_grad():
            pre = fresh_modelft.xyz().detach().clone()

        fresh_modelft.use_rigid_xyz()
        # Mutate the rigid transform so commit=False has something to discard.
        with torch.no_grad():
            fresh_modelft.xyz.euler_angles.fill_(0.1)
            fresh_modelft.xyz.translations.fill_(0.3)

        fresh_modelft.restore_xyz_from_rigid(commit=False)
        # The exact same MixedTensor object should be back in place.
        assert fresh_modelft.xyz is original_container
        with torch.no_grad():
            diff = (fresh_modelft.xyz() - pre).abs().max().item()
        assert diff == 0.0

    @pytest.mark.unit
    def test_multi_chain_grouping(self, pdb_dir):
        m = ModelFT(verbose=0)
        m.load_pdb(str(pdb_dir / "3E98.pdb"))
        chain_ids_pdb = list(m.pdb["chainid"].unique())
        m.use_rigid_xyz()
        assert m.xyz.n_chains == len(chain_ids_pdb)
        assert m.xyz.chain_id_order == chain_ids_pdb


class TestRigidFreezeRestore:
    """use_rigid_xyz freezes adp/u/occupancy; restore must re-enable exactly
    the groups that were refinable beforehand so per-atom / ADP refinement can
    resume (regression: the handoff left them frozen → empty LBFGS param set →
    ``max()`` crash in the next refine_adp)."""

    @pytest.mark.unit
    def test_use_rigid_freezes_then_restore_reenables(self, fresh_modelft):
        m = fresh_modelft
        adp0 = m.adp.refinable_params.numel()
        occ0 = m.occupancy.refinable_params.numel()
        assert adp0 > 0  # sanity: adp is refinable to start

        m.use_rigid_xyz()
        # Only rigid leaves refine during the step.
        assert m.adp.refinable_params.numel() == 0
        assert m.u.refinable_params.numel() == 0
        assert m.occupancy.refinable_params.numel() == 0

        m.restore_xyz_from_rigid(commit=True)
        # Pre-rigid refinable state is restored.
        assert m.adp.refinable_params.numel() == adp0
        assert m.occupancy.refinable_params.numel() == occ0

    @pytest.mark.unit
    def test_restore_no_commit_also_reenables(self, fresh_modelft):
        m = fresh_modelft
        adp0 = m.adp.refinable_params.numel()
        m.use_rigid_xyz()
        m.restore_xyz_from_rigid(commit=False)
        assert m.adp.refinable_params.numel() == adp0

    @pytest.mark.unit
    def test_pre_frozen_group_stays_frozen(self, fresh_modelft):
        m = fresh_modelft
        m.freeze("occupancy")
        adp0 = m.adp.refinable_params.numel()
        assert m.occupancy.refinable_params.numel() == 0

        m.use_rigid_xyz()
        m.restore_xyz_from_rigid(commit=True)
        # adp (refinable before) comes back; occupancy (frozen before) stays frozen.
        assert m.adp.refinable_params.numel() == adp0
        assert m.occupancy.refinable_params.numel() == 0
