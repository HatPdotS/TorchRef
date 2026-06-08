"""Unit tests for RigidXYZTensor and the Model.use_rigid_xyz swap."""

import pytest
import torch

from torchref.base.alignment.rotation import rotation_matrix_euler_zyz
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

        ang = torch.tensor([[0.0, 0.05, 0.0]], dtype=dtype, device=device)
        trans = torch.tensor([[0.2, -0.3, 0.5]], dtype=dtype, device=device)
        with torch.no_grad():
            xyz.euler_angles.copy_(ang)
            xyz.translations.copy_(trans)

        center = xyz.chain_centers[0]
        R = rotation_matrix_euler_zyz(ang.squeeze(0))
        expected = (xyz.original_xyz - center) @ R.T + center + trans

        with torch.no_grad():
            diff = (xyz() - expected).abs().max().item()
        assert diff < 1e-4

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
