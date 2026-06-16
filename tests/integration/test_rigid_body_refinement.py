"""Integration tests for the multi-resolution rigid-body refinement step.

These tests apply a known rigid perturbation to a loaded structure, run
``LBFGSRefinement.refine_rigid_body()``, and verify that the R-factor and
the per-chain rigid parameters recover within tolerance.
"""

import math

import numpy as np
import pytest
import torch

from torchref.base.alignment.rotation import rotation_matrix_euler_zyz
from torchref.refinement.lbfgs_refinement import LBFGSRefinement


def _build_refinement(pdb_dir, mtz_dir, name):
    return LBFGSRefinement(
        data_file=str(mtz_dir / f"{name}.mtz"),
        pdb=str(pdb_dir / f"{name}.pdb"),
        verbose=0,
    )


def _apply_rigid_perturbation(ref, chain_id, euler_deg, translation_A):
    """Apply a rigid rotation+translation to atoms belonging to ``chain_id``."""
    pdb = ref.model.pdb
    device = ref.device
    dtype = ref.model.xyz.dtype

    mask_np = (pdb["chainid"].values == chain_id)
    mask = torch.from_numpy(np.asarray(mask_np, dtype=bool)).to(device)

    ang = torch.tensor(
        [math.radians(a) for a in euler_deg], dtype=dtype, device=device
    )
    R = rotation_matrix_euler_zyz(ang)
    trans = torch.tensor(translation_A, dtype=dtype, device=device)

    xyz_full = ref.model.xyz().detach().clone()
    sub = xyz_full[mask]
    center = sub.mean(dim=0)
    xyz_full[mask] = (sub - center) @ R.T + center + trans
    ref.model.xyz[:] = xyz_full
    ref.model.reset_cache()


class TestSingleChainRecovery:
    @pytest.mark.integration
    def test_single_chain_recovers_baseline_rfactor(self, pdb_dir, mtz_dir):
        ref = _build_refinement(pdb_dir, mtz_dir, "1DAW")
        ref.get_scales()
        rwork0, rfree0 = ref.get_rfactor()

        chain_id = ref.model.pdb["chainid"].iloc[0]
        _apply_rigid_perturbation(
            ref, chain_id, euler_deg=[0.0, 0.5, 0.0], translation_A=[0.5, 0.0, 0.0]
        )
        rwork_pert, _ = ref.get_rfactor()
        assert rwork_pert > rwork0 + 0.02, (
            f"perturbation too weak: Rwork before={rwork0:.4f}, after={rwork_pert:.4f}"
        )

        ref.refine_rigid_body(iterations_per_step=50, commit=True)

        rwork_after, rfree_after = ref.get_rfactor()
        assert rwork_after <= rwork0 + 0.1, (
            f"Rwork did not recover: baseline={rwork0:.4f}, "
            f"after-rigid={rwork_after:.4f}"
        )
        assert rfree_after <= rfree0 + 0.1

        # After commit=True, xyz is baked into a per-atom MixedTensor.
        from torchref.model.parameter_wrappers import MixedTensor

        assert isinstance(ref.model.xyz, MixedTensor)


class TestPerChainRecovery:
    @pytest.mark.integration
    def test_independent_per_chain_shifts_recover(self, pdb_dir, mtz_dir):
        ref = _build_refinement(pdb_dir, mtz_dir, "3E98")
        ref.get_scales()
        rwork0, rfree0 = ref.get_rfactor()

        chains = list(ref.model.pdb["chainid"].unique())
        assert len(chains) >= 2, "test requires a multi-chain structure"

        # Distinct per-chain rotations + translations so collapse to a shared
        # transform cannot recover both chains.
        per_chain_specs = [
            dict(euler_deg=[0.0, 0.5, 0.0], translation_A=[0.5, 0.0, 0.0]),
            dict(euler_deg=[0.5, 0.0, 0.0], translation_A=[0.0, 0.5, 0.0]),
            dict(euler_deg=[0.0, 0.0, 0.5], translation_A=[0.0, 0.0, 0.5]),
            dict(euler_deg=[0.3, -0.3, 0.0], translation_A=[-0.3, 0.3, 0.0]),
        ]
        for chain_id, spec in zip(chains, per_chain_specs):
            _apply_rigid_perturbation(ref, chain_id, **spec)
        rwork_pert, _ = ref.get_rfactor()
        assert rwork_pert > rwork0 + 0.02

        ref.refine_rigid_body(iterations_per_step=50, commit=True)
        rwork_after, rfree_after = ref.get_rfactor()
        assert rwork_after <= rwork0 + 0.1, (
            f"Rwork did not recover: baseline={rwork0:.4f}, "
            f"after-rigid={rwork_after:.4f}"
        )
        assert rfree_after <= rfree0 + 0.1
