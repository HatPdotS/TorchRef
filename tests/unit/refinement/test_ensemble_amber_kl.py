"""
Unit tests for the entropy-only branch of
:class:`~torchref.experimental.ensemble.ensemble_amber_kl.EnsembleAmberKLTarget`.

The Amber-energy branch is skipped because the underlying ``AmberTarget``
has known blockers on real structures (template matching). The entropy
estimator is the critical anti-collapse mechanism and is tested here.
"""

import os

import pytest
import torch

from torchref.experimental.ensemble import EnsembleModel
from torchref.experimental.ensemble import EnsembleAmberKLTarget


TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "1DAW.pdb"
)


@pytest.fixture
def ensemble() -> EnsembleModel:
    return EnsembleModel.from_single(
        TEST_PDB, n_members=8, perturb_sigma=0.5, b_const=5.0,
        seed=42, verbose=0,
    )


def test_entropy_only_mode_is_finite(ensemble):
    target = EnsembleAmberKLTarget(model=ensemble, lam=1.0, kT=0.0, verbose=0)
    loss = target.forward()
    assert torch.isfinite(loss)


def test_collapsed_ensemble_yields_higher_loss(ensemble):
    target = EnsembleAmberKLTarget(model=ensemble, lam=1.0, kT=0.0, verbose=0)
    loss_spread = target.forward().item()

    # Collapse all members to member 0.
    flat = ensemble.xyz.refinable_params
    view = flat.view(ensemble.n_members, ensemble.n_atoms_per_member, 3)
    member0 = view[0].clone()
    with torch.no_grad():
        for i in range(1, ensemble.n_members):
            view[i] = member0
    if hasattr(ensemble, "reset_cache"):
        ensemble.reset_cache()
    loss_collapsed = target.forward().item()

    assert loss_collapsed > loss_spread, (
        f"Collapsed ensemble (loss={loss_collapsed}) should exceed "
        f"spread (loss={loss_spread})"
    )


def test_gradient_flows_back_to_xyz(ensemble):
    target = EnsembleAmberKLTarget(model=ensemble, lam=1.0, kT=0.0, verbose=0)
    ensemble.xyz.refinable_params.grad = None
    loss = target.forward()
    loss.backward()
    g = ensemble.xyz.refinable_params.grad
    assert g is not None
    assert torch.isfinite(g).all()
