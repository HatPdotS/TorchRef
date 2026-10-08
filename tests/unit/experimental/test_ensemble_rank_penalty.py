"""
Unit tests for :class:`~torchref.experimental.ensemble.rank_penalty.RankPenaltyTarget`.

Pins that the penalty sees only the alive members of a population pool.
"""

import os

import pytest
import torch

from torchref.experimental.ensemble import EnsembleModel, RankPenaltyTarget

TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "1DAW.pdb"
)
_KW = dict(perturb_sigma=0.2, b_const=5.0, seed=42, verbose=0)
MODES = ["nuclear", "subspace", "entropy", "maxent", "diverse"]


@pytest.fixture(scope="module")
def pool_and_plain():
    pool = EnsembleModel.from_single(TEST_PDB, n_members=4, n_max=6, **_KW)
    plain = EnsembleModel.from_single(TEST_PDB, n_members=4, **_KW)
    return pool, plain


@pytest.mark.parametrize("mode", MODES)
def test_dead_slots_do_not_enter_the_penalty(pool_and_plain, mode):
    pool, plain = pool_and_plain
    kw = dict(mode=mode, target_rank=1)
    loss = RankPenaltyTarget(model=pool, **kw).forward()
    assert torch.allclose(loss, RankPenaltyTarget(model=plain, **kw).forward())
    pool.xyz.refinable_params.grad = None
    loss.backward()
    grad = pool.xyz.refinable_params.grad.view(pool.n_members, -1)
    assert torch.all(grad[4:] == 0)

