"""A context remembers whether TorchRef completed its atom set's hydrogens.

Derived models never generate again (their policy reads ``keep``), so the policy alone
cannot say whether the hydrogens are TorchRef's. ``ctx.hydrogens_generated`` records it:
set by ``hydrogens="add"``, kept through derivation and checkpoints, cleared by
stripping, absent for a file read with ``keep``.
"""

import pytest

from torchref.model.model import Model


@pytest.fixture(scope="module")
def added(pdb_dir):
    return Model(verbose=0, device="cpu", hydrogens="add").load_pdb(
        str(pdb_dir / "1DAW.pdb")
    )


@pytest.mark.unit
def test_add_sets_and_derivation_keeps_it(added):
    assert added.ctx.hydrogens_generated
    stripped_altlocs = added.strip_altlocs()
    assert stripped_altlocs.ctx.hydrogens == "keep"
    assert stripped_altlocs.ctx.hydrogens_generated
    assert added.ctx.copy().hydrogens_generated


@pytest.mark.unit
def test_keep_and_strip_do_not_claim_it(pdb_dir, added):
    kept = Model(verbose=0, device="cpu", hydrogens="keep").load_pdb(
        str(pdb_dir / "1DAW.pdb")
    )
    assert not kept.ctx.hydrogens_generated
    assert kept.hydrogenate().ctx.hydrogens_generated
    assert not added.strip_hydrogens().ctx.hydrogens_generated


@pytest.mark.unit
def test_checkpoint_round_trip(added):
    restored = Model.create_from_state_dict(
        added.state_dict(), device=added.device, verbose=0
    )
    assert restored.ctx.hydrogens_generated
