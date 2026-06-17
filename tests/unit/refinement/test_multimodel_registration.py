"""Regression tests for MultiModel restraint registration.

MultiModelGeometryTarget / MultiModelADPTarget expose per-model sub-targets via
items() keyed as ``model_i/<sub>``. LossState.register_targets used to re-key
each entry on the leaf target's fixed ``.name`` (e.g. ``geometry/bond``),
discarding the ``model_i/`` prefix, so every base model collided on one key per
leaf type and all but the last were dropped. The fix honors hierarchical dict
keys (those containing ``/``). See TORCHREF_AUDIT.md cluster 4.
"""

import pytest
import torch
from torch import nn


@pytest.mark.unit
def test_register_targets_honors_hierarchical_keys():
    """Two leaves sharing a .name but distinct model_i/ dict keys must not collide."""
    from torchref.refinement.loss_state import LossState

    class Leaf(nn.Module):
        name = "geometry/bond"  # identical across "models" — the collision trigger

        def __init__(self, v):
            super().__init__()
            self._v = float(v)

        def forward(self):
            return torch.tensor(self._v)

    class MultiLike(nn.Module):
        name = "multi_model_geometry"

        def __init__(self):
            super().__init__()
            self.a = Leaf(1.0)
            self.b = Leaf(2.0)

        def items(self):
            return {"model_0/bond": self.a, "model_1/bond": self.b}.items()

        def forward(self):
            return self.a() + self.b()

    state = LossState(device=torch.device("cpu"))
    state.register_target("geometry", MultiLike(), probe=False)

    keys = set(state.targets.keys())
    # Pre-fix both collapsed onto "geometry/geometry/bond" (one survivor).
    assert "geometry/model_0/bond" in keys
    assert "geometry/model_1/bond" in keys
    assert len([k for k in keys if k.endswith("/bond")]) == 2
    # Group weight still rooted at "geometry" so set_weight("geometry") applies.
    assert all(k.startswith("geometry/") for k in keys)


@pytest.mark.integration
def test_multimodel_geometry_registration_keeps_all_base_models(pdb_dir, mtz_dir):
    """Real 2-base-model MultiModelGeometryTarget keeps both models' restraints."""
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    from torchref import LBFGSRefinement
    from torchref.model.model_collection import ModelCollection
    from torchref.refinement.loss_state import create_loss_state
    from torchref.refinement.targets.collection.multimodel import (
        MultiModelGeometryTarget,
    )

    ref_a = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0)
    ref_b = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0)
    mc = ModelCollection([ref_a.model, ref_b.model], dark_key="dark", verbose=0)

    geom = MultiModelGeometryTarget(mc, verbose=0)

    # items() carries the per-model index.
    item_keys = [k for k, _ in geom.items()]
    assert any(k.startswith("model_0/") for k in item_keys)
    assert any(k.startswith("model_1/") for k in item_keys)

    state = create_loss_state(device=ref_a.model.device)
    state.register_target("geometry", geom, probe=False)

    keys = set(state.targets.keys())
    n0 = sum(1 for k in keys if "model_0" in k)
    n1 = sum(1 for k in keys if "model_1" in k)
    # Both base models' restraint sub-targets survive (equal, non-empty);
    # pre-fix only the last base model's keys existed.
    assert n0 >= 1 and n0 == n1
    assert any(k.endswith("/bond") and "model_0" in k for k in keys)
    assert any(k.endswith("/bond") and "model_1" in k for k in keys)
