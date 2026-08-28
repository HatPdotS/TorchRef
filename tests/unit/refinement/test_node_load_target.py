"""The node load-balancing barrier.

Two properties carry the design. It must be **one-sided** -- an abandoned node is
penalised, an over-loaded one is not -- because the symmetric choice (maximising load
entropy) is optimal at uniform load and would flatten the multiscale kernel-width spread
that a working field genuinely has. And it must act through the *weights* only, so its
gradient reaches node positions and widths but never the node values: it removes the
opportunity to place an extreme B rather than penalising the B.
"""

import pytest
import torch

from torchref.model.model import Model
from torchref.refinement.targets.adp import NodeLoadTarget


@pytest.fixture(scope="module")
def pdb_path(pdb_dir):
    return str(pdb_dir / "3GR5.pdb")


def _field_model(pdb_path, n_nodes=32, **kw):
    model = Model(verbose=0)
    model.load_pdb(pdb_path)
    model.set_adp_mode("field", n_nodes=n_nodes, k_neighbors=8, **kw)
    return model


@pytest.mark.unit
def test_inert_outside_field_mode(pdb_path):
    """Registered unconditionally, so it must cost nothing on the per-atom path."""
    model = Model(verbose=0)
    model.load_pdb(pdb_path)
    model.set_adp_mode("isotropic")
    target = NodeLoadTarget(model)
    assert float(target()) == 0.0
    assert target.stats()["node_load_active"].value == 0.0


@pytest.mark.unit
def test_balanced_field_is_barely_penalised(pdb_path):
    """A freshly fitted field has near-even load, so the barrier starts near its floor."""
    model = _field_model(pdb_path)
    target = NodeLoadTarget(model)
    rel = target._relative_load().detach()
    # Mean relative load is 1 by construction.
    assert torch.allclose(rel.mean(), torch.ones((), dtype=rel.dtype), atol=1e-6)
    per_node = float(target()) / rel.numel()
    assert per_node < 0.5, f"per-node penalty {per_node:.3f} on a balanced field"


@pytest.mark.unit
def test_abandoning_a_node_raises_the_penalty(pdb_path):
    """Collapsing one node's kernel starves it, and the barrier must notice."""
    model = _field_model(pdb_path)
    target = NodeLoadTarget(model)
    before = float(target())

    # Narrow one node far below the others: it loses every softmax contest except
    # against an atom sitting on top of it.
    with torch.no_grad():
        model.adp.refinable_params[0, 1] -= 6.0
    model.adp.reset_forward_cache()

    after = float(target())
    rel = target._relative_load().detach()
    assert rel.min() < 0.25, "the node was not actually starved"
    assert after > before + 1.0, f"barrier missed it: {before:.3f} -> {after:.3f}"


@pytest.mark.unit
def test_penalty_is_one_sided(pdb_path):
    """Over-loading a node must not be penalised; only abandonment is.

    This is what separates the barrier from a load-entropy term, which is optimal at
    uniform load and would push back on a legitimately broad node.
    """
    model = _field_model(pdb_path)
    target = NodeLoadTarget(model)
    baseline = float(target())

    with torch.no_grad():
        model.adp.refinable_params[0, 1] += 3.0  # widen one node -> it gains load
    model.adp.reset_forward_cache()
    widened = float(target())
    rel = target._relative_load().detach()

    assert float(rel.max()) > 1.5, "the node did not actually gain load"
    # Widening one node necessarily takes load from others, so the total may rise a
    # little; what must not happen is the over-loaded node itself being charged.
    per_node_change = (widened - baseline) / rel.numel()
    assert per_node_change < 0.5, (
        f"over-loading was penalised like abandonment ({per_node_change:.3f}/node)"
    )


@pytest.mark.unit
def test_gradient_reaches_geometry_but_not_values(pdb_path):
    """Acts on the weights: positions and widths get gradient, node B does not."""
    model = _field_model(pdb_path)
    target = NodeLoadTarget(model)
    target().backward()

    grad = model.adp.refinable_params.grad
    assert grad is not None
    # Columns are [log B, log sigma, dx, dy, dz].
    assert float(grad[:, 0].abs().sum()) == pytest.approx(0.0, abs=1e-12), (
        "the barrier must not push the node VALUES"
    )
    assert float(grad[:, 1].abs().sum()) > 0, "no gradient to the kernel widths"
    assert float(grad[:, 2:5].abs().sum()) > 0, "no gradient to the node positions"


@pytest.mark.unit
def test_position_gradient_absent_when_positions_are_fixed(pdb_path):
    """With positions fixed the barrier can only act through the widths."""
    model = _field_model(pdb_path, refine_node_positions=False)
    target = NodeLoadTarget(model)
    target().backward()
    grad = model.adp.refinable_params.grad
    assert grad.shape[1] == 2
    assert float(grad[:, 1].abs().sum()) > 0


@pytest.mark.unit
def test_registered_in_the_total_adp_target(pdb_path):
    """Reachable under the weight path 'adp/node_load'."""
    from torchref.refinement.targets.combined import TotalADPTarget

    model = _field_model(pdb_path)
    total = TotalADPTarget(model, verbose=0)
    assert "node_load" in total.target_losses()
    assert torch.isfinite(torch.as_tensor(float(total["node_load"]())))


@pytest.mark.unit
def test_default_weight_exists_for_the_component(pdb_path):
    """A component with no weight entry would silently inherit the group weight."""
    from torchref.refinement.base_refinement import DEFAULT_GROUP_WEIGHTS

    assert "adp/node_load" in DEFAULT_GROUP_WEIGHTS
    assert DEFAULT_GROUP_WEIGHTS["adp/node_load"] > 0
