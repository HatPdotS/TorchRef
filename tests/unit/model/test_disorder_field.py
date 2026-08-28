"""The node-field ADP wrapper: weights, positivity, cache correctness, lifecycle.

Two properties here are load-bearing rather than cosmetic. The weights must be a
normalised mixture over each atom's candidate nodes, because that is what makes the
per-atom B a convex combination of positive node values and therefore positive without
a clamp. And the forward cache must notice that the coordinates moved: the wrapper reads
them through an injected accessor rather than a call argument, so
``CachedForwardMixin``'s own fingerprint cannot see them and
``DisorderFieldTensor._fingerprint_state`` has to fold them in.
"""

import math

import pytest
import torch

from torchref.model.disorder_field import (
    DisorderFieldTensor,
    build_neighbor_list,
    farthest_point_anchors,
)
from torchref.model.parameter_wrappers import MixedTensor


@pytest.fixture
def coords():
    """A compact 3-D blob of atoms on a deterministic lattice."""
    g = torch.arange(6, dtype=torch.float64)
    x, y, z = torch.meshgrid(g, g, g, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=1) * 1.7


@pytest.fixture
def target_b(coords):
    """A smooth B field with a real spatial gradient for the fit to chase."""
    return 20.0 + 1.5 * coords[:, 0] + 0.8 * coords[:, 2]


def _field(coords, target_b, **kw):
    xyz = MixedTensor(coords.clone(), name="xyz")
    kw.setdefault("n_nodes", 12)
    kw.setdefault("k_neighbors", 6)
    return DisorderFieldTensor(
        initial_values=target_b, xyz_fn=xyz, dtype=torch.float64, **kw
    ), xyz


@pytest.mark.unit
def test_anchor_selection_is_deterministic(coords):
    """Same coordinates, same anchors -- no RNG anywhere in placement."""
    a = farthest_point_anchors(coords, 10)
    b = farthest_point_anchors(coords, 10)
    assert torch.equal(a, b)
    assert a.shape[0] == 10
    assert a.dtype == torch.int64
    # Anchors are atom indices, and distinct.
    assert int(a.max()) < coords.shape[0]
    assert torch.unique(a).shape[0] == a.shape[0]


@pytest.mark.unit
def test_neighbor_list_is_nearest_first(coords):
    """``build_neighbor_list`` returns each atom's k nearest nodes, closest first."""
    node_pos = coords[farthest_point_anchors(coords, 8)]
    nl = build_neighbor_list(coords, node_pos, 4)
    assert nl.shape == (coords.shape[0], 4)
    d = torch.cdist(coords, node_pos)
    gathered = torch.gather(d, 1, nl)
    assert bool((gathered.diff(dim=1) >= -1e-12).all()), "not sorted by distance"
    assert torch.equal(nl[:, 0], d.argmin(dim=1))


@pytest.mark.unit
def test_weights_are_a_normalised_mixture(coords, target_b):
    """Rows sum to one and are non-negative -- what makes the output positive."""
    field, _ = _field(coords, target_b)
    W = field.weights()
    assert W.shape == (coords.shape[0], 6)
    assert bool((W >= 0).all())
    assert torch.allclose(W.sum(dim=1), torch.ones(coords.shape[0], dtype=W.dtype))


@pytest.mark.unit
def test_output_is_per_atom_and_positive(coords, target_b):
    """Public space is per-atom even though storage is per-node."""
    field, _ = _field(coords, target_b)
    out = field()
    assert out.shape == (coords.shape[0],)
    assert field.shape == (coords.shape[0],)
    assert field.node_shape == (12, 2)
    assert bool((out > 0).all())
    assert bool(torch.isfinite(out).all())


@pytest.mark.unit
def test_single_node_flat_kernel_is_a_constant(coords):
    """K=1 reproduces a constant B exactly -- the analytic control.

    With one node every atom's weight vector is ``[1.0]`` whatever the distance, so the
    field degenerates to a single scalar and must return it uniformly.
    """
    b = torch.full((coords.shape[0],), 37.5, dtype=torch.float64)
    field, _ = _field(coords, b, n_nodes=1, k_neighbors=1)
    out = field()
    assert torch.allclose(out, b, atol=1e-9), f"got spread {out.min()}..{out.max()}"


@pytest.mark.unit
def test_fit_tracks_a_smooth_gradient(coords, target_b):
    """A 12-node field on a linear B ramp beats the best constant by a wide margin."""
    field, _ = _field(coords, target_b)
    resid = (field() - target_b).pow(2).mean().sqrt()
    constant = (target_b - target_b.mean()).pow(2).mean().sqrt()
    assert resid < 0.25 * constant, f"rmse {resid:.3f} vs constant {constant:.3f}"


@pytest.mark.unit
def test_more_nodes_fit_better(coords, target_b):
    """Reconstruction improves monotonically with node count on a smooth target."""
    errors = []
    for n in (2, 8, 32):
        field, _ = _field(coords, target_b, n_nodes=n, k_neighbors=min(6, n))
        errors.append(float((field() - target_b).detach().pow(2).mean().sqrt()))
    assert errors[0] > errors[1] > errors[2], errors


@pytest.mark.unit
def test_fingerprint_sees_the_coordinates(coords, target_b):
    """The load-bearing test for the injected accessor, and it proves its own point.

    ``forward()`` takes no arguments, so the coordinates reach it through ``_xyz_fn``.
    The inherited fingerprint covers only this module's own parameters and buffers, so
    it cannot see them -- asserted directly here, by checking the base implementation
    does NOT change while the override does. Without
    ``DisorderFieldTensor._fingerprint_state`` the cache would serve a B computed at
    coordinates that have since moved.
    """
    from torchref.utils.caching import CachedForwardMixin

    field, xyz = _field(coords, target_b)
    base_before = CachedForwardMixin._fingerprint_state(field)
    full_before = field._fingerprint_state()

    with torch.no_grad():
        xyz.refinable_params[:8, 0] += 4.0

    assert CachedForwardMixin._fingerprint_state(field) == base_before, (
        "the field's own parameters and buffers did not move, so the inherited "
        "fingerprint is blind to this change -- which is why the override exists"
    )
    assert field._fingerprint_state() != full_before, "override missed the coordinates"


@pytest.mark.unit
def test_cache_returns_a_fresh_value_after_a_non_rigid_move(coords, target_b):
    """A change of relative geometry must reach the output, not a stale cache.

    The perturbation has to be non-rigid: the field is translation-invariant by
    construction, so shifting every atom equally moves the nodes with them and is
    *correctly* a no-op.
    """
    field, xyz = _field(coords, target_b)
    before = field().clone()

    with torch.no_grad():
        xyz.refinable_params[: coords.shape[0] // 2, 0] += 4.0

    after = field()
    assert not torch.allclose(before, after), "stale cache: coordinates were ignored"


@pytest.mark.unit
def test_rigid_translation_leaves_the_field_unchanged(coords, target_b):
    """The invariance that makes the previous test need a non-rigid perturbation."""
    field, xyz = _field(coords, target_b)
    before = field().clone()

    with torch.no_grad():
        xyz.refinable_params += 9.0

    assert torch.allclose(field(), before, atol=1e-9)


@pytest.mark.unit
def test_node_positions_follow_the_coordinates(coords, target_b):
    """Node positions are derived from the atoms, so a rigid shift carries them along."""
    field, xyz = _field(coords, target_b)
    before = field.node_positions().clone()

    with torch.no_grad():
        xyz.refinable_params += 2.5

    after = field.node_positions()
    assert torch.allclose(after - before, torch.full_like(before, 2.5), atol=1e-9)


@pytest.mark.unit
def test_gradient_reaches_nodes_and_coordinates(coords, target_b):
    """Both channels are live through the plain zero-arg forward path."""
    field, xyz = _field(coords, target_b)
    field().sum().backward()
    assert field.refinable_params.grad is not None
    assert bool(field.refinable_params.grad.abs().sum() > 0)
    assert xyz.refinable_params.grad is not None
    assert bool(xyz.refinable_params.grad.abs().sum() > 0)


@pytest.mark.unit
def test_gradcheck_on_both_channels(coords, target_b):
    """Analytic gradients match finite differences, for node values and coordinates.

    Checked through :meth:`DisorderFieldTensor.evaluate`, which is the field's
    arithmetic without the accessor or the cache. Rebinding ``refinable_params`` inside
    a gradcheck closure would not work: ``nn.Parameter(p)`` is a fresh leaf, so the
    graph back to ``p`` is severed and there is nothing to check.
    """
    small = coords[:20].clone()
    field, _ = _field(small, target_b[:20], n_nodes=3, k_neighbors=3)

    raw = field.node_values().detach().clone().requires_grad_(True)
    xyz_in = small.detach().clone().requires_grad_(True)

    assert torch.autograd.gradcheck(
        field.evaluate, (xyz_in, raw), eps=1e-6, atol=1e-5
    )


@pytest.mark.unit
def test_list_adequacy_invariant_is_exposed(coords, target_b):
    """The smallest candidate weight is reported, and shrinks as k grows."""
    tight, _ = _field(coords, target_b, n_nodes=16, k_neighbors=2)
    loose, _ = _field(coords, target_b, n_nodes=16, k_neighbors=12)
    assert loose.smallest_candidate_weight() < tight.smallest_candidate_weight()
    assert 0.0 <= loose.smallest_candidate_weight() <= 1.0


@pytest.mark.unit
def test_node_load_is_in_node_space_and_conserves_total_weight(coords, target_b):
    """Load must be scattered into node space, not summed over the candidate axis.

    ``weights()`` is ``(n_atoms, k)`` over CANDIDATES, so ``weights().sum(0)`` is a
    length-k vector of per-slot totals with no meaning -- a trap worth pinning, because
    it silently returns a plausible-looking tensor of the wrong length.
    """
    field, _ = _field(coords, target_b, n_nodes=12, k_neighbors=6)
    load = field.node_load()

    assert load.shape == (12,), "load must be per node, not per candidate slot"
    assert field.weights().sum(dim=0).shape == (6,), "the trap this method avoids"
    # Rows of W sum to 1, so the total load is exactly the atom count.
    assert torch.allclose(
        load.sum(), torch.tensor(float(coords.shape[0]), dtype=load.dtype)
    )
    assert bool((load >= 0).all())


@pytest.mark.unit
def test_rebuild_neighbor_list_is_explicit_and_refreshes(coords, target_b):
    """Membership only changes when the caller asks; the rebuild then takes effect."""
    field, xyz = _field(coords, target_b)
    original = field.neighbor_list.clone()

    with torch.no_grad():
        xyz.refinable_params[:, 0] += 40.0  # move atoms far past the nodes

    assert torch.equal(field.neighbor_list, original), "list changed without a rebuild"
    field.rebuild_neighbor_list()
    assert field.neighbor_list.shape == original.shape
    assert bool(torch.isfinite(field()).all())


@pytest.mark.unit
def test_atom_space_mask_collapses_to_node_space(coords, target_b):
    """Masks arrive in ATOM space and are collapsed with OR onto the nodes."""
    field, _ = _field(coords, target_b)
    n_atoms = coords.shape[0]

    mask = torch.zeros(n_atoms, dtype=torch.bool)
    mask[0] = True
    field.update_refinable_mask(mask)

    assert field.refinable_mask.shape == (field.n_nodes,)
    served = field.neighbor_list[0]
    assert bool(field.refinable_mask[served].all()), "atom 0's nodes must be refinable"
    assert int(field.refinable_mask.sum()) == int(torch.unique(served).numel())


@pytest.mark.unit
def test_wrong_sized_mask_is_rejected(coords, target_b):
    """A node-space mask passed as atom space (or vice versa) is an error, not a guess."""
    field, _ = _field(coords, target_b)
    with pytest.raises(ValueError, match="Atom-space mask"):
        field.update_refinable_mask(torch.ones(field.n_nodes, dtype=torch.bool))
    with pytest.raises(ValueError, match="Node-space mask"):
        field.update_refinable_mask(
            torch.ones(coords.shape[0], dtype=torch.bool), in_node_space=True
        )


@pytest.mark.unit
def test_freezing_all_nodes_leaves_the_output_intact(coords, target_b):
    """Repartitioning moves values between storage halves without changing them."""
    field, _ = _field(coords, target_b)
    before = field().clone()
    field.update_refinable_mask(
        torch.zeros(field.n_nodes, dtype=torch.bool), in_node_space=True
    )
    assert int(field.get_refinable_count()) == 0
    assert torch.allclose(field(), before, atol=1e-12)


@pytest.mark.unit
def test_per_atom_assignment_is_refused(coords, target_b):
    """K nodes cannot represent arbitrary per-atom values, so writing is not silent."""
    field, _ = _field(coords, target_b)
    with pytest.raises(NotImplementedError, match="per-atom assignment"):
        field[0] = 42.0


@pytest.mark.unit
def test_refit_moves_the_field_to_a_new_target(coords, target_b):
    """``refit`` is the representable alternative to per-atom assignment."""
    field, _ = _field(coords, target_b)
    new_target = target_b * 0.5 + 5.0
    field.refit(new_target)
    resid = (field() - new_target).pow(2).mean().sqrt()
    baseline = (new_target - new_target.mean()).pow(2).mean().sqrt()
    assert resid < 0.25 * baseline


@pytest.mark.unit
def test_copy_is_independent_but_shares_the_accessor(coords, target_b):
    """Parameters are copied; the coordinate accessor is deliberately shared.

    Deep-copying the accessor is what breaks ``Restraints.copy()`` today: ``deepcopy``
    walks the borrowed wrapper whose cache can hold a graph-attached tensor.
    """
    field, xyz = _field(coords, target_b)
    clone = field.copy()

    assert clone._xyz_fn.module is xyz, "accessor must reference the same wrapper"
    assert clone.refinable_params.data_ptr() != field.refinable_params.data_ptr()
    assert torch.allclose(clone(), field())

    with torch.no_grad():
        clone.refinable_params[:, 0] += 1.0
    clone.reset_forward_cache()
    field.reset_forward_cache()
    assert not torch.allclose(clone(), field()), "copy is not independent"


@pytest.mark.unit
def test_copy_survives_an_evaluated_forward(coords, target_b):
    """``copy()`` works after ``forward()`` has populated the accessor's cache."""
    field, _ = _field(coords, target_b)
    field()  # populate xyz's forward cache with a graph-attached tensor
    clone = field.copy()
    assert torch.allclose(clone(), field())


@pytest.mark.unit
def test_state_dict_excludes_the_accessor_and_round_trips(coords, target_b):
    """The callable is not state; everything else survives a save/load exactly.

    Restored the way ``Model.create_from_state_dict`` does it: build a wrapper with real
    values to fix the shapes and masks, then let ``load_state_dict`` overwrite them. The
    empty shell exists for shape-less construction, not as a load target.
    """
    field, xyz = _field(coords, target_b)
    # Cloned deliberately: state_dict() hands back detached REFERENCES, so a later
    # in-place edit of refinable_params would silently rewrite the saved state too.
    sd = {key: value.clone() for key, value in field.state_dict().items()}
    assert not any("xyz_fn" in key for key in sd), sd.keys()
    expected = field().clone()

    restored, _ = _field(coords, target_b)
    with torch.no_grad():
        restored.refinable_params[:, 0] += 0.75  # move it off the saved state
    restored.reset_forward_cache()
    assert not torch.allclose(restored(), expected), "perturbation did not take"

    restored.load_state_dict(sd)
    restored.reset_forward_cache()

    assert torch.allclose(restored(), expected, atol=1e-12)


@pytest.mark.unit
def test_empty_shell_needs_no_accessor(coords):
    """The ``load_state_dict`` entry point constructs without coordinates."""
    shell = DisorderFieldTensor(dtype=torch.float64)
    assert shell.neighbor_list is None
    assert shell.shape == (0,)


@pytest.mark.unit
def test_accessor_is_required_when_values_are_given(coords, target_b):
    """Nodes cannot be placed without coordinates, and that fails loudly."""
    with pytest.raises(ValueError, match="xyz_fn"):
        DisorderFieldTensor(initial_values=target_b, dtype=torch.float64)


@pytest.mark.unit
def test_accessor_is_not_a_submodule_or_buffer(coords, target_b):
    """It must stay out of module traversal, or a device move duplicates the graph."""
    field, xyz = _field(coords, target_b)
    from torchref.utils.utils import ModuleReference

    assert isinstance(field._xyz_fn, ModuleReference)
    assert xyz not in list(field.modules())
    assert not any(m is xyz for _, m in field.named_modules())
    # xyz's parameters must not appear among the field's own.
    field_ptrs = {p.data_ptr() for p in field.parameters()}
    assert xyz.refinable_params.data_ptr() not in field_ptrs


@pytest.mark.unit
def test_device_round_trip_keeps_buffers_together(coords, target_b):
    """A ``.to()`` moves the node storage and the index buffers as one."""
    field, _ = _field(coords, target_b)
    before = field().clone()
    field.to(torch.device("cpu"))
    assert field.neighbor_list.device.type == "cpu"
    assert field.anchor_atom.device.type == "cpu"
    assert torch.allclose(field(), before, atol=1e-12)


@pytest.mark.unit
def test_float32_and_float64_both_work(coords, target_b):
    """The field works in either dtype without requiring one."""
    for dtype in (torch.float32, torch.float64):
        xyz = MixedTensor(coords.to(dtype).clone(), name="xyz")
        field = DisorderFieldTensor(
            initial_values=target_b.to(dtype),
            xyz_fn=xyz,
            n_nodes=8,
            k_neighbors=4,
            dtype=dtype,
        )
        out = field()
        assert out.dtype == dtype
        assert bool(torch.isfinite(out).all())


@pytest.mark.unit
def test_ragged_anchor_neighbourhoods_average_their_atoms(coords, target_b):
    """A node anchored on several atoms sits at their centroid."""
    xyz = MixedTensor(coords.clone(), name="xyz")
    anchor_atom = torch.tensor([0, 1, 2, 10, 11], dtype=torch.int64)
    anchor_node = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
    field = DisorderFieldTensor(
        initial_values=target_b,
        xyz_fn=xyz,
        k_neighbors=2,
        anchor_rows=(anchor_atom, anchor_node),
        dtype=torch.float64,
    )
    pos = field.node_positions()
    assert pos.shape == (2, 3)
    assert torch.allclose(pos[0], coords[[0, 1, 2]].mean(dim=0))
    assert torch.allclose(pos[1], coords[[10, 11]].mean(dim=0))
