"""Every registered target must arrive in the LossState under a key its weight can reach.

``LossState.register_targets`` keys each component off ``target.name``, falling back to
``Target.name`` -- which is the literal string ``"model_target"``. A target that forgets
to declare its own name therefore registers under that, collides with every other target
that forgot, and no hierarchical weight can address it. The term is constructed, callable
and correct in isolation; it simply never enters the loss.

That is what happened to ``adp/node_load`` and ``adp/node_smoothness``: both shipped
unnamed, so the node-coverage barrier was never in any refinement's loss despite having a
weight of 10.0 in ``DEFAULT_GROUP_WEIGHTS`` and passing every test that called it
directly. These tests check the plumbing rather than the arithmetic.
"""

import pytest

from torchref.refinement.targets.base import ModelTarget


def _leaf_target_classes():
    """Every concrete ModelTarget subclass that is a loss component, not a container."""
    import inspect
    import pkgutil
    import importlib

    import torchref.refinement.targets as pkg
    from torchref.refinement.targets.combined import CombinedModelTargets

    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception:
            continue
    seen = {}
    stack = [ModelTarget]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            stack.append(sub)
            # Skip group bases (ADPTarget, GeometryTarget, ...). They are not formally
            # abstract -- nothing in them is an abstractmethod -- so the only reliable
            # marker is that other targets derive from them. A base legitimately carries
            # the placeholder name because it is never registered itself.
            if sub.__subclasses__():
                continue
            if inspect.isabstract(sub) or issubclass(sub, CombinedModelTargets):
                continue
            seen[sub.__qualname__] = sub
    return seen


@pytest.mark.unit
def test_no_component_target_inherits_the_placeholder_name():
    """A component using the base placeholder cannot be addressed by any weight."""
    offenders = {
        name: cls.name
        for name, cls in _leaf_target_classes().items()
        if getattr(cls, "name", None) in (None, "base_target", "model_target",
                                          "data_target")
    }
    assert not offenders, (
        "these targets would register under the base placeholder name, colliding with "
        "each other and unreachable by any hierarchical weight:\n  "
        + "\n  ".join(f"{k}: name={v!r}" for k, v in sorted(offenders.items()))
    )


@pytest.mark.unit
def test_adp_component_names_are_hierarchical():
    """An ADP component must sit under the ``adp`` group or the group weight misses it."""
    import torchref.refinement.targets.adp as adp_pkg

    bad = {}
    for attr in dir(adp_pkg):
        cls = getattr(adp_pkg, attr)
        if not isinstance(cls, type) or not issubclass(cls, ModelTarget):
            continue
        if cls.__subclasses__():
            continue  # a group base, never registered itself
        name = getattr(cls, "name", "")
        if not isinstance(name, str) or not name.startswith("adp/"):
            bad[attr] = name
    assert not bad, f"ADP targets not under the adp group: {bad}"


@pytest.mark.unit
@pytest.mark.parametrize("mode_set", [None, "rigid_dilation"])
def test_field_components_are_registered_and_weighted(pdb_dir, mtz_dir, mode_set):
    """End to end: the components the representation declares must be in the loss.

    Compares the combined target's own component set against the LossState's keys, so a
    component that exists but never registers is caught -- which is the failure mode that
    a direct ``adp_target['node_load']()`` call cannot see.
    """
    from torchref.refinement.lbfgs_refinement import LBFGSRefinement

    ref = LBFGSRefinement(
        data_file=str(mtz_dir / "1DAW.mtz"), pdb=str(pdb_dir / "1DAW.pdb"),
        verbose=0, adp_mode="field_aniso", adp_mode_set=mode_set,
    )
    components = set(ref.adp_target.target_losses())
    state = ref.complete_loss_state()

    for component in components:
        key = f"adp/{component}"
        assert key in state.targets, (
            f"{component!r} is a component of TotalADPTarget but never reached the "
            f"LossState. Registered adp keys: "
            f"{sorted(k for k in state.targets if k.startswith('adp'))}"
        )
        # And the weight has to be addressable, not merely present.
        assert state.get_effective_weight(key) is not None

    assert "node_load" in components, "field mode must carry the coverage barrier"
    assert state.get_effective_weight("adp/node_load") > 0.0, (
        "the coverage barrier registered but at zero effective weight, so it is inert"
    )
