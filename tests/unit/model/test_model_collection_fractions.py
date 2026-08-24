"""Characterisation of how ``ModelCollection`` stores population fractions.

Nothing else in the suite asserts anything about fraction storage -- not the softmax
parametrisation, not the sum-to-1 validation, not the freeze flags, not the override path.
These tests pin the observable contract so a change of storage has to reproduce it rather
than merely still run.

Deliberately fileless: the fractions live on ``_SharedMixedModel`` and depend on the base
models only for ``dtype_float`` and device, so a stub is enough and the whole file runs in
well under a second.
"""

import pytest
import torch
from torch import nn


class _StubModel(nn.Module):
    """Minimal stand-in for ``ModelFT`` for fraction bookkeeping.

    Carries a real parameter so ``.to()`` and ``resolve_device`` behave, exposes the two
    attributes ``_SharedMixedModel.__init__`` reads, and returns structure factors that
    differ per instance so a weighted sum can be checked against its parts.
    """

    def __init__(self, seed: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self._seed = seed

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype_float(self):
        return self.anchor.dtype

    def forward(self, hkl, recalc: bool = False):
        # Distinct per model, and a function of the parameter so a gradient can reach it.
        n = hkl.shape[0]
        base = torch.arange(1, n + 1, dtype=self.anchor.dtype, device=self.anchor.device)
        amp = (base * float(self._seed + 1)) + self.anchor
        return amp.to(torch.complex64)


@pytest.fixture
def two_model_collection():
    """A dark + one-timepoint collection over two distinguishable stub models."""
    from torchref.model.model_collection import ModelCollection

    mc = ModelCollection([_StubModel(0), _StubModel(1)], dark_key="dark", verbose=0)
    mc.add_dark()
    mc.add_timepoint("light", [0.7, 0.3])
    return mc


@pytest.fixture
def hkl():
    return torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 1]])


class TestSoftmaxStorage:
    @pytest.mark.unit
    def test_fractions_are_the_softmax_of_the_stored_logits(self, two_model_collection):
        mc = two_model_collection
        mixed = mc["light"]
        expected = torch.softmax(mixed.fraction_params, dim=0)
        assert torch.allclose(mixed.fractions, expected)

    @pytest.mark.unit
    @pytest.mark.parametrize("f", [0.01, 0.22, 0.3, 0.5, 0.99])
    def test_requested_fractions_round_trip(self, f):
        """What you pass to ``add_timepoint`` is what ``fractions`` reports back."""
        from torchref.model.model_collection import ModelCollection

        mc = ModelCollection([_StubModel(0), _StubModel(1)], verbose=0)
        mc.add_timepoint("t", [1.0 - f, f])
        assert mc["t"].fractions[1].item() == pytest.approx(f, abs=1e-6)
        assert mc["t"].fractions.sum().item() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.unit
    def test_dark_excited_fraction_is_the_clamp_floor_not_zero(
        self, two_model_collection
    ):
        """``add_dark`` asks for exactly 0, but the log-clamp at 1e-6 means the stored
        value is 1e-6. Anything deriving a bound from the dark's fraction inherits that
        floor rather than a true zero.
        """
        dark = two_model_collection["dark"]
        assert dark.fractions[1].item() == pytest.approx(1e-6, rel=1e-3)
        assert dark.fractions[1].item() > 0.0

    @pytest.mark.unit
    def test_fraction_dtype_and_device_follow_the_base_models(self):
        from torchref.model.model_collection import ModelCollection

        models = [_StubModel(0), _StubModel(1)]
        mc = ModelCollection(models, verbose=0)
        mc.add_timepoint("t", [0.6, 0.4])
        assert mc["t"].fraction_params.dtype == models[0].dtype_float
        assert mc["t"].fraction_params.device == models[0].device


class TestValidation:
    @pytest.mark.unit
    def test_fractions_must_sum_to_one(self):
        from torchref.model.model_collection import ModelCollection

        mc = ModelCollection([_StubModel(0), _StubModel(1)], verbose=0)
        with pytest.raises(ValueError, match="sum to 1"):
            mc.add_timepoint("t", [0.5, 0.2])

    @pytest.mark.unit
    def test_fraction_count_must_match_the_model_count(self):
        from torchref.model.model_collection import ModelCollection

        mc = ModelCollection([_StubModel(0), _StubModel(1)], verbose=0)
        with pytest.raises(ValueError, match="must match"):
            mc.add_timepoint("t", [1.0])

    @pytest.mark.unit
    def test_duplicate_timepoint_names_are_rejected(self, two_model_collection):
        with pytest.raises(ValueError, match="already exists"):
            two_model_collection.add_timepoint("light", [0.5, 0.5])


class TestFreezing:
    @pytest.mark.unit
    def test_dark_is_frozen_and_timepoints_are_not(self, two_model_collection):
        mc = two_model_collection
        assert mc["dark"].fraction_params.requires_grad is False
        assert mc["light"].fraction_params.requires_grad is True

    @pytest.mark.unit
    def test_freeze_and_unfreeze_flip_the_flag(self, two_model_collection):
        mixed = two_model_collection["light"]
        mixed.freeze_fractions()
        assert mixed.fraction_params.requires_grad is False
        mixed.unfreeze_fractions()
        assert mixed.fraction_params.requires_grad is True

    @pytest.mark.unit
    def test_unfreeze_all_leaves_the_dark_frozen(self, two_model_collection):
        """The dark reference must not become refinable through the bulk call."""
        mc = two_model_collection
        mc.unfreeze_all_fractions()
        assert mc["dark"].fraction_params.requires_grad is False
        assert mc["light"].fraction_params.requires_grad is True

    @pytest.mark.unit
    def test_freeze_all_freezes_every_timepoint(self, two_model_collection):
        mc = two_model_collection
        mc.freeze_all_fractions()
        assert all(
            mc[k].fraction_params.requires_grad is False for k in mc.keys()
        )


class TestOverride:
    @pytest.mark.unit
    def test_override_replaces_fractions_and_clears_back(self, two_model_collection):
        mixed = two_model_collection["light"]
        forced = torch.tensor([0.1, 0.9])

        mixed.set_fraction_override(forced)
        assert mixed.fractions is forced

        mixed.clear_fraction_override()
        assert torch.allclose(
            mixed.fractions, torch.softmax(mixed.fraction_params, dim=0)
        )

    @pytest.mark.unit
    def test_override_reaches_the_forward(self, two_model_collection, hkl):
        """The override is the point at which an external (kinetic) population enters
        the structure-factor sum, so it has to change ``forward``, not just the property.
        """
        mixed = two_model_collection["light"]
        before = mixed(hkl, recalc=True)

        mixed.set_fraction_override(torch.tensor([0.1, 0.9]))
        after = mixed(hkl, recalc=True)

        assert not torch.allclose(before, after)

    @pytest.mark.unit
    def test_override_carries_gradient(self, two_model_collection, hkl):
        """Gradients must flow through the override to whatever produced it."""
        mixed = two_model_collection["light"]
        forced = torch.tensor([0.4, 0.6], requires_grad=True)
        mixed.set_fraction_override(forced)

        mixed(hkl, recalc=True).abs().sum().backward()

        assert forced.grad is not None
        assert torch.isfinite(forced.grad).all()


class TestCollectionLevelViews:
    @pytest.mark.unit
    def test_fractions_matrix_rows_follow_insertion_order(self, two_model_collection):
        mc = two_model_collection
        matrix = mc.get_fractions_matrix()
        assert matrix.shape == (2, 2)
        for row, key in enumerate(mc.keys()):
            assert torch.allclose(matrix[row], mc[key].fractions)

    @pytest.mark.unit
    def test_timepoint_names_excludes_the_dark_key(self, two_model_collection):
        mc = two_model_collection
        assert mc.dark_key == "dark"
        assert mc.timepoint_names == ["light"]
        assert mc.keys() == ["dark", "light"]

    @pytest.mark.unit
    def test_base_models_are_shared_not_copied(self, two_model_collection):
        mc = two_model_collection
        assert mc.n_base_models == 2
        for i in range(2):
            assert mc["dark"].models[i] is mc["light"].models[i]
            assert mc["dark"].models[i] is mc.base_models[i]

    @pytest.mark.unit
    def test_a_timepoint_owns_only_its_fractions(self, two_model_collection):
        """``_SharedMixedModel`` holds the base models in a plain list, not a
        ``ModuleList``, so a timepoint must not re-register their parameters.
        """
        mixed = two_model_collection["light"]
        owned = list(mixed.parameters())
        assert len(owned) == 1
        assert owned[0] is mixed.fraction_params

    @pytest.mark.unit
    def test_shared_base_parameters_are_counted_once(self, two_model_collection):
        """Two timepoints over two shared models: two base parameters plus one set of
        fractions each. Double-registration would make the optimizer step a base model
        once per timepoint.
        """
        mc = two_model_collection
        params = list(mc.parameters())
        assert len(params) == 2 + 2

        base_anchors = [m.anchor for m in mc.base_models]
        for anchor in base_anchors:
            assert sum(p is anchor for p in params) == 1


class TestForwardAndGradient:
    @pytest.mark.unit
    def test_forward_is_the_fraction_weighted_sum_of_the_parts(
        self, two_model_collection, hkl
    ):
        mixed = two_model_collection["light"]
        parts = mixed.get_individual_fcalc(hkl, recalc=True)
        w = mixed.fractions

        expected = w[0] * parts[0] + w[1] * parts[1]
        assert torch.allclose(mixed(hkl, recalc=True), expected)

    @pytest.mark.unit
    def test_gradient_reaches_the_fraction_params(self, two_model_collection, hkl):
        mixed = two_model_collection["light"]
        mixed(hkl, recalc=True).abs().sum().backward()

        grad = mixed.fraction_params.grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    @pytest.mark.unit
    def test_frozen_dark_fractions_receive_no_gradient(
        self, two_model_collection, hkl
    ):
        dark = two_model_collection["dark"]
        dark(hkl, recalc=True).abs().sum().backward()
        assert dark.fraction_params.grad is None
