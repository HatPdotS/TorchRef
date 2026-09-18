"""Population fractions, shared ownership, constraints and gradients."""

import pytest
import torch
from torch import nn

from torchref.config import (
    get_complex_dtype,
    get_default_device,
    get_float_dtype,
    get_int_dtype,
)


class _StubModel(nn.Module):
    """Minimal stand-in for ``ModelFT`` for fraction bookkeeping."""

    def __init__(self, seed: int):
        super().__init__()
        self.anchor = nn.Parameter(
            torch.zeros(1, device=get_default_device(), dtype=get_float_dtype())
        )
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
        base = torch.arange(
            1, n + 1, dtype=self.anchor.dtype, device=self.anchor.device
        )
        amp = (base * float(self._seed + 1)) + self.anchor
        return amp.to(get_complex_dtype())


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
    return torch.tensor(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 1]],
        device=get_default_device(),
        dtype=get_int_dtype(),
    )


class TestPopulationFactorisation:
    @pytest.mark.unit
    def test_fractions_are_the_activation_times_the_branching(
        self, two_model_collection
    ):
        mc = two_model_collection
        alpha = mc.alpha_mean
        expected = torch.stack([1.0 - alpha, alpha * mc.branching()[0][0]])
        assert torch.allclose(mc["light"].fractions, expected)

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
    def test_the_reference_row_is_exactly_e_ref(self, two_model_collection):
        """The dark is the alpha = 0 evaluation, so its excited fraction is exactly zero
        -- not a clamp floor."""
        dark = two_model_collection["dark"]
        assert dark.fractions[1].item() == 0.0
        assert dark.fractions[0].item() == 1.0

    @pytest.mark.unit
    def test_fraction_dtype_and_device_follow_the_base_models(self, any_device):
        from torchref.model.model_collection import ModelCollection

        models = [_StubModel(0).to(any_device), _StubModel(1).to(any_device)]
        mc = ModelCollection(models, verbose=0)
        mc.add_timepoint("t", [0.6, 0.4])
        assert mc._activation_logit.dtype == models[0].dtype_float
        assert mc._activation_logit.device == models[0].device


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
        # Frozen by default: population refinement is opt-in.
        assert mc._activation_logit.requires_grad is False
        assert mc.fraction_parameters() == [
            mc._activation_logit,
            mc._branching_logits[0],
        ]

    @pytest.mark.unit
    def test_freeze_and_unfreeze_flip_the_flag(self, two_model_collection):
        mixed = two_model_collection["light"]
        mixed.freeze_fractions()
        assert mixed.collection._activation_logit.requires_grad is False
        mixed.unfreeze_fractions()
        assert mixed.collection._activation_logit.requires_grad is True

    @pytest.mark.unit
    def test_the_reference_carries_no_population_parameter(self, two_model_collection):
        """The reference is the alpha = 0 evaluation, not a row with pinned logits,
        so there is nothing of its own to freeze or refine."""
        mc = two_model_collection
        mc.unfreeze_all_fractions()
        assert "dark" not in mc._branching_rows
        assert "light" in mc._branching_rows
        assert mc._activation_logit.requires_grad is True

    @pytest.mark.unit
    def test_freeze_all_freezes_every_timepoint(self, two_model_collection):
        mc = two_model_collection
        mc.freeze_all_fractions()
        assert all(not p.requires_grad for p in mc.fraction_parameters())


class TestOverride:
    @pytest.mark.unit
    def test_override_replaces_fractions_and_clears_back(self, two_model_collection):
        mixed = two_model_collection["light"]
        forced = torch.tensor(
            [0.1, 0.9], device=get_default_device(), dtype=get_float_dtype()
        )

        mixed.set_fraction_override(forced)
        assert mixed.fractions is forced

        mixed.clear_fraction_override()
        assert torch.allclose(
            mixed.fractions, mixed.collection.fractions_matrix()[mixed._index]
        )

    @pytest.mark.unit
    def test_override_reaches_the_forward(self, two_model_collection, hkl):
        """The override is the point at which an external (kinetic) population enters
        the structure-factor sum, so it has to change ``forward``, not just the property.
        """
        mixed = two_model_collection["light"]
        before = mixed(hkl, recalc=True)

        mixed.set_fraction_override(
            torch.tensor(
                [0.1, 0.9], device=get_default_device(), dtype=get_float_dtype()
            )
        )
        after = mixed(hkl, recalc=True)

        assert not torch.allclose(before, after)

    @pytest.mark.unit
    def test_override_carries_gradient(self, two_model_collection, hkl):
        """Gradients must flow through the override to whatever produced it."""
        mixed = two_model_collection["light"]
        forced = torch.tensor(
            [0.4, 0.6],
            requires_grad=True,
            device=get_default_device(),
            dtype=get_float_dtype(),
        )
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
        assert (
            list(mixed.parameters()) == []
        ), "a timepoint view registered a parameter of its own"

    @pytest.mark.unit
    def test_shared_base_parameters_are_counted_once(self, two_model_collection):
        """Two timepoints over two shared models: two base parameters plus one set of
        fractions each. Double-registration would make the optimizer step a base model
        once per timepoint.
        """
        mc = two_model_collection
        params = list(mc.parameters())
        # two base anchors + activation + lambda + one branching row
        assert len(params) == 2 + 3

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
    def test_gradient_reaches_the_activation(self, two_model_collection, hkl):
        mc = two_model_collection
        mc.unfreeze_all_fractions()
        mixed = mc["light"]
        mixed(hkl, recalc=True).abs().sum().backward()

        grad = mc._activation_logit.grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    @pytest.mark.unit
    def test_the_reference_contributes_no_activation_gradient(
        self, two_model_collection, hkl
    ):
        """The dark dataset carries no activation information, so its row must be
        exactly e_ref with no path back to the shared parameter."""
        mc = two_model_collection
        mc.unfreeze_all_fractions()
        mc["dark"](hkl, recalc=True).abs().sum().backward()
        assert (
            mc._activation_logit.grad is None
            or float(mc._activation_logit.grad.abs().max()) == 0.0
        )


class TestSharedActivation:
    """One activation serves every timepoint; only the branching varies with time."""

    @pytest.mark.unit
    def test_a_second_timepoint_may_rebranch_at_the_same_activation(self):
        """Three components, two timepoints, same 30% activated but split differently
        between the two excited states."""
        from torchref.model.model_collection import ModelCollection

        mc = ModelCollection([_StubModel(i) for i in range(3)], verbose=0)
        mc.add_dark()
        mc.add_timepoint("early", [0.7, 0.3, 0.0])
        mc.add_timepoint("late", [0.7, 0.0, 0.3])

        assert float(mc.alpha_mean) == pytest.approx(0.3, abs=1e-5)
        assert torch.allclose(
            mc["early"].fractions,
            torch.tensor(
                [0.7, 0.3, 0.0], device=get_default_device(), dtype=get_float_dtype()
            ),
            atol=1e-5,
        )
        assert torch.allclose(
            mc["late"].fractions,
            torch.tensor(
                [0.7, 0.0, 0.3], device=get_default_device(), dtype=get_float_dtype()
            ),
            atol=1e-5,
        )

    @pytest.mark.unit
    def test_a_conflicting_activation_is_rejected_not_projected(self):
        """A silent least-squares projection here would produce populations nobody
        asked for, so this raises and names the escape hatch."""
        from torchref.model.model_collection import ModelCollection

        mc = ModelCollection([_StubModel(0), _StubModel(1)], verbose=0)
        mc.add_dark()
        mc.add_timepoint("early", [0.7, 0.3])

        with pytest.raises(ValueError, match="set_fraction_override"):
            mc.add_timepoint("late", [0.5, 0.5])

    @pytest.mark.unit
    def test_adding_the_reference_after_a_timepoint_leaves_activation_alone(self):
        """A pure-reference row carries no activation information."""
        from torchref.model.model_collection import ModelCollection

        mc = ModelCollection([_StubModel(0), _StubModel(1)], verbose=0)
        mc.add_timepoint("light", [0.78, 0.22])
        mc.add_dark()
        assert float(mc.alpha_mean) == pytest.approx(0.22, abs=1e-5)


class TestActivationJacobian:
    @pytest.mark.unit
    def test_rows_sum_to_zero_and_the_reference_row_vanishes(
        self, two_model_collection
    ):
        """Fractions stay on the simplex, so the derivative is tangent to it; and the
        reference does not move with the activation at all."""
        mc = two_model_collection
        jac = mc.activation_jacobian()

        assert jac.shape == (len(mc), mc.n_base_models)
        assert torch.allclose(
            jac.sum(dim=1),
            torch.zeros(len(mc), device=get_default_device(), dtype=get_float_dtype()),
            atol=1e-6,
        )
        assert torch.equal(
            jac[0],
            torch.zeros(
                mc.n_base_models, device=get_default_device(), dtype=get_float_dtype()
            ),
        )
        assert float(jac[1][0]) == pytest.approx(-1.0)

    @pytest.mark.unit
    def test_fractions_matrix_is_e_ref_plus_alpha_times_the_jacobian(
        self, two_model_collection
    ):
        mc = two_model_collection
        e_ref = torch.zeros(
            mc.n_base_models, device=get_default_device(), dtype=get_float_dtype()
        )
        e_ref[0] = 1.0
        expected = e_ref.unsqueeze(0) + mc.alpha_mean * mc.activation_jacobian()
        assert torch.allclose(mc.fractions_matrix(), expected)

    @pytest.mark.unit
    def test_the_mixture_is_exactly_linear_in_the_activation(
        self, two_model_collection
    ):
        """The property the second moment rests on: the secant equals the derivative,
        so there is no truncation term anywhere downstream."""
        mc = two_model_collection
        jac = mc.activation_jacobian()

        mc.set_activation(0.6)
        w1 = mc.fractions_matrix().clone()
        mc.set_activation(0.1)
        w2 = mc.fractions_matrix().clone()

        assert torch.allclose(w1 - w2, (0.6 - 0.1) * jac, atol=1e-6)


class TestActivationDispersion:
    @pytest.mark.unit
    def test_lambda_is_exactly_zero_by_default(self, two_model_collection):
        """Exactly, not approximately: sigmoid can never return 0, so a fixed float is
        the only way to reproduce the coherent single-moment model."""
        mc = two_model_collection
        assert float(mc.lambda_twin) == 0.0
        assert float(mc.sigma_alpha_sq) == 0.0

    @pytest.mark.unit
    def test_lambda_is_not_a_live_parameter_until_asked_for(self, two_model_collection):
        mc = two_model_collection

        def _present():
            # Identity, not ``in``: ``==`` on tensors is elementwise.
            return any(p is mc._lambda_logit for p in mc.fraction_parameters())

        assert not _present()

        mc.set_lambda_twin(0.3, refinable=True)
        assert _present()
        assert mc._lambda_logit.requires_grad is True

    @pytest.mark.unit
    @pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 1.0])
    def test_the_variance_bound_holds_by_construction(self, two_model_collection, lam):
        mc = two_model_collection
        mc.set_lambda_twin(lam)
        alpha = float(mc.alpha_mean)
        bound = alpha * (1.0 - alpha)

        # The bound holds algebraically; the tolerance is float32 ulp, since the two
        # sides reach alpha (1 - alpha) by different arithmetic.
        sigma_sq = float(mc.sigma_alpha_sq)
        assert 0.0 <= sigma_sq <= bound * (1.0 + 1e-6)
        assert sigma_sq == pytest.approx(bound * lam, rel=1e-5)

    @pytest.mark.unit
    def test_lambda_one_saturates_the_bound(self, two_model_collection):
        """The fully incoherent limit: every crystal either fully activated or dark."""
        mc = two_model_collection
        mc.set_lambda_twin(1.0)
        alpha = float(mc.alpha_mean)
        assert float(mc.sigma_alpha_sq) == pytest.approx(
            alpha * (1.0 - alpha), rel=1e-5
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_lambda_is_rejected(self, two_model_collection, bad):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            two_model_collection.set_lambda_twin(bad)

    @pytest.mark.unit
    def test_refined_lambda_stays_strictly_interior(self, two_model_collection):
        """Once refinable it is a sigmoid, so it can approach but never reach the
        bounds -- which is why the fixed path exists."""
        mc = two_model_collection
        mc.set_lambda_twin(0.0, refinable=True)
        value = float(mc.lambda_twin.detach())
        assert 0.0 < value < 1.0
