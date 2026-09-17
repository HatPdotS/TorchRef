"""``forward_batched`` must agree with ``forward_mixed`` row by row, and stay affine.

The batched form exists so ``T`` mixtures share one pass through the scale parameters.
Two properties make it usable:

* **row agreement** -- row ``i`` of the batch must equal the unbatched call on row ``i``,
  otherwise the saving is bought with wrong numbers;
* **affinity in the mixing weights** -- ``ScalerBase.forward`` is
  ``K * b * (aniso * F_calc + f_sol)`` and the mixed solvent is linear in the weights, so
  scaling a *derivative* of the fractions returns the derivative of the scaled structure
  factors. That is what lets a second moment be built from the same machinery instead of
  a separate differentiation path.

Affinity is tested with a **secant**, not a finite difference. Because the mixture is
exactly linear in the activation fraction, ``S(a1) - S(a2)`` equals
``(a1 - a2) * dS/da`` exactly, with no truncation term to tolerate -- so the assertion
is at float precision rather than ``O(h^2)``.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def scaled_collection(pdb_dir, mtz_dir):
    """A dark/light collection on 1DAW with an initialized shared scaler."""
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData
    from torchref.cli._common import load_model
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.collection_scaler import CollectionScaler

    d_min = 2.05
    dark = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    light = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))

    model_dark = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    model_light = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    with torch.no_grad():
        model_light.xyz.refinable_params += 0.2

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", dark, set_as_reference=True)
    dc.add_dataset("light", light)

    mc = ModelCollection([model_dark, model_light], dark_key="dark", verbose=0)
    mc.add_dark()
    mc.add_timepoint("light", [0.78, 0.22])

    scaler = CollectionScaler(dc, mc, verbose=0)
    scaler.initialize()
    return dc, mc, scaler


def _weights(alpha: float) -> torch.Tensor:
    """Single-row activation weights ``[[1 - a, a]]``."""
    return torch.tensor([[1.0 - alpha, alpha]])


@pytest.mark.integration
class TestBatchedMatchesUnbatched:
    def test_every_row_matches_forward_mixed(self, scaled_collection):
        dc, mc, scaler = scaled_collection
        components = dc.component_structure_factors(mc, recalc=True)

        w = torch.tensor([[1.0, 0.0], [0.78, 0.22], [0.3, 0.7]])
        fcalc = mc.mix_component_fcalcs(components, w)

        batched = scaler.forward_batched(fcalc, w)
        assert batched.shape == fcalc.shape

        for i in range(w.shape[0]):
            single = scaler.forward_mixed(fcalc[i], w[i])
            assert torch.allclose(batched[i], single, rtol=1e-6, atol=1e-6), (
                f"batched row {i} disagrees with forward_mixed"
            )

    def test_component_solvent_stack_shape(self, scaled_collection):
        dc, mc, scaler = scaled_collection
        stack = scaler.compute_component_solvent_raw()
        assert stack.shape == (mc.n_base_models, len(dc.hkl))
        assert stack.is_complex()

    def test_solvent_stack_rows_are_the_per_component_solvents(
        self, scaled_collection
    ):
        """A transposed or misordered stack would still have the right shape."""
        _, mc, scaler = scaled_collection
        stack = scaler.compute_component_solvent_raw()
        for k in range(mc.n_base_models):
            assert torch.equal(stack[k], scaler._get_component_f_sol_raw(k))


@pytest.mark.integration
class TestAffineInTheMixingWeights:
    def test_secant_in_alpha_equals_the_scaled_jacobian(self, scaled_collection):
        """The property the two-moment forward model rests on.

        ``forward_batched(dF, J)`` with ``J = dW/da`` is the derivative of the scaled
        mixture, including the solvent term. Exact, because everything between the
        weights and the output is affine.
        """
        dc, mc, scaler = scaled_collection
        components = dc.component_structure_factors(mc, recalc=True)

        a1, a2 = 0.60, 0.10
        w1, w2 = _weights(a1), _weights(a2)
        jac = torch.tensor([[-1.0, 1.0]])  # d/da of [1 - a, a]

        s1 = scaler.forward_batched(mc.mix_component_fcalcs(components, w1), w1)
        s2 = scaler.forward_batched(mc.mix_component_fcalcs(components, w2), w2)
        deriv = scaler.forward_batched(
            mc.mix_component_fcalcs(components, jac), jac
        )

        secant = s1 - s2
        expected = (a1 - a2) * deriv

        rel = (secant - expected).abs().max() / expected.abs().max()
        assert rel < 1e-5, (
            f"secant and scaled Jacobian disagree by {rel:.2e}; the scaler is not "
            f"affine in the mixing weights, so a derivative cannot be scaled this way"
        )

    def test_the_solvent_term_is_included_in_the_derivative(self, scaled_collection):
        """Anti-vacuity: if the per-component solvents were identical, the solvent
        would cancel out of the Jacobian and the test above would hold even with the
        solvent term dropped."""
        _, mc, scaler = scaled_collection
        stack = scaler.compute_component_solvent_raw()
        if mc.n_base_models < 2:
            pytest.skip("needs at least two components")
        assert not torch.allclose(stack[0], stack[1]), (
            "per-component solvents are identical, so this fixture cannot detect a "
            "dropped solvent derivative"
        )

    def test_scaling_is_linear_in_the_structure_factors(self, scaled_collection):
        """The other half of affinity: doubling F_calc at fixed weights doubles the
        F_calc-dependent part, leaving the solvent offset behind."""
        dc, mc, scaler = scaled_collection
        components = dc.component_structure_factors(mc, recalc=True)
        w = _weights(0.22)
        fcalc = mc.mix_component_fcalcs(components, w)

        s1 = scaler.forward_batched(fcalc, w)
        s2 = scaler.forward_batched(2.0 * fcalc, w)
        zero = scaler.forward_batched(torch.zeros_like(fcalc), w)

        # (S(2F) - S(0)) == 2 * (S(F) - S(0))
        lhs, rhs = s2 - zero, 2.0 * (s1 - zero)
        rel = (lhs - rhs).abs().max() / rhs.abs().max()
        assert rel < 1e-5


@pytest.mark.integration
class TestSolventCacheIsNotPoisoned:
    def test_batched_calls_leave_the_cache_alone(self, scaled_collection):
        """Two batched calls with different weights, then a plain one.

        The Jacobian-weighted call carries negative weights, so a leaked cache would
        show up as a sign error rather than a small perturbation.
        """
        dc, mc, scaler = scaled_collection
        components = dc.component_structure_factors(mc, recalc=True)
        w = _weights(0.22)
        jac = torch.tensor([[-1.0, 1.0]])
        fcalc = mc.mix_component_fcalcs(components, w)

        before = scaler.forward_mixed(fcalc[0], w[0]).clone()

        scaler.forward_batched(fcalc, w)
        scaler.forward_batched(mc.mix_component_fcalcs(components, jac), jac)

        after = scaler.forward_mixed(fcalc[0], w[0])
        assert torch.allclose(after, before, rtol=1e-6, atol=1e-6), (
            "a batched call changed what a later forward_mixed returns"
        )
