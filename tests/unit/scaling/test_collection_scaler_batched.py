"""Batched scaling preserves individual results, affinity and solvent caches."""

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
            assert torch.allclose(
                batched[i], single, rtol=1e-6, atol=1e-6
            ), f"batched row {i} disagrees with forward_mixed"

    def test_solvent_stack_rows_are_the_per_component_solvents(self, scaled_collection):
        """A transposed or misordered stack would still have the right shape."""
        _, mc, scaler = scaled_collection
        stack = scaler.compute_component_solvent_raw()
        assert stack.shape == (mc.n_base_models, len(scaler.hkl))
        assert stack.is_complex()
        for k in range(mc.n_base_models):
            assert torch.equal(stack[k], scaler._get_component_f_sol_raw(k))


@pytest.mark.integration
class TestAffineInTheMixingWeights:
    def test_secant_in_alpha_equals_the_scaled_jacobian(self, scaled_collection):
        """The property the two-moment forward model rests on."""
        dc, mc, scaler = scaled_collection
        components = dc.component_structure_factors(mc, recalc=True)

        solvent = scaler.compute_component_solvent_raw()
        assert not torch.allclose(solvent[0], solvent[1])
        a1, a2 = 0.60, 0.10
        w1, w2 = _weights(a1), _weights(a2)
        jac = torch.tensor([[-1.0, 1.0]])  # d/da of [1 - a, a]

        s1 = scaler.forward_batched(mc.mix_component_fcalcs(components, w1), w1)
        s2 = scaler.forward_batched(mc.mix_component_fcalcs(components, w2), w2)
        deriv = scaler.forward_batched(mc.mix_component_fcalcs(components, jac), jac)

        secant = s1 - s2
        expected = (a1 - a2) * deriv

        # Subtraction roundoff scales with its operands, not the smaller secant.
        scale = (s1.abs() + s2.abs() + expected.abs()).max()
        tolerance = 16 * torch.finfo(s1.real.dtype).eps * scale
        assert (secant - expected).abs().max() <= tolerance

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
        """Two batched calls with different weights, then a plain one."""
        dc, mc, scaler = scaled_collection
        components = dc.component_structure_factors(mc, recalc=True)
        w = _weights(0.22)
        jac = torch.tensor([[-1.0, 1.0]])
        fcalc = mc.mix_component_fcalcs(components, w)

        before = scaler.forward_mixed(fcalc[0], w[0]).clone()

        scaler.forward_batched(fcalc, w)
        scaler.forward_batched(mc.mix_component_fcalcs(components, jac), jac)

        after = scaler.forward_mixed(fcalc[0], w[0])
        assert torch.allclose(
            after, before, rtol=1e-6, atol=1e-6
        ), "a batched call changed what a later forward_mixed returns"
