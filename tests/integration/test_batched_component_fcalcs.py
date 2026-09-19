"""Batched structure factors preserve mixed-model values and Friedel phases."""

import pytest
import torch

from torchref.config import get_default_device, get_float_dtype

pytestmark = pytest.mark.integration


class TestBatchedMatchesTheLoop:
    def test_component_stack_matches_per_model_structure_factors(
        self, difference_models
    ):
        dc, mc = difference_models
        data = dc["dark"]

        stacked = dc.component_structure_factors(mc, recalc=True)
        assert stacked.shape == (mc.n_base_models, len(data.hkl))

        for k, model in enumerate(mc.base_models):
            reference = data.structure_factors(model, recalc=False)
            assert torch.equal(
                stacked[k], reference
            ), f"component {k} differs from data.structure_factors"

    def test_mixture_matches_the_per_timepoint_forward(self, difference_models):
        """A batched contraction agrees with each mixed-model forward."""
        dc, mc = difference_models
        data = dc["dark"]

        stacked = dc.component_structure_factors(mc, recalc=True)
        mixed = mc.mix_component_fcalcs(stacked, mc.get_fractions_matrix())
        assert mixed.shape == (len(mc), len(data.hkl))

        for row, key in enumerate(mc.keys()):
            reference = data.structure_factors(mc[key], recalc=False)
            assert torch.allclose(
                mixed[row], reference, rtol=1e-6, atol=1e-6
            ), f"timepoint {key!r} differs from its own mixed forward"

    def test_compute_all_fcalc_agrees_on_the_signed_index(self, difference_models):
        """``compute_all_fcalc`` takes the caller's indices verbatim, so handed the
        signed ones it must reproduce the Friedel-corrected mixture up to the
        conjugation that ``component_structure_factors`` applies."""
        dc, mc = difference_models
        data = dc["dark"]

        direct = mc.compute_all_fcalc(data._hkl_for_sf(), recalc=True)
        corrected = data.conjugate_friedel(direct)

        stacked = dc.component_structure_factors(mc, recalc=False)
        mixed = mc.mix_component_fcalcs(stacked, mc.get_fractions_matrix())

        assert torch.allclose(corrected, mixed, rtol=1e-6, atol=1e-6)


@pytest.fixture
def flagged_pair(difference_models):
    """Pair deposited models with both signed Miller-index conventions."""
    from torchref import ReflectionData
    from torchref.io.datasets.collection import DatasetCollection

    dc_ref, mc = difference_models
    src = dc_ref["dark"]

    hkl = src.hkl.clone()
    half = torch.zeros(len(hkl), dtype=torch.bool, device=get_default_device())
    half[::2] = True
    hkl[half] = -hkl[half]

    data = ReflectionData.from_tensors(
        hkl=hkl,
        F=src.F.clone(),
        F_sigma=src.F_sigma.clone(),
        cell=src.cell,
        spacegroup=src.spacegroup,
        rfree_flags=src.rfree_flags.clone(),
        device=get_default_device(),
        verbose=0,
    )

    assert data.friedel_flags.any() and (~data.friedel_flags).any()
    dc = DatasetCollection(verbose=0, device=get_default_device())
    dc.add_dataset("dark", data, set_as_reference=True)
    return dc, mc


class TestConventionIsNotSkipped:

    def test_component_stack_is_conjugated_where_flagged(self, flagged_pair):
        """``component_structure_factors`` must apply the conjugation, not skip it."""
        dc, mc = flagged_pair
        data = dc["dark"]

        stacked = dc.component_structure_factors(mc, recalc=True)
        signed = mc.compute_component_fcalcs(data._hkl_for_sf(), recalc=False)
        flagged = data.friedel_flags
        assert torch.equal(stacked[:, flagged], signed[:, flagged].conj())
        assert torch.equal(stacked[:, ~flagged], signed[:, ~flagged])
        assert not torch.allclose(stacked[:, flagged], signed[:, flagged])


class TestContraction:
    def test_weights_matrix_is_applied_row_wise(self, difference_models):
        """A transposed einsum would still return the right shape when T == K."""
        dc, mc = difference_models
        stacked = dc.component_structure_factors(mc, recalc=True)

        w = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            device=get_default_device(),
            dtype=get_float_dtype(),
        )
        mixed = mc.mix_component_fcalcs(stacked, w)

        assert torch.equal(mixed[0], stacked[0])
        assert torch.equal(mixed[1], stacked[1])

    def test_gradient_flows_through_the_contraction(self, difference_models):
        dc, mc = difference_models
        mc.unfreeze_all_fractions()
        stacked = dc.component_structure_factors(mc, recalc=True)
        w = mc.get_fractions_matrix()

        mc.mix_component_fcalcs(stacked, w).abs().sum().backward()

        grad = mc._activation_logit.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
