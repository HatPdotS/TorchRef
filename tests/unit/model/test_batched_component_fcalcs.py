"""The batched component/mixture structure factors must equal the per-timepoint loop.

``compute_component_fcalcs`` + ``mix_component_fcalcs`` exist to evaluate each shared base
model once instead of once per timepoint. That is only a saving if it produces the same
numbers as the loop it replaces, including the signed-index and Friedel bookkeeping that
:meth:`ReflectionData.structure_factors` owns.

Both sides are built from a single set of model forwards (one ``recalc=True``, then cache
hits) because repeated structure-factor evaluation is not bit-reproducible: two
``recalc=True`` calls on identical input differ by ~4e-3 on individual ``F_calc``. Comparing
two independent evaluations would measure that noise instead of the contraction.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def pair(pdb_dir, mtz_dir):
    """A 2-component, 2-timepoint collection on 1DAW with unequal fractions."""
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData
    from torchref.cli._common import load_model
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection

    d_min = 2.05
    data = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    model_a = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    model_b = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    with torch.no_grad():
        model_b.xyz.refinable_params += 0.2

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", data, set_as_reference=True)

    mc = ModelCollection([model_a, model_b], dark_key="dark", verbose=0)
    mc.add_dark()
    mc.add_timepoint("light", [0.65, 0.35])
    return dc, mc


@pytest.mark.integration
class TestBatchedMatchesTheLoop:
    def test_component_stack_matches_per_model_structure_factors(self, pair):
        dc, mc = pair
        data = dc["dark"]

        stacked = dc.component_structure_factors(mc, recalc=True)
        assert stacked.shape == (mc.n_base_models, len(data.hkl))

        for k, model in enumerate(mc.base_models):
            reference = data.structure_factors(model, recalc=False)
            assert torch.equal(stacked[k], reference), (
                f"component {k} differs from data.structure_factors"
            )

    def test_mixture_matches_the_per_timepoint_forward(self, pair):
        """The whole point: one contraction standing in for T mixed forwards."""
        dc, mc = pair
        data = dc["dark"]

        stacked = dc.component_structure_factors(mc, recalc=True)
        mixed = mc.mix_component_fcalcs(stacked, mc.get_fractions_matrix())
        assert mixed.shape == (len(mc), len(data.hkl))

        for row, key in enumerate(mc.keys()):
            reference = data.structure_factors(mc[key], recalc=False)
            assert torch.allclose(mixed[row], reference, rtol=1e-6, atol=1e-6), (
                f"timepoint {key!r} differs from its own mixed forward"
            )

    def test_compute_all_fcalc_agrees_on_the_signed_index(self, pair):
        """``compute_all_fcalc`` takes the caller's indices verbatim, so handed the
        signed ones it must reproduce the Friedel-corrected mixture up to the
        conjugation that ``component_structure_factors`` applies."""
        dc, mc = pair
        data = dc["dark"]

        direct = mc.compute_all_fcalc(data._hkl_for_sf(), recalc=True)
        corrected = data.conjugate_friedel(direct)

        stacked = dc.component_structure_factors(mc, recalc=False)
        mixed = mc.mix_component_fcalcs(stacked, mc.get_fractions_matrix())

        assert torch.allclose(corrected, mixed, rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def flagged_pair(pair):
    """The same models against data with **manufactured** Friedel-flagged rows.

    Every reflection file under ``tests/files/`` is already inside the CCP4 ASU, so
    ``friedel_flags.any()`` is False on all of them and any assertion about the index
    convention is silently vacuous. Negating half the Miller indices forces
    canonicalisation to flip them back, reproducing the ~50% flagged fraction real
    P1 data carries. Cell and space group are preserved, so the same models apply.
    """
    from torchref import ReflectionData
    from torchref.io.datasets.collection import DatasetCollection

    dc_ref, mc = pair
    src = dc_ref["dark"]

    hkl = src.hkl.clone()
    half = torch.zeros(len(hkl), dtype=torch.bool)
    half[::2] = True
    hkl[half] = -hkl[half]

    data = ReflectionData.from_tensors(
        hkl=hkl,
        F=src.F.clone(),
        F_sigma=src.F_sigma.clone(),
        cell=src.cell,
        spacegroup=src.spacegroup,
        rfree_flags=src.rfree_flags.clone(),
        device="cpu",
        verbose=0,
    )

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", data, set_as_reference=True)
    return dc, mc


@pytest.mark.integration
class TestConventionIsNotSkipped:
    def test_the_fixture_actually_has_flagged_rows(self, flagged_pair):
        """The precondition, asserted rather than assumed."""
        data = flagged_pair[0]["dark"]
        assert data.friedel_flags is not None
        frac = data.friedel_flags.float().mean().item()
        assert 0.2 < frac < 0.8, f"expected a mixed flag population, got {frac:.3f}"

    def test_conjugation_moves_phases_and_leaves_amplitudes(self, flagged_pair):
        dc, mc = flagged_pair
        data = dc["dark"]

        raw = mc.compute_component_fcalcs(data._hkl_for_sf(), recalc=True)
        corrected = data.conjugate_friedel(raw)

        assert torch.allclose(raw.abs(), corrected.abs())
        assert not torch.allclose(torch.angle(raw), torch.angle(corrected))

    def test_component_stack_is_conjugated_where_flagged(self, flagged_pair):
        """``component_structure_factors`` must apply the conjugation, not skip it.

        Compared against the per-model supported entry point, which is the definition
        of the convention.
        """
        dc, mc = flagged_pair
        data = dc["dark"]

        stacked = dc.component_structure_factors(mc, recalc=True)
        for k, model in enumerate(mc.base_models):
            reference = data.structure_factors(model, recalc=False)
            assert torch.equal(stacked[k], reference), f"component {k} phases differ"

    def test_skipping_the_conjugation_would_be_detected(self, flagged_pair):
        """Anti-vacuity: the naive call this method exists to replace disagrees."""
        dc, mc = flagged_pair
        data = dc["dark"]

        correct = dc.component_structure_factors(mc, recalc=True)
        naive = mc.compute_component_fcalcs(data.hkl, recalc=True)

        assert not torch.allclose(correct, naive), (
            "evaluating on the canonical index gives the same answer as the signed "
            "index plus conjugation -- this fixture cannot detect a convention bug"
        )


@pytest.mark.integration
class TestContraction:
    def test_weights_matrix_is_applied_row_wise(self, pair):
        """A transposed einsum would still return the right shape when T == K."""
        dc, mc = pair
        stacked = dc.component_structure_factors(mc, recalc=True)

        w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mixed = mc.mix_component_fcalcs(stacked, w)

        assert torch.equal(mixed[0], stacked[0])
        assert torch.equal(mixed[1], stacked[1])

    def test_gradient_flows_through_the_contraction(self, pair):
        dc, mc = pair
        stacked = dc.component_structure_factors(mc, recalc=True)
        w = mc.get_fractions_matrix()

        mc.mix_component_fcalcs(stacked, w).abs().sum().backward()

        grad = mc["light"].fraction_params.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
