"""Characterisation of the collection X-ray targets' observable contract.

Written to protect a change of fraction storage and a move to batched
``[T, R]`` accessors. The three regressions worth catching are all silent:

* reading **raw** ``ReflectionData.F`` instead of the scaled ``get_corrected_data()``,
  which drops the inter-dataset scaling;
* masking with the 2-way ``rfree_flags`` instead of the 3-way ``work``/``free``/
  ``validation`` subset, which lets validation reflections back into the loss;
* returning a **mean** where the target returns a **sum**, which reweights the X-ray
  term by 1/N against every restraint.

Each is pinned by a *deterministic invariant* rather than a stored number.
``model.forward`` is run-to-run nondeterministic even inside one process (threaded
reduction order; ~4e-3 absolute on individual ``F_calc``), so "the loss equals 3.6e4"
is a weaker statement than "the loss responds to this input the way only a correct
implementation can". Measured for reference: the summed losses here vary by ~4e-7
relative across repeated calls, so the literal checks that remain are given a
tolerance three orders of magnitude above that.

The fixture deliberately makes the two datasets and the two models **differ**. With one
``ReflectionData`` added twice -- as the sigma_A collection fixture does -- the observed
difference is identically zero and a raw-vs-scaled regression is invisible.
"""

import pytest
import torch

# Repeated-call spread of the summed losses, measured on this fixture. The literal
# assertions below sit far above it; tightening past ~1e-6 would flake.
LOSS_RTOL = 1e-4


@pytest.fixture(scope="module")
def collection(pdb_dir, mtz_dir):
    """``(dc, mc, scaler)`` for a dark/light pair with a real difference in both
    the data and the models.

    The light dataset carries a shared scale view, so ``F_obs_light != F_obs_dark``
    only through the *corrected* accessor -- which is what makes the raw-vs-scaled
    invariant below bite. The light model is displaced, so ``ΔF_calc != 0`` too.
    """

    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData
    from torchref.cli._common import load_model
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.collection_scaler import CollectionScaler

    data_dark = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    data_light = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))

    # max_res is required: without it the FFT grid setup has no resolution to size from.
    d_min = 2.05
    model_dark = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    model_light = load_model(str(pdb), max_res=d_min, device="cpu", verbose=0)
    with torch.no_grad():
        # A real displacement, so the calculated difference is not degenerate.
        xyz = model_light.xyz.refinable_params
        xyz += 0.15 * torch.ones_like(xyz)

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", data_dark, set_as_reference=True)
    dc.add_dataset("light", data_light)

    dc.scale(nsteps=1)

    mc = ModelCollection([model_dark, model_light], dark_key="dark", verbose=0)
    mc.add_dark()
    mc.add_timepoint("light", [0.7, 0.3])

    scaler = CollectionScaler(dc, mc, verbose=0)
    scaler.initialize()
    return dc, mc, scaler


def _targets(dc, mc, scaler):
    from torchref.refinement.targets import (
        CollectionDifferenceIntensityTarget,
        CollectionDifferenceTarget,
        CollectionMLTarget,
    )

    return {
        "difference": CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0),
        "difference_i": CollectionDifferenceIntensityTarget(
            dc, mc, scaler=scaler, verbose=0
        ),
        "ml": CollectionMLTarget(dc, mc, scaler=scaler, verbose=0),
    }


@pytest.mark.integration
class TestObservedAmplitudesAreScaled:
    """The loss must move when a dataset's own scale moves.

    ``DatasetCollection.scale()`` fits a per-dataset shared corrections that
    exists only in ``get_corrected_data()``. A target reading raw ``.F`` is completely
    blind to it, so this is a direct test of which accessor is in use.
    """

    @pytest.mark.parametrize("name", ["difference", "difference_i", "ml"])
    def test_loss_responds_to_the_datasets_own_log_scale(self, collection, name):
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)[name]

        before = target.forward().item()
        light = dc["light"]
        original = dc.scaler.raw_parameters[1, 0].detach().clone()
        try:
            with torch.no_grad():
                dc.scaler.raw_parameters[1, 0] += 0.25  # ~28% on amplitudes
            target.maintenance() if hasattr(target, "maintenance") else None
            after = target.forward().item()
        finally:
            with torch.no_grad():
                dc.scaler.raw_parameters[1, 0].copy_(original)

        rel = abs(after - before) / abs(before)
        assert rel > 1e-3, (
            f"{name}: changing the light dataset's log_scale moved the loss by only "
            f"{rel:.2e}; the target is reading raw amplitudes, not the scaled ones"
        )

    def test_corrected_and_raw_amplitudes_actually_differ(self, collection):
        """Anti-vacuity: the invariant above is only meaningful if the two accessors
        disagree on this fixture."""
        dc, _, _ = collection
        light = dc["light"]
        with torch.no_grad():
            dc.scaler.raw_parameters[1, 0] += 0.25
            corrected, _ = light.get_corrected_data()
            raw = light.F_raw
            differ = not torch.allclose(corrected, raw)
            dc.scaler.raw_parameters[1, 0] -= 0.25
        assert differ


@pytest.mark.integration
class TestSubsetSelectionIsThreeWay:
    """Work / free / validation, with validation carved out of both."""

    def test_subsets_are_disjoint_and_cover_the_valid_reflections(self, collection):
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)["difference"]
        data = dc["dark"]

        work = data.work.mask
        free = data.free.mask
        val = data.validation.mask

        assert not (work & free).any()
        assert not (work & val).any()
        assert not (free & val).any()
        assert torch.equal(work | free | val, data.masks().to(torch.bool))
        assert target.use_set == "work"

    def test_carving_a_validation_set_shrinks_the_work_and_free_sets(self, collection):
        """A 2-way ``rfree_flags`` implementation cannot see a validation set at all,
        so the reported ``n`` would not move.
        """
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)["difference"]
        data = dc["dark"]

        n_before = target._n_reflections()
        free_before = data.free.n
        flags = None if data.validation_flags is None else data.validation_flags.clone()
        try:
            data.generate_validation_set(val_fraction_of_free=0.5, seed=0)
            assert data.validation.n > 0, "no validation reflections were carved"
            assert data.free.n < free_before, "free set did not shrink"
            assert target._n_reflections() <= n_before
        finally:
            data.validation_flags = flags
            data._subset_fp = None

    @pytest.mark.parametrize("use_set", ["work", "free"])
    def test_loss_is_restricted_to_the_selected_subset(self, collection, use_set):
        """Work and free are different sizes here, so a target that ignored
        ``use_set`` would return the same number for both."""
        from torchref.refinement.targets import CollectionDifferenceTarget

        dc, mc, scaler = collection
        target = CollectionDifferenceTarget(
            dc, mc, scaler=scaler, use_set=use_set, verbose=0
        )
        assert target.use_set == use_set
        n = target._n_reflections()
        expected = sum(
            (dc[k].work if use_set == "work" else dc[k].free).n
            for k in target._keys()
        )
        assert n == expected


@pytest.mark.integration
class TestLossesAreSummedNotAveraged:
    """A summed X-ray term grows with the data; a meaned one does not.

    This is the invariant that catches a 1/N reweight, which is otherwise invisible
    -- it looks exactly like a change of X-ray weight.
    """

    def test_adding_a_dataset_grows_the_absolute_loss(self, collection, pdb_dir, mtz_dir):
        """The expected ratio is n_after / n_before, and that is 3/2, not 2.

        The fixture already holds two datasets (dark + light), so adding a third takes
        the absolute target from 2 to 3. This test used to expect 2.0 because it ran on
        ``CollectionRiceTarget``, which overrode ``_keys()`` to drop the dark reference
        and so went from 1 to 2. ``ml`` fits every dataset including the dark.

        A meaned target would stay near 1.0 either way, which is what this is for.
        """
        from torchref import ReflectionData
        from torchref.refinement.targets import CollectionMLTarget

        dc, mc, scaler = collection
        target_before = CollectionMLTarget(dc, mc, scaler=scaler, verbose=0)
        n_before = len(target_before._keys())
        one = target_before.forward().item()

        extra = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz_dir / "1DAW.mtz"))
        dc.add_dataset("light2", extra)
        mc.add_timepoint("light2", [0.7, 0.3])
        try:
            target_after = CollectionMLTarget(dc, mc, scaler=scaler, verbose=0)
            n_after = len(target_after._keys())
            two = target_after.forward().item()
        finally:
            dc._datasets.pop("light2")
            dc._dataset_order.remove("light2")
            del mc._timepoints["light2"]
            mc._order.remove("light2")

        assert (n_before, n_after) == (2, 3)
        ratio = two / one
        # Tolerance is loose because the shared Luzzati beta is REFITTED on the pooled
        # free reflections of the larger collection, so the per-reflection loss moves a
        # little too. That is a property of the target, not slack: the two hypotheses
        # this test separates are 1.5 and 1.0, which are far apart.
        assert ratio == pytest.approx(n_after / n_before, rel=0.15), (
            f"{n_after} datasets gave {ratio:.3f}x the loss of {n_before}; a summed "
            f"target should scale with the count and a meaned one stay near 1.0"
        )


@pytest.mark.integration
class TestReportedNumbers:
    """The shape of what ``get_rfactor`` / ``stats`` promise, plus reproducibility."""

    @pytest.mark.parametrize("name", ["difference", "difference_i", "ml"])
    def test_forward_is_finite_and_reproducible(self, collection, name):
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)[name]
        first = target.forward().item()
        second = target.forward().item()
        assert torch.isfinite(torch.tensor(first))
        assert second == pytest.approx(first, rel=LOSS_RTOL)

    def test_rfactor_shape_and_range(self, collection):
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)["difference"]

        rf = target.get_rfactor()
        assert set(rf) == {"per_dataset", "rwork_pct", "rfree_pct"}
        assert set(rf["per_dataset"]) == {"dark", "light"}
        for key, (rwork, rfree) in rf["per_dataset"].items():
            assert 0.0 < rwork < 1.0, f"{key} rwork out of range: {rwork}"
            assert 0.0 < rfree < 1.0, f"{key} rfree out of range: {rfree}"
        assert set(rf["rwork_pct"]) == {"p10", "p25", "p50", "p75", "p90"}

    def test_stats_reports_the_median_of_the_per_dataset_distribution(self, collection):
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)["difference"]

        stats = target.stats()
        rf = target.get_rfactor()
        for key in ("loss", "n", "rwork", "rfree"):
            assert key in stats, f"missing stat: {key}"
        assert stats["rwork"].value == pytest.approx(
            rf["rwork_pct"]["p50"], rel=LOSS_RTOL
        )
        assert stats["n"].value == target._n_reflections()

    def test_gradient_reaches_the_light_model(self, collection):
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)["difference"]

        target.forward().backward()
        grad = mc.base_models[1].xyz.refinable_params.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().max() > 0
