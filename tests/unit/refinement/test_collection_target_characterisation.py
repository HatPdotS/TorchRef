"""Collection targets consume live scaled data, selected subsets and summed losses."""

import pytest
import torch

# Allow float32 differences from threaded structure-factor reductions.
LOSS_RTOL = 1e-4


@pytest.fixture(scope="module")
def collection(pdb_dir, mtz_dir):
    """``(dc, mc, scaler)`` for a dark/light pair with a real difference in both the
    data and the models."""

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
    """The loss must move when a dataset's own scale moves."""

    @pytest.mark.parametrize("name", ["difference", "difference_i", "ml"])
    def test_loss_responds_to_the_datasets_own_log_scale(self, collection, name):
        dc, mc, scaler = collection
        target = _targets(dc, mc, scaler)[name]

        before = target.forward().item()
        original = dc.scaler.raw_parameters[1, 0].detach().clone()
        try:
            with torch.no_grad():
                dc.scaler.raw_parameters[1, 0] += 0.25
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


@pytest.mark.integration
class TestSubsetSelectionIsThreeWay:
    """Work / free / validation, with validation carved out of both."""

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
            (dc[k].work if use_set == "work" else dc[k].free).n for k in target._keys()
        )
        assert n == expected


@pytest.mark.integration
class TestLossesAreSummedNotAveraged:
    """A summed X-ray term grows with the data; a meaned one does not."""

    def test_adding_a_dataset_grows_the_absolute_loss(
        self, collection, pdb_dir, mtz_dir
    ):
        """Summed loss grows in proportion to the number of datasets."""
        from torchref import ReflectionData
        from torchref.refinement.targets import CollectionMLTarget

        dc, mc, scaler = collection
        target_before = CollectionMLTarget(dc, mc, scaler=scaler, verbose=0)
        n_before = len(target_before._keys())
        one = target_before.forward().item()

        extra = ReflectionData(device="cpu", verbose=0).load_mtz(
            str(mtz_dir / "1DAW.mtz")
        )
        saved = (
            dc._datasets,
            list(dc._dataset_order),
            dc.hkl,
            dc.scaler,
            dc.scaling_metrics,
        )
        branching = torch.nn.ParameterList(mc._branching_logits)
        dc.add_dataset("light2", extra)
        mc.add_timepoint("light2", [0.7, 0.3])
        try:
            target_after = CollectionMLTarget(dc, mc, scaler=scaler, verbose=0)
            n_after = len(target_after._keys())
            two = target_after.forward().item()
        finally:
            (
                dc._datasets,
                dc._dataset_order,
                dc._common_hkl,
                dc.scaler,
                dc.scaling_metrics,
            ) = saved
            del mc._timepoints["light2"]
            mc._order.remove("light2")
            mc._branching_rows.pop("light2")
            mc._branching_logits = branching

        assert (n_before, n_after) == (2, 3)
        ratio = two / one
        # Refitting shared beta on pooled free reflections shifts the per-row loss.
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
