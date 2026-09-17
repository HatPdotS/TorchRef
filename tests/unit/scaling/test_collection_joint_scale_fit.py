"""Joint model-to-data scaling objectives and shared parameter ownership."""

import inspect

import pytest
import torch


@pytest.fixture(scope="module")
def collection(pdb_dir, mtz_dir):
    pdb, mtz = pdb_dir / "1DAW.pdb", mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    from torchref import LBFGSRefinement, ReflectionData
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection

    ref = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0)
    extra = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", ref.reflection_data, set_as_reference=True)
    dc.add_dataset("t1", extra)

    mc = ModelCollection([ref.model], dark_key="dark", verbose=0)
    mc.add_dark()
    mc.add_timepoint("t1", [1.0])
    return dc, mc


def _fresh_scaler(collection):
    from torchref.scaling.collection_scaler import CollectionScaler

    dc, mc = collection
    return CollectionScaler(dc, mc, verbose=0).initialize()


@pytest.mark.unit
def test_it_offers_exactly_the_selectable_objectives():
    from torchref.scaling.collection_scaler import CollectionScaler
    from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET, SCALE_TARGETS

    sig = inspect.signature(CollectionScaler.refine_lbfgs_joint)
    assert sig.parameters["scale_target"].default == DEFAULT_SCALE_TARGET
    assert (
        DEFAULT_SCALE_TARGET == "ls"
    ), "the joint fit's default must track the single-dataset one"
    assert "nll" in SCALE_TARGETS and "ml_noalpha" in SCALE_TARGETS


@pytest.mark.integration
def test_unknown_objective_fails_closed(collection):
    scaler = _fresh_scaler(collection)
    with pytest.raises(ValueError, match="scale_target must be one of"):
        scaler.refine_lbfgs_joint(scale_target="nll_i")


@pytest.mark.integration
@pytest.mark.parametrize("scale_target", ["ls", "nll", "ml_noalpha"])
def test_every_objective_fits_finite_parameters(collection, scale_target):
    """Every selectable row must drive the joint fit to finite parameters."""
    scaler = _fresh_scaler(collection)
    m = scaler.refine_lbfgs_joint(
        nsteps=2, max_iter=20, verbose=False, scale_target=scale_target
    )
    for p in scaler.parameters():
        assert torch.isfinite(p).all(), f"{scale_target}: non-finite scale parameter"
    assert m["rwork"] and all(0.0 < r < 1.0 for r in m["rwork"]), m["rwork"]


@pytest.mark.integration
def test_the_dataset_view_shares_the_parents_parameters(collection):
    """A row's scaler must be a view, not a copy."""
    from torchref.scaling.collection_scaler import _DatasetScalerView

    scaler = _fresh_scaler(collection)
    dc, mc = collection
    fracs = mc[mc.dark_key].fractions.detach()
    view = _DatasetScalerView(scaler, fracs)

    # No parameters of its own -- only the bound fractions buffer.
    assert list(view.parameters()) == []
    assert view.device == scaler.device
    # And it routes through the parent's mixed-solvent path.
    with torch.no_grad():
        fcalc = mc[mc.dark_key](dc[mc.dark_key].hkl)
        got = view(fcalc)
        want = scaler.forward_mixed(fcalc, fracs)
    assert torch.equal(got, want)
