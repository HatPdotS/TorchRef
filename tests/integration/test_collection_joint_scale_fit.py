"""Joint model-to-data scaling objectives and shared parameter ownership."""

import inspect

import pytest
import torch

pytestmark = pytest.mark.integration


@pytest.fixture
def collection(loaded_model_ft, loaded_reflection_data):
    """One structural model paired with independent dark/timepoint observations."""
    from torchref.io import DatasetCollection
    from torchref.model import ModelCollection

    data = loaded_reflection_data
    dc = DatasetCollection(device=data.device, verbose=0)
    dc.add_dataset("dark", data, set_as_reference=True).add_dataset("t1", data)
    mc = ModelCollection([loaded_model_ft], dark_key="dark", verbose=0)
    mc.add_dark().add_timepoint("t1", [1.0])
    return dc, mc


def _fresh_scaler(collection):
    from torchref.scaling.collection_scaler import CollectionScaler

    dc, mc = collection
    return CollectionScaler(dc, mc, verbose=0).initialize()


def test_it_offers_exactly_the_selectable_objectives():
    from torchref.scaling.collection_scaler import CollectionScaler
    from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET, SCALE_TARGETS

    sig = inspect.signature(CollectionScaler.refine_lbfgs_joint)
    assert sig.parameters["scale_target"].default == DEFAULT_SCALE_TARGET
    assert (
        DEFAULT_SCALE_TARGET == "ls"
    ), "the joint fit's default must track the single-dataset one"
    assert "nll" in SCALE_TARGETS and "ml_noalpha" in SCALE_TARGETS


def test_unknown_objective_fails_closed(collection):
    scaler = _fresh_scaler(collection)
    with pytest.raises(ValueError, match="scale_target must be one of"):
        scaler.refine_lbfgs_joint(scale_target="nll_i")


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
