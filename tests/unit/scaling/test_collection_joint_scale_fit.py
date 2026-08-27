"""``CollectionScaler.refine_lbfgs_joint`` -- the model-to-data fit, per row.

It used to hand-roll a Rice likelihood inline with ``beta = sigma_obs**2`` and no
normaliser at all. Both are now gone: it builds a row of ``XRAY_TARGETS``, exactly as
``ScalerBase.refine_lbfgs`` does, and normalises the objective because L-BFGS converges on
absolute tolerances.

The tests here are mostly about what must NOT differ between the two scale fits, since
"one likelihood, two call sites" is the whole claim.
"""

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
    from torchref.scaling.collection_scaler import CollectionScaler

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
def test_the_hand_rolled_rice_is_gone():
    """No private likelihood in the scaling package.

    It set ``beta = sigma_obs**2`` -- pairing a measurement sigma with a Rice ``Sigma``,
    which asserts an isotropic *complex* error where sigma_obs carries no phase at all.
    ``xray_likelihoods`` records that no regime makes that correct, and the taxonomy
    deliberately offers no such row; a copy inside the scaler bypassed both.
    """
    import torchref.scaling.collection_scaler as cs

    src = inspect.getsource(cs)
    assert "rice_math" not in src, "the scaler grew a private Rice likelihood again"
    assert "SigmaAEstimator" not in src, (
        "the scaler precomputes beta again; rows own their own estimator"
    )
    assert "create_xray_target" in src, "the joint fit must build a taxonomy row"


@pytest.mark.unit
def test_it_offers_exactly_the_selectable_objectives():
    from torchref.scaling.collection_scaler import CollectionScaler
    from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET, SCALE_TARGETS

    sig = inspect.signature(CollectionScaler.refine_lbfgs_joint)
    assert sig.parameters["scale_target"].default == DEFAULT_SCALE_TARGET
    assert DEFAULT_SCALE_TARGET == "ls", (
        "the joint fit's default must track the single-dataset one"
    )
    assert "nll" in SCALE_TARGETS and "ml_noalpha" in SCALE_TARGETS


@pytest.mark.integration
def test_unknown_objective_fails_closed(collection):
    scaler = _fresh_scaler(collection)
    with pytest.raises(ValueError, match="scale_target must be one of"):
        scaler.refine_lbfgs_joint(scale_target="nll_i")


@pytest.mark.integration
@pytest.mark.parametrize("scale_target", ["ls", "nll", "ml_noalpha"])
def test_every_objective_fits_finite_parameters(collection, scale_target):
    """Every selectable row must drive the joint fit to finite parameters.

    The old fit produced non-finite scales on real data; the objective it handed L-BFGS
    was unnormalised against absolute tolerances, which is a documented way to get there.
    """
    scaler = _fresh_scaler(collection)
    m = scaler.refine_lbfgs_joint(
        nsteps=2, max_iter=20, verbose=False, scale_target=scale_target
    )
    for p in scaler.parameters():
        assert torch.isfinite(p).all(), f"{scale_target}: non-finite scale parameter"
    assert m["rwork"] and all(0.0 < r < 1.0 for r in m["rwork"]), m["rwork"]


@pytest.mark.integration
def test_the_dataset_view_shares_the_parents_parameters(collection):
    """A row's scaler must be a view, not a copy.

    If the view registered the parent as a submodule, the parent's parameters would be
    counted twice and L-BFGS would see duplicate leaves. If it copied them, the fit would
    optimise something the collection never reads.
    """
    from torchref.scaling.collection_scaler import _DatasetScalerView

    scaler = _fresh_scaler(collection)
    dc, mc = collection
    fracs = mc[mc.dark_key].fractions.detach()
    view = _DatasetScalerView(scaler, fracs)

    # No parameters of its own -- only the bound fractions buffer.
    assert list(view.parameters()) == []
    # And it routes through the parent's mixed-solvent path.
    with torch.no_grad():
        fcalc = mc[mc.dark_key](dc[mc.dark_key].hkl)
        got = view(fcalc)
        want = scaler.forward_mixed(fcalc, fracs)
    assert torch.equal(got, want)


@pytest.mark.integration
def test_the_objective_is_normalised(collection):
    """The loss L-BFGS sees must be O(1), not the data's own magnitude.

    ``tolerance_grad``/``tolerance_change`` are absolute. This fit had no normaliser at
    all, so on a large work set under unit weights the loss reached a magnitude where its
    own float32 ulp exceeded the decrease the line search was trying to resolve.
    """
    src = inspect.getsource(
        __import__(
            "torchref.scaling.collection_scaler", fromlist=["CollectionScaler"]
        ).CollectionScaler.refine_lbfgs_joint
    )
    assert "_norm" in src, "the joint objective is unnormalised again"
    # The U penalty must NOT follow the observable/objective: sharing a normaliser that
    # moves with the objective silently changes the regularisation strength.
    assert "work.F" in src, "the normaliser must be built from amplitudes"
