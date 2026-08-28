"""``DatasetCollection.scale()`` -- the data-to-data scale fit.

The only fit in the library with no model on either side: it puts one dataset onto
another, both of them measurements of the same quantity. That is why it is least squares
and takes no objective at all -- there is no model error for a sigma_A row to account for,
and sigma weighting collapses on a scale fit.

The load-bearing test here is the free-set one. That fit runs *upstream of every target*,
so a leak there compromises every free-set number the pipeline later reports, and no
downstream test would notice.
"""

import pytest
import torch


@pytest.fixture
def pair(mtz_dir):
    """Two copies of 1DAW as a reference + one dataset to scale onto it."""
    mtz = mtz_dir / "1DAW.mtz"
    if not mtz.exists():
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData
    from torchref.io.datasets.collection import DatasetCollection

    ref = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    other = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("ref", ref, set_as_reference=True)
    dc.add_dataset("other", other)
    return dc


def _fitted(dc):
    ds = dc["other"]
    return ds.log_scale.detach().clone(), ds.U_aniso.detach().clone()


@pytest.mark.integration
def test_scale_never_touches_the_free_set(pair):
    """Corrupting the free reflections must not move the fitted parameters at all.

    Two *different* garbage values, because a single one could coincide with a
    no-op: if the fit sees the free set, two different corruptions give two different
    answers. ``torch.equal``, not ``allclose`` -- the free reflections must contribute
    exactly nothing, not merely little.

    This fit used to mask with ``ReflectionData.masks()``, which is validity only
    (``TensorMasks.__call__`` ANDs the validity masks and has no work/free notion), so
    the free reflections went into the scale parameters via 10 x LBFGS(max_iter=100).
    """
    dc = pair
    free = dc["other"].free.mask
    assert free.sum() > 0, "fixture has no free reflections; the test would be vacuous"

    results = []
    for filler in (3.0, 900.0):
        d = dc["other"]
        # Reset the parameters so each arm starts from the same place.
        with torch.no_grad():
            d.log_scale.zero_()
            d.U_aniso.zero_()
            d.F[free] = filler
            d.F_sigma[free] = filler
            d._corrected_fp = None       # drop the cached corrected view
            d._corrected_cache = None
        dc.scale()
        results.append(_fitted(dc))

    (ls_a, u_a), (ls_b, u_b) = results
    assert torch.equal(ls_a, ls_b), (
        f"log_scale moved when only the FREE reflections changed: {ls_a} vs {ls_b}"
    )
    assert torch.equal(u_a, u_b), (
        f"U_aniso moved when only the FREE reflections changed: {u_a} vs {u_b}"
    )


@pytest.mark.integration
def test_identical_datasets_fit_a_unit_scale(pair):
    """Two copies of one dataset must scale onto each other with no correction.

    The sanity check the objectives have to pass before any comparison between them
    means anything.
    """
    dc = pair
    dc.scale()
    log_scale, U = _fitted(dc)
    assert float(log_scale.abs().max()) < 1e-3, log_scale
    assert float(U.abs().max()) < 1e-3, U


@pytest.mark.integration
def test_a_known_scale_is_recovered(pair):
    """Scale one dataset by a known factor and check the fit undoes it."""
    dc = pair
    k = 2.5
    with torch.no_grad():
        d = dc["other"]
        d.F *= k
        d.F_sigma *= k
        d._corrected_fp = None
        d._corrected_cache = None
    dc.scale()
    log_scale, _ = _fitted(dc)
    # log_scale multiplies the observations, so recovering 1/k means log_scale = -log(k).
    import math
    assert float(log_scale.reshape(-1)[0]) == pytest.approx(-math.log(k), abs=0.02)


@pytest.mark.unit
def test_the_objective_is_not_selectable():
    """No objective parameter, and specifically no sigma-weighted one.

    Two separate reasons, both worth keeping written down. There is no model in this fit,
    so a sigma_A or Rice likelihood has no model error to account for. And
    inverse-variance weighting *collapses* on a scale fit -- down-weighting the weak
    shells is exactly what lets the scale run away in them -- which is why the
    model-to-data fit's default came back to unit-weight ``ls`` as well. A sigma-weighted
    variant was built and measured here: it scored slightly better on held-out
    reflections for one dataset pair, and was still removed, because a small gain on one
    pair does not outweigh a failure mode found across a panel.
    """
    import inspect

    from torchref.io.datasets.collection import DatasetCollection

    assert "objective" not in inspect.signature(DatasetCollection.scale).parameters
    src = inspect.getsource(DatasetCollection.scale)
    assert "sigma" in src, "the reason sigma weighting is absent must stay documented"


@pytest.mark.integration
def test_the_objective_is_normalised(pair):
    """The loss handed to L-BFGS must be O(1), because its tolerances are absolute.

    Not a style point: ``tolerance_grad``/``tolerance_change`` are absolute, so an
    objective carrying the data's own magnitude (~1e9 on a large work set under unit
    weights) puts the float32 ulp of the loss above the decrease the line search is
    trying to resolve. Probed by scaling the data by 1e3 and checking the fit still
    recovers the same answer.
    """
    import math

    dc = pair
    with torch.no_grad():
        d = dc["other"]
        d.F *= 1000.0
        d.F_sigma *= 1000.0
        d._corrected_fp = None
        d._corrected_cache = None
    dc.scale()
    log_scale, _ = _fitted(dc)
    assert float(log_scale.reshape(-1)[0]) == pytest.approx(-math.log(1000.0), abs=0.05)
