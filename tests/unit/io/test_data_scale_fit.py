"""``DatasetCollection.scale()`` -- the data-to-data scale fit.

The only fit in the library with no model on either side: it puts one dataset onto
another, both of them measurements of the same quantity. That is why its objectives are
least squares and there is no sigma_A row -- there is no model error to account for.

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
@pytest.mark.parametrize("objective", ["ls", "ls_sigma"])
def test_scale_never_touches_the_free_set(pair, objective):
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
        dc.scale(objective=objective)
        results.append(_fitted(dc))

    (ls_a, u_a), (ls_b, u_b) = results
    assert torch.equal(ls_a, ls_b), (
        f"log_scale moved when only the FREE reflections changed: {ls_a} vs {ls_b}"
    )
    assert torch.equal(u_a, u_b), (
        f"U_aniso moved when only the FREE reflections changed: {u_a} vs {u_b}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("objective", ["ls", "ls_sigma"])
def test_identical_datasets_fit_a_unit_scale(pair, objective):
    """Two copies of one dataset must scale onto each other with no correction.

    The sanity check the objectives have to pass before any comparison between them
    means anything.
    """
    dc = pair
    dc.scale(objective=objective)
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
def test_unknown_objective_fails_closed():
    from torchref.io.datasets.collection import (
        DATA_SCALE_OBJECTIVES,
        DatasetCollection,
    )

    dc = DatasetCollection(verbose=0, device="cpu")
    with pytest.raises(ValueError, match="objective must be one of"):
        dc.scale(objective="ml")
    # No sigma_A / Rice row is offered, and that is a modelling statement: this fit has
    # no model, so there is no model error for such a likelihood to account for.
    assert DATA_SCALE_OBJECTIVES == ("ls", "ls_sigma")


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
