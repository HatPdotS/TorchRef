"""The collection X-ray taxonomy, as contracts.

Mirrors ``tests/unit/refinement/test_nll_beta.py``'s registry tests for the multi-dataset
table. Same thesis, same invariants: one class per row, the observable declared rather than
passed, and no row that pairs an amplitude distribution with intensities.
"""

import inspect

import pytest


@pytest.mark.unit
def test_each_row_has_its_own_class():
    """One class per selectable row, and the test and the table must agree on the set."""
    from torchref.refinement.targets.collection import COLLECTION_XRAY_TARGETS
    from torchref.refinement.targets.collection.intensity import (
        CollectionTwoMomentIntensityTarget,
    )
    from torchref.refinement.targets.collection.xray import (
        CollectionDifferenceIntensityTarget,
        CollectionDifferenceTarget,
        CollectionMLTarget,
    )

    expected = {
        "difference": CollectionDifferenceTarget,
        "difference_i": CollectionDifferenceIntensityTarget,
        "two_moment": CollectionTwoMomentIntensityTarget,
        "ml": CollectionMLTarget,
    }
    assert set(expected) == set(COLLECTION_XRAY_TARGETS.names), (
        "table and test disagree on the rows"
    )
    for name, cls in expected.items():
        assert COLLECTION_XRAY_TARGETS.by_name(name).target_cls is cls, name

    seen = {}
    for name in COLLECTION_XRAY_TARGETS.names:
        cls = COLLECTION_XRAY_TARGETS.by_name(name).target_cls
        assert cls not in seen, f"{name} and {seen[cls]} share {cls.__name__}"
        seen[cls] = name


@pytest.mark.unit
def test_the_observable_is_declared_not_passed():
    """The spec's claim and the class's own attribute must agree.

    A row advertising intensities while reading amplitudes would be wrong by ``2|F|``,
    which is resolution-dependent -- so it reads as a scale or B error rather than as a
    bug, and nothing downstream would flag it.
    """
    from torchref.refinement.targets.collection import (
        COLLECTION_XRAY_TARGETS,
        CollectionXrayTargetSpec,
    )
    from torchref.refinement.targets.collection.xray import CollectionDifferenceTarget

    by_obs = {}
    for spec in COLLECTION_XRAY_TARGETS.specs:
        assert spec.observable in ("amplitude", "intensity"), spec.name
        assert getattr(spec.target_cls, "observable", "amplitude") == spec.observable
        by_obs.setdefault(spec.observable, []).append(spec.name)
        assert "observable" not in inspect.signature(
            spec.target_cls.__init__
        ).parameters

    assert set(by_obs["intensity"]) == {"difference_i", "two_moment"}
    assert set(by_obs["amplitude"]) == {"difference", "ml"}

    with pytest.raises(ValueError, match="observable"):
        CollectionXrayTargetSpec(
            name="bogus",
            target_cls=CollectionDifferenceTarget,
            doc="",
            observable="intensity",
        )


@pytest.mark.unit
def test_both_difference_observables_are_offered():
    """Neither difference row is privileged.

    Which one is better is a property of a dataset's signal-to-noise: amplitudes keep the
    loss in the same space as the output DED coefficients, intensities avoid the
    French-Wilson posterior reshaping the weak tail. The abstraction exists so that
    carrying both is cheap -- and the intensity row proves it, being nothing but an
    ``observable`` declaration over the amplitude one.
    """
    from torchref.refinement.targets.collection import COLLECTION_XRAY_TARGETS
    from torchref.refinement.targets.collection.xray import (
        CollectionDifferenceIntensityTarget,
        CollectionDifferenceTarget,
    )

    assert {"difference", "difference_i"} <= set(COLLECTION_XRAY_TARGETS.names)
    assert issubclass(CollectionDifferenceIntensityTarget, CollectionDifferenceTarget)
    # The subclass adds no likelihood of its own: only the name and the observable.
    own = set(vars(CollectionDifferenceIntensityTarget)) - {
        # `__annotations__` is present because `name`/`observable` are annotated
        # assignments, not because the class defines behaviour.
        "__doc__", "__module__", "__qualname__", "__annotations__",
        "name", "observable",
    }
    assert not own, f"the intensity difference row grew a body: {sorted(own)}"


@pytest.mark.unit
def test_there_is_no_intensity_rice_row():
    """Rice is amplitude-only by nature, so the axis is not square.

    Rice and the folded normal are distributions *of an amplitude*; the intensity analogue
    is the exponential / chi-square_1 Wilson distribution, a different primitive rather
    than a different variance. A row pairing the ML class with intensities would be a
    modelling error, not a new feature.
    """
    from torchref.refinement.targets.collection import COLLECTION_XRAY_TARGETS
    from torchref.refinement.targets.collection.xray import CollectionMLTarget

    for spec in COLLECTION_XRAY_TARGETS.specs:
        if spec.observable == "intensity":
            assert not issubclass(spec.target_cls, CollectionMLTarget), spec.name


@pytest.mark.unit
def test_unknown_rows_fail_closed():
    from torchref.refinement.targets.collection import COLLECTION_XRAY_TARGETS

    with pytest.raises(ValueError, match="Unknown collection X-ray target"):
        COLLECTION_XRAY_TARGETS.by_name("no_such_row")


@pytest.mark.unit
def test_every_row_goes_through_the_seam():
    """No row may hand-write a ``forward``.

    That is what this refactor bought: 265 lines of per-row forwards, each re-deriving
    stacking, masking, the sigma floor and the summing, collapsed into one. A row that
    reintroduces its own ``forward`` also reintroduces the possibility of it disagreeing
    with ``residuals``, which nothing else would notice.
    """
    from torchref.refinement.targets.collection import COLLECTION_XRAY_TARGETS
    from torchref.refinement.targets.collection.base import CollectionXrayTarget

    for spec in COLLECTION_XRAY_TARGETS.specs:
        cls = spec.target_cls
        assert cls.forward is CollectionXrayTarget.forward, (
            f"{spec.name} overrides forward; the likelihood belongs in _per_refl"
        )
        assert cls._per_refl is not CollectionXrayTarget._per_refl, (
            f"{spec.name} has no _per_refl of its own"
        )
