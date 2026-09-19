"""Collection target registry classes, observables and invalid selections."""

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
        CollectionDifferenceSigmaDTarget,
        CollectionDifferenceTarget,
        CollectionMLTarget,
    )

    expected = {
        "difference": CollectionDifferenceTarget,
        "difference_i": CollectionDifferenceIntensityTarget,
        "difference_sd": CollectionDifferenceSigmaDTarget,
        "two_moment": CollectionTwoMomentIntensityTarget,
        "ml": CollectionMLTarget,
    }
    assert set(expected) == set(
        COLLECTION_XRAY_TARGETS.names
    ), "table and test disagree on the rows"
    for name, cls in expected.items():
        assert COLLECTION_XRAY_TARGETS.by_name(name).target_cls is cls, name

    seen = {}
    for name in COLLECTION_XRAY_TARGETS.names:
        cls = COLLECTION_XRAY_TARGETS.by_name(name).target_cls
        assert cls not in seen, f"{name} and {seen[cls]} share {cls.__name__}"
        seen[cls] = name


@pytest.mark.unit
def test_the_observable_is_declared_not_passed():
    """The spec's claim and the class's own attribute must agree."""
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
        assert (
            "observable" not in inspect.signature(spec.target_cls.__init__).parameters
        )

    assert set(by_obs["intensity"]) == {"difference_i", "two_moment"}
    assert set(by_obs["amplitude"]) == {"difference", "difference_sd", "ml"}

    with pytest.raises(ValueError, match="observable"):
        CollectionXrayTargetSpec(
            name="bogus",
            target_cls=CollectionDifferenceTarget,
            doc="",
            observable="intensity",
        )


@pytest.mark.unit
def test_there_is_no_intensity_rice_row():
    """Rice is amplitude-only by nature, so the axis is not square."""
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
