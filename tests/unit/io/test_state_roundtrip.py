"""Dataset state round-trips, and surviving a field that no longer exists.

``_from_state`` splats the saved state straight into the constructor, so removing a
dataclass field would otherwise make every checkpoint written before the removal
permanently unloadable with a bare ``TypeError``. ``outlier_flags`` was the first field
this actually happened to (removed with outlier rejection); the guard is generic so the
next removal is not a second incident.
"""

import warnings

import pytest
import torch

from torchref.io import ReflectionData
from torchref.io.datasets.fcalc_data import FcalcDataset


def _minimal_state(cls, n=8):
    """A `_get_state`-shaped dict for `cls`, on CPU."""
    obj = cls(
        hkl=torch.randint(-4, 5, (n, 3), dtype=torch.int32),
        device=torch.device("cpu"),
        verbose=0,
    )
    if hasattr(obj, "F"):
        obj.F = torch.rand(n)
        obj.F_sigma = torch.rand(n) + 0.1
    state = obj._get_state()
    state["device"] = "cpu"
    return state


@pytest.mark.unit
@pytest.mark.parametrize("cls", [ReflectionData, FcalcDataset])
def test_state_roundtrip(cls):
    """The baseline: a state dict this version wrote loads back unchanged in shape."""
    state = _minimal_state(cls)
    n = state["hkl"].shape[0]
    obj = cls._from_state(dict(state), device="cpu")
    assert isinstance(obj, cls)
    assert len(obj.hkl) == n


@pytest.mark.unit
@pytest.mark.parametrize("cls", [ReflectionData, FcalcDataset])
def test_state_with_a_removed_field_still_loads(cls):
    """A key that is no longer a dataclass field is dropped, loudly, not fatally.

    ``outlier_flags`` is the real case: checkpoints written before outlier rejection was
    removed carry it. Uses a deliberately invented name too, so the test keeps testing the
    mechanism rather than one historical field.
    """
    state = _minimal_state(cls)
    n = state["hkl"].shape[0]
    state["outlier_flags"] = torch.zeros(n, dtype=torch.bool)
    state["a_field_that_never_existed"] = 42

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = cls._from_state(dict(state), device="cpu")

    assert isinstance(obj, cls)
    assert len(obj.hkl) == n
    assert not hasattr(obj, "outlier_flags")

    msgs = [str(w.message) for w in caught if "no longer fields" in str(w.message)]
    assert msgs, "dropping a removed field must warn, not happen silently"
    assert "outlier_flags" in msgs[0] and "a_field_that_never_existed" in msgs[0]


@pytest.mark.unit
def test_masks_survive_a_removed_field():
    """The stale-key filter must not eat ``masks``, which is not a dataclass field and is
    popped separately -- the one key that is legitimately absent from ``fields()``."""
    state = _minimal_state(ReflectionData)
    n = state["hkl"].shape[0]
    keep = torch.ones(n, dtype=torch.bool)
    keep[0] = False
    state["masks"] = {"probe": keep}
    state["outlier_flags"] = torch.zeros(n, dtype=torch.bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = ReflectionData._from_state(dict(state), device="cpu")

    assert "probe" in obj.masks
    assert bool(obj.masks()[0]) is False
