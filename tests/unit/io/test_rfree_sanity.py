"""The sanity check ``ReflectionData.load`` applies to a file's R-free flags.

A deposited free set is judged by its free/work ratio over the measured
reflections, whichever reader delivered it, and honoured whenever it can be
used: only a column without a single work reflection is replaced.
"""

import warnings

import numpy as np
import pytest
import torch

from torchref.io.datasets.reflection_data import ReflectionData
from torchref.io.mtz import MTZReader


@pytest.fixture(scope="module")
def daw(mtz_dir):
    """1DAW as ``MTZReader`` delivers it: intensities, 5 % free, all measured."""
    return MTZReader(verbose=0).read(str(mtz_dir / "1DAW.mtz"))()


class _Reader:
    """Hands ``load`` a copy of 1DAW with some entries replaced."""

    def __init__(self, daw, **replace):
        data, self.cell, self.spacegroup = daw
        self.data = {**data, **replace}

    def __call__(self):
        return dict(self.data), self.cell, self.spacegroup


def _load(reader):
    """Load ``reader``, returning the dataset and the R-free warnings raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = ReflectionData(verbose=0, device="cpu").load(reader)
    return data, [str(w.message) for w in caught if "R-free flags" in str(w.message)]


@pytest.mark.unit
def test_a_deposited_free_set_loads_without_a_warning(daw):
    data, messages = _load(_Reader(daw))

    assert messages == []
    assert data.rfree_source == "_Reader FreeR"


@pytest.mark.unit
def test_a_column_without_work_reflections_is_replaced(daw):
    flags = np.zeros_like(daw[0]["R-free-flags"])

    data, messages = _load(_Reader(daw, **{"R-free-flags": flags}))

    assert len(messages) == 1 and "no measured reflection as work" in messages[0]
    assert data.rfree_source != "_Reader FreeR"
    free = float((~data.rfree_flags.to(torch.bool)).float().mean())
    assert 0.0 < free < 0.1


@pytest.mark.unit
def test_a_column_without_free_reflections_is_kept(daw):
    flags = np.ones_like(daw[0]["R-free-flags"])

    data, messages = _load(_Reader(daw, **{"R-free-flags": flags}))

    assert len(messages) == 1 and "no measured reflection as free" in messages[0]
    assert data.rfree_source == "_Reader FreeR"
    assert bool(data.rfree_flags.all())


@pytest.mark.unit
def test_a_free_set_above_a_quarter_of_the_work_set_is_kept(daw):
    flags = (np.arange(len(daw[0]["HKL"])) % 3 != 0).astype(np.int32)

    data, messages = _load(_Reader(daw, **{"R-free-flags": flags}))

    assert len(messages) == 1 and "more than a quarter" in messages[0]
    assert data.rfree_source == "_Reader FreeR"
    free = float((~data.rfree_flags.to(torch.bool)).float().mean())
    assert free == pytest.approx(1.0 / 3.0, abs=0.01)


@pytest.mark.unit
def test_only_measured_reflections_with_a_flag_count(daw):
    """Unmeasured and excluded rows would make the free set look oversized."""
    data, _, _ = daw
    n = len(data["HKL"])
    unmeasured = np.arange(n) % 5 == 1
    excluded = np.arange(n) % 5 == 2
    flags = data["R-free-flags"].copy()
    flags[unmeasured] = 0
    flags[excluded] = -1
    I = np.where(unmeasured, np.nan, data["I"]).astype(np.float32)
    sigma_I = np.where(unmeasured, np.nan, data["SIGI"]).astype(np.float32)

    loaded, messages = _load(
        _Reader(daw, **{"R-free-flags": flags, "I": I, "SIGI": sigma_I})
    )

    assert messages == []
    assert loaded.rfree_source == "_Reader FreeR"
