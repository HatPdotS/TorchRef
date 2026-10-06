"""MTZReader column selection and free-set interpretation, on 1DAW.mtz."""

import numpy as np
import pytest
import reciprocalspaceship as rs

from torchref.io.datasets.reflection_data import ReflectionData
from torchref.io.mtz import MTZReader


def _rewrite(mtz_dir, tmp_path, edit):
    """Write a copy of 1DAW.mtz after ``edit(ds)`` and return its path."""
    ds = rs.read_mtz(str(mtz_dir / "1DAW.mtz"))
    ds = edit(ds)
    path = tmp_path / "edited.mtz"
    ds.write_mtz(str(path))
    return str(path)


def _load(path, column_names=None):
    data = ReflectionData(verbose=0)
    data.load_mtz(path, column_names=column_names)
    return data


@pytest.mark.unit
def test_free_value_is_the_minority_among_valid_rows(mtz_dir, tmp_path):
    """A Phenix column (1 = free) keeps its free set when most rows are excluded.

    Zeros are the work set here although they are fewer than half of all rows:
    the excluded rows do not vote, as in rfree.read_free_set.
    """
    phenix = {}

    def edit(ds):
        flags = np.where(ds["FreeR_flag"].to_numpy() == 0, 1, 0)
        flags[np.arange(len(flags)) % 5 < 3] = -1
        phenix["flags"] = flags
        ds["FreeR_flag"] = rs.DataSeries(flags, index=ds.index).astype("I")
        return ds

    data, _, _ = MTZReader().read(_rewrite(mtz_dir, tmp_path, edit))()
    flags = phenix["flags"]
    expected = np.where(flags < 0, -1, np.where(flags == 1, 0, 1))
    np.testing.assert_array_equal(data["R-free-flags"], expected)


@pytest.mark.unit
def test_amplitude_pin_loads_amplitudes_beside_intensities(mtz_dir):
    """Pinning F turns off the intensity search, so French-Wilson does not run."""
    data = _load(str(mtz_dir / "1DAW.mtz"), {"F": "FP", "SIGF": "SIGFP"})
    assert data.amplitude_source == "FP"
    assert data.intensity_source is None


@pytest.mark.unit
def test_pinned_intensity_column_is_read_as_intensities(mtz_dir, tmp_path):
    """A column pinned under "F" whose MTZ type is J is an intensity pin."""
    path = _rewrite(
        mtz_dir,
        tmp_path,
        lambda ds: ds.rename(columns={"I": "I_light", "SIGI": "SIGI_light"}),
    )
    data = _load(path, {"F": "I_light", "SIGF": "SIGI_light"})
    assert data.intensity_source == "I_light"
    assert data.amplitude_source is None


@pytest.mark.unit
def test_amplitude_pin_wins_over_stacked_intensities(mtz_dir, tmp_path):
    """Stacking's choice of I(+)/I(-) yields to a pinned amplitude column."""
    path = _rewrite(
        mtz_dir,
        tmp_path,
        lambda ds: ds.stack_anomalous().unstack_anomalous(["I", "SIGI"]),
    )
    stacked = _load(path)
    assert not stacked.friedel_merged
    assert stacked.intensity_source == "I"

    pinned = _load(path, {"F": "FP", "SIGF": "SIGFP"})
    assert pinned.amplitude_source == "FP"
    assert pinned.intensity_source is None
