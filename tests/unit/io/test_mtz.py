"""MTZReader column selection and free-set interpretation, on 1DAW.mtz."""

import numpy as np
import pytest
import reciprocalspaceship as rs

from torchref.io.mtz import MTZReader


def _rewrite(mtz_dir, tmp_path, edit):
    """Write a copy of 1DAW.mtz after ``edit(ds)`` and return its path."""
    ds = rs.read_mtz(str(mtz_dir / "1DAW.mtz"))
    ds = edit(ds)
    path = tmp_path / "edited.mtz"
    ds.write_mtz(str(path))
    return str(path)


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
