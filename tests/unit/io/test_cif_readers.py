"""The CIF readers on edited copies of deposited SF-mmCIF and mmCIF files."""

import numpy as np
import pytest

from torchref.io.cif_readers import ReflectionCIFReader


def _write_edited(source, tmp_path, edit):
    """Write ``source`` with ``edit`` applied to each line; return the new path."""
    lines = source.read_text().splitlines()
    path = tmp_path / source.name
    path.write_text("\n".join(edit(line) for line in lines) + "\n")
    return str(path)


@pytest.mark.unit
def test_f_squared_columns_load_as_intensities(cif_sf_dir, tmp_path):
    source = cif_sf_dir / "1DAW-sf.cif"
    renamed = {
        "_refln.F_meas_au": "_refln.F_unrecognised",
        "_refln.F_meas_sigma_au": "_refln.F_unrecognised_sigma",
        "_refln.intensity_meas": "_refln.F_squared_meas",
        "_refln.intensity_sigma": "_refln.F_squared_sigma",
    }
    path = _write_edited(source, tmp_path, lambda line: renamed.get(line.strip(), line))

    original = ReflectionCIFReader(str(source)).data
    data = ReflectionCIFReader(path).data

    assert "F" not in data and "SIGF" not in data
    np.testing.assert_array_equal(data["I"], original["I"])
    np.testing.assert_array_equal(data["SIGI"], original["SIGI"])
