"""The CIF readers on edited copies of deposited SF-mmCIF and mmCIF files."""

import numpy as np
import pytest

from torchref.io.cif_readers import ModelCIFReader, ReflectionCIFReader


def _write_edited(source, tmp_path, edit):
    """Write ``source`` with its list of lines passed through ``edit``; return the path."""
    lines = edit(source.read_text().splitlines())
    path = tmp_path / source.name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _blank_line_after_row(row, first_tag):
    """An ``edit`` that puts a blank line after data row ``row`` of a loop."""

    def edit(lines):
        start = next(i for i, line in enumerate(lines) if line.strip() == first_tag)
        while lines[start].lstrip().startswith("_"):
            start += 1
        return lines[: start + row] + [""] + lines[start + row :]

    return edit


@pytest.mark.unit
def test_f_squared_columns_load_as_intensities(cif_sf_dir, tmp_path):
    source = cif_sf_dir / "1DAW-sf.cif"
    renamed = {
        "_refln.F_meas_au": "_refln.F_unrecognised",
        "_refln.F_meas_sigma_au": "_refln.F_unrecognised_sigma",
        "_refln.intensity_meas": "_refln.F_squared_meas",
        "_refln.intensity_sigma": "_refln.F_squared_sigma",
    }
    path = _write_edited(
        source,
        tmp_path,
        lambda lines: [renamed.get(line.strip(), line) for line in lines],
    )

    original = ReflectionCIFReader(str(source)).data
    data = ReflectionCIFReader(path).data

    assert "F" not in data and "SIGF" not in data
    np.testing.assert_array_equal(data["I"], original["I"])
    np.testing.assert_array_equal(data["SIGI"], original["SIGI"])


@pytest.mark.unit
def test_blank_line_inside_a_loop_keeps_the_rows_after_it(
    cif_dir, cif_sf_dir, tmp_path
):
    reflections = _write_edited(
        cif_sf_dir / "1DAW-sf.cif",
        tmp_path,
        _blank_line_after_row(100, "_refln.crystal_id"),
    )
    atoms = _write_edited(
        cif_dir / "1DAW.cif", tmp_path, _blank_line_after_row(100, "_atom_site.id")
    )

    assert len(ReflectionCIFReader(reflections).data["HKL"]) == 23356
    assert len(ModelCIFReader(atoms).dataframe) == 3051
