"""The CIF readers on deposited SF-mmCIF, mmCIF and monomer files, often edited."""

import numpy as np
import pandas as pd
import pytest

from torchref.io import ReflectionData
from torchref.io.cif_readers import (
    CIFReader,
    ModelCIFReader,
    ReflectionCIFReader,
    RestraintCIFReader,
)
from torchref.topology.monomer.library import get_library_manager


def _write_edited(source, tmp_path, edit):
    """Write ``source`` with its lines passed through ``edit``; return the new path."""
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


def _first_anisotropic_atom_as_pairs(lines):
    """An ``edit`` that cuts the ``_atom_site_anisotrop`` loop to its first row,
    written as key-value pairs, as wwPDB writes a category with one row."""
    start = next(
        i for i, line in enumerate(lines) if line.strip() == "_atom_site_anisotrop.id"
    )
    end = start
    while lines[end].strip().startswith("_atom_site_anisotrop."):
        end += 1
    tags = [line.strip() for line in lines[start:end]]
    pairs = [f"{tag} {value}" for tag, value in zip(tags, lines[end].split())]
    after = next(i for i in range(end, len(lines)) if lines[i].startswith("#"))
    return lines[: start - 1] + pairs + lines[after:]


def _rewrite_status(value, tag="_refln.status"):
    """An ``edit`` of 1DAW-sf.cif that renames the status column to ``tag`` and
    gives reflection ``row`` the token ``value(row, status)``."""

    def edit(lines):
        edited, row = [], 0
        for line in lines:
            tokens = line.split()
            if line.strip() == "_refln.status":
                line = tag
            elif len(tokens) == 11 and tokens[6] in ("o", "f"):
                tokens[6] = str(value(row, tokens[6]))
                line = " ".join(tokens)
                row += 1
            edited.append(line)
        return edited

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


@pytest.mark.unit
def test_a_second_load_replaces_the_first_file(cif_sf_dir):
    second = str(cif_sf_dir / "3GR5-sf.cif")
    reader = CIFReader(str(cif_sf_dir / "1DAW-sf.cif"))
    reader.load(second)
    fresh = CIFReader(second)

    assert reader.available_blocks == fresh.available_blocks == ["r3gr5sf"]
    assert reader.data_block == "r3gr5sf"
    assert reader.keys() == fresh.keys()
    pd.testing.assert_frame_equal(reader["refln"], fresh["refln"])


@pytest.mark.unit
def test_one_anisotropic_atom_written_as_pairs_is_read(cif_dir, tmp_path):
    source = cif_dir / "4BX9.cif"
    path = _write_edited(source, tmp_path, _first_anisotropic_atom_as_pairs)
    u = ["u11", "u22", "u33", "u12", "u13", "u23"]

    original = ModelCIFReader(str(source)).dataframe
    table = ModelCIFReader(path).dataframe

    assert table["anisou_flag"].sum() == 1 and table.loc[0, "anisou_flag"]
    pd.testing.assert_series_equal(table.loc[0, u], original.loc[0, u])


@pytest.mark.unit
def test_space_group_is_read_from_the_space_group_category(
    cif_dir, cif_sf_dir, tmp_path
):
    def rename(lines):
        old, new = "_symmetry.space_group_name_H-M", "_space_group.name_H-M_alt"
        return [line.replace(old, new) for line in lines]

    reflections = _write_edited(cif_sf_dir / "1DAW-sf.cif", tmp_path, rename)
    atoms = _write_edited(cif_dir / "1DAW.cif", tmp_path, rename)

    assert ReflectionCIFReader(reflections).spacegroup == "C 1 2 1"
    assert ModelCIFReader(atoms).spacegroup == "C 1 2 1"


@pytest.mark.unit
def test_a_cell_without_all_three_lengths_is_not_read(cif_dir, cif_sf_dir, tmp_path):
    def drop_b(lines):
        return [line for line in lines if not line.startswith("_cell.length_b")]

    reflections = _write_edited(cif_sf_dir / "1DAW-sf.cif", tmp_path, drop_b)
    atoms = _write_edited(cif_dir / "1DAW.cif", tmp_path, drop_b)

    with pytest.raises(ValueError, match="Unit cell parameters not found"):
        ReflectionCIFReader(reflections)
    assert ModelCIFReader(atoms).cell is None


@pytest.mark.unit
def test_provenance_keys_are_tag_names(cif_sf_dir):
    path = str(cif_sf_dir / "3GR5-sf.cif")
    data = ReflectionCIFReader(path).data

    assert data["HKL_key"] == "_refln.index_h,_refln.index_k,_refln.index_l"
    assert data["F_col"] == "_refln.F_meas_au"
    assert data["SIGF_col"] == "_refln.F_meas_sigma_au"
    assert data["R-free-source"] == "_refln.status"
    assert "F=_refln.F_meas_au" in repr(ReflectionData(verbose=0).load_cif(path))


@pytest.mark.unit
@pytest.mark.parametrize(
    "free, work",
    [(0, lambda row: 1 + row % 19), (1, lambda row: 0)],
    ids=["ccp4-0-free", "phenix-1-free"],
)
def test_numeric_free_flags_give_the_status_split(cif_sf_dir, tmp_path, free, work):
    source = cif_sf_dir / "1DAW-sf.cif"
    path = _write_edited(
        source,
        tmp_path,
        _rewrite_status(
            lambda row, status: free if status == "f" else work(row),
            tag="_refln.pdbx_r_free_flag",
        ),
    )

    from_status = ReflectionCIFReader(str(source)).data["R-free-flags"]
    data = ReflectionCIFReader(path).data

    np.testing.assert_array_equal(data["R-free-flags"], from_status)
    assert data["R-free-source"] == "_refln.pdbx_r_free_flag"


@pytest.mark.unit
def test_status_letters_other_than_o_and_f_exclude_their_rows(cif_sf_dir, tmp_path):
    letters = ["x", "<", "-", "h", "l"]
    path = _write_edited(
        cif_sf_dir / "1DAW-sf.cif",
        tmp_path,
        _rewrite_status(lambda row, status: letters[row] if row < 5 else status),
    )

    flags = ReflectionCIFReader(path).data["R-free-flags"]

    assert flags[:5].tolist() == [-1] * 5
    assert set(flags[5:].tolist()) == {0, 1}


@pytest.mark.unit
@pytest.mark.parametrize("status, flag", [("f", 0), ("o", 1)])
def test_a_one_sided_free_set_is_passed_on(cif_sf_dir, tmp_path, status, flag):
    path = _write_edited(
        cif_sf_dir / "1DAW-sf.cif", tmp_path, _rewrite_status(lambda row, _: status)
    )

    assert set(ReflectionCIFReader(path).data["R-free-flags"].tolist()) == {flag}


@pytest.mark.unit
def test_an_empty_quoted_value_keeps_its_column():
    """``''`` in a loop row is a value: the columns after it stay in place."""
    from torchref.topology.monomer.cif import read_component_groups, read_library_blocks

    comps = CIFReader.from_string(read_library_blocks()["comp_list"])["chem_comp"]

    assert comps.set_index("_chem_comp.id").loc["01G", "_chem_comp.name"] == ""
    assert dict(zip(comps["_chem_comp.id"], comps["_chem_comp.group"])) == (
        read_component_groups()
    )


@pytest.mark.unit
def test_a_quote_inside_a_value_does_not_open_a_string():
    """A quote opens a string only at the start of a value and closes it only before
    whitespace, so O5' and 'it's' are one value each, as gemmi reads them."""
    import gemmi

    content = "\n".join(
        [
            "data_t",
            "loop_",
            "_atom_site.id",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "1 O5' ANP",
            '2 "O5\'" ANP',
            "3 'it's' ANP",
            "4 '' ANP",
            "5 C5' 'A B'",
        ]
    )
    columns = ["id", "label_atom_id", "label_comp_id"]
    block = gemmi.cif.read_string(content).sole_block()
    expected = [
        [gemmi.cif.as_string(value) for value in row]
        for row in block.find("_atom_site.", columns)
    ]

    assert CIFReader.from_string(content)["atom_site"].values.tolist() == expected
    assert expected[0] == ["1", "O5'", "ANP"] and expected[2][1] == "it's"


@pytest.mark.unit
def test_torsions_keep_their_id():
    path = get_library_manager(verbose=0).get_cif_file("DA")
    torsions = RestraintCIFReader(str(path)).get_all_restraints()["DA"]["torsions"]

    puckers = {f"{form}-nyu{i}" for form in ("C2e", "C3e") for i in range(5)}
    assert puckers <= set(torsions["id"])


@pytest.mark.unit
def test_plane_atoms_without_an_esd_get_the_link_plane_default(tmp_path):
    """A component plane row with no ``dist_esd`` gets 0.02 Å, as link planes do."""

    def drop_plane_esds(lines):
        return [
            line.rsplit(maxsplit=1)[0] if line.startswith("PHE plan-") else line
            for line in lines
            if line.strip() != "_chem_comp_plane_atom.dist_esd"
        ]

    source = get_library_manager(verbose=0).get_cif_file("PHE")
    path = _write_edited(source, tmp_path, drop_plane_esds)
    planes = RestraintCIFReader(path).get_all_restraints()["PHE"]["planes"]

    assert len(planes) > 0
    np.testing.assert_allclose(planes["sigma"].to_numpy(dtype=float), 0.02)


@pytest.mark.unit
def test_atoms_without_a_type_symbol_raise(cif_dir, tmp_path):
    def unknown_elements(lines):
        edited = []
        for line in lines:
            tokens = line.split()
            if tokens[:1] in (["ATOM"], ["HETATM"]) and len(tokens) == 21:
                tokens[2] = "?"
                line = " ".join(tokens)
            edited.append(line)
        return edited

    path = _write_edited(cif_dir / "1DAW.cif", tmp_path, unknown_elements)
    with pytest.raises(ValueError, match="no element in _atom_site.type_symbol"):
        ModelCIFReader(path)
