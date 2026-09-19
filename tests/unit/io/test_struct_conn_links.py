"""``_struct_conn`` rows become the LINK-record table the PDB reader produces."""

import numpy as np
import pytest

from torchref.io.cif_readers import ModelCIFReader
from torchref.io.pdb import LINK_COLUMNS


@pytest.mark.unit
def test_3e98_peptide_links_to_selenomethionine(cif_dir):
    links = ModelCIFReader(str(cif_dir / "3E98.cif")).links
    assert list(links.columns) == list(LINK_COLUMNS)
    assert len(links) == 8
    assert set(links["name1"]) == {"C"} and set(links["name2"]) == {"N"}
    assert set(links["resname2"]) <= {"MSE", "ARG", "ASP"}
    assert (links["altloc1"] == "").all() and (links["icode1"] == "").all()
    assert links["resseq1"].dtype.kind == "i"
    assert np.isfinite(links["length"]).all()


@pytest.mark.unit
def test_1daw_metal_contacts_are_kept(cif_dir):
    links = ModelCIFReader(str(cif_dir / "1DAW.cif")).links
    assert len(links) == 14
    magnesium = links[(links["resname2"] == "MG") | (links["resname1"] == "MG")]
    assert len(magnesium) > 0
    pairs = set(zip(links["name1"], links["resname1"], links["resseq1"]))
    assert ("OD2", "ASP", 175) in pairs


@pytest.mark.unit
def test_disulfides_are_left_to_distance_detection(cif_dir):
    links = ModelCIFReader(str(cif_dir / "3A5V.cif")).links
    assert len(links) == 12
    assert "SG" not in set(links["name1"]) | set(links["name2"])


@pytest.mark.unit
def test_file_without_struct_conn_gives_empty_table(tmp_path):
    minimal = """\
data_test
_cell.length_a 10.0
_cell.length_b 10.0
_cell.length_c 10.0
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
_symmetry.space_group_name_H-M 'P 1'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
ATOM 1 N N . ALA A 1 ? 0.0 0.0 0.0 1.0 20.0 1 A
ATOM 2 C CA . ALA A 1 ? 1.5 0.0 0.0 1.0 20.0 1 A
"""
    path = tmp_path / "no_links.cif"
    path.write_text(minimal)
    links = ModelCIFReader(str(path)).links
    assert len(links) == 0
    assert list(links.columns) == list(LINK_COLUMNS)
