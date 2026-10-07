"""
Tests for the top-level object-creation readers (read_mtz / read_cif / read_pdb).
"""
import pytest


@pytest.mark.unit
class TestReadMtz:
    def test_read_mtz_returns_reflectiondata(self, mtz_dir):
        from torchref import read_mtz
        from torchref.io import ReflectionData

        data = read_mtz(str(mtz_dir / "1DAW.mtz"), verbose=0)
        assert isinstance(data, ReflectionData)
        assert data.hkl.shape[0] > 0


@pytest.mark.unit
class TestReadPdb:
    def test_read_pdb_defaults_to_modelft(self, pdb_dir):
        from torchref import read_pdb
        from torchref.model import ModelFT

        model = read_pdb(str(pdb_dir / "1DAW.pdb"))
        assert isinstance(model, ModelFT)
        assert len(model.pdb) > 0

    def test_read_pdb_model_class_override(self, pdb_dir):
        from torchref import read_pdb
        from torchref.model import Model, ModelFT

        model = read_pdb(str(pdb_dir / "1DAW.pdb"), model_class=Model)
        assert isinstance(model, Model)
        assert not isinstance(model, ModelFT)


@pytest.mark.unit
class TestReadCif:
    def test_read_cif_model(self, cif_dir):
        from torchref import read_cif
        from torchref.model import ModelFT

        obj = read_cif(str(cif_dir / "1DAW.cif"), verbose=0)
        assert isinstance(obj, ModelFT)
        assert len(obj.pdb) > 0

    def test_read_cif_reflections(self, cif_sf_dir):
        from torchref import read_cif
        from torchref.io import ReflectionData

        obj = read_cif(str(cif_sf_dir / "1DAW-sf.cif"), verbose=0)
        assert isinstance(obj, ReflectionData)
        assert obj.hkl.shape[0] > 0


# Minimal restraint dictionary with a single ``data_comp_LIG`` block and NO
# ``comp_list`` header (the layout commonly emitted by eLBOW / AceDRG / Grade).
_NO_COMPLIST_CIF = """\
data_comp_LIG
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
LIG C1 C
LIG C2 C
LIG O1 O

loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
LIG C1 C2 SINGLE 1.530 0.010
LIG C2 O1 SINGLE 1.430 0.010
"""


# One compound written CCD style: a category with a single row (_chem_comp,
# _chem_comp_chir) as key-value pairs instead of a loop.
_KEY_VALUE_CIF = """\
data_LIG
_chem_comp.id LIG
_chem_comp.type non-polymer
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
LIG C1 C
LIG C2 C
LIG O1 O
LIG N1 N
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
LIG C1 C2 1.530 0.020
LIG C1 O1 1.430 0.020
LIG C1 N1 1.470 0.020
_chem_comp_chir.comp_id LIG
_chem_comp_chir.id chir_01
_chem_comp_chir.atom_id_centre C1
_chem_comp_chir.atom_id_1 C2
_chem_comp_chir.atom_id_2 O1
_chem_comp_chir.atom_id_3 N1
_chem_comp_chir.volume_sign positive
"""


@pytest.mark.unit
class TestRestraintCompIdExtraction:
    """A restraint dict without ``comp_list`` must still resolve its compound
    ID from the data blocks (not the filename stem), otherwise user-supplied
    restraints are silently overshadowed by the bundled monomer library.
    """

    def test_compound_id_from_data_block(self, tmp_path):
        from torchref.io.cif_readers import RestraintCIFReader

        # Deliberately give the file a stem that does NOT match the comp id,
        # so a filename-stem fallback would key the restraints incorrectly.
        cif = tmp_path / "my_custom_ligand.cif"
        cif.write_text(_NO_COMPLIST_CIF)

        reader = RestraintCIFReader(str(cif))
        assert reader.compounds == ["LIG"]

        restraints = reader.get_all_restraints()
        assert "LIG" in restraints
        bonds = restraints["LIG"]["bonds"]
        assert len(bonds) == 2  # restraints survive the comp-id filter

    def test_categories_written_as_key_value_pairs(self, tmp_path):
        from torchref.io.cif_readers import RestraintCIFReader

        cif = tmp_path / "ccd_style.cif"
        cif.write_text(_KEY_VALUE_CIF)

        reader = RestraintCIFReader(str(cif))
        assert reader.compounds == ["LIG"]
        chirals = reader.get_all_restraints()["LIG"]["chirals"]
        assert chirals.to_dict("records") == [
            {
                "atom_centre": "C1",
                "atom1": "C2",
                "atom2": "O1",
                "atom3": "N1",
                "volume_sign": "positive",
            }
        ]
