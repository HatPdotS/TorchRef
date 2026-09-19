"""A restraint dictionary defining several compounds must restrain all of them.

``CIFReader`` stores loops under their CIF category, so every ``data_comp_*``
block in a multi-compound dictionary writes to the same key (``chem_comp_bond``,
...). Replacing rather than accumulating kept only whichever block came last,
leaving every other compound silently unrestrained -- silently, because an empty
bond table still carries the ``value``/``sigma`` columns that validation checks
for.

The reference here is the monomer library itself: each compound is read once from
a combined file and once from its own single-block file, and the two must agree.
That makes the assertion an equality against real data rather than a golden
number transcribed by hand.
"""

import numpy as np
import pandas as pd
import pytest

from torchref.io.cif_readers import RestraintCIFReader
from torchref.topology.monomer.library import get_library_manager
from torchref.topology.monomer.cif import (
    split_data_blocks,
    validate_restraint_data,
)

#: Bundled compounds, chosen to differ in every section: GLY has no chirals, ARG
#: has the most planes, ALA sits between them.
CODES = ("ALA", "ARG", "GLY")

SECTIONS = ("bonds", "angles", "torsions", "planes", "chirals", "atoms")

#: Two ligands, no ``comp_id`` column anywhere -- the layout some eLBOW/Grade
#: dictionaries use. Only the block name says which compound a row belongs to.
_NO_COMP_ID_CIF = """
data_comp_list
loop_
_chem_comp.id
_chem_comp.three_letter_code
_chem_comp.name
_chem_comp.group
LIGA LIGA 'ligand A' non-polymer
LIGB LIGB 'ligand B' non-polymer

data_comp_LIGA
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
A1 C
A2 O
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
A1 A2 1.230 0.020

data_comp_LIGB
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
B1 N
B2 C
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
B1 B2 1.470 0.015
"""


def _library_cif(code):
    """Path to ``code``'s bundled single-compound dictionary."""
    path = get_library_manager(verbose=0).get_cif_file(code)
    assert path is not None, f"{code} is not in the bundled monomer library"
    return path


@pytest.fixture(scope="module")
def combined_dictionary(tmp_path_factory):
    """The three library compounds concatenated into one multi-block file.

    Built rather than committed so the single-compound originals stay the
    reference: any divergence is the reader's, not a stale fixture's.
    """
    header, rows, bodies = None, [], []
    for code in CODES:
        blocks = split_data_blocks(_library_cif(code).read_text())
        listing = blocks["comp_list"].splitlines()
        if header is None:
            header = [
                line
                for line in listing
                if line.strip().startswith(("data_", "loop_", "_"))
            ]
        rows += [
            line
            for line in listing
            if line.strip() and not line.strip().startswith(("data_", "loop_", "_"))
        ]
        bodies.append(blocks[f"comp_{code}"])

    path = tmp_path_factory.mktemp("restraints") / "three_compounds.cif"
    path.write_text("\n".join(header + rows) + "\n\n" + "\n\n".join(bodies) + "\n")
    return path


@pytest.fixture(scope="module")
def combined_restraints(combined_dictionary):
    return RestraintCIFReader(str(combined_dictionary)).get_all_restraints()


@pytest.fixture(scope="module")
def separate_restraints():
    """Each compound read from its own file -- the control."""
    return {
        code: RestraintCIFReader(str(_library_cif(code))).get_all_restraints()[code]
        for code in CODES
    }


@pytest.mark.unit
class TestMultiComponentDictionary:
    def test_preconditions(
        self, combined_dictionary, combined_restraints, separate_restraints
    ):
        """Without these the comparison below could pass on two empty dicts."""
        blocks = [
            line
            for line in combined_dictionary.read_text().splitlines()
            if line.startswith("data_comp_") and line.strip() != "data_comp_list"
        ]
        assert len(blocks) == len(CODES), (
            f"fixture must hold {len(CODES)} compound blocks, found {blocks}"
        )
        assert list(combined_restraints) == list(CODES)
        for code in CODES:
            for section in SECTIONS:
                if section == "chirals" and code == "GLY":
                    continue  # glycine is achiral; nothing to compare
                assert len(separate_restraints[code][section]) > 0, (
                    f"control for {code}/{section} is empty, so it proves nothing"
                )

    @pytest.mark.parametrize("code", CODES)
    @pytest.mark.parametrize("section", SECTIONS)
    def test_compound_matches_its_own_file(
        self, combined_restraints, separate_restraints, code, section
    ):
        """Reading a compound alongside others must not change what it yields."""
        pd.testing.assert_frame_equal(
            combined_restraints[code][section],
            separate_restraints[code][section],
        )

    def test_no_compound_inherits_another_blocks_restraints(
        self, combined_restraints, separate_restraints
    ):
        """The old failure mode: a section absent from the last block was filled
        in from an earlier one, so a compound carried a neighbour's restraints."""
        assert len(combined_restraints["GLY"]["chirals"]) == 0
        assert len(separate_restraints["GLY"]["chirals"]) == 0


@pytest.mark.unit
class TestCompoundsWithoutCompIdColumn:
    """With no ``comp_id`` column the rows are told apart by their source block."""

    @pytest.fixture(scope="class")
    def restraints(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("nocompid") / "two_ligands.cif"
        path.write_text(_NO_COMP_ID_CIF)
        return RestraintCIFReader(str(path)).get_all_restraints()

    def test_preconditions(self, tmp_path_factory):
        """The fixture must really lack comp_id, or the block fallback is untested."""
        path = tmp_path_factory.mktemp("nocompid_check") / "two_ligands.cif"
        path.write_text(_NO_COMP_ID_CIF)
        data = RestraintCIFReader(str(path)).cif.data
        for category in ("chem_comp_atom", "chem_comp_bond"):
            assert not any("comp_id" in c for c in data[category].columns), (
                f"{category} carries a comp_id column; the fixture is wrong"
            )

    def test_each_ligand_keeps_only_its_own_bonds(self, restraints):
        assert list(restraints) == ["LIGA", "LIGB"]
        for comp, atoms in (("LIGA", ("A1", "A2")), ("LIGB", ("B1", "B2"))):
            bonds = restraints[comp]["bonds"]
            assert len(bonds) == 1
            assert (bonds.loc[0, "atom1"], bonds.loc[0, "atom2"]) == atoms

    def test_each_ligand_keeps_only_its_own_atoms(self, restraints):
        assert sorted(restraints["LIGA"]["atoms"]["atom_id"]) == ["A1", "A2"]
        assert sorted(restraints["LIGB"]["atoms"]["atom_id"]) == ["B1", "B2"]


@pytest.mark.unit
class TestColumnIndexAlignment:
    """Values that are present must not be lost to index misalignment.

    ``_extract_col`` keeps the frame's index for a column it finds but returns a
    fresh ``RangeIndex`` for one it does not. Assembling a result from both makes
    pandas align the mismatch away to NaN, so a compound sitting at a non-zero
    index -- which is every compound after the first once blocks are combined --
    would lose the columns that *are* there.
    """

    def test_present_columns_survive_a_missing_first_column(self):
        reader = RestraintCIFReader.__new__(RestraintCIFReader)
        frame = pd.DataFrame(
            {
                "_chem_comp_bond.comp_id": ["A", "A", "B", "B"],
                # atom_id_1 deliberately absent: it is what _standardize_bonds
                # extracts first, so its fallback would set the result's index.
                "_chem_comp_bond.atom_id_2": ["A2", "A3", "B2", "B3"],
                "_chem_comp_bond.value_dist": ["1.1", "1.2", "1.3", "1.4"],
                "_chem_comp_bond.value_dist_esd": ["0.01"] * 4,
            }
        )
        bonds = reader._standardize_bonds(reader._filter_by_comp(frame, "B"))

        assert bonds["atom2"].tolist() == ["B2", "B3"]
        assert bonds["value"].tolist() == [1.3, 1.4]
        assert bonds["atom1"].isna().all()  # genuinely absent, correctly NaN


@pytest.mark.unit
class TestChiralitySpellings:
    """The CCP4 library writes both ``positive`` and the truncated ``positiv``."""

    def test_short_spellings_are_not_dropped(self):
        from torchref.topology.builders import PreprocessedCIF

        chirals = pd.DataFrame(
            {
                "atom_centre": ["CA"] * 4,
                "atom1": ["N"] * 4,
                "atom2": ["C"] * 4,
                "atom3": ["CB"] * 4,
                "volume_sign": ["positive", "positiv", "negativ", "both"],
            }
        )
        signs = PreprocessedCIF({})._preprocess_chirals(chirals)["volume_sign"]

        # NaN here is not a rounding detail: builders_numba skips those rows, so
        # an unrecognised spelling deletes the restraint outright.
        assert not np.isnan(signs).any()
        assert signs.tolist() == [1.0, 1.0, -1.0, 0.0]


@pytest.mark.unit
class TestEmptyCompoundIsReported:
    """A compound that yields no restraints is named rather than passed over."""

    @pytest.fixture
    def dictionary_with_undefined_compound(self, tmp_path):
        text = _library_cif("ALA").read_text().replace(
            "ALA ALA ALANINE peptide 13 6 .",
            "ALA ALA ALANINE peptide 13 6 .\nGHOST GHOST GHOSTLY non-polymer 1 1 .",
        )
        path = tmp_path / "with_ghost.cif"
        path.write_text(text)
        return path

    def test_warns_without_raising(self, dictionary_with_undefined_compound, capsys):
        restraints = RestraintCIFReader(
            str(dictionary_with_undefined_compound)
        ).get_all_restraints()
        assert len(restraints["GHOST"]["bonds"]) == 0  # precondition
        assert len(restraints["ALA"]["bonds"]) > 0

        validate_restraint_data(restraints, str(dictionary_with_undefined_compound))

        out = capsys.readouterr().out
        assert "GHOST" in out and "no bond restraints" in out
        assert "ALA," not in out  # only the empty one is named
