"""Restraint dictionaries keep each bond's order and aromatic flag.

The order is chemistry the force-field parameterisation of a ligand needs (it fixes
the net charge and the atom types), so the reader carries it next to the restraint
values in one vocabulary, whichever dictionary spelling the file uses.
"""

import pandas as pd
import pytest

from torchref.io.cif_readers import RestraintCIFReader
from torchref.topology.monomer.library import get_library_manager


def _bonds(columns):
    reader = RestraintCIFReader.__new__(RestraintCIFReader)
    frame = pd.DataFrame(
        {
            "_chem_comp_bond.comp_id": ["L"] * 3,
            "_chem_comp_bond.atom_id_1": ["C1", "C2", "C3"],
            "_chem_comp_bond.atom_id_2": ["C2", "C3", "O1"],
            "_chem_comp_bond.value_dist": ["1.5", "1.4", "1.2"],
            "_chem_comp_bond.value_dist_esd": ["0.02"] * 3,
            **columns,
        }
    )
    return reader._standardize_bonds(reader._filter_by_comp(frame, "L"))


@pytest.mark.unit
def test_monomer_library_spelling():
    """``type`` SINGLE/DOUBLE with a y/n ``aromatic`` flag (acedrg dictionaries)."""
    bonds = _bonds(
        {
            "_chem_comp_bond.type": ["SINGLE", "DOUBLE", "deloc"],
            "_chem_comp_bond.aromatic": ["n", "y", "n"],
        }
    )
    assert bonds["order"].tolist() == ["single", "double", "deloc"]
    assert bonds["aromatic"].tolist() == [False, True, False]


@pytest.mark.unit
def test_component_dictionary_spelling():
    """``value_order`` SING/DOUB/AROM with ``pdbx_aromatic_flag`` (wwPDB CCD)."""
    bonds = _bonds(
        {
            "_chem_comp_bond.value_order": ["SING", "DOUB", "AROM"],
            "_chem_comp_bond.pdbx_aromatic_flag": ["N", "N", "N"],
        }
    )
    assert bonds["order"].tolist() == ["single", "double", "aromatic"]
    assert bonds["aromatic"].tolist() == [False, False, True]


@pytest.mark.unit
def test_missing_order_reads_blank():
    """A dictionary without bond types gives ``""`` and no aromatic flag."""
    bonds = _bonds({})
    assert bonds["order"].tolist() == ["", "", ""]
    assert not bonds["aromatic"].any()
    assert bonds["value"].tolist() == [1.5, 1.4, 1.2]


@pytest.mark.unit
def test_bundled_phenylalanine_ring():
    """The bundled PHE dictionary gives a Kekulé ring with every ring bond aromatic."""
    path = get_library_manager(verbose=0).get_cif_file("PHE")
    bonds = RestraintCIFReader(path).get_compound_restraints("PHE")["bonds"]
    ring = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
    in_ring = bonds["atom1"].isin(ring) & bonds["atom2"].isin(ring)
    assert in_ring.sum() == 6
    assert bonds.loc[in_ring, "aromatic"].all()
    assert sorted(bonds.loc[in_ring, "order"]) == ["double"] * 3 + ["single"] * 3
    assert set(bonds.loc[~in_ring, "order"]) <= {"single", "double"}
