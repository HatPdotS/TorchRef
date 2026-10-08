"""AmberTarget's residue classification, which needs no OpenMM."""

import pandas as pd
import pytest

from torchref.experimental.targets.amber_target import AmberTarget


class _Chemistry:
    def __init__(self, records, resnames):
        self._atoms = pd.DataFrame({"ATOM": records, "resname": resnames})

    def to_dataframe(self):
        return self._atoms.copy()


def _target(records, resnames, residue_charges=None):
    target = AmberTarget.__new__(AmberTarget)
    target._chem_model = _Chemistry(records, resnames)
    target._residue_charges = dict(residue_charges or {})
    return target


@pytest.mark.unit
def test_hetatm_ligand_is_parameterised():
    """A non-standard HETATM residue is returned with its supplied charge."""
    target = _target(["ATOM", "HETATM", "HETATM"], ["ALA", "LIG", "HOH"], {"LIG": -1})
    assert target._detect_nonstandard_residues() == [("LIG", -1)]


@pytest.mark.unit
def test_unknown_atom_residue_raises():
    """An ATOM residue AMBER14 does not know raises, whatever charges are given."""
    target = _target(["ATOM", "ATOM"], ["ALA", "XYZ"], {"XYZ": 0})
    with pytest.raises(ValueError, match="XYZ"):
        target._detect_nonstandard_residues()
