"""GAFF2 templates made from the model's own atoms and the monomer dictionaries.

Pins the chemistry handed to antechamber (net charge from dictionary formal charges,
corrected per hydrogen the model lacks or adds; residue_charges override), that the
template's atoms are the model's atoms in model order, that one chemistry is
parameterised once and then read from the cache, and the refusals: a residue without
a dictionary, and a ligand covalently bonded to another residue.
"""

import io

import numpy as np
import pytest

from torchref.experimental.mm import ligands
from torchref.experimental.mm.ligands import gaff2_forcefield, ligand_chemistry
from torchref.experimental.mm.topology import build_asu_topology
from torchref.model.model import Model

pytestmark = pytest.mark.amber


@pytest.fixture(scope="module")
def sulfate(pdb_dir):
    model = (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "3GR5.pdb"))
        .strip_altlocs()
    )
    return model, build_asu_topology(model)


@pytest.fixture(scope="module")
def anp(pdb_dir):
    model = (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "1DAW.pdb"))
        .strip_altlocs()
    )
    return model, build_asu_topology(model)


def _residues(asu, name):
    return [r for r in range(asu.n_residues) if asu.residue_name[r] == name]


def test_sulfate_charge_comes_from_the_dictionary(sulfate):
    model, asu = sulfate
    chemistry = ligand_chemistry(
        asu, _residues(asu, "SO4")[0], model.xyz().detach().numpy()
    )
    assert chemistry.net_charge == -2
    assert sorted(chemistry.symbols) == ["O", "O", "O", "O", "S"]


def test_each_missing_hydrogen_lowers_the_charge(anp):
    """Removing one hydrogen of ANP deprotonates its parent: one charge lower."""
    model, asu = anp
    r = _residues(asu, "ANP")[0]
    full = ligand_chemistry(asu, r, model.xyz().detach().numpy())
    table = model.to_dataframe()
    start, end = asu.residue_start[r], asu.residue_start[r + 1]
    names = asu.name[start:end]
    hydrogen = int(
        asu.rows[start + int(np.flatnonzero(asu.symbol[start:end] == "H")[0])]
    )
    reduced = model._derive(table.drop(index=hydrogen), hydrogens="keep")
    reduced_asu = build_asu_topology(reduced)
    chemistry = ligand_chemistry(
        reduced_asu, _residues(reduced_asu, "ANP")[0], reduced.xyz().detach().numpy()
    )
    assert chemistry.net_charge == full.net_charge - 1
    assert len(chemistry.names) == len(names) - 1


def test_residue_charges_override(sulfate):
    model, asu = sulfate
    chemistry = ligand_chemistry(
        asu, _residues(asu, "SO4")[0], model.xyz().detach().numpy(), {"SO4": -1}
    )
    assert chemistry.net_charge == -1


def test_template_atoms_are_the_model_atoms(sulfate):
    """The XML template carries the model's atom names in model order."""
    import openmm.app as app

    model, asu = sulfate
    residues = _residues(asu, "SO4")
    xml = gaff2_forcefield(asu, residues, model.xyz().detach().numpy())
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    ff.loadFile(io.StringIO(xml))
    templates = [t for name, t in ff._templates.items() if name.startswith("SO4-gaff2")]
    assert len(templates) == 1
    start, end = asu.residue_start[residues[0]], asu.residue_start[residues[0] + 1]
    assert [a.name for a in templates[0].atoms] == list(asu.name[start:end])


def test_one_chemistry_is_parameterised_once(sulfate, monkeypatch):
    """Every SO4 shares one antechamber run; a second build reads the cache."""
    model, asu = sulfate
    residues = _residues(asu, "SO4")
    assert len(residues) > 1
    gaff2_forcefield(asu, residues, model.xyz().detach().numpy())

    def no_programs(*args, **kwargs):
        raise AssertionError("antechamber ran despite a cached result")

    monkeypatch.setattr(ligands, "_run", no_programs)
    assert "SO4-gaff2" in gaff2_forcefield(asu, residues, model.xyz().detach().numpy())


def test_residue_without_a_dictionary_is_refused(pdb_dir):
    model = (
        Model(verbose=0, device="cpu", hydrogens="add")
        .load_pdb(str(pdb_dir / "3GR5.pdb"))
        .strip_altlocs()
    )
    asu = build_asu_topology(model)
    asu.restraints.cif_dict = {
        k: v for k, v in asu.restraints.cif_dict.items() if k != "SO4"
    }
    with pytest.raises(ValueError, match="no monomer dictionary"):
        ligand_chemistry(asu, _residues(asu, "SO4")[0], model.xyz().detach().numpy())


def test_covalent_ligand_is_refused(sulfate):
    model, asu = sulfate
    r = _residues(asu, "SO4")[0]
    linked = build_asu_topology(model)
    sulfur = int(asu.residue_start[r])
    linked.bonds = np.concatenate([asu.bonds, [[0, sulfur]]])
    with pytest.raises(ValueError, match="covalently bonded"):
        ligand_chemistry(linked, r, model.xyz().detach().numpy())
