"""What ModelContext.from_atoms accepts as the atom table of one model, and which
residues its chain sequences hold."""

import gemmi
import pandas as pd
import pytest

from torchref.io.cif_readers import ModelCIFReader
from torchref.model import Model
from torchref.model.context import ModelContext
from torchref.topology import Topology


@pytest.fixture(scope="module")
def ensemble_path(cif_dir):
    """Two models of the same 15 atoms, numbered by ``pdbx_PDB_model_num``."""
    return str(cif_dir / "test_ihm_ensemble.cif")


@pytest.mark.unit
def test_a_table_of_several_models_is_refused(ensemble_path):
    """Every atom once per model is an ensemble, not one crystal's model; the error
    names the model numbers."""
    with pytest.raises(ValueError, match=r"model_num \[1, 2\]"):
        Model(verbose=0).load_cif(ensemble_path)


@pytest.mark.unit
def test_each_model_of_the_file_loads_on_its_own(ensemble_path):
    """One model's rows, as the IHM reader splits them, load as a model."""
    reader = ModelCIFReader(ensemble_path, verbose=0)
    _, cell, spacegroup = reader()
    for table in reader.get_atom_data_by_model().values():
        model = Model(verbose=0).load(lambda: (table, cell, spacegroup))
        assert model.n_atoms == len(table) == 15


@pytest.mark.unit
def test_chain_sequence_is_the_deposited_one_without_ligands(pdb_dir):
    """1DAW chain A reads as its SEQRES; the AMP-PNP, magnesium and waters stay out."""
    path = str(pdb_dir / "1DAW.pdb")
    structure = gemmi.read_structure(path)
    structure.setup_entities()
    seqres = gemmi.one_letter_code(structure.entities[0].full_sequence)

    assert Model(verbose=0).load_pdb(path).ctx.chain_sequences == [("A", seqres)]


@pytest.mark.unit
def test_hetatm_selenomethionines_read_as_methionine(pdb_dir):
    """3E98's selenomethionines 65 and 73, written as HETATM, are protein residues
    of both chains, not gaps."""
    model = Model(verbose=0).load_pdb(str(pdb_dir / "3E98.pdb"))
    sequences = dict(model.ctx.chain_sequences)

    for chain in "AB":
        assert "RNIEMRHRLSQLMDVAR" in sequences[chain]
    assert "?" not in sequences["A"]


@pytest.mark.unit
def test_chains_without_peptide_links_keep_their_sequences():
    """DNA and RNA chains, a CA-only trace and a one-residue chain are polymer by
    their residues' component type, with no coordinates or links consulted; a water
    is not, whatever its record type."""
    residues = [
        ("B", 1, "DA", ["P", "C1'", "N9"]),
        ("B", 2, "DC", ["P", "C1'", "N1"]),
        ("B", 3, "DG", ["P", "C1'", "N9"]),
        ("R", 1, "A", ["P", "C1'"]),
        ("R", 2, "U", ["P", "C1'"]),
        ("C", 1, "GLY", ["CA"]),
        ("C", 2, "ALA", ["CA"]),
        ("C", 4, "VAL", ["CA"]),
        ("D", 7, "LYS", ["N", "CA", "C", "O"]),
        ("D", 8, "HOH", ["O"]),
    ]
    table = pd.DataFrame(
        [(c, r, n, a) for c, r, n, names in residues for a in names],
        columns=["chainid", "resseq", "resname", "name"],
    )
    ctx = ModelContext(topology=Topology.from_table(table))

    assert ctx.chain_sequences == [
        ("B", "XXX"),
        ("R", "XX"),
        ("C", "GA?V"),
        ("D", "K"),
    ]
