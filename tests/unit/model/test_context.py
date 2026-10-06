"""What ModelContext.from_atoms accepts as the atom table of one model."""

import pytest

from torchref.io.cif_readers import ModelCIFReader
from torchref.model import Model


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
