"""The pipeline evaluates the search model in the crystal's cell, whatever cell
the model file carried."""
import pytest

from torchref.symmetry.cell import Cell

pytestmark = pytest.mark.alignment


@pytest.fixture(scope="module")
def case(pdb_dir, mtz_dir):
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model import ModelFT

    pdb, mtz = pdb_dir / "1DAW.pdb", mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW not available")
    return ModelFT(verbose=0).load_pdb(str(pdb)), ReflectionData(verbose=0).load_mtz(str(mtz))


def test_pipeline_adopts_the_crystal_cell(case):
    from torchref.experimental.alignment import MolecularReplacementPipeline

    model, data = case
    boxed = model.copy()
    boxed.spacegroup = "P 1"
    boxed.cell = Cell([40.0, 30.0, 35.0, 90.0, 90.0, 90.0])
    pipe = MolecularReplacementPipeline(data, boxed, d_min=4.0, d_max=15.0, n_shells=20)
    assert pipe.model.cell == data.cell
    assert boxed.cell != data.cell, "the caller's model is left alone"


def test_pipeline_keeps_a_model_already_in_the_crystal_cell(case):
    from torchref.experimental.alignment import MolecularReplacementPipeline

    model, data = case
    search = model.copy()
    search.spacegroup = "P 1"
    pipe = MolecularReplacementPipeline(data, search, d_min=4.0, d_max=15.0, n_shells=20)
    assert pipe.model is search
