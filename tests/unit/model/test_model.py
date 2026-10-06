"""
Unit tests for torchref.model.model

Tests the Model class for atomic structure representation.
Note: Unit tests use mock data, not real file I/O.
"""

import pytest
import torch
import torch.nn as nn


class TestModelInitialization:
    """Tests for Model class initialization."""

    @pytest.mark.unit
    def test_model_empty_initialization(self):
        """Test Model can be initialized without files."""
        from torchref.model.model import Model

        model = Model()

        assert model.ctx.initialized is False
        assert model.pdb is None
        assert model.xyz is None
        assert model.adp is None

    @pytest.mark.unit
    def test_model_is_nn_module(self):
        """Model should be a nn.Module."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert isinstance(model, nn.Module)

    @pytest.mark.unit
    def test_model_default_dtype(self):
        """Test default dtype is float32."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert model.dtype_float == torch.float32

    @pytest.mark.unit
    def test_model_custom_dtype(self):
        """Test custom dtype specification."""
        from torchref.model.model import Model
        
        model = Model(dtype_float=torch.float64)
        
        assert model.dtype_float == torch.float64

    @pytest.mark.unit
    def test_model_hydrogen_default(self):
        """Hydrogen stripping and generation are both opt-in; hydrogens are atoms."""
        from torchref.model.model import Model

        model = Model()

        assert model.ctx.hydrogens == "keep"
        assert model.ctx.hydrogen_mode == "atoms"

    @pytest.mark.unit
    def test_model_bool_uninitialized(self):
        """Uninitialized model should be falsy."""
        from torchref.model.model import Model
        
        model = Model()
        
        assert bool(model) == False


class TestModelDeviceHandling:
    """Tests for device handling in Model."""

    @pytest.mark.unit
    def test_model_default_device(self):
        """Test default device matches the package-wide configured default."""
        from torchref.config import get_default_device
        from torchref.model.model import Model

        model = Model()

        assert model.device == get_default_device()

    @pytest.mark.unit
    def test_model_custom_device(self):
        """Test custom device specification."""
        from torchref.model.model import Model
        
        model = Model(device=torch.device('cpu'))
        
        assert model.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_model_gpu_device(self, gpu_device):
        """Test GPU device specification."""
        from torchref.model.model import Model
        
        model = Model(device=gpu_device)
        
        assert model.device.type == gpu_device.type


class TestModelGetSelectionMask:
    """Tests for Model.get_selection_mask() method."""

    @pytest.mark.unit
    def test_get_selection_mask_uninitialized_raises(self):
        """Test that get_selection_mask() raises RuntimeError on uninitialized model."""
        from torchref.model.model import Model
        
        model = Model()
        
        with pytest.raises(RuntimeError, match="uninitialized"):
            model.get_selection_mask("chain A")


@pytest.mark.unit
def test_dropped_rows_leave_a_positional_index(pdb_dir, tmp_path):
    """A model losing atoms to the NaN drop must still index its own tensors.

    ``load`` derives the ``index`` column from the DataFrame index, and every
    consumer uses it to address length-N per-atom tensors positionally. Dropping rows
    without reindexing leaves gaps, so the largest value exceeds N-1 and
    ``_create_occupancy_groups`` walks off the end of ``initial_occ``. Roughly one
    PDB-REDO entry in six carries an atom with no coordinates or no B and hit this.
    """
    import pandas as pd

    from torchref.model.model import Model

    src = Model(verbose=0)
    src.load_pdb(str(pdb_dir / "3GR5.pdb"))
    df = src.pdb.copy()
    n_before = len(df)

    # Blank the B of a few interior atoms so the dropna removes them.
    victims = [5, 100, 500]
    df.loc[victims, "tempfactor"] = float("nan")
    cell = src.cell.data.cpu().numpy()
    sg = src.spacegroup

    model = Model(verbose=0)
    model.load(lambda: (df, cell, sg))

    assert len(model.pdb) == n_before - len(victims)
    idx = model.pdb["index"].to_numpy()
    assert idx.min() == 0
    assert idx.max() == len(model.pdb) - 1, "index must stay positional after a drop"
    assert sorted(idx) == list(range(len(model.pdb)))
    # The occupancy grouping is what actually indexed past the end.
    assert model.occupancy().shape[0] == len(model.pdb)


SELECTION = "resseq 10:20"


@pytest.fixture
def daw_model(pdb_dir):
    """1DAW, loaded per test: the selection methods mutate the model in place."""
    from torchref.model.model import Model

    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model


@pytest.mark.unit
@pytest.mark.parametrize("start", ["full", "partial"])
@pytest.mark.parametrize("freeze", [True, False])
def test_selection_edits_the_refinable_set(daw_model, start, freeze):
    """Freezing subtracts the selection from the set and unfreezing adds it."""
    model = daw_model
    if start == "partial":
        model.xyz_mask = model.get_selection_mask("resseq 15:60").to(model.device)
    current = model.xyz_mask.clone()
    selected = model.get_selection_mask(SELECTION).to(model.device)

    model.update_mask_from_selection(SELECTION, "xyz", freeze=freeze)
    model.apply_mask_to_parameter("xyz")

    expected = current & ~selected if freeze else current | selected
    assert torch.equal(model.xyz_mask, expected)
    assert model.xyz.get_refinable_count() == int(expected.sum())


@pytest.mark.unit
def test_unfreeze_selection_keeps_the_rest_refinable(daw_model):
    """Unfreezing a selection never freezes the atoms outside it."""
    model = daw_model
    model.unfreeze_selection(SELECTION, targets="xyz")
    assert int(model.xyz_mask.sum()) == model.n_atoms
    assert model.xyz.get_refinable_count() == model.n_atoms


@pytest.mark.unit
def test_unfreeze_all_reapplies_the_set_after_a_selection(daw_model):
    """``freeze_all`` is a toggle, so ``unfreeze_all`` brings the whole set back."""
    model = daw_model
    model.freeze_all()
    model.unfreeze_selection(SELECTION, targets="xyz")
    model.unfreeze_all()
    assert model.xyz.get_refinable_count() == model.n_atoms


@pytest.mark.unit
def test_refining_only_a_selection_starts_from_an_empty_set(daw_model):
    """The documented idiom: freeze everything by selection, then add one back."""
    model = daw_model
    model.freeze_selection("all", targets="xyz")
    model.unfreeze_selection(SELECTION, targets="xyz")
    n_selected = int(model.get_selection_mask(SELECTION).sum())
    assert model.xyz.get_refinable_count() == n_selected
