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
    def test_model_strip_h_default(self):
        """strip_H defaults to False, and hydrogen generation is on."""
        from torchref.model.model import Model

        model = Model()

        assert model.ctx.strip_H is False
        assert model.ctx.add_hydrogens is True

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
    model.load(lambda: (df, cell, sg), add_hydrogens=False)

    assert len(model.pdb) == n_before - len(victims)
    idx = model.pdb["index"].to_numpy()
    assert idx.min() == 0
    assert idx.max() == len(model.pdb) - 1, "index must stay positional after a drop"
    assert sorted(idx) == list(range(len(model.pdb)))
    # The occupancy grouping is what actually indexed past the end.
    assert model.occupancy().shape[0] == len(model.pdb)
