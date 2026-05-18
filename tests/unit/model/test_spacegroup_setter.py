"""
Regression tests for the cell / spacegroup setters on Model, ModelFT, and SfFFT.

Background: PyTorch's `nn.Module.__setattr__` intercepts assignments of
nn.Module values to attribute names and registers them in `_modules[name]`,
bypassing class-level `@property` descriptors. Without the `__setattr__`
override on our nn.Module-derived classes, assigning a SpaceGroup *object*
to e.g. `fft.spacegroup` creates a phantom `_modules['spacegroup']` entry
while leaving the canonical `_modules['_spacegroup']` (and the property
return value) stale.

These tests exercise every public path for changing the spacegroup and
verify that |F_calc| actually reflects the new symmetry.
"""
import math
from pathlib import Path

import pytest
import torch

from torchref.model import ModelFT
from torchref.symmetry import Cell, SpaceGroup


TEST_PDB = Path(__file__).resolve().parents[2] / "files" / "pdb" / "1DAW.pdb"


@pytest.fixture
def model_c2():
    """A fresh ModelFT loaded from 1DAW (C2 spacegroup)."""
    return ModelFT().load_pdb(str(TEST_PDB))


@pytest.fixture
def test_hkl():
    return torch.tensor([[5, 3, 2], [4, 0, 5], [2, 2, 3]], dtype=torch.int64)


def _F_calc(model, hkl):
    with torch.no_grad():
        return model(hkl).abs()


@pytest.mark.unit
def test_spacegroup_string_input(model_c2, test_hkl):
    """`M.spacegroup = 'P 1'` (string) must take effect on F_calc."""
    F_c2 = _F_calc(model_c2, test_hkl)
    model_c2.spacegroup = "P 1"
    F_p1 = _F_calc(model_c2, test_hkl)
    assert not torch.allclose(F_c2, F_p1), \
        "F_calc must change after spacegroup change (string input)"
    assert str(model_c2.spacegroup) == "SpaceGroup('P1', number=1, n_ops=1)"
    assert str(model_c2.fft.spacegroup) == "SpaceGroup('P1', number=1, n_ops=1)"
    # No phantom _modules['spacegroup'] entry from PyTorch's auto-registration.
    assert model_c2._modules.get("spacegroup") is None
    assert model_c2.fft._modules.get("spacegroup") is None


@pytest.mark.unit
def test_spacegroup_module_input(model_c2, test_hkl):
    """
    `M.spacegroup = SpaceGroup('P 1')` (Module input) must take effect.
    This is the case that previously failed because PyTorch's
    `nn.Module.__setattr__` would intercept the Module assignment and
    bypass the property setter.
    """
    F_c2 = _F_calc(model_c2, test_hkl)
    model_c2.spacegroup = SpaceGroup("P 1")
    F_p1 = _F_calc(model_c2, test_hkl)
    assert not torch.allclose(F_c2, F_p1), \
        "F_calc must change after spacegroup change (Module input)"
    assert str(model_c2.spacegroup) == "SpaceGroup('P1', number=1, n_ops=1)"
    assert str(model_c2.fft.spacegroup) == "SpaceGroup('P1', number=1, n_ops=1)"
    assert model_c2._modules.get("spacegroup") is None
    assert model_c2.fft._modules.get("spacegroup") is None


@pytest.mark.unit
def test_fft_spacegroup_setter_module_input(model_c2, test_hkl):
    """Low-level `M.fft.spacegroup = SpaceGroup('P 1')` must take effect."""
    F_c2 = _F_calc(model_c2, test_hkl)
    model_c2.fft.spacegroup = SpaceGroup("P 1")
    F_p1 = _F_calc(model_c2, test_hkl)
    assert not torch.allclose(F_c2, F_p1)
    assert str(model_c2.fft.spacegroup) == "SpaceGroup('P1', number=1, n_ops=1)"


@pytest.mark.unit
def test_fft_set_cell_and_spacegroup_module_input(model_c2, test_hkl):
    """`M.fft.set_cell_and_spacegroup(cell, SpaceGroup('P 1'))` must take effect."""
    F_c2 = _F_calc(model_c2, test_hkl)
    model_c2.fft.set_cell_and_spacegroup(model_c2.cell, SpaceGroup("P 1"))
    F_p1 = _F_calc(model_c2, test_hkl)
    assert not torch.allclose(F_c2, F_p1)
    assert str(model_c2.fft.spacegroup) == "SpaceGroup('P1', number=1, n_ops=1)"


@pytest.mark.unit
def test_fft_spacegroup_setter_invalidates_caches(model_c2):
    """The SfFFT's `map_symmetry` and reciprocal-symmetry extractor must be
    rebuilt on spacegroup change (otherwise the next density-map build would
    apply the OLD symmetry)."""
    fft = model_c2.fft
    # Force a state where caches exist.
    fft.setup_grid()
    old_map_sym_id = id(fft.map_symmetry)
    fft.spacegroup = SpaceGroup("P 1")
    # map_symmetry was rebuilt by setup_grid (called from the setter).
    new_map_sym_id = id(fft.map_symmetry)
    assert new_map_sym_id != old_map_sym_id, \
        "map_symmetry must be rebuilt after spacegroup change"


@pytest.mark.unit
def test_fft_cell_setter_invalidates_grid(model_c2):
    """Setting `M.fft.cell = new_cell` must invalidate / rebuild the grid."""
    fft = model_c2.fft
    fft.setup_grid()
    old_grid_size = tuple(int(x) for x in fft.gridsize)
    # Use a cell with very different parameters.
    new_cell = Cell([80.0, 80.0, 80.0, 90.0, 90.0, 90.0])
    fft.cell = new_cell
    # Grid was re-set up automatically with the new cell.
    new_grid_size = tuple(int(x) for x in fft.gridsize)
    assert new_grid_size != old_grid_size, \
        "Grid must be rebuilt after cell change"
    assert fft.cell is new_cell
