"""The FFT grid is derived from the model's context and cached on a value key."""

import pytest
import torch

from torchref.config import dtypes
from torchref.model import ModelFT
from torchref.model.context import ModelContext
from torchref.model.sf_fft import SfFFT
from torchref.symmetry import Cell, SpaceGroup


@pytest.fixture(scope="module")
def pdb_path(pdb_dir):
    path = pdb_dir / "1DAW.pdb"
    if not path.exists():
        pytest.skip("1DAW.pdb fixture not present")
    return str(path)


def _model(pdb_path, **kwargs) -> ModelFT:
    return ModelFT(max_res=2.5, verbose=0, device="cpu", **kwargs).load_pdb(pdb_path)


def _hkl(model, n=64):
    gen = torch.Generator().manual_seed(0)
    return torch.randint(-8, 9, (n, 3), generator=gen).to(
        dtype=dtypes.int, device=model.device
    )


def _count_calls(monkeypatch, cls, name):
    counter = {"n": 0}
    original = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        counter["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, wrapped)
    return counter


@pytest.mark.unit
@pytest.mark.parametrize("strip_H", [False, True])
def test_one_engine_and_one_spacegroup_per_load(pdb_path, monkeypatch, strip_H):
    import torchref.model.model_ft as model_ft_module

    engines = _count_calls(monkeypatch, model_ft_module.SfFFT, "__init__")
    spacegroups = _count_calls(monkeypatch, SpaceGroup, "__init__")

    model = ModelFT(max_res=2.5, verbose=0, device="cpu", strip_H=strip_H)
    model.load_pdb(pdb_path)
    assert engines["n"] == 1
    assert spacegroups["n"] == 1

    # Later crystal changes are followed by the key, not by rebuilding the engine.
    model.cell = model.cell.clone()
    model.max_res = 3.0
    assert model.grid_shape is not None
    assert engines["n"] == 1


@pytest.mark.unit
def test_engine_reads_the_model_context(pdb_path):
    model = _model(pdb_path)

    def bound(m):
        return (
            m.fft.ctx is m.ctx
            and m.fft.cell is m.ctx.cell
            and m.fft.spacegroup is m.ctx.spacegroup
        )

    assert bound(model)

    copied = model.copy()
    assert bound(copied)
    assert copied.ctx is not model.ctx
    assert copied.fft.cell is not model.fft.cell

    selected = model.select("all")
    assert bound(selected)

    restored = ModelFT.create_from_state_dict(model.state_dict(), device="cpu", verbose=0)
    assert bound(restored)
    assert restored.grid_shape == model.grid_shape


@pytest.mark.unit
def test_explicit_gridsize_survives_every_path(pdb_path):
    explicit = (64, 32, 24)
    model = _model(pdb_path, gridsize=explicit)
    assert model.grid_shape == explicit

    model.cell = model.cell.clone()
    assert model.grid_shape == explicit

    assert model.copy().grid_shape == explicit
    assert model.select("all").grid_shape == explicit

    restored = ModelFT.create_from_state_dict(model.state_dict(), device="cpu", verbose=0)
    assert restored.explicit_gridsize == explicit
    assert restored.grid_shape == explicit


@pytest.mark.unit
def test_grid_follows_its_key(pdb_path):
    model = _model(pdb_path)
    shape0 = model.grid_shape
    key0 = model.grid_key
    ptr0 = model.fft.gridsize.data_ptr()

    # Same values, different object: nothing to rebuild.
    model.cell = model.cell.clone()
    assert model.grid_key == key0
    assert model.fft.gridsize.data_ptr() == ptr0

    scale = torch.tensor([1.25, 1.25, 1.25, 1.0, 1.0, 1.0])
    model.cell = Cell(model.cell.data * scale, dtype=model.dtype_float, device="cpu")
    assert model.grid_key != key0
    assert model.grid_shape != shape0
    assert all(n > m for n, m in zip(model.grid_shape, shape0))

    model.max_res = 4.0
    coarse = model.grid_shape
    assert all(n < m for n, m in zip(coarse, model.grid_shape)) is False
    model.max_res = 2.5
    assert all(n > m for n, m in zip(model.grid_shape, coarse))


@pytest.mark.unit
def test_first_fcalc_after_cell_reassignment_uses_the_late_path(pdb_path, monkeypatch):
    model = _model(pdb_path)
    hkl = _hkl(model)
    with torch.no_grad():
        f0 = model(hkl).clone()

    model.cell = model.cell.clone()
    model.reset_cache()
    assert model.fft.late_symmetry_compatible is True

    symmetrised = _count_calls(monkeypatch, SpaceGroup, "symmetrize_map")
    with torch.no_grad():
        f1 = model(hkl)
    assert symmetrised["n"] == 0
    assert torch.allclose(f1, f0)


@pytest.mark.unit
def test_forward_cache_invalidates_on_a_spacegroup_change(pdb_path):
    model = _model(pdb_path)
    hkl = _hkl(model)
    with torch.no_grad():
        f0 = model(hkl)
        assert model(hkl) is f0  # cached
        assert model._fwd_cached_state_fp[-1] == model.grid_key

        model.spacegroup = "P 1 2 1"  # same cell, centring dropped
        f1 = model(hkl)
    assert f1 is not f0
    assert not torch.allclose(f1, f0)
    assert model._fwd_cached_state_fp[-1] == model.grid_key


@pytest.mark.unit
def test_to_moves_the_shared_context_once(pdb_path, monkeypatch):
    model = _model(pdb_path)
    cell = model.ctx.cell
    resets = {"n": 0}
    original = Cell.reset_cache

    def counting(self):
        if self is cell:
            resets["n"] += 1
        return original(self)

    monkeypatch.setattr(Cell, "reset_cache", counting)
    model.to(torch.device("cpu"))
    assert resets["n"] == 1


@pytest.mark.unit
def test_engine_without_a_crystal():
    sf = SfFFT(ModelContext(), max_res=1.0)
    assert sf.gridsize is None
    assert sf.grid_shape is None
    assert sf.grid_key is None

    empty = torch.zeros((0, 3))
    with pytest.raises(RuntimeError, match="no cell or space group"):
        sf.compute_structure_factors(
            torch.zeros((1, 3), dtype=dtypes.int),
            empty, torch.zeros(0), torch.zeros(0), torch.zeros((0, 5)), torch.zeros((0, 5)),
        )


@pytest.mark.unit
def test_legacy_gridsize_is_adopted_only_when_it_differs(pdb_path):
    model = _model(pdb_path)

    same = model.state_dict()
    same["_fft.gridsize"] = torch.tensor(model.grid_shape)
    restored = ModelFT.create_from_state_dict(same, device="cpu", verbose=0)
    assert restored.explicit_gridsize is None
    assert restored.grid_shape == model.grid_shape

    different = model.state_dict()
    different["_fft.gridsize"] = torch.tensor([64, 32, 24])
    restored = ModelFT.create_from_state_dict(different, device="cpu", verbose=0)
    assert restored.explicit_gridsize == (64, 32, 24)
    assert restored.grid_shape == (64, 32, 24)
