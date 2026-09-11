"""Value identity of :class:`~torchref.symmetry.cell.Cell`: ``key``, ``__eq__``, ``__hash__``."""

import pytest
import torch

from torchref.symmetry import Cell

PARAMS = [50.0, 60.0, 70.0, 90.0, 90.0, 90.0]


@pytest.mark.unit
def test_equal_parameters_compare_and_hash_equal():
    a = Cell(PARAMS, dtype=torch.float32, device="cpu")
    b = Cell(PARAMS, dtype=torch.float32, device="cpu")
    assert a is not b
    assert a == b
    assert hash(a) == hash(b)
    assert a.key == tuple(PARAMS)


@pytest.mark.unit
def test_clone_compares_equal():
    a = Cell(PARAMS)
    assert a.clone() == a
    assert hash(a.clone()) == hash(a)


@pytest.mark.unit
def test_different_parameters_compare_unequal():
    a = Cell(PARAMS)
    perturbed = list(PARAMS)
    perturbed[0] += 1e-3
    b = Cell(perturbed)
    assert a != b
    assert a.key != b.key


@pytest.mark.unit
def test_comparison_with_other_types_is_false():
    a = Cell(PARAMS)
    assert (a == "50 60 70 90 90 90") is False
    assert (a == PARAMS) is False
    assert a != None  # noqa: E711


@pytest.mark.unit
def test_usable_as_dict_key_and_in_set():
    a = Cell(PARAMS)
    b = Cell(PARAMS)
    cache = {a: "grid"}
    assert cache[b] == "grid"
    assert len({a, b}) == 1


@pytest.mark.unit
def test_key_is_cached_and_reset_with_the_other_derived_quantities():
    cell = Cell(PARAMS, dtype=torch.float32, device="cpu")
    assert "key" not in cell._cache
    first = cell.key
    assert cell._cache["key"] is first

    cell.to(dtype=torch.float64)
    assert "key" not in cell._cache, "reset_cache must drop the key with the rest"
    assert cell.key == first


@pytest.mark.unit
def test_in_place_edit_is_refused_at_the_next_derived_read():
    cell = Cell(PARAMS)
    _ = cell.volume
    cell.data[0] = 51.0
    with pytest.raises(RuntimeError, match="create a new one"):
        cell.fractional_matrix
    with pytest.raises(RuntimeError, match="Please don't edit Cell objects"):
        cell.key


@pytest.mark.unit
def test_in_place_edit_before_any_read_is_refused_too():
    cell = Cell(PARAMS)
    cell.data.mul_(2.0)
    with pytest.raises(RuntimeError, match="edited in place"):
        cell.volume


@pytest.mark.unit
def test_constructor_owns_its_tensor():
    t = torch.tensor(PARAMS, dtype=torch.float32)
    cell = Cell(t, dtype=torch.float32, device="cpu")
    t[0] = 99.0
    assert cell.key[0] == 50.0
    assert float(cell.volume) == pytest.approx(210000.0)


@pytest.mark.unit
def test_device_and_dtype_moves_are_not_edits():
    cell = Cell(PARAMS, dtype=torch.float32, device="cpu")
    _ = cell.fractional_matrix
    cell.to(dtype=torch.float64)
    assert cell.fractional_matrix.dtype == torch.float64
    assert cell.key == tuple(PARAMS)
    assert cell.clone() == cell
