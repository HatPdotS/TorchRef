"""Tests for ReflectionData.__getitem__ / __select__."""

import gemmi
import pytest
import torch

from torchref.io.datasets.reflection_data import ReflectionData
from torchref.symmetry import Cell
from torchref.utils.utils import TensorMasks


def _make_reflection_data(n=20):
    """Build a minimal ReflectionData with all common fields populated."""
    rd = ReflectionData(verbose=0, device="cpu")

    # Core per-reflection tensors
    rd.hkl = torch.randint(-5, 6, (n, 3), dtype=torch.int32)
    rd.F = torch.rand(n, dtype=torch.float32) * 100 + 1.0
    rd.F_sigma = torch.rand(n, dtype=torch.float32) * 5 + 0.5
    rd.I = torch.rand(n, dtype=torch.float32) * 1000 + 10.0
    rd.I_sigma = torch.rand(n, dtype=torch.float32) * 10 + 1.0
    rd.rfree_flags = torch.randint(0, 2, (n,), dtype=torch.int32).to(torch.bool)
    rd.resolution = torch.rand(n, dtype=torch.float32) * 3 + 1.0
    rd.phase = torch.rand(n, dtype=torch.float32) * 6.28
    rd.fom = torch.rand(n, dtype=torch.float32)

    # Non-per-reflection tensor (should NOT be indexed)
    rd.U_aniso = torch.rand(6, dtype=torch.float32)

    # Cell and spacegroup
    rd.cell = Cell(
        torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0]),
        dtype=torch.float32,
    )
    rd.spacegroup = gemmi.SpaceGroup("P212121")

    # Masks
    mask_all_true = torch.ones(n, dtype=torch.bool)
    rd.masks = TensorMasks(device="cpu")
    rd.masks["sanity"] = mask_all_true

    return rd


class TestGetitemBooleanMask:
    """Select subset via boolean mask."""

    def test_basic_selection(self):
        rd = _make_reflection_data(20)
        mask = torch.zeros(20, dtype=torch.bool)
        mask[:10] = True

        sel = rd[mask]

        assert len(sel.hkl) == 10
        assert len(sel.F) == 10
        assert len(sel.F_sigma) == 10
        assert len(sel.resolution) == 10
        assert len(sel.phase) == 10
        assert len(sel.fom) == 10
        torch.testing.assert_close(sel.hkl, rd.hkl[:10])
        torch.testing.assert_close(sel.F, rd.F[:10])

    def test_masks_propagated(self):
        rd = _make_reflection_data(20)
        mask = torch.zeros(20, dtype=torch.bool)
        mask[:10] = True

        sel = rd[mask]

        assert "sanity" in sel.masks
        assert len(sel.masks["sanity"]) == 10


class TestGetitemIntegerIndices:
    """Reorder via integer index tensor."""

    def test_permutation(self):
        rd = _make_reflection_data(10)
        perm = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        sel = rd[perm]

        assert len(sel.hkl) == 10
        torch.testing.assert_close(sel.hkl, rd.hkl[perm])
        torch.testing.assert_close(sel.F, rd.F[perm])
        torch.testing.assert_close(sel.phase, rd.phase[perm])


class TestNonMatchingTensors:
    """Tensors whose first dim != n_refl should be cloned, not indexed."""

    def test_u_aniso_copied(self):
        rd = _make_reflection_data(20)
        mask = torch.zeros(20, dtype=torch.bool)
        mask[:10] = True

        sel = rd[mask]

        # U_aniso has shape (6,), not (20,), so it should be copied as-is
        torch.testing.assert_close(sel.U_aniso, rd.U_aniso)
        assert sel.U_aniso.shape == (6,)


class TestCellAndSpacegroup:
    """Cell is cloned; spacegroup is copied by reference."""

    def test_cell_cloned(self):
        rd = _make_reflection_data(20)
        sel = rd[torch.arange(10)]

        assert sel.cell is not rd.cell
        torch.testing.assert_close(sel.cell.data, rd.cell.data)

    def test_spacegroup_preserved(self):
        rd = _make_reflection_data(20)
        sel = rd[torch.arange(10)]

        assert sel.spacegroup.hm == rd.spacegroup.hm


class TestAllFieldsCovered:
    """Verify that __select__ handles every tensor field without dropping any."""

    def test_no_field_dropped(self):
        from dataclasses import fields as dc_fields

        rd = _make_reflection_data(20)
        sel = rd[torch.arange(10)]

        for f in dc_fields(rd):
            orig = getattr(rd, f.name)
            selected = getattr(sel, f.name)
            if orig is None:
                continue
            if isinstance(orig, torch.Tensor):
                assert selected is not None, f"Field {f.name} was dropped"


class TestUnsupportedIndex:
    """Non-tensor index should raise TypeError."""

    def test_int_raises(self):
        rd = _make_reflection_data(10)
        with pytest.raises(TypeError):
            _ = rd[5]

    def test_list_raises(self):
        rd = _make_reflection_data(10)
        with pytest.raises(TypeError):
            _ = rd[[0, 1, 2]]
