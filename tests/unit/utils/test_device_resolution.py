"""Unit tests for ``torchref.utils.resolve_device``.

Covers the resolution rules: explicit override wins, empty call returns
the configured default, single module wins, consistent modules return
the first device, inconsistent modules emit a UserWarning and move the
others. Uses real ``Cell`` (non-Module DeviceMixin) and ``SpaceGroup``
(nn.Module) instances so both code paths are exercised.
"""

import pytest
import torch

from torchref.config import get_default_device
from torchref.symmetry import Cell, SpaceGroup
from torchref.utils import require_cell_dtype, resolve_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_cell():
    return Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], device="cpu")


def _cpu_sg():
    return SpaceGroup("P 21 21 21", device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# CPU-only tests
# ---------------------------------------------------------------------------

class TestNoModules:
    def test_no_args_returns_default(self):
        assert resolve_device() == get_default_device()

    def test_all_none_returns_default(self):
        assert resolve_device(None, None, None) == get_default_device()

    def test_explicit_device_no_modules(self):
        assert resolve_device(device="cpu") == torch.device("cpu")

    def test_explicit_device_str_coerced(self):
        result = resolve_device(device="cpu")
        assert isinstance(result, torch.device)
        assert result.type == "cpu"


class TestSingleModule:
    def test_single_cell(self):
        cell = _cpu_cell()
        assert resolve_device(cell) == torch.device("cpu")

    def test_single_spacegroup(self):
        sg = _cpu_sg()
        assert resolve_device(sg) == torch.device("cpu")

    def test_none_then_module(self):
        """None entries are skipped; first non-None wins."""
        cell = _cpu_cell()
        assert resolve_device(None, cell) == torch.device("cpu")


class TestConsistentModules:
    def test_two_modules_no_warning(self, recwarn):
        cell, sg = _cpu_cell(), _cpu_sg()
        result = resolve_device(cell, sg)
        assert result == torch.device("cpu")
        assert len(recwarn) == 0

    def test_three_modules_no_warning(self, recwarn):
        cell, sg, cell2 = _cpu_cell(), _cpu_sg(), _cpu_cell()
        result = resolve_device(cell, sg, cell2)
        assert result == torch.device("cpu")
        assert len(recwarn) == 0


class TestExplicitOverride:
    def test_explicit_overrides_modules(self, recwarn):
        cell = _cpu_cell()
        result = resolve_device(cell, device="cpu")
        assert result == torch.device("cpu")
        assert cell.device == torch.device("cpu")
        # No warning: explicit override is silent
        assert len(recwarn) == 0

    def test_explicit_overrides_skips_nones(self):
        cell = _cpu_cell()
        result = resolve_device(None, cell, None, device="cpu")
        assert result == torch.device("cpu")
        assert cell.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# GPU tests (cuda required for the cross-device branches)
# ---------------------------------------------------------------------------

@pytest.mark.cuda
class TestMixedDevices:
    def test_inconsistent_warns_and_moves_to_first(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cell_cuda = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], device="cuda")
        sg_cpu = SpaceGroup("P 21 21 21", device=torch.device("cpu"))

        with pytest.warns(UserWarning, match="differing devices"):
            result = resolve_device(cell_cuda, sg_cpu)

        assert result == cell_cuda.device  # first input wins
        assert result.type == "cuda"
        assert sg_cpu.matrices.device.type == "cuda"  # sg got moved

    def test_explicit_override_silently_moves_all(self, recwarn):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cell_cuda = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], device="cuda")
        sg_cpu = SpaceGroup("P 21 21 21", device=torch.device("cpu"))

        result = resolve_device(cell_cuda, sg_cpu, device="cpu")
        assert result.type == "cpu"
        assert cell_cuda.device.type == "cpu"  # cuda module pulled down to cpu
        assert sg_cpu.matrices.device.type == "cpu"
        # Override path is silent even when inputs disagreed
        assert len(recwarn) == 0

    def test_first_module_is_target_three_way(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cell_cpu = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], device="cpu")
        sg_cuda = SpaceGroup("P 21 21 21", device=torch.device("cuda"))
        cell2_cpu = Cell([10.0, 20.0, 30.0, 90.0, 90.0, 90.0], device="cpu")

        with pytest.warns(UserWarning):
            result = resolve_device(cell_cpu, sg_cuda, cell2_cpu)

        assert result.type == "cpu"  # cell_cpu wins
        assert sg_cuda.matrices.device.type == "cpu"
        assert cell2_cpu.device.type == "cpu"  # already cpu, unaffected


# ---------------------------------------------------------------------------
# The dtype axis: refused, not reconciled
# ---------------------------------------------------------------------------
class TestRequireCellDtype:
    """``require_cell_dtype`` is the dtype counterpart to ``resolve_device`` -- and the
    opposite policy, deliberately. Moving a tensor between devices is lossless, so
    reconciling is right there; casting is not, so this refuses and leaves the choice with
    the caller.
    """

    def test_agreement_is_silent(self, recwarn):
        cell = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float32)
        require_cell_dtype(cell, torch.float32, "SfDS")
        assert len(recwarn) == 0

    def test_none_cell_is_ignored(self):
        """So a caller can run this beside its own "is the cell set" check, in either order."""
        require_cell_dtype(None, torch.float32, "SfDS")

    def test_disagreement_raises_naming_both_dtypes(self):
        cell = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float64)
        with pytest.raises(RuntimeError) as exc:
            require_cell_dtype(cell, torch.float32, "SfDS")
        message = str(exc.value)
        assert "float64" in message and "float32" in message
        assert "SfDS" in message

    def test_the_cell_is_not_repaired(self):
        """The refusal must not quietly cast on the way out.

        A helper that raised *and* mutated would be the worst of both designs: the caller
        sees an error, retries, and silently gets the precision it was warned about.
        """
        cell = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float64)
        with pytest.raises(RuntimeError):
            require_cell_dtype(cell, torch.float32, "SfDS")
        assert cell.dtype == torch.float64


def test_sfds_refuses_a_cell_recast_after_construction():
    """The reason this is checked at point of use rather than in ``__init__``.

    ``Cell.to()`` mutates in place, so a cell can be recast long after the module that owns
    it was built -- and every cell-derived quantity (the fractional matrices, the reciprocal
    basis) is read straight off the cell. A constructor check cannot see this at all.

    Without the guard the symptom is a bare ``RuntimeError: expected mat1 and mat2 to have
    the same dtype`` from whichever matmul runs first, with nothing pointing at the cell.
    """
    from torchref.model.sf_ds import SfDS

    cell = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float32, device="cpu")
    sf = SfDS(cell=cell, spacegroup="P 1", dtype_float=torch.float32,
              device=torch.device("cpu"))
    xyz = torch.zeros(3, 3, dtype=torch.float32)
    sf._cartesian_to_fractional(xyz)  # consistent: fine

    cell.to(dtype=torch.float64)  # in place, behind the module's back
    with pytest.raises(RuntimeError, match="float64"):
        sf._cartesian_to_fractional(xyz)
