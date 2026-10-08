"""The bulk-solvent mask's voxel index arithmetic."""

import pytest
import torch
from torch.overrides import TorchFunctionMode

from torchref.config import get_float_dtype, get_int_dtype
from torchref.scaling.solvent import _voxel_offsets_within


@pytest.mark.unit
def test_voxel_offsets_take_the_configured_int_dtype():
    """The ball of voxel offsets is built in the configured integer dtype, which the
    dilation's voxel indices inherit."""
    frac = torch.diag(torch.tensor([30.0, 40.0, 50.0], dtype=get_float_dtype()))
    grid_dims = torch.tensor([30, 40, 50], dtype=get_int_dtype())

    offsets = _voxel_offsets_within(2.5, grid_dims, frac, torch.device("cpu"))

    assert offsets.dtype == get_int_dtype()
    assert offsets.abs().max().item() == 2


class _Float32Recorder(TorchFunctionMode):
    """Record every torch call that returns a float32 tensor."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor) and out.dtype == torch.float32:
            self.calls.append(getattr(func, "__name__", repr(func)))
        return out


@pytest.mark.integration
def test_solvent_mask_stays_in_float64(double_cpu, sample_structure_pair):
    """Under a float64 configuration the dilation's voxel positions are not rounded
    through float32."""
    from torchref.model.model_ft import ModelFT
    from torchref.scaling.solvent import SolventModel

    model = ModelFT(verbose=0)
    model.load_cif(str(sample_structure_pair["model"]))
    solvent = SolventModel(model, verbose=0)

    with _Float32Recorder() as recorder:
        mask = solvent.get_solvent_mask()

    assert recorder.calls == []
    assert 0 < int(mask.sum()) < mask.numel()
