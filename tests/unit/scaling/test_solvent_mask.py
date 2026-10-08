"""The bulk-solvent mask's voxel index arithmetic."""

import pytest
import torch

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
