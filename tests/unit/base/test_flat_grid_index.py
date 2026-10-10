"""Flat reciprocal-grid indices stay exact on grids larger than int32 can address.

``h*Ny*Nz + k*Nz + l`` is a packed key: formed in the default int32 it wraps past
2**31 voxels and addresses aliased voxels, so it is built in int64 whatever dtype the
Miller indices arrive in.
"""

import pytest
import torch

from torchref.base.reciprocal.symmetry import _equiv_hkls_to_flat_indices
from torchref.config import get_int_dtype

pytestmark = pytest.mark.unit


def test_flat_indices_are_exact_past_int32():
    n = 2048  # 2048**3 voxels; the helper never allocates the grid
    hkl = torch.tensor([[[1000, -3, 7], [-1, 0, 2047]]], dtype=get_int_dtype())
    flat = _equiv_hkls_to_flat_indices(hkl, n, n, n)
    expected = [(h % n) * n * n + (k % n) * n + (l % n) for h, k, l in hkl[0].tolist()]
    assert flat.dtype == torch.int64
    assert flat.tolist() == expected
