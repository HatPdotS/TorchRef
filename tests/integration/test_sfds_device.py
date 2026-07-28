"""Regression test: SfDS must reconcile the input hkl with its own device.

SfDS.compute_structure_factors allocates the accumulator on self.device but
derives equiv_hkls/phases/s_vectors from the input hkl's device. A caller
passing hkl on a different device than the module used to hit a cross-device
add. The fix normalizes hkl onto the module device (via resolve_device). See
TORCHREF_AUDIT.md cluster (sf_ds device).
"""

import pytest
import torch

from torchref.model.sf_ds import SfDS
from torchref.symmetry import Cell

_CELL = [50.0, 60.0, 70.0, 90.0, 90.0, 90.0]


def _atoms(device, n=8):
    torch.manual_seed(0)
    A = torch.rand(n, 5, device=device)
    B = torch.rand(n, 5, device=device) + 0.5
    xyz = torch.rand(n, 3, device=device) * 20
    occ = torch.rand(n, device=device) * 0.4 + 0.6
    adp = torch.rand(n, device=device) * 10 + 5
    return xyz, adp, occ, A, B


@pytest.mark.integration
def test_sfds_same_device_cpu():
    """Sanity: hkl already on the module device works and stays on it."""
    cell = Cell(_CELL, device="cpu")
    sf = SfDS(cell, spacegroup="P212121").to("cpu")
    xyz, adp, occ, A, B = _atoms("cpu")
    hkl = torch.randint(-6, 7, (50, 3)).float()
    F, _ = sf.compute_structure_factors(hkl, xyz, adp, occ, A, B)
    assert F.device.type == "cpu"
    assert torch.isfinite(F.real).all() and torch.isfinite(F.imag).all()


@pytest.mark.cuda
@pytest.mark.integration
def test_sfds_hkl_on_different_device():
    """hkl on CPU while the module + atoms are on CUDA must not crash."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cuda = torch.device("cuda")
    cell = Cell(_CELL, device=cuda)
    sf = SfDS(cell, spacegroup="P212121").to(cuda)
    xyz, adp, occ, A, B = _atoms(cuda)

    hkl_cpu = torch.randint(-6, 7, (50, 3)).float()  # deliberately on CPU
    assert hkl_cpu.device.type == "cpu"

    # Pre-fix: cross-device add in the symmetry accumulation raised here.
    F, _ = sf.compute_structure_factors(hkl_cpu, xyz, adp, occ, A, B)
    assert F.device.type == "cuda"
    assert torch.isfinite(F.real).all() and torch.isfinite(F.imag).all()
