"""Triton-vs-eager equivalence for direct-summation structure factors.

The custom Triton kernels (float32) are compared against the checkpointed
eager backend evaluated in float64 (downcast for comparison), for both the
isolated P1 kernels and the full ``SfDS`` symmetry loop.

Markers: ``@pytest.mark.cuda`` (auto-skipped without CUDA) and
``@pytest.mark.integration``. Requires a CUDA device.
"""

import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.integration]

_RTOL = 1e-3


def _rel(a, b):
    num = (a - b.to(a.dtype)).abs().max()
    den = b.abs().max().to(num.dtype) + 1e-6
    return (num / den).item()


@pytest.fixture
def cuda():
    # No availability check: the module-level ``cuda`` marker is the only gate
    # (see conftest.pytest_collection_modifyitems). A second check here could
    # only turn a forgotten marker into a silent pass.
    return torch.device("cuda", 0)


def _asym_loss(F, wr, wi):
    return (wr * F.real).sum() + (wi * F.imag).sum()


def test_iso_triton_vs_eager(cuda):
    from torchref.base.direct_summation.triton_ds import ds_iso_triton
    from torchref.base.direct_summation import dispatch as D

    torch.manual_seed(0)
    R, N = 400, 23
    hkl = torch.randint(-6, 7, (R, 3), device=cuda).float()
    s = torch.rand(R, device=cuda) * 0.6
    A = torch.rand(N, 5, device=cuda)
    B = torch.rand(N, 5, device=cuda) + 0.5
    wr = torch.randn(R, device=cuda)
    wi = torch.randn(R, device=cuda)  # asymmetric -> exposes a wrong imag sign

    xyz0 = torch.rand(N, 3, device=cuda, dtype=torch.float64)
    occ0 = torch.rand(N, device=cuda, dtype=torch.float64) * 0.4 + 0.6
    adp0 = torch.rand(N, device=cuda, dtype=torch.float64) * 10 + 5

    xt, ot, at = (t.float().detach().requires_grad_() for t in (xyz0, occ0, adp0))
    Ft = ds_iso_triton(hkl, s, xt, ot, at, A, B)
    _asym_loss(Ft, wr, wi).backward()

    xr, orf, arf = (t.detach().requires_grad_() for t in (xyz0, occ0, adp0))
    Fr = D._checkpointed_iso(hkl, s, xr, orf, arf, A.double(), B.double(), max_memory_gb=None)
    _asym_loss(Fr, wr.double(), wi.double()).backward()

    assert _rel(Ft, Fr) < _RTOL
    assert _rel(xt.grad, xr.grad) < _RTOL
    assert _rel(ot.grad, orf.grad) < _RTOL
    assert _rel(at.grad, arf.grad) < _RTOL


def test_aniso_triton_vs_eager(cuda):
    from torchref.base.direct_summation.triton_ds import ds_aniso_triton
    from torchref.base.direct_summation import dispatch as D

    torch.manual_seed(1)
    R, N = 400, 23
    hkl = torch.randint(-6, 7, (R, 3), device=cuda).float()
    svec = torch.randn(R, 3, device=cuda) * 0.3
    A = torch.rand(N, 5, device=cuda)
    B = torch.rand(N, 5, device=cuda) + 0.5
    wr = torch.randn(R, device=cuda)
    wi = torch.randn(R, device=cuda)

    xyz0 = torch.rand(N, 3, device=cuda, dtype=torch.float64)
    occ0 = torch.rand(N, device=cuda, dtype=torch.float64) * 0.4 + 0.6
    U0 = torch.rand(N, 6, device=cuda, dtype=torch.float64) * 0.04 + 0.01

    xt, ot, Ut = (t.float().detach().requires_grad_() for t in (xyz0, occ0, U0))
    Ft = ds_aniso_triton(hkl, svec, xt, ot, Ut, A, B)
    _asym_loss(Ft, wr, wi).backward()

    xr, orf, Ur = (t.detach().requires_grad_() for t in (xyz0, occ0, U0))
    Fr = D._checkpointed_aniso(hkl, svec, xr, orf, Ur, A.double(), B.double(), max_memory_gb=None)
    _asym_loss(Fr, wr.double(), wi.double()).backward()

    assert _rel(Ft, Fr) < _RTOL
    assert _rel(xt.grad, xr.grad) < _RTOL
    assert _rel(ot.grad, orf.grad) < _RTOL
    assert _rel(Ut.grad, Ur.grad) < _RTOL


def test_sfds_engine_toggle_end_to_end(cuda):
    """Full SfDS symmetry loop: Engine.TRITON vs Engine.EAGER agree."""
    from torchref.symmetry import Cell
    from torchref.model.sf_ds import SfDS
    from torchref.base.direct_summation import Engine

    torch.manual_seed(2)
    cell = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
    N, R = 30, 200
    hkl = torch.randint(-6, 7, (R, 3), device=cuda).float()
    A = torch.rand(N, 5, device=cuda)
    B = torch.rand(N, 5, device=cuda) + 0.5
    xyz0 = torch.rand(N, 3, device=cuda) * 20
    occ0 = torch.rand(N, device=cuda) * 0.4 + 0.6
    adp0 = torch.rand(N, device=cuda) * 10 + 5
    wr = torch.randn(R, device=cuda)
    wi = torch.randn(R, device=cuda)

    def run(engine):
        sf = SfDS(cell, spacegroup="P212121", engine=engine, max_memory_gb=2.0)
        xyz = xyz0.clone().requires_grad_()
        occ = occ0.clone().requires_grad_()
        adp = adp0.clone().requires_grad_()
        F, _ = sf.compute_structure_factors(hkl, xyz, adp, occ, A, B)
        _asym_loss(F, wr, wi).backward()
        return F.detach(), xyz.grad, occ.grad, adp.grad

    Ft = run(Engine.TRITON)
    Fe = run(Engine.EAGER)
    for t, e in zip(Ft, Fe):
        assert _rel(t, e) < _RTOL
