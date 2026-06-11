"""Unit tests for the direct-summation dispatch / checkpointed backend.

CPU-only (no GPU required). Verifies that the recompute-on-backward
checkpointed backend matches the eager-autograd reference to machine
precision, that reflection-chunking is exact, that gradients pass
``gradcheck``, and that the engine guards behave.
"""

import pytest
import torch

from torchref.base.direct_summation import Engine, ds_aniso, ds_iso
from torchref.base.direct_summation import dispatch as D
from torchref.base.direct_summation.anisotropic import aniso_structure_factor_torched
from torchref.base.direct_summation.isotropic import iso_structure_factor_torched
from torchref.config import dtypes

pytestmark = pytest.mark.unit

# The eager reference functions internally cast hkl to ``dtypes.float`` so the
# eager-parity comparisons run in the configured float dtype (float32 by
# default). gradcheck, which never calls the eager fn, runs in float64.
_F = dtypes.float
_eager_atol = 1e-4 if _F == torch.float32 else 1e-10


def _p1(c):
    return c.unsqueeze(2)


def _inputs(N=4, R=7, seed=0, dtype=None):
    dtype = dtype or _F
    torch.manual_seed(seed)
    hkl = torch.randint(-3, 4, (R, 3)).to(dtype)
    s = torch.rand(R, dtype=dtype) * 0.5
    svec = torch.randn(R, 3, dtype=dtype) * 0.3
    A = torch.rand(N, 5, dtype=dtype)
    B = torch.rand(N, 5, dtype=dtype) + 0.5
    return hkl, s, svec, A, B


def _leaves(N=4, seed=1, dtype=None):
    dtype = dtype or _F
    torch.manual_seed(seed)
    xyz = torch.rand(N, 3, dtype=dtype, requires_grad=True)
    occ = (torch.rand(N, dtype=dtype) * 0.4 + 0.6).requires_grad_()
    adp = (torch.rand(N, dtype=dtype) * 10 + 5).requires_grad_()
    U = (torch.rand(N, 6, dtype=dtype) * 0.04 + 0.01).requires_grad_()
    return xyz, occ, adp, U


def test_checkpointed_iso_matches_eager():
    hkl, s, _, A, B = _inputs()
    xyz, occ, adp, _ = _leaves()
    F = D._checkpointed_iso(hkl, s, xyz, occ, adp, A, B, max_memory_gb=None)
    ((F.real**2 + 2 * F.imag).sum()).backward()
    gck = (xyz.grad.clone(), occ.grad.clone(), adp.grad.clone())

    xyz2, occ2, adp2, _ = _leaves()
    Fe = iso_structure_factor_torched(
        hkl=hkl, s=s, xyz_fractional=xyz2, occ=occ2, scattering_factors=None,
        adp=adp2, spacegroup=_p1, max_memory_gb=2.0, A=A, B_coeff=B,
    )
    ((Fe.real**2 + 2 * Fe.imag).sum()).backward()
    assert torch.allclose(F, Fe, atol=_eager_atol, rtol=1e-3)
    for g, ge in zip(gck, (xyz2.grad, occ2.grad, adp2.grad)):
        assert torch.allclose(g, ge, atol=_eager_atol, rtol=1e-3)


def test_checkpointed_aniso_matches_eager():
    hkl, _, svec, A, B = _inputs()
    xyz, occ, _, U = _leaves()
    F = D._checkpointed_aniso(hkl, svec, xyz, occ, U, A, B, max_memory_gb=None)
    ((F.real**2 + 2 * F.imag).sum()).backward()
    gck = (xyz.grad.clone(), occ.grad.clone(), U.grad.clone())

    xyz2, occ2, _, U2 = _leaves()
    Fe = aniso_structure_factor_torched(
        hkl=hkl, s_vector=svec, xyz_fractional=xyz2, occ=occ2,
        scattering_factors=None, U=U2, spacegroup=_p1, max_memory_gb=2.0,
        A=A, B_coeff=B,
    )
    ((Fe.real**2 + 2 * Fe.imag).sum()).backward()
    assert torch.allclose(F, Fe, atol=_eager_atol, rtol=1e-3)
    for g, ge in zip(gck, (xyz2.grad, occ2.grad, U2.grad)):
        assert torch.allclose(g, ge, atol=_eager_atol, rtol=1e-3)


def test_checkpointed_chunking_is_exact():
    hkl, s, _, A, B = _inputs(R=11, dtype=torch.float64)
    xyz, occ, adp, _ = _leaves(dtype=torch.float64)
    F_full = D._checkpointed_iso(hkl, s, xyz, occ, adp, A, B, max_memory_gb=None)
    F_chunk = D._checkpointed_iso(hkl, s, xyz, occ, adp, A, B, max_memory_gb=1e-7)
    assert torch.equal(F_full, F_chunk)


def test_checkpointed_iso_gradcheck():
    hkl, s, _, A, B = _inputs(dtype=torch.float64)
    xyz, occ, adp, _ = _leaves(dtype=torch.float64)

    def f(x, o, a):
        return D._checkpointed_iso(hkl, s, x, o, a, A, B, max_memory_gb=1e-7)

    assert torch.autograd.gradcheck(f, (xyz, occ, adp), eps=1e-6, atol=1e-5)


def test_empty_atoms_returns_zeros():
    hkl, s, _, _, _ = _inputs()
    empty = torch.zeros(0, 3, dtype=torch.float64)
    z = torch.zeros(0, dtype=torch.float64)
    z5 = torch.zeros(0, 5, dtype=torch.float64)
    F = ds_iso(hkl, s, empty, z, z, z5, z5, engine=Engine.AUTO)
    assert F.shape == (hkl.shape[0],)
    assert F.abs().sum().item() == 0.0


def test_explicit_triton_engine_rejects_cpu():
    hkl, s, _, A, B = _inputs()
    xyz, occ, adp, _ = _leaves()
    with pytest.raises(RuntimeError):
        ds_iso(hkl, s, xyz, occ, adp, A, B, engine=Engine.TRITON)
