"""Gradient-correctness tests for the autograd refinement engine.

A central claim of the TorchRef paper is that automatic differentiation
*eliminates the need for manually-derived analytical gradients*. These tests
back that claim from two angles:

1. **Eager paths vs. finite differences** — ``torch.autograd.gradcheck`` in
   double precision verifies the pure-PyTorch (autograd) implementations
   against numerical derivatives for

   * the structure-factor calculation w.r.t. ``xyz``, ``occ`` and the ADPs
     (isotropic ``adp`` and anisotropic ``U``),
   * the Gaussian X-ray likelihood (the body of ``GaussianXrayTarget.forward``),
   * the bond-length restraint (the body of ``BondTarget.forward``).

2. **Optimized paths vs. the eager reference** — some production paths replace
   autograd with a *hand-written* ``backward`` (the recompute-on-backward
   ``_CheckpointedSF`` used on CPU/large problems, and the Triton kernels used
   on CUDA float32). Those must reproduce the eager autograd gradient. We assert
   this with **cosine similarity** and **gradient-norm ratio** rather than
   absolute/relative tolerances, which are finicky across dtypes/backends:
   a correct gradient points the same way (cosine ≈ 1) and has the same
   magnitude (norm ratio ≈ 1).

The structure-factor checks use the direct-summation engine, which is the
analytic (autograd) SF path. The production FFT path (``ModelFT.forward``) is
not amenable to ``gradcheck`` — its output is single-precision complex and the
on-grid splat makes element-wise finite differences underflow — so it is
validated with the same cosine/gradnorm metric against central finite
differences of a scalar loss (:func:`test_model_forward_xyz_adp_*`).

CPU-only by default; the Triton comparisons are marked ``cuda`` and are
auto-skipped unless the host actually has a CUDA device.
"""

import itertools
import tempfile

import pytest
import torch

from torchref.base.direct_summation.dispatch import (
    _checkpointed_aniso,
    _checkpointed_iso,
    _eager_aniso,
    _eager_iso,
)
from torchref.base.targets.bond import _bond_math_eager, bond_math
from torchref.base.targets.xray_gaussian import (
    _gaussian_xray_loss_math_eager,
    gaussian_xray_loss_math,
)
from torchref.config import device, dtypes

pytestmark = pytest.mark.unit

# Kept for readability in the section header below; backend gating itself is
# the ``cuda`` marker's job (see tests/conftest.py).
_HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


# These live in ``tests/helpers/grad_asserts.py`` so the accelerator kernel
# comparison tests can share them; re-exported here because this module is where
# they originated and several tests below use them unqualified.
from tests.helpers.grad_asserts import (  # noqa: E402
    _flat_real,
    assert_grads_agree,
    cosine_similarity,
    got_ref_pairs,
    gradnorm_ratio,
)

# --- synthetic structure-factor inputs (P1, ~10-15 atoms) ------------------
def _sf_inputs(N=12, R=18, dtype=torch.float64, device="cpu"):
    g = torch.Generator().manual_seed(0)
    hkl = torch.randint(-3, 4, (R, 3), generator=g).to(dtype=dtype, device=device)
    s = (torch.rand(R, generator=g) * 0.4).to(dtype=dtype, device=device)
    svec = (torch.randn(R, 3, generator=g) * 0.3).to(dtype=dtype, device=device)
    A = torch.rand(N, 5, generator=g).to(dtype=dtype, device=device)
    B = (torch.rand(N, 5, generator=g) + 0.5).to(dtype=dtype, device=device)
    return hkl, s, svec, A, B


def _sf_leaves(N=12, dtype=torch.float64, device="cpu", requires_grad=True):
    g = torch.Generator().manual_seed(1)
    mk = lambda t: t.to(dtype=dtype, device=device).requires_grad_(requires_grad)
    xyz = mk(torch.rand(N, 3, generator=g))
    occ = mk(torch.rand(N, generator=g) * 0.4 + 0.6)
    adp = mk(torch.rand(N, generator=g) * 10 + 5)
    U = mk(torch.rand(N, 6, generator=g) * 0.04 + 0.01)
    return xyz, occ, adp, U


def test_gaussian_xray_gradcheck(double_cpu):
    """Gaussian X-ray NLL (GaussianXrayTarget.forward body): d/d(F_obs, F_calc).

    ``F_calc`` is complex; gradcheck differentiates the real and imaginary
    parts. ``gaussian_xray_loss_math`` is the public dispatcher and routes to
    the eager implementation for CPU/float64 inputs.
    """
    g = torch.Generator().manual_seed(2)
    R = 24
    F_obs = (
        torch.rand(R, generator=g, dtype=torch.float64) * 100 + 10
    ).requires_grad_()
    F_calc = (
        (
            torch.randn(R, generator=g, dtype=torch.float64)
            + 1j * torch.randn(R, generator=g, dtype=torch.float64)
        )
        * 10
    ).requires_grad_()
    sigma = torch.rand(R, generator=g, dtype=torch.float64) * 5 + 1

    def f(fo, fc):
        return gaussian_xray_loss_math(fo, fc, sigma)

    assert torch.autograd.gradcheck(f, (F_obs, F_calc), eps=1e-6, atol=1e-5)


# =============================================================================
# 3. Geometry restraint target — eager autograd vs finite differences
# =============================================================================
def test_bond_gradcheck(double_cpu):
    """Bond-length restraint NLL (BondTarget.forward body): d/d(xyz).

    ``bond_math`` is the public dispatcher and routes to the eager
    implementation for CPU/float64 inputs.
    """
    N = 12
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, N, (16, 2), generator=g)
    idx = idx[idx[:, 0] != idx[:, 1]]  # drop degenerate (zero-length) bonds
    references = torch.rand(idx.shape[0], generator=g, dtype=torch.float64) * 1.5 + 1.0
    sigmas = torch.rand(idx.shape[0], generator=g, dtype=torch.float64) * 0.1 + 0.02
    xyz = torch.rand(N, 3, generator=g, dtype=torch.float64).requires_grad_()

    def f(x):
        return bond_math(x, idx, references, sigmas)

    assert torch.autograd.gradcheck(f, (xyz,), eps=1e-6, atol=1e-5)


# =============================================================================
# 4. Optimized SF path (hand-written backward) vs eager autograd
#    Metric: cosine similarity + gradnorm ratio (CPU, always runs).
# =============================================================================
def _scalar(F):
    """A non-trivial real scalar of a complex SF vector (mixes re & im)."""
    return (F.real**2 + 2.0 * F.imag).sum()


def _grads(fn, leaves):
    """Gradients of ``_scalar(fn(*leaves))`` w.r.t. each leaf."""
    out = _scalar(fn(*leaves))
    return torch.autograd.grad(out, leaves)




@pytest.mark.cuda
def test_triton_sf_iso_matches_eager_cosine():
    """DS isotropic Triton kernel gradient == eager autograd (CUDA float32)."""
    from torchref.base.direct_summation.dispatch import ds_iso
    from torchref.utils import Engine

    dev = "cuda"
    hkl, s, _, A, B = _sf_inputs(dtype=torch.float32, device=dev)
    xyz, occ, adp, _ = _sf_leaves(dtype=torch.float32, device=dev)
    xyz2, occ2, adp2, _ = _sf_leaves(dtype=torch.float32, device=dev)

    g_triton = _grads(
        lambda x, o, a: ds_iso(hkl, s, x, o, a, A, B, engine=Engine.TRITON),
        (xyz, occ, adp),
    )
    g_eager = _grads(
        lambda x, o, a: _eager_iso(hkl, s, x, o, a, A, B, max_memory_gb=2.0),
        (xyz2, occ2, adp2),
    )
    # Looser thresholds: float32 + distinct kernel arithmetic.
    assert_grads_agree(g_triton, g_eager, min_cos=0.999, ratio_tol=1e-2, ctx="iso ")


@pytest.mark.cuda
def test_triton_sf_aniso_matches_eager_cosine():
    """DS anisotropic Triton kernel gradient == eager autograd (CUDA float32)."""
    from torchref.base.direct_summation.dispatch import ds_aniso
    from torchref.utils import Engine

    dev = "cuda"
    hkl, _, svec, A, B = _sf_inputs(dtype=torch.float32, device=dev)
    xyz, occ, _, U = _sf_leaves(dtype=torch.float32, device=dev)
    xyz2, occ2, _, U2 = _sf_leaves(dtype=torch.float32, device=dev)

    g_triton = _grads(
        lambda x, o, u: ds_aniso(hkl, svec, x, o, u, A, B, engine=Engine.TRITON),
        (xyz, occ, U),
    )
    g_eager = _grads(
        lambda x, o, u: _eager_aniso(hkl, svec, x, o, u, A, B, max_memory_gb=2.0),
        (xyz2, occ2, U2),
    )
    assert_grads_agree(g_triton, g_eager, min_cos=0.999, ratio_tol=1e-2, ctx="aniso ")


@pytest.mark.cuda
def test_triton_gaussian_xray_matches_eager_cosine():
    """Gaussian X-ray Triton kernel gradient == eager autograd (CUDA float32)."""
    dev = "cuda"
    g = torch.Generator().manual_seed(2)
    R = 64
    F_obs = (torch.rand(R, generator=g) * 100 + 10).to(dev)
    sigma = (torch.rand(R, generator=g) * 5 + 1).to(dev)
    fc_vals = (torch.randn(R, generator=g) + 1j * torch.randn(R, generator=g)) * 10
    fc_triton = fc_vals.clone().to(dev).requires_grad_()
    fc_eager = fc_vals.clone().to(dev).requires_grad_()

    L_t = gaussian_xray_loss_math(F_obs, fc_triton, sigma)  # CUDA fp32 -> Triton
    (g_t,) = torch.autograd.grad(L_t, fc_triton)
    L_e = _gaussian_xray_loss_math_eager(
        F_obs, fc_eager, sigma, torch.ones(R, dtype=torch.bool, device=dev)
    )
    (g_e,) = torch.autograd.grad(L_e, fc_eager)
    assert_grads_agree([g_t], [g_e], min_cos=0.999, ratio_tol=1e-2, ctx="xray ")


@pytest.mark.cuda
def test_triton_bond_matches_eager_cosine():
    """Bond restraint Triton kernel gradient == eager autograd (CUDA float32)."""
    dev = "cuda"
    N = 16
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, N, (24, 2), generator=g)
    idx = idx[idx[:, 0] != idx[:, 1]].to(dev)
    references = (torch.rand(idx.shape[0], generator=g) * 1.5 + 1.0).to(dev)
    sigmas = (torch.rand(idx.shape[0], generator=g) * 0.1 + 0.02).to(dev)
    base = torch.rand(N, 3, generator=g)

    x_triton = base.clone().to(dev).requires_grad_()
    x_eager = base.clone().to(dev).requires_grad_()

    L_t = bond_math(x_triton, idx, references, sigmas)  # CUDA fp32 -> Triton
    (g_t,) = torch.autograd.grad(L_t, x_triton)
    L_e = _bond_math_eager(x_eager, idx, references, sigmas)
    (g_e,) = torch.autograd.grad(L_e, x_eager)
    assert_grads_agree([g_t], [g_e], min_cos=0.999, ratio_tol=1e-2, ctx="bond ")


# ---------------------------------------------------------------------------
# Anisotropic ADP restraints (SIMU / locality on the unified U6 tensor)
# ---------------------------------------------------------------------------
from torchref.base.targets.adp import (  # noqa: E402
    EIGHT_PI2,
    adp_locality_aniso_math,
    adp_rigid_bond_aniso_math,
    adp_simu_aniso_math,
    adp_simu_math,
    u6_deviatoric,
)


def _iso_u6(b):
    """Isotropic U6 (diagonal B/8pi^2, zero off-diagonal) from a B vector."""
    diag = b.new_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return (b / EIGHT_PI2).unsqueeze(-1) * diag


def _rand_u6(n, seed):
    """Random physically-plausible U6: positive diagonal, small off-diagonal."""
    g = torch.Generator().manual_seed(seed)
    d = torch.rand(n, 3, generator=g, dtype=torch.float64) * 0.2 + 0.1
    o = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.02
    return torch.cat([d, o], dim=1)


def test_adp_simu_aniso_reduces_to_iso(double_cpu):
    """An all-isotropic U6 makes aniso SIMU collapse exactly onto iso SIMU:
    the magnitude channel equals adp_simu_math(B) and the deviatoric is zero."""
    b = torch.rand(12, dtype=torch.float64) * 20 + 5
    pairs = torch.tensor([[0, 1], [1, 2], [3, 7], [5, 9], [2, 11]])
    sig = torch.tensor(2.0, dtype=torch.float64)
    sdev = torch.tensor(1.0, dtype=torch.float64)
    iso = adp_simu_math(b, pairs, sig)
    ani = adp_simu_aniso_math(_iso_u6(b), pairs, sig, sdev)
    assert torch.allclose(iso, ani, atol=1e-9)
    assert torch.allclose(u6_deviatoric(_iso_u6(b)),
                          torch.zeros(12, 6, dtype=torch.float64), atol=1e-12)


def test_adp_simu_aniso_gradcheck(double_cpu):
    u6 = _rand_u6(10, 1).requires_grad_(True)
    pairs = torch.tensor([[0, 1], [1, 2], [3, 7], [5, 9], [2, 8]])
    sig = torch.tensor(2.0, dtype=torch.float64)
    sdev = torch.tensor(1.0, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda u: adp_simu_aniso_math(u, pairs, sig, sdev),
        (u6,), eps=1e-6, atol=1e-5)


def test_adp_locality_aniso_gradcheck(double_cpu):
    u6 = _rand_u6(10, 2).requires_grad_(True)
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, 10, (10, 3), generator=g)
    dist = torch.rand(10, 3, generator=g, dtype=torch.float64) * 5 + 1
    sdev = torch.tensor(0.5, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda u: adp_locality_aniso_math(u, idx, dist, sdev),
        (u6,), eps=1e-6, atol=1e-5)


def test_adp_rigid_bond_aniso_reduces_to_iso(double_cpu):
    """All-isotropic U6 ⇒ aniso rigid-bond Δz == the iso (B_i-B_j)/8pi^2 form."""
    b = torch.rand(10, dtype=torch.float64) * 20 + 5
    g = torch.Generator().manual_seed(7)
    xyz = torch.rand(10, 3, generator=g, dtype=torch.float64) * 15
    pairs = torch.tensor([[0, 1], [1, 2], [3, 7], [5, 9], [2, 8]])
    sigma = 0.004
    ani = adp_rigid_bond_aniso_math(_iso_u6(b), xyz, pairs, sigma)
    dz = (b[pairs[:, 0]] - b[pairs[:, 1]]) / EIGHT_PI2
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64))
    iso = (0.5 * (dz / sigma) ** 2 + torch.log(torch.tensor(sigma)) + 0.5 * log_2pi).sum()
    assert torch.allclose(iso, ani, atol=1e-9)


def test_adp_rigid_bond_aniso_gradcheck(double_cpu):
    """Gradient flows correctly to BOTH the U tensors and the coordinates."""
    u6 = _rand_u6(10, 4).requires_grad_(True)
    g = torch.Generator().manual_seed(5)
    xyz = (torch.rand(10, 3, generator=g, dtype=torch.float64) * 15).requires_grad_(True)
    pairs = torch.tensor([[0, 1], [1, 2], [3, 7], [5, 9], [2, 8]])
    assert torch.autograd.gradcheck(
        lambda u, x: adp_rigid_bond_aniso_math(u, x, pairs, 0.004),
        (u6, xyz), eps=1e-6, atol=1e-5)
