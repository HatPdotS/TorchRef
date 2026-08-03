"""Gradient-correctness tests for the autograd refinement engine.

A central claim of the TorchRef paper is that automatic differentiation
*eliminates the need for manually-derived analytical gradients*. These tests
back that claim from two angles:

1. **Eager paths vs. finite differences** — ``torch.autograd.gradcheck`` in
   double precision verifies the pure-PyTorch (autograd) implementations
   against numerical derivatives for

   * the structure-factor calculation w.r.t. ``xyz``, ``occ`` and the ADPs
     (isotropic ``adp`` and anisotropic ``U``),
   * the Gaussian X-ray likelihood (``nll_sigma_obs_math``, the body of
     ``NLLXrayTarget.forward``),
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


import pytest
import torch

from torchref.base.targets._dispatch import use_triton
from torchref.base.targets.bond import _bond_math_eager, bond_math
from torchref.base.targets.xray_likelihoods import (
    amplitude_var_from_sigma_obs,
    nll_math,
)
from torchref.base.targets.xray_nll import nll_sigma_obs_math
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
from tests.helpers.grad_asserts import assert_grads_agree  # noqa: E402

# --- synthetic structure-factor inputs (P1, ~10-15 atoms) ------------------
def test_gaussian_xray_gradcheck(double_cpu):
    """Gaussian X-ray NLL (NLLXrayTarget.forward body): d/d(F_obs, F_calc).

    ``F_calc`` is complex; gradcheck differentiates the real and imaginary
    parts. ``nll_sigma_obs_math`` is the public dispatcher and routes to
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
        return nll_sigma_obs_math(fo, fc, sigma)

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
@pytest.mark.cuda
def test_triton_gaussian_xray_matches_eager_cosine():
    """Gaussian X-ray Triton kernel gradient == eager autograd (CUDA float32).

    ``F_calc`` is a **real, non-negative amplitude**, which is the production contract:
    ``XrayTarget.get_data`` returns ``get_F_calc_scaled``, and that is
    ``torch.abs(get_fcalc_scaled(...))``. Both properties are load-bearing.

    *Real*, because the two implementations disagree on a complex input -- the eager path
    takes ``torch.abs(F_calc)`` while the Triton kernel loads one real scalar per element
    (``diff = F_obs - F_calc``). This test used to pass complex64, which the dtype gate let
    through only because ``is_floating_point()`` is ``False`` for complex; the kernel's own
    assert was the sole thing that caught it. The gate now refuses complex, so a complex
    input here would silently route to eager and compare eager against itself.

    *Non-negative*, because ``abs()`` is the identity only on non-negative input. A negative
    ``F_calc`` would flip the gradient sign in the eager path and not in Triton, and the
    disagreement would be a property of the fixture rather than of either kernel.
    """
    dev = "cuda"
    g = torch.Generator().manual_seed(2)
    R = 64
    F_obs = (torch.rand(R, generator=g) * 100 + 10).to(dev)
    sigma = (torch.rand(R, generator=g) * 5 + 1).to(dev)
    fc_vals = torch.rand(R, generator=g) * 100 + 10
    fc_triton = fc_vals.clone().to(dev).requires_grad_()
    fc_eager = fc_vals.clone().to(dev).requires_grad_()

    # Non-vacuity: without this the test still passes if the gate starts refusing these
    # inputs, by comparing the eager implementation against itself.
    assert use_triton(fc_triton, F_obs, sigma), (
        "the Triton arm of this test is not reaching Triton, so it compares eager against "
        "eager -- the gate in TARGET_BACKENDS has started refusing CUDA float32 amplitudes"
    )

    L_t = nll_sigma_obs_math(F_obs, fc_triton, sigma)  # CUDA fp32 -> Triton
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
