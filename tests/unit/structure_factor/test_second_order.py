"""Second derivatives: FD -> direct summation -> the FFT/splat route.

Why direct summation and not finite differences for the map route
----------------------------------------------------------------
The two tests this module replaces --
``test_kernel_fixes.py::test_cpu_plain_scatter_second_derivative_correct`` and
``::test_cpu_cpp_scatter_second_derivative_correct`` -- compared the map route's HVP
against central differences *of the map route*, and were failing.

The reference was the problem, not the kernel -- but not for the reason an earlier draft
of this module claimed. That draft argued finite differences *diverge* on a truncated
density, because the cull surface moves with the atom and each voxel crossing contributes
a fixed jump. Measured, that does not happen: FD converges normally, reaching rel 5.2e-04
against the map route's own autograd HVP. At a 3 sigma cutoff the surface density is ~1%
of peak, so the jumps are too small to matter.

The actual problem is that FD asks the wrong question. It differentiates the *same*
discretized function the kernel does, so it can only confirm that autograd correctly
differentiates whatever the kernel computes. Measured here: the map route's HVP agrees
with itself to 5e-04 while sitting 2.3e-02 from the analytic answer. An FD-based gate at
5e-03 -- what the replaced tests used -- passes comfortably while the derivative under
test is 2.3e-02 wrong.

Direct summation has no grid and no truncation, so FD *is* a valid reference for it
(measured 1.19e-10), and it is then an independent reference for the map route.
:func:`test_finite_differences_cannot_detect_map_route_error` measures all of this, so the
reasoning is pinned by numbers rather than asserted in prose.

The regression the original two tests guarded is preserved: a graph-less C++ backward
that produced a measured cosine of ~0.57 while first-order gradients stayed correct.
That is still caught here, against a reference that converges.
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers.grad_asserts import cosine_similarity, hvp, hvp_central_fd, rel_error
from torchref.base.direct_summation.dispatch import _eager_aniso, _eager_iso
from torchref.utils import Engine, use_engine

from . import (
    ATOL_GRADCHECK,
    COS_MIN,
    EPS_GRADCHECK,
    RTOL_DS_HVP_VS_FD,
    RTOL_GRADCHECK,
    RTOL_HVP,
)
from . import helpers as H
from .conftest import DEVICE_DTYPE_KERNELS

pytestmark = pytest.mark.unit

_DTYPES = [torch.float32, torch.float64]  # float32 first: production dtype


# ---------------------------------------------------------------------------
# 1. The oracle is sound at second order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_ds_gradgradcheck(scene_small, kind):
    """``gradgradcheck`` on the eager DS path -- the first in this repo.

    ``grep gradgradcheck`` over ``torchref/`` and ``tests/`` previously returned nothing,
    so no second derivative anywhere had been checked against finite differences at the
    autograd level. Everything second-order was either cross-backend parity or an HVP
    against an FD reference on the truncated map route.
    """
    s = scene_small
    if kind == "iso":
        fn = lambda x, o, a: _eager_iso(s.hkl, s.s, x, o, a, s.A, s.B, None)  # noqa: E731
        third = s.adp
    else:
        fn = lambda x, o, u: _eager_aniso(  # noqa: E731
            s.hkl, s.s_vec, x, o, u, s.A, s.B, None
        )
        third = s.u6

    args = tuple(t.clone().requires_grad_(True) for t in (s.xyz_frac, s.occ, third))
    assert torch.autograd.gradgradcheck(
        fn, args, eps=EPS_GRADCHECK, atol=ATOL_GRADCHECK, rtol=RTOL_GRADCHECK
    )


@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_ds_hvp_matches_finite_differences(scene_small, kind):
    """The oracle's HVP against central differences of its own gradient.

    Complements ``gradgradcheck``: that checks the full double-backward machinery
    elementwise on a tiny input, this checks the specific contraction the map-route tests
    use, so the two links are compared like for like.
    """
    s = scene_small
    fn = H.ds_aniso_oracle if kind == "aniso" else H.ds_iso_oracle
    third = s.u6 if kind == "aniso" else s.adp
    with torch.no_grad():
        obs = H.synthetic_obs(fn(s, s.xyz, s.occ, third))
    loss = lambda x: H.ls_target(fn(s, x, s.occ, third), obs)  # noqa: E731

    g = torch.Generator().manual_seed(17)
    v = torch.randn(s.xyz.shape, generator=g, dtype=s.xyz.dtype)
    v /= v.norm()

    got = hvp(loss, s.xyz, v)
    ref = hvp_central_fd(loss, s.xyz, v, eps=1e-6)
    rel, cos = rel_error(got, ref), cosine_similarity(got, ref)
    print(f"\n  {kind}: DS HVP vs central FD -- rel {rel:.3e}  cos {cos:.10f}")
    assert rel < RTOL_DS_HVP_VS_FD
    assert cos > 1.0 - 1e-8


# ---------------------------------------------------------------------------
# 2. The first-order-only contract of the public DS API, made explicit
# ---------------------------------------------------------------------------
def test_public_ds_api_raises_on_double_backward(scene_small):
    """``ds_iso`` and ``SfDS`` are first-order only. Assert it, so it is documented.

    ``_CheckpointedSF.backward`` calls ``torch.autograd.grad`` without
    ``create_graph=True`` on detached copies
    (``torchref/base/direct_summation/dispatch.py:197``), so the returned gradients carry
    no graph. It is *not* decorated ``@once_differentiable``, so nothing warns you; the
    failure surfaces only when a second derivative is requested.

    It does at least fail loudly rather than returning zeros -- verified: it raises
    ``element 0 of tensors does not require grad``. This test pins that, so if anyone
    later makes the checkpointed path double-differentiable it fails here and the oracle
    guidance in this package's docstring gets revisited rather than quietly going stale.
    """
    from torchref.base.direct_summation.dispatch import ds_iso

    s = scene_small
    x = s.xyz_frac.clone().requires_grad_(True)
    F = ds_iso(s.hkl, s.s, x, s.occ, s.adp, s.A, s.B)
    with torch.no_grad():
        obs = H.synthetic_obs(F)
    (g1,) = torch.autograd.grad(H.ls_target(F, obs), x, create_graph=True)

    v = torch.ones_like(x)
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad((g1 * v).sum(), x)


def test_eager_ds_survives_double_backward_where_public_api_does_not(scene_small):
    """The positive half of the pair above: the eager path *does* compose.

    Together these two tests state the rule this package depends on -- use ``_eager_*``
    as the second-order oracle, never ``ds_*``/``SfDS`` -- as executable fact rather than
    as a comment that can drift.
    """
    s = scene_small
    x = s.xyz_frac.clone().requires_grad_(True)
    F = _eager_iso(s.hkl, s.s, x, s.occ, s.adp, s.A, s.B, None)
    with torch.no_grad():
        obs = H.synthetic_obs(F.detach())
    (g1,) = torch.autograd.grad(H.ls_target(F, obs), x, create_graph=True)
    (out,) = torch.autograd.grad((g1 * torch.ones_like(x)).sum(), x)
    assert torch.isfinite(out).all() and out.abs().sum() > 0


# ---------------------------------------------------------------------------
# 3. DS -> FFT at second order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", [Engine.AUTO, Engine.EAGER], ids=["auto", "eager"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_fft_hvp_matches_ds(scene_fine, oracle_fine, kind, dtype, engine):
    """The map route's HVP against the DS oracle's HVP.

    This is the replacement for the two failing FD-based tests, and it covers strictly
    more: both dtypes and both engines, where the originals covered float64-plain and
    float32-C++ only, each with its own hand-tuned ``eps``.

    Both sides contract with the same seeded direction ``v`` (from the oracle fixture),
    since an HVP is only comparable along a shared direction.
    """
    aniso = kind == "aniso"
    sf = H.sf_fft_for(scene_fine, dtype)
    v = oracle_fine[f"{kind}_v"]
    _, occ, third = scene_fine.leaves(aniso=aniso, requires_grad=False)

    def loss(x):
        return H.ls_target(
            H.fft_sf(scene_fine, sf, x, occ, third, aniso=aniso),
            oracle_fine[f"{kind}_obs"],
        )

    with use_engine(engine):
        got = hvp(loss, scene_fine.xyz, v)
    ref = oracle_fine[f"{kind}_hvp"]

    rel, cos = rel_error(got, ref), cosine_similarity(got, ref)
    print(f"\n  {kind}/{dtype}/{engine.value}: HVP vs DS -- rel {rel:.3e}  cos {cos:.8f}")
    assert cos > COS_MIN, f"{kind}/{dtype}/{engine.value}: HVP direction, cos {cos:.6f}"
    assert rel < RTOL_HVP, f"{kind}/{dtype}/{engine.value}: HVP magnitude, rel {rel:.3e}"


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
def test_fused_cpu_kernel_uses_the_double_backward_fallback(
    scene_fine, oracle_fine, dtype, monkeypatch
):
    """Non-vacuity guard for the AUTO second-order path.

    The fused C++ sphere splat has a hand-written first-order backward with no graph, so
    under ``create_graph=True`` it routes through ``_double_backward_vjp``, which re-runs
    the forward through the portable torch splat on the *saved* leaves and differentiates
    that. Without this guard, ``test_fft_hvp_matches_ds[auto]`` would still pass if AUTO
    silently fell back to the eager path for the whole forward -- and would then be
    testing EAGER twice under two names.

    Asserts the fallback is entered, which is only meaningful alongside a separate
    assertion that the fused kernel was used in the first place.
    """
    from torchref.base.electron_density.kernels.cpu import sphere_splat

    if not sphere_splat.sphere_splat_available():
        pytest.skip(f"fused CPU sphere splat unavailable: {sphere_splat.last_error()}")

    calls = {"fallback": 0}
    original = sphere_splat._double_backward_vjp

    def counting(*args, **kwargs):
        calls["fallback"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sphere_splat, "_double_backward_vjp", counting)

    sf = H.sf_fft_for(scene_fine, dtype)
    v = torch.ones_like(scene_fine.xyz)
    _, occ, adp = scene_fine.leaves(requires_grad=False)
    obs = oracle_fine["iso_obs"]

    def loss(x):
        return H.ls_target(H.fft_sf(scene_fine, sf, x, occ, adp), obs)

    with use_engine(Engine.AUTO):
        out = hvp(loss, scene_fine.xyz, v)

    assert calls["fallback"] > 0, (
        "AUTO produced an HVP without entering _double_backward_vjp, so the fused "
        "kernel was not on the path and this dtype's second-order coverage is vacuous"
    )
    assert torch.isfinite(out).all() and out.abs().sum() > 0


# ---------------------------------------------------------------------------
# 4. The measurement that justifies choosing DS over finite differences
# ---------------------------------------------------------------------------
def test_finite_differences_cannot_detect_map_route_error(scene_fine, oracle_fine):
    """Finite differences confirm self-consistency, not correctness. Measured.

    This test was originally written to demonstrate an *FD plateau* on the map route --
    the argument being that a hard truncation whose cull surface moves with the atom
    contributes a fixed jump per crossing, so the spurious term shrinks no faster than the
    divisor. That argument is inherited from an earlier draft and the measurement does not
    support it: FD converges normally here, reaching rel 5.2e-04 against the map route's
    own autograd HVP at ``eps = 1e-2``, with the usual V-shape (truncation-dominated at
    large ``eps``, round-off at small). At the 3 sigma cutoff the density at the surface is
    ~1% of peak, so the jumps are evidently too small to matter.

    The measurement supports a stronger conclusion instead. The map route's HVP agrees
    with *itself* to 5e-04 while sitting 2.3e-02 away from the analytic answer. So an
    FD-based test gated at the 5e-03 the replaced tests used passes comfortably while the
    derivative under test is 2.3e-02 wrong -- FD cannot see that error at all, because it
    differentiates the same discretized function. Only an independent reference can.

    That is why this package uses DS. Not because FD diverges, but because FD is measuring
    the wrong thing: it validates that autograd correctly differentiates whatever the
    kernel computes, and is blind by construction to whether the kernel computes the right
    thing.
    """
    sf = H.sf_fft_for(scene_fine)
    v = oracle_fine["iso_v"]
    _, occ, adp = scene_fine.leaves(requires_grad=False)
    obs = oracle_fine["iso_obs"]

    def loss(x):
        return H.ls_target(H.fft_sf(scene_fine, sf, x, occ, adp), obs)

    auto = hvp(loss, scene_fine.xyz, v)
    ref_ds = oracle_fine["iso_hvp"]

    print("\n  eps        rel vs autograd(map)   rel vs DS oracle")
    rows = []
    for eps in (1e-4, 1e-3, 1e-2, 1e-1):
        fd = hvp_central_fd(loss, scene_fine.xyz, v, eps=eps)
        r_self, r_ds = rel_error(fd, auto), rel_error(fd, ref_ds)
        rows.append((eps, r_self, r_ds))
        print(f"  {eps:<9.0e}  {r_self:18.3e}   {r_ds:16.3e}")

    print(f"  autograd(map) vs DS oracle: {rel_error(auto, ref_ds):.3e}")

    best_self = min(r for _, r, _ in rows)
    best_vs_ds = min(r for _, _, r in rows)
    map_vs_ds = rel_error(auto, ref_ds)

    # 1. FD does converge against the map route's own autograd.
    assert best_self < 5e-3, (
        f"FD only reached rel {best_self:.2e} against the map route's autograd HVP. This "
        "test's premise is that it converges; if that has stopped being true the "
        "docstring's reasoning needs rewriting, not the tolerance."
    )
    # 2. And is nonetheless blind to the real error, which is what matters.
    assert best_vs_ds > 4 * best_self, (
        f"FD sits {best_vs_ds:.2e} from the analytic answer but {best_self:.2e} from the "
        "map route's autograd. Those being comparable would mean the map route is now "
        "accurate enough that FD and DS are interchangeable references, and this test "
        "no longer demonstrates anything -- re-examine rather than retune."
    )
    assert map_vs_ds > 5e-3, (
        f"the map route's HVP is now within {map_vs_ds:.2e} of the analytic answer, "
        "which is inside the 5e-3 gate the replaced FD-based tests used -- so those "
        "tests would no longer have been misleading and this rationale is stale"
    )


def test_fft_hvp_matches_ds_real_structure(gemmi_aniso_grad, oracle_aniso_grad):
    """Second derivatives on 7L84 at production sampling, in the production dtype.

    The synthetic HVP sweep above covers dtype and engine combinations; this is the one
    that speaks to production, for the same reason as its first-order counterpart in
    ``test_gradients.py``.
    """
    scene, _ = gemmi_aniso_grad
    sf = H.sf_fft_for(scene, torch.float32)
    v = oracle_aniso_grad["aniso_v"]
    _, occ, u6 = scene.leaves(aniso=True, requires_grad=False)
    obs = oracle_aniso_grad["aniso_obs"]

    def loss(x):
        return H.ls_target(H.fft_sf(scene, sf, x, occ, u6, aniso=True), obs)

    got = hvp(loss, scene.xyz, v)
    ref = oracle_aniso_grad["aniso_hvp"]
    rel, cos = rel_error(got, ref), cosine_similarity(got, ref)
    print(f"\n7L84 P1 HVP vs DS: rel {rel:.3e}  cos {cos:.8f}")
    assert cos > COS_MIN, f"7L84 HVP direction: cos {cos:.6f}"
    assert rel < RTOL_HVP, f"7L84 HVP magnitude: rel {rel:.3e}"



# ---------------------------------------------------------------------------
# 5. Every production kernel, on every device it ships on
# ---------------------------------------------------------------------------
# Second order is where the kernels genuinely differ, so this section is not simply the
# gradient section with another derivative. Only two of the four families are
# double-differentiable:
#
#   add_*_cpu_sphere_var   yes -- via ``_double_backward_vjp``, which re-derives the VJP
#                          through the portable splat on the saved leaves
#   add_*_plain_var        yes -- pure autograd over ``scatter_add``
#   add_*_mps_var          no  -- ``mps/variable_radius.py:12``: "Backward is first-order
#                          only (like CUDA); double backward must use Engine.EAGER"
#   WorkQueueGridDensity*  no  -- same, no ``create_graph`` path in the Triton backward
#
# So an accelerator HVP cannot be compared against the oracle: there is no HVP to compare.
# What is testable, and tested below, is that those kernels **raise** rather than returning
# a silently wrong (or zero) second derivative -- the same failure mode this repo already
# fixed once for the C++ scatter, where a graph-less backward gave cosine 0.57 while first
# derivatives stayed correct.

_DOUBLE_DIFFERENTIABLE = {"cpu_sphere", "portable"}


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DEVICE_DTYPE_KERNELS)
def test_kernel_hvp_matches_ds(scene_fine, oracle_fine, device, dtype, kernel, kind):
    """HVP of a double-differentiable kernel against the oracle's, on every device.

    Covers the portable splat running *on an accelerator*, which nothing tested before:
    it is what ``Engine.EAGER`` and the CUDA/MPS float64 fallthrough actually execute, and
    it is the path a Hessian-based optimizer lands on there.
    """
    if kernel not in _DOUBLE_DIFFERENTIABLE:
        pytest.skip(f"{kernel} is first-order only; see test_kernel_rejects_double_backward")
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    obs = oracle_fine[f"{kind}_obs"].to(device=device, dtype=dtype)
    v = oracle_fine[f"{kind}_v"].to(device=device, dtype=dtype)
    _, occ, third = scene.leaves(aniso=aniso, requires_grad=False)

    def loss(x):
        return H.ls_target(
            H.density_to_F(scene, H.splat_direct(scene, kernel, x, occ, third, aniso=aniso)),
            obs,
        )

    got = hvp(loss, scene.xyz, v)
    ref = oracle_fine[f"{kind}_hvp"]
    rel, cos = rel_error(got, ref), cosine_similarity(got, ref)
    print(f"\n  {device.type}/{dtype}/{kernel}/{kind}: HVP rel {rel:.3e}  cos {cos:.8f}")
    assert cos > COS_MIN, f"{device.type}/{dtype}/{kernel}/{kind}: HVP cos {cos:.6f}"
    assert rel < RTOL_HVP, f"{device.type}/{dtype}/{kernel}/{kind}: HVP rel {rel:.3e}"


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DEVICE_DTYPE_KERNELS)
def test_kernel_rejects_double_backward(scene_fine, oracle_fine, device, dtype, kernel, kind):
    """First-order-only kernels must raise under ``create_graph=True``, not return garbage.

    The accelerator kernels have hand-written backwards that return tensors carrying no
    graph, and neither is decorated ``@once_differentiable`` -- so nothing warns, and the
    only signal is whatever autograd does when asked for a second derivative. Pinning that
    it *raises* is what distinguishes "unsupported" from "silently wrong".

    The positive half of the pair is :func:`test_kernel_hvp_matches_ds`; together they say
    which kernels an optimizer may take a Hessian through.
    """
    if kernel in _DOUBLE_DIFFERENTIABLE:
        pytest.skip(f"{kernel} supports double backward; see test_kernel_hvp_matches_ds")
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    obs = oracle_fine[f"{kind}_obs"].to(device=device, dtype=dtype)
    _, occ, third = scene.leaves(aniso=aniso, requires_grad=False)

    x = scene.xyz.clone().requires_grad_(True)
    F = H.density_to_F(scene, H.splat_direct(scene, kernel, x, occ, third, aniso=aniso))
    (g1,) = torch.autograd.grad(H.ls_target(F, obs), x, create_graph=True)

    with pytest.raises(RuntimeError) as exc:
        torch.autograd.grad((g1 * torch.ones_like(x)).sum(), x)
    print(f"\n  {device.type}/{dtype}/{kernel}/{kind} raised: {str(exc.value)[:70]}")
