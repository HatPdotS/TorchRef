"""First-order gradients: FD -> direct summation -> the FFT/splat route.

Three links, in order:

1. ``gradcheck`` proves the eager DS path's gradients against finite differences. DS is
   analytic, so FD converges against it and this gate is tight.
2. The *production* DS path (``_checkpointed_*``, what ``ds_iso`` and ``SfDS`` actually
   call) is tied to the eager oracle. This link is load-bearing: without it, "the oracle
   is correct" would say nothing about the code that ships.
3. The FFT/splat route is compared against the oracle, at the loose gate the truncation
   and sampling error actually warrants.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.helpers.grad_asserts import (
    assert_grads_agree,
    cosine_similarity,
    gradnorm_ratio,
    rel_error,
)
from torchref.base.direct_summation.dispatch import (
    _checkpointed_aniso,
    _checkpointed_iso,
    _eager_aniso,
    _eager_iso,
)
from torchref.utils import Engine, use_engine

from . import (
    COS_MIN_DS,
    RTOL_DS_F32,
    RTOL_DS_F64,
    ATOL_GRADCHECK,
    COS_MIN_GRADIENT,
    COS_MIN_GRADIENT_SYNTHETIC,
    EPS_GRADCHECK,
    RTOL_BACKEND_GRAD_F32,
    RTOL_BACKEND_GRAD_F64,
    RTOL_GRADCHECK,
    RTOL_GRADIENT,
    RTOL_GRADIENT_SYNTHETIC,
)
from . import helpers as H
from .conftest import DEVICE_DTYPE_KERNELS, DS_DEVICE_DTYPE_KERNELS

pytestmark = pytest.mark.unit

_DTYPES = [torch.float32, torch.float64]  # float32 first: production dtype
_LEAF_NAMES = ("xyz", "occ", "adp_or_u")


# ---------------------------------------------------------------------------
# 1. FD -> DS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_ds_gradcheck(scene_small, kind):
    """The oracle's own gradients, against finite differences.

    Run on the deliberately tiny ``scene_small`` because ``gradcheck`` costs
    O(n_params) forward evaluations -- 430 ms at 10 atoms, 75 s at 60. The property being
    proved (that eager DS differentiates correctly) is scene-independent, so there is
    nothing to gain from a larger scene and a great deal of wall-clock to lose.
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

    args = tuple(
        t.clone().requires_grad_(True) for t in (s.xyz_frac, s.occ, third)
    )
    assert torch.autograd.gradcheck(
        fn, args, eps=EPS_GRADCHECK, atol=ATOL_GRADCHECK, rtol=RTOL_GRADCHECK
    )


# ---------------------------------------------------------------------------
# 2. The production DS path matches the oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunked", [False, True], ids=["one_chunk", "chunked"])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_checkpointed_matches_eager_oracle(scene_fine, kind, chunked):
    """``_checkpointed_*`` -- reached by every ``ds_iso``/``ds_aniso``/``SfDS`` call on
    CPU, MPS or float64 -- must reproduce the eager oracle in both value and gradient.

    This is the link that makes ``_eager_*`` a legitimate stand-in for the shipping code.
    ``_CheckpointedSF`` recomputes each chunk under ``enable_grad`` in its backward, so
    the gradient path is genuinely different from eager even though the maths is the
    same; ``chunked`` forces multiple chunks via a tiny memory budget so that recompute
    loop is actually exercised rather than degenerating to one pass.

    Measured agreement is 2.3e-16, so this is gated near machine precision. It is
    deliberately far tighter than anything involving the map route: any drift here is a
    bug, not an approximation.
    """
    s = scene_fine
    max_mem = 1e-7 if chunked else None
    if kind == "iso":
        eager = lambda x, o, a: _eager_iso(s.hkl, s.s, x, o, a, s.A, s.B, max_mem)  # noqa: E731
        prod = lambda x, o, a: _checkpointed_iso(  # noqa: E731
            s.hkl, s.s, x, o, a, s.A, s.B, max_mem
        )
        third = s.adp
    else:
        eager = lambda x, o, u: _eager_aniso(  # noqa: E731
            s.hkl, s.s_vec, x, o, u, s.A, s.B, max_mem
        )
        prod = lambda x, o, u: _checkpointed_aniso(  # noqa: E731
            s.hkl, s.s_vec, x, o, u, s.A, s.B, max_mem
        )
        third = s.u6

    with torch.no_grad():
        obs = H.synthetic_obs(eager(s.xyz_frac, s.occ, third))

    def run(fn):
        leaves = tuple(
            t.clone().requires_grad_(True) for t in (s.xyz_frac, s.occ, third)
        )
        F = fn(*leaves)
        grads = torch.autograd.grad(H.ls_target(F, obs), leaves)
        return F.detach(), grads

    F_eager, g_eager = run(eager)
    F_prod, g_prod = run(prod)

    assert rel_error(F_prod, F_eager) < 1e-13, "checkpointed forward drifted from eager"
    assert_grads_agree(
        dict(zip(_LEAF_NAMES, g_prod)),
        dict(zip(_LEAF_NAMES, g_eager)),
        min_cos=1.0 - 1e-12,
        ratio_tol=1e-10,
        ctx=f"{kind}/{'chunked' if chunked else 'one_chunk'} ",
    )


def test_public_ds_api_matches_oracle(scene_fine):
    """The same check one level up, through ``ds_iso``'s own dispatch.

    Guards against the dispatch layer -- not the kernel -- introducing a discrepancy:
    a wrong ``max_memory_gb`` default, or an ``Engine`` gate that silently routes
    somewhere unintended on CPU.
    """
    from torchref.base.direct_summation.dispatch import ds_iso

    s = scene_fine
    with torch.no_grad():
        obs = H.synthetic_obs(_eager_iso(s.hkl, s.s, s.xyz_frac, s.occ, s.adp, s.A, s.B, None))

    leaves = tuple(t.clone().requires_grad_(True) for t in (s.xyz_frac, s.occ, s.adp))
    F = ds_iso(s.hkl, s.s, *leaves, s.A, s.B)
    g_public = torch.autograd.grad(H.ls_target(F, obs), leaves)

    leaves2 = tuple(t.clone().requires_grad_(True) for t in (s.xyz_frac, s.occ, s.adp))
    F2 = _eager_iso(s.hkl, s.s, *leaves2, s.A, s.B, None)
    g_oracle = torch.autograd.grad(H.ls_target(F2, obs), leaves2)

    assert rel_error(F, F2) < 1e-13
    assert_grads_agree(
        dict(zip(_LEAF_NAMES, g_public)),
        dict(zip(_LEAF_NAMES, g_oracle)),
        min_cos=1.0 - 1e-12,
        ratio_tol=1e-10,
        ctx="ds_iso ",
    )


# ---------------------------------------------------------------------------
# 3. DS -> FFT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", [Engine.AUTO, Engine.EAGER], ids=["auto", "eager"])
@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_fft_gradients_match_ds(scene_fine, oracle_fine, kind, dtype, engine):
    """Gradients of the map route against the analytic oracle, for all three leaves.

    ``occ`` is included deliberately. Before this package, occupancy gradients through
    the FFT route were untested at any order -- the previous FFT gradient test
    parametrized only ``xyz`` and ``adp`` -- and ``occ`` is the one leaf whose gradient a
    scene with ``occ == 1`` everywhere cannot constrain, which is why
    :func:`helpers.synthetic_scene` never uses unit occupancy.

    Gradients are roughly 10x more truncation-sensitive than amplitudes, because each
    ``xyz`` derivative brings a factor ~``2*pi*h``. Hence ``RTOL_GRADIENT`` an order of
    magnitude above ``RTOL_AMPLITUDE`` rather than the same number.
    """
    aniso = kind == "aniso"
    sf = H.sf_fft_for(scene_fine, dtype)
    leaves = scene_fine.leaves(aniso=aniso)
    with use_engine(engine):
        F = H.fft_sf(scene_fine, sf, *leaves, aniso=aniso)
        got = torch.autograd.grad(H.ls_target(F, oracle_fine[f"{kind}_obs"]), leaves)
    ref = oracle_fine[f"{kind}_grads"]

    gate, cos_gate = RTOL_GRADIENT_SYNTHETIC, COS_MIN_GRADIENT_SYNTHETIC
    print(f"\n{kind}, {dtype}, {engine.value}  (rel gate {gate:.0e}, cos gate {cos_gate})")
    for name, g, r in zip(_LEAF_NAMES, got, ref):
        rel, cos = rel_error(g, r), cosine_similarity(g, r)
        print(f"  {name:10s} rel {rel:.3e}  cos {cos:.8f}  ratio {gradnorm_ratio(g, r):.4f}")
        assert rel < gate, f"{kind}/{dtype}/{engine.value} {name}: rel {rel:.3e}"
        assert cos > cos_gate, f"{kind}/{dtype}/{engine.value} {name}: cos {cos:.6f}"


@pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_backend_parity_of_gradients(scene_fine, oracle_fine, kind, dtype):
    """AUTO and EAGER share one truncation contract, so their gradients must agree far
    more tightly than either agrees with DS. Separates a kernel-specific backward bug
    from the shared truncation residual."""
    aniso = kind == "aniso"
    sf = H.sf_fft_for(scene_fine, dtype)

    def run(engine):
        leaves = scene_fine.leaves(aniso=aniso)
        with use_engine(engine):
            F = H.fft_sf(scene_fine, sf, *leaves, aniso=aniso)
            return torch.autograd.grad(H.ls_target(F, oracle_fine[f"{kind}_obs"]), leaves)

    auto, eager = run(Engine.AUTO), run(Engine.EAGER)
    tol = RTOL_BACKEND_GRAD_F32 if dtype is torch.float32 else RTOL_BACKEND_GRAD_F64
    for name, a, e in zip(_LEAF_NAMES, auto, eager):
        rel = rel_error(a, e)
        print(f"  {kind}/{dtype} {name:10s} AUTO vs EAGER rel {rel:.3e} (gate {tol:.0e})")
        assert rel < tol, f"{kind}/{dtype} {name}: AUTO vs EAGER rel {rel:.3e}"


def test_occupancy_gradient_is_not_degenerate(scene_fine, oracle_fine):
    """Non-vacuity guard: the ``occ`` gradient must be non-trivial and atom-dependent.

    An all-equal or all-zero ``occ`` gradient would satisfy the cosine and relative gates
    above while carrying no information, which is exactly what a scene with uniform
    occupancy produces.
    """
    g_occ = oracle_fine["iso_grads"][1]
    assert g_occ.abs().min() > 0, "some occupancy gradient is exactly zero"
    spread = float(g_occ.std() / g_occ.abs().mean())
    print(f"\n  occ gradient relative spread across atoms: {spread:.3f}")
    assert spread > 0.05, (
        "occupancy gradients are nearly identical across atoms, so the comparison "
        "cannot distinguish a correct per-atom gradient from a constant"
    )


# ---------------------------------------------------------------------------
# 4. The authoritative gate: a real structure
# ---------------------------------------------------------------------------
def test_fft_gradients_match_ds_real_structure(gemmi_aniso_grad, oracle_aniso_grad):
    """Gradients on 7L84 -- 1209 atoms, all with ANISOU -- at production sampling.

    This is the gate that states something about production; the synthetic sweeps above
    exist for dtype/engine/cell coverage and are calibrated loosely because small scenes
    overstate the discretization residual (see ``__init__.py``).

    Run in float32, the production dtype, against the float64 oracle. Nothing here is
    tuned: the numbers came out at 1.35e-02 (xyz) and 2.62e-02 (U) against the 1e-01 gate
    you specified, so there is ~4x headroom without the gate being vacuous -- the
    under-sampled control in ``test_forward.py`` shows what breaching it looks like.
    """
    scene, _ = gemmi_aniso_grad
    sf = H.sf_fft_for(scene, torch.float32)
    leaves = scene.leaves(aniso=True)
    got = torch.autograd.grad(
        H.ls_target(H.fft_sf(scene, sf, *leaves, aniso=True), oracle_aniso_grad["aniso_obs"]),
        leaves,
    )
    ref = oracle_aniso_grad["aniso_grads"]

    print(f"\n7L84 P1, {scene.n_atoms} ANISOU atoms, {scene.n_refl} refl, float32")
    print(f"  grid {tuple(int(v) for v in sf.gridsize)}")
    for name, g, r in zip(_LEAF_NAMES, got, ref):
        rel, cos, ratio = rel_error(g, r), cosine_similarity(g, r), gradnorm_ratio(g, r)
        print(f"  {name:10s} rel {rel:.3e}  cos {cos:.8f}  ratio {ratio:.4f}")
        assert rel < RTOL_GRADIENT, f"7L84 {name}: rel {rel:.3e}"
        assert cos > COS_MIN_GRADIENT, f"7L84 {name}: cos {cos:.6f}"

    # The deviatoric part specifically -- an earlier revision of this package recorded a
    # 54% bias here from a 10-atom synthetic scene. On a real structure it is not present,
    # and this assertion is what keeps that claim from creeping back.
    dev = lambda g: g[:, :3] - g[:, :3].mean(dim=1, keepdim=True)  # noqa: E731
    dev_ratio = gradnorm_ratio(dev(got[2]), dev(ref[2]))
    print(f"  deviatoric U ratio {dev_ratio:.4f}")
    assert abs(dev_ratio - 1.0) < 0.05, (
        f"deviatoric anisotropic ADP gradient is biased by {abs(dev_ratio - 1.0):.1%} on a "
        "real structure. That would justify revisiting the grid oversampling or a B_extra "
        "smearing correction; it was measured at 0.9995 when this test was written."
    )



# ---------------------------------------------------------------------------
# 5. Every production kernel, on every device it ships on
# ---------------------------------------------------------------------------
# Direct kernel calls, not dispatch -- see the note in ``test_forward.py``. Worst case
# over the whole matrix when written: gradient rel 9.51e-02, cos 0.995544, and every
# device agreeing to the printed digits, so the CPU constants bound the accelerators too.


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DEVICE_DTYPE_KERNELS)
def test_kernel_gradients_match_ds(scene_fine, oracle_fine, device, dtype, kernel, kind):
    """Gradients from one production kernel against the DS oracle, all three leaves.

    The candidate's leaves live on the target device in the target dtype, while the
    oracle's are CPU float64. Both describe the same structure because
    :meth:`Scene.to` moves and casts a single source scene, and the comparison helpers
    promote to CPU float64 before measuring.
    """
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    obs = oracle_fine[f"{kind}_obs"].to(device=device, dtype=dtype)
    leaves = scene.leaves(aniso=aniso)
    F = H.density_to_F(scene, H.splat_direct(scene, kernel, *leaves, aniso=aniso))
    got = torch.autograd.grad(H.ls_target(F, obs), leaves)
    ref = oracle_fine[f"{kind}_grads"]

    print(f"\n  {device.type}/{dtype}/{kernel}/{kind}")
    for name, g, r in zip(_LEAF_NAMES, got, ref):
        rel, cos = rel_error(g, r), cosine_similarity(g, r)
        print(f"    {name:10s} rel {rel:.3e}  cos {cos:.8f}  ratio {gradnorm_ratio(g, r):.4f}")
        assert rel < RTOL_GRADIENT_SYNTHETIC, (
            f"{device.type}/{dtype}/{kernel}/{kind} {name}: rel {rel:.3e}"
        )
        assert cos > COS_MIN_GRADIENT_SYNTHETIC, (
            f"{device.type}/{dtype}/{kernel}/{kind} {name}: cos {cos:.6f}"
        )


@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DEVICE_DTYPE_KERNELS)
def test_kernel_gradients_agree_with_portable(scene_fine, oracle_fine, device, dtype, kernel, kind):
    """Kernel-vs-portable gradients on the same device, gated tightly.

    The oracle gate above is loose by necessity -- it absorbs the real discretization
    error. This one is tight, because two kernels applying the same truncation contract to
    the same inputs differ only in accumulation order. It is what would catch a backward
    that is wrong in the same direction as the discretization residual.
    """
    if kernel == "portable":
        pytest.skip("portable is the reference for this comparison")
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    obs = oracle_fine[f"{kind}_obs"].to(device=device, dtype=dtype)

    def grads(name):
        leaves = scene.leaves(aniso=aniso)
        F = H.density_to_F(scene, H.splat_direct(scene, name, *leaves, aniso=aniso))
        return torch.autograd.grad(H.ls_target(F, obs), leaves)

    got, ref = grads(kernel), grads("portable")
    tol = RTOL_BACKEND_GRAD_F32 if dtype is torch.float32 else RTOL_BACKEND_GRAD_F64
    for name, g, r in zip(_LEAF_NAMES, got, ref):
        rel = rel_error(g, r)
        print(f"  {device.type}/{dtype}/{kernel}/{kind} {name:10s} vs portable rel {rel:.3e}")
        assert rel < tol, f"{device.type}/{dtype}/{kernel}/{kind} {name}: rel {rel:.3e}"


# ---------------------------------------------------------------------------
# 6. Direct-summation kernels vs the eager oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["iso", "aniso"])
@pytest.mark.parametrize("device,dtype,kernel", DS_DEVICE_DTYPE_KERNELS)
def test_ds_kernel_gradients_match_eager_oracle(scene_fine, device, dtype, kernel, kind):
    """Gradients from one DS kernel against the eager oracle, all three leaves.

    Absorbs ``test_gradient_correctness.py``'s two Triton DS gradient tests, which lived
    beside geometry and ADP-restraint tests and compared against ``_eager_*`` at
    ``min_cos=0.999, ratio_tol=1e-2`` while never checking forward values. Here the forward
    comparison is :func:`test_forward.test_ds_kernel_matches_eager_oracle` and the gate is
    near precision, because these are two analytic implementations with no discretization
    between them.

    Coordinates are **fractional** -- direct summation's convention, unlike the splat
    kernels' Cartesian.
    """
    aniso = kind == "aniso"
    scene = scene_fine.to(device=device, dtype=dtype)
    third_ref = scene_fine.u6 if aniso else scene_fine.adp

    with torch.no_grad():
        obs = H.synthetic_obs(H.ds_direct(scene_fine, "eager", aniso=aniso))

    def grads(sc, name, third_src, obs_local):
        leaves = tuple(
            t.clone().requires_grad_(True)
            for t in (sc.xyz_frac, sc.occ, third_src)
        )
        F = H.ds_direct(sc, name, *leaves, aniso=aniso)
        return torch.autograd.grad(H.ls_target(F, obs_local), leaves)

    ref = grads(scene_fine, "eager", third_ref, obs)
    got = grads(
        scene, kernel, scene.u6 if aniso else scene.adp, obs.to(device=device, dtype=dtype)
    )

    tol = RTOL_DS_F32 if dtype is torch.float32 else RTOL_DS_F64
    print(f"\n  {device.type}/{dtype}/{kernel}/{kind}")
    for name, g, r in zip(("xyz_frac", "occ", "adp_or_u"), got, ref):
        rel, cos = rel_error(g, r), cosine_similarity(g, r)
        print(f"    {name:10s} rel {rel:.3e}  cos {cos:.9f}")
        assert rel < tol, f"{device.type}/{dtype}/{kernel}/{kind} {name}: rel {rel:.3e}"
        assert cos > COS_MIN_DS, f"{device.type}/{dtype}/{kernel}/{kind} {name}: cos {cos:.7f}"
