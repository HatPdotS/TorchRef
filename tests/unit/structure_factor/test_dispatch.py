"""The dispatch ladder itself: which engine selects which kernel, and failure policy.

Deliberately separate from the accuracy tests. Those call each kernel **directly**, so
they say nothing about whether ``build_electron_density`` would have chosen it — and that
separation is the point. Entangling the two is what forced the old accelerator tests to
carry a monkeypatched call recorder just to prove they were not measuring the fallback.

What is pinned here:

* under ``Engine.AUTO`` *and* the strict engine, an accelerator host really reaches its
  native kernel rather than the portable splat;
* a strict engine **raises** when its kernel is unavailable, while ``AUTO`` degrades
  quietly — the two halves of the never-silently-degrade contract;
* ``Engine.EAGER`` reaches the portable splat on every device.

Ported from ``tests/integration/test_variable_radius_{gpu,mps}.py``, which are deleted:
their accuracy coverage is superseded by the oracle legs in this package, but these
dispatch contracts are not accuracy and had no replacement.
"""

from __future__ import annotations

import pytest
import torch

from torchref.base.electron_density.main import build_electron_density
from torchref.base.electron_density.radius_policy import per_atom_radius_iso
from torchref.utils import Engine, use_engine

from tests.helpers.grad_asserts import rel_error

from . import RTOL_DS_F32
from . import helpers as H

pytestmark = pytest.mark.unit


def _build(scene, device, dtype, engine, aniso=False):
    """Run one dispatch through ``build_electron_density`` on ``device``.

    ``dtype`` is passed explicitly rather than inherited from the ambient config. That is
    load-bearing for the accelerator branches: under this package's float64 pin the Metal
    and Triton gates would never fire, which is a silent way for every assertion below to
    become vacuous.
    """
    s = scene.to(device=device, dtype=dtype)
    dims = H._grid_dims(s)
    grid = torch.zeros(*dims, 3, dtype=dtype, device=device)
    voxel = torch.tensor(
        [float(s.cell.data[i]) / dims[i] for i in range(3)], dtype=dtype, device=device
    )
    empty1 = s.xyz.new_zeros(0)
    empty3 = s.xyz.new_zeros(0, 3)
    empty5 = s.A.new_zeros(0, 5)
    kw = dict(
        real_space_grid=grid,
        inv_frac_matrix=s.inv_frac_matrix,
        frac_matrix=s.frac_matrix,
        voxel_size=voxel,
        dtype=dtype,
    )
    with use_engine(engine):
        if aniso:
            return build_electron_density(
                xyz_iso=empty3, adp_iso=empty1, occ_iso=empty1,
                A_iso=empty5, B_iso=empty5,
                xyz_aniso=s.xyz, u_aniso=s.u6, occ_aniso=s.occ,
                A_aniso=s.A, B_aniso=s.B, **kw,
            )
        return build_electron_density(
            xyz_iso=s.xyz, adp_iso=s.adp, occ_iso=s.occ,
            A_iso=s.A, B_iso=s.B, **kw,
        )


# ---------------------------------------------------------------------------
# Provenance: the named kernel actually runs
# ---------------------------------------------------------------------------
@pytest.mark.mps
@pytest.mark.parametrize("engine", [Engine.AUTO, Engine.METAL])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_metal_kernel_is_actually_dispatched(scene_small, engine, kind, monkeypatch):
    """On MPS float32, both ``AUTO`` and ``METAL`` must call the Metal kernel.

    Verified with a call recorder, not by comparing maps: the portable reference uses
    ``scatter_add`` on MPS, whose accumulation order is not reproducible, so two portable
    runs already differ at ~1e-7 and an equality check against one would prove nothing
    either way.

    **The patch target matters.** ``_add_isotropic`` does a *function-local*
    ``from ...kernels.mps import add_isotropic_mps_var`` on every call, so it re-reads the
    package each time. Patching the ``main`` module's namespace would silently no-op and
    leave this test as vacuous as the bug it guards against.

    The recorder delegates to the real kernel rather than stubbing it, so the dispatch is
    still exercised end to end.

    The submodule is imported explicitly rather than reached as ``kernels.mps``: the parent
    package does not import it eagerly, so that attribute only exists once something has
    already triggered the function-local import. Relying on that ordering is how this test
    would pass in a full run and fail in isolation.
    """
    from torchref.base.electron_density.kernels import mps as kernels_mps

    name = f"add_{'anisotropic' if kind == 'aniso' else 'isotropic'}_mps_var"
    real = getattr(kernels_mps, name)
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kernels_mps, name, recording)
    _build(scene_small, torch.device("mps"), torch.float32, engine, aniso=kind == "aniso")
    assert calls, f"{engine} did not dispatch to {name}"


@pytest.mark.cuda
@pytest.mark.parametrize("engine", [Engine.AUTO, Engine.TRITON])
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_triton_kernel_is_actually_dispatched(scene_small, engine, kind, monkeypatch):
    """CUDA float32 equivalent of the Metal provenance test.

    Patch target differs: ``main.py`` imports the CUDA wrappers at module scope, not
    per-call, so the name to patch is the one bound in ``main`` -- the mirror image of the
    Metal case, and the reason each needs its own test rather than one parametrized helper.
    """
    from torchref.base.electron_density import main as ed_main

    name = f"add_{'anisotropic' if kind == 'aniso' else 'isotropic'}_cuda_var"
    real = getattr(ed_main, name)
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ed_main, name, recording)
    _build(scene_small, torch.device("cuda"), torch.float32, engine, aniso=kind == "aniso")
    assert calls, f"{engine} did not dispatch to {name}"


@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_eager_reaches_the_portable_splat(scene_small, kind, monkeypatch):
    """``Engine.EAGER`` must reach the portable splat, on any device.

    The counterpart to the provenance tests above: EAGER is the documented escape hatch
    for double backward and debugging, so it has to be the *portable* kernel that runs,
    not whatever AUTO would have picked.
    """
    from torchref.base.electron_density import main as ed_main

    name = f"add_{'anisotropic' if kind == 'aniso' else 'isotropic'}_plain_var"
    real = getattr(ed_main, name)
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ed_main, name, recording)
    _build(scene_small, torch.device("cpu"), torch.float32, Engine.EAGER, aniso=kind == "aniso")
    assert calls, f"Engine.EAGER did not dispatch to {name}"


# ---------------------------------------------------------------------------
# Failure policy: strict engines raise, AUTO degrades
# ---------------------------------------------------------------------------
@pytest.mark.mps
def test_metal_raises_when_shader_unavailable(scene_small, monkeypatch):
    """``Engine.METAL`` must raise, never degrade, when the shader is missing.

    ``monkeypatch`` reverts the module globals at teardown, so this needs no
    ``try``/``finally`` -- and must not have one, since a swallowed failure here is
    precisely the behaviour under test.

    The AUTO half is the other side of the contract: a user gets a correct answer from the
    portable splat, while a benchmark or a test asking for METAL gets an error instead of a
    quietly slower result.
    """
    from torchref.base.electron_density.kernels.mps import compile as mps_compile

    monkeypatch.setattr(mps_compile, "_lib", None)
    monkeypatch.setattr(mps_compile, "_lib_failed", True)
    monkeypatch.setattr(mps_compile, "_lib_error", ("forced test failure", ""))

    mps = torch.device("mps")
    with pytest.raises(RuntimeError, match="forced test failure"):
        _build(scene_small, mps, torch.float32, Engine.METAL)

    # ...while AUTO still degrades to a working splat. Compared against EAGER on the same
    # device, not against the oracle: with the shader forced to fail both engines now reach
    # the *same* portable kernel, so they must agree closely, and an oracle comparison here
    # would instead be measuring this scene's discretization error (which is large -- it is
    # deliberately tiny and coarse for gradcheck) and tell us nothing about the fallback.
    # Loose rather than bitwise because MPS ``scatter_add`` accumulation order is not
    # reproducible, so two portable runs already differ at ~1e-7.
    auto = _build(scene_small, mps, torch.float32, Engine.AUTO)
    eager = _build(scene_small, mps, torch.float32, Engine.EAGER)
    rel = H.rel_l2(auto.cpu().to(torch.float64), eager.cpu().to(torch.float64))
    print(f"\n  AUTO vs EAGER after forced shader failure: relL2 {rel:.3e}")
    assert rel < 1e-5, (
        f"AUTO did not degrade to the portable splat after the shader failed (rel {rel:.3e})"
    )


@pytest.mark.parametrize("engine", [Engine.TRITON, Engine.METAL])
def test_strict_engine_rejects_cpu(scene_small, engine):
    """A strict accelerator engine on CPU must raise, not fall back.

    Both gates refuse CPU inputs, by different routes: ``should_use_triton`` raises
    directly, and ``should_use_metal`` raises on non-MPS. Either way the failure is loud,
    which is what lets the provenance tests above be meaningful — a strict engine that
    quietly degraded would make them untestable.
    """
    with pytest.raises(RuntimeError):
        _build(scene_small, torch.device("cpu"), torch.float32, engine)


# ---------------------------------------------------------------------------
# Non-vacuity
# ---------------------------------------------------------------------------
def test_aniso_scene_exercises_off_diagonal_u(scene_small, scene_fine):
    """The anisotropic scenes must have non-zero off-diagonal U.

    Ported from ``test_variable_radius_gpu.py``. Every aniso comparison in this package
    would pass on axis-aligned ellipsoids while leaving the cross-term arithmetic
    untouched -- the ``p01``/``p02``/``p12`` entries of the inverted 3x3, and the backward's
    off-diagonal U gradients, which carry a ``4*pi^2`` factor where the diagonal ones carry
    ``2*pi^2``. A scene-level assertion, since the scenes are what the tests share.
    """
    for name, scene in (("scene_small", scene_small), ("scene_fine", scene_fine)):
        off = scene.u6[:, 3:]
        assert off.abs().max() > 0, f"{name}: off-diagonal U is all zero"
        assert (off < 0).any() and (off > 0).any(), (
            f"{name}: off-diagonal U is single-signed; a sign error in the cross terms "
            "would not show up"
        )


def test_radius_policy_is_the_same_on_every_device(scene_small):
    """The per-atom truncation radius must not depend on device or dtype.

    ``splat_direct`` computes the radius itself because it bypasses the dispatch. If that
    ever diverged from what ``build_electron_density`` computes, every accuracy test in
    this package would be measuring a different truncation than production uses -- and
    would still pass, because both sides would share it.
    """
    from torchref.config import get_sigma_cutoff_ed

    ns = get_sigma_cutoff_ed()
    ref = per_atom_radius_iso(scene_small.adp, scene_small.B, n_sigma=ns)
    for device in (torch.device("cpu"), torch.device("mps")):
        if device.type == "mps" and not torch.backends.mps.is_available():
            continue
        s = scene_small.to(device=device, dtype=torch.float32)
        got = per_atom_radius_iso(s.adp, s.B, n_sigma=ns)
        assert torch.equal(got.cpu().to(ref.dtype), ref), (
            f"radius policy differs on {device.type}: the truncation contract is not "
            "device-independent"
        )


# ---------------------------------------------------------------------------
# Under-probed gates (CUDA only)
# ---------------------------------------------------------------------------
# Both of these assert the *intended* contract, not current behaviour. They were written
# from reading the gates on a host with no CUDA, so they have never run. If one fails on a
# GPU host, that is the finding -- do not loosen it to match what the code does.


@pytest.mark.cuda
def test_triton_density_gate_probes_every_tensor(scene_small):
    """A float64 ``density_map`` must not reach the float32 Triton kernel.

    ``main.py`` gates the density Triton branch on ``should_use_triton(xyz)`` -- **only
    xyz** -- while its sibling gates probe six tensors
    (``should_use_sphere_splat(density_map, xyz, adp, occ, A, B)``, and the same for Metal).
    So a float32 ``xyz`` with a float64 ``density_map`` passes, and
    ``WorkQueueGridDensity.forward`` hands that float64 buffer to a kernel doing float32
    ``tl.atomic_add``.

    The contract asserted here is that such a call fails loudly. Either outcome is
    acceptable -- a raise from the gate, or a raise from the kernel -- but silently
    returning a number is not.
    """
    cuda = torch.device("cuda")
    s = scene_small.to(device=cuda, dtype=torch.float32)
    dims = H._grid_dims(s)
    dm_f64 = torch.zeros(*dims, dtype=torch.float64, device=cuda)
    from torchref.base.electron_density.kernels.cuda.variable_radius import (
        add_isotropic_cuda_var,
    )
    from torchref.config import get_sigma_cutoff_ed

    radius = per_atom_radius_iso(s.adp, s.B, n_sigma=get_sigma_cutoff_ed())

    with pytest.raises(Exception):
        out = add_isotropic_cuda_var(
            dm_f64, s.xyz, s.adp, s.occ, s.A, s.B,
            s.inv_frac_matrix, s.frac_matrix, radius,
        )
        # If it did not raise, it must at least not have silently produced garbage.
        assert torch.isfinite(out).all() and out.abs().sum() > 0, (
            "float64 density_map + float32 Triton kernel returned a non-finite or empty "
            "map instead of raising"
        )


@pytest.mark.cuda
def test_triton_ds_does_not_silently_truncate_hkl(scene_small):
    """The Triton DS kernel must not silently downcast a float64 ``hkl``.

    ``triton_ds.py`` gates on ``xyz_frac`` alone and then force-casts every other input to
    float32 (``_cols_f32``), including ``hkl``. ``hkl`` feeds the phase
    ``phi = 2*pi*(h.r)``, the most precision-sensitive quantity in the calculation, and a
    float64 ``hkl`` is exactly what a float64 config produces (``sf_ds.py:473``).

    Asserted against the eager oracle rather than against the float32 result: the question
    is not whether the two Triton calls agree with each other, it is whether the truncation
    costs accuracy that the caller asked for by supplying float64.
    """
    from torchref.base.direct_summation.dispatch import _eager_iso
    from torchref.base.direct_summation.triton_ds import ds_iso_triton

    cuda = torch.device("cuda")
    s64 = scene_small.to(device=cuda, dtype=torch.float64)
    s32 = scene_small.to(device=cuda, dtype=torch.float32)

    ref = _eager_iso(
        s64.hkl, s64.s, s64.xyz_frac, s64.occ, s64.adp, s64.A, s64.B, None
    )
    # float64 hkl with float32 xyz_frac: the gate passes, and hkl gets truncated.
    got = ds_iso_triton(
        s64.hkl, s32.s, s32.xyz_frac, s32.occ, s32.adp, s32.A, s32.B
    )
    rel = H.rel_l2(got.cpu().to(torch.complex128), ref.cpu().to(torch.complex128))
    print(f"\n  Triton DS with float64 hkl: relL2 vs eager oracle {rel:.3e}")
    assert rel < 1e-4, (
        f"a float64 hkl into the Triton DS kernel costs {rel:.3e} against the oracle. The "
        "kernel truncates it to float32 with no warning; either it should preserve the "
        "precision or the gate should refuse the call."
    )


@pytest.mark.cuda
def test_sfds_engine_toggle_end_to_end(scene_fine):
    """``SfDS`` through a full symmetry loop: ``Engine.TRITON`` vs ``Engine.EAGER``.

    Restores the end-to-end coverage lost with ``test_ds_triton_vs_eager.py``. Distinct
    from the per-kernel DS legs in ``test_forward.py`` / ``test_gradients.py``: those call
    the kernels directly in P1, so nothing there exercises ``SfDS``'s symmetry accumulation
    or its per-call ``engine=`` argument, which **overrides** ``use_engine`` rather than
    deferring to it.

    A non-P1 group on purpose. Both engines share the symmetry algebra, so this does not
    validate the convention -- ``test_forward.py::test_sfds_matches_gemmi_with_symmetry``
    does that against gemmi. What it validates is that swapping the engine underneath a
    symmetry loop changes nothing.
    """
    from torchref.model.sf_ds import SfDS

    cuda = torch.device("cuda")
    s = scene_fine.to(device=cuda, dtype=torch.float32)
    obs = H.synthetic_obs(H.ds_direct(scene_fine, "eager").detach()).to(cuda, torch.float32)

    def run(engine):
        sf = SfDS(
            cell=s.cell, spacegroup="P212121", engine=engine,
            dtype_float=torch.float32, device=cuda, max_memory_gb=2.0,
        )
        leaves = tuple(t.clone().requires_grad_(True) for t in (s.xyz, s.adp, s.occ))
        xyz, adp, occ = leaves
        F, _ = sf.compute_structure_factors(s.hkl, xyz, adp, occ, s.A, s.B)
        grads = torch.autograd.grad(H.ls_target(F, obs), leaves)
        return F.detach(), grads

    F_t, g_t = run(Engine.TRITON)
    F_e, g_e = run(Engine.EAGER)

    rel_F = H.rel_l2(F_t.cpu().to(torch.complex128), F_e.cpu().to(torch.complex128))
    print(f"\n  SfDS P212121 TRITON vs EAGER: F relL2 {rel_F:.3e}")
    assert rel_F < RTOL_DS_F32, f"SfDS forward differs by engine: rel {rel_F:.3e}"
    for name, a, b in zip(("xyz", "adp", "occ"), g_t, g_e):
        rel = rel_error(a, b)
        print(f"    {name:4s} grad relL2 {rel:.3e}")
        assert rel < RTOL_DS_F32, f"SfDS {name} gradient differs by engine: rel {rel:.3e}"
