"""Dispatch end to end, as distinct from the selection *decision*.

Deliberately separate from the accuracy tests. Those call each kernel **directly**, so they
say nothing about whether ``build_electron_density`` would have chosen it — and that
separation is the point.

Also distinct from ``tests/unit/utils/test_backend_tables.py``, which asserts what ``select``
returns. That is a pure function and can be interrogated for any ``(device, dtype)``; what it
cannot show is that the chosen kernel was then actually *called*. So what is pinned here is
provenance and fallback behaviour, both of which need a real dispatch on a real device:

* on an accelerator host, the default really reaches the native kernel rather than the
  portable splat (call recorder, patched at the kernel's defining module);
* ``force_portable`` reaches the portable splat on every device;
* with an accelerator forced dead, the fallback produces a numerically correct map.

Ported from ``tests/integration/test_variable_radius_{gpu,mps}.py``, which are deleted: their
accuracy coverage is superseded by the oracle legs in this package, but these dispatch
contracts are not accuracy and had no replacement.
"""

from __future__ import annotations

import pytest
import torch

from torchref.base.electron_density.main import build_electron_density
from torchref.base.electron_density.radius_policy import per_atom_radius_iso

from tests.helpers.grad_asserts import rel_error

from . import RTOL_DS_F32
from . import helpers as H

pytestmark = pytest.mark.unit


def _build(scene, device, dtype, force_portable=False, aniso=False):
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
    kw["force_portable"] = force_portable
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
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_metal_kernel_is_actually_dispatched(scene_small, kind, monkeypatch):
    """On MPS float32, the default must call the Metal kernel.

    Verified with a call recorder, not by comparing maps: the portable reference uses
    ``scatter_add`` on MPS, whose accumulation order is not reproducible, so two portable
    runs already differ at ~1e-7 and an equality check against one would prove nothing
    either way.

    **The patch target is the defining module**, and it is the same rule for every backend:
    dispatch resolves each kernel by ``(module_path, attr)`` at call time, so the name to
    patch is the one the table points at. That uniformity is new -- this test and the CUDA
    one used to need different targets (a package here, ``main`` there) because the two
    ladders bound their kernels differently.

    The recorder delegates to the real kernel rather than stubbing it, so the dispatch is
    still exercised end to end.
    """
    from torchref.base.electron_density.kernels.mps import variable_radius as kernels_mps

    name = f"add_{'anisotropic' if kind == 'aniso' else 'isotropic'}_mps_var"
    real = getattr(kernels_mps, name)
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kernels_mps, name, recording)
    _build(scene_small, torch.device("mps"), torch.float32, aniso=kind == "aniso")
    assert calls, f"the default did not dispatch to {name}"


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_triton_kernel_is_actually_dispatched(scene_small, kind, monkeypatch):
    """CUDA float32 equivalent of the Metal provenance test.

    Same patch target rule as the Metal case -- the module that defines the kernel. The two
    used to differ, which is why they are separate tests rather than one parametrized
    helper; that reason is gone, and merging them is a follow-up worth doing on a host that
    has both backends to run it on.
    """
    from torchref.base.electron_density.kernels.cuda import variable_radius as kernels_cuda

    name = f"add_{'anisotropic' if kind == 'aniso' else 'isotropic'}_cuda_var"
    real = getattr(kernels_cuda, name)
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kernels_cuda, name, recording)
    _build(scene_small, torch.device("cuda"), torch.float32, aniso=kind == "aniso")
    assert calls, f"the default did not dispatch to {name}"


@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_force_portable_reaches_the_portable_splat(scene_small, kind, monkeypatch):
    """``force_portable`` must reach the portable splat, on any device.

    The counterpart to the provenance tests above, and the only override the dispatcher has:
    it exists so you can pin the reference implementation when you suspect an accelerator is
    returning wrong numbers, so it has to be the *portable* kernel that runs and not whatever
    the default would have picked.
    """
    from torchref.base.electron_density.kernels.cpu import (
        variable_radius as kernels_portable,
    )

    name = f"add_{'anisotropic' if kind == 'aniso' else 'isotropic'}_plain_var"
    real = getattr(kernels_portable, name)
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kernels_portable, name, recording)
    _build(scene_small, torch.device("cpu"), torch.float32, force_portable=True,
           aniso=kind == "aniso")
    assert calls, f"force_portable did not dispatch to {name}"


# ---------------------------------------------------------------------------
# Failure policy: strict engines raise, AUTO degrades
# ---------------------------------------------------------------------------
@pytest.mark.mps
def test_default_degrades_correctly_when_the_shader_is_unavailable(scene_small, monkeypatch):
    """With the shader dead, dispatch must still produce a *correct* map.

    Selection tests prove the *decision*; this proves the fallback's numbers. It is the only
    place that runs the degradation end to end rather than asserting which row was chosen.

    The half of this test that asserted ``Engine.METAL`` raises is gone with the enum. Its
    job -- making a dead accelerator loud rather than silently slow -- is now done by two
    other things: ``test_backend_is_available_where_it_is_expected`` fails on any run on an
    MPS host where the shader should work and doesn't, and the degradation warning is promoted
    to an error under pytest. Both are strictly earlier and broader than a forced engine was.

    ``monkeypatch`` reverts the module globals at teardown, so this needs no
    ``try``/``finally`` -- and must not have one, since a swallowed failure here is precisely
    the behaviour under test.
    """
    from torchref.base.electron_density.kernels.mps import compile as mps_compile

    monkeypatch.setattr(mps_compile, "_lib", None)
    monkeypatch.setattr(mps_compile, "_lib_failed", True)
    monkeypatch.setattr(mps_compile, "_lib_error", ("forced test failure", ""))

    mps = torch.device("mps")
    # The default degrades to a working splat. Compared against the pinned portable path
    # device, not against the oracle: with the shader forced to fail both engines now reach
    # the *same* portable kernel, so they must agree closely, and an oracle comparison here
    # would instead be measuring this scene's discretization error (which is large -- it is
    # deliberately tiny and coarse for gradcheck) and tell us nothing about the fallback.
    # Loose rather than bitwise because MPS ``scatter_add`` accumulation order is not
    # reproducible, so two portable runs already differ at ~1e-7.
    auto = _build(scene_small, mps, torch.float32)
    portable = _build(scene_small, mps, torch.float32, force_portable=True)
    rel = H.rel_l2(auto.cpu().to(torch.float64), portable.cpu().to(torch.float64))
    print(f"\n  default vs portable after forced shader failure: relL2 {rel:.3e}")
    assert rel < 1e-5, (
        f"dispatch did not degrade to the portable splat after the shader failed "
        f"(rel {rel:.3e})"
    )


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
def test_triton_density_gate_probes_every_tensor(scene_small, monkeypatch):
    """A float64 ``density_map`` must not reach the float32 Triton kernel.

    Written when the density Triton branch gated on ``should_use_triton(xyz)`` -- **only
    xyz** -- while its sibling gates probed six tensors. A float32 ``xyz`` with a float64
    ``density_map`` therefore passed, and ``WorkQueueGridDensity.forward`` handed that
    float64 buffer to a kernel doing float32 ``tl.atomic_add``.

    The ``cuda_triton`` row now declares ``probes`` covering all six, so the mixed set
    resolves to ``portable`` instead. That is asserted at **two** layers, because they can
    fail independently: the table can name the right probes while the dispatch site forgets
    to consult it, and the site can consult a table whose probes are wrong.

    This originally wrapped a direct ``add_isotropic_cuda_var`` call in
    ``pytest.raises(Exception)``, which was the wrong question in two ways. That wrapper is
    documented not to re-gate ("CUDA float32 only -- the gate is ``should_use_triton``; this
    wrapper does not re-check"), so *not* raising is its correct behaviour -- and the
    ``pytest.raises`` block made its own follow-up assertion unreachable. The guard the test
    was written for lives at selection, so that is where it is asked.

    What is *not* asserted is that some kernel then serves the mixed set. None does, and none
    should: a float64 map beside float32 atoms is a caller error, and ``portable`` -- which
    takes its working dtype from the map -- raises on the float32 cell matrices rather than
    quietly picking one precision. That raise is the "fails loudly" half of the original
    contract; the half worth pinning is that it comes from the reference kernel on the host
    side and not from a float32 ``tl.atomic_add`` walking a float64 buffer.
    """
    from torchref.base.electron_density._backends import DENSITY_BACKENDS
    from torchref.base.electron_density.kernels.cuda import (
        variable_radius as kernels_cuda,
    )
    from torchref.base.electron_density.main import _add_isotropic
    from torchref.config import get_sigma_cutoff_ed
    from torchref.utils.backends import select

    cuda = torch.device("cuda")
    s = scene_small.to(device=cuda, dtype=torch.float32)
    dims = H._grid_dims(s)
    dm_f64 = torch.zeros(*dims, dtype=torch.float64, device=cuda)
    radius = per_atom_radius_iso(s.adp, s.B, n_sigma=get_sigma_cutoff_ed())

    # Layer 1: the decision. Position 0 (the map) is probed, so float64 fails
    # ``dtypes=(float32,)``; ``mps_metal``/``cpu_sphere`` are device-disjoint from CUDA, so
    # the only row left is the base case.
    args = (dm_f64, s.xyz, s.adp, s.occ, s.A, s.B,
            s.inv_frac_matrix, s.frac_matrix, radius)
    chosen = select(DENSITY_BACKENDS, args)
    assert chosen.name == "portable", (
        f"a float64 density map beside float32 atoms selected {chosen.name!r}; the "
        "float32-only Triton kernel would have received a float64 buffer for its "
        "tl.atomic_add"
    )

    # Layer 2: the dispatch site honours that decision. Separable from layer 1 -- the table
    # can name the right probes while the site forgets to consult it -- and asserted by
    # provenance rather than by the return value, since nothing serves this input.
    reached_triton = []
    monkeypatch.setattr(
        kernels_cuda,
        "add_isotropic_cuda_var",
        lambda *a, **k: reached_triton.append(1),
    )
    with pytest.raises(RuntimeError):
        _add_isotropic(
            dm_f64, s.xyz, s.adp, s.occ, s.A, s.B,
            s.inv_frac_matrix, s.frac_matrix,
        )
    assert not reached_triton, (
        "dispatch called the CUDA Triton splat with a float64 density map, so it is not "
        "consulting the probes declared on the cuda_triton row"
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
def test_sfds_backend_toggle_end_to_end(scene_fine):
    """``SfDS`` through a full symmetry loop: the Triton DS kernel vs the reference.

    Restores the end-to-end coverage lost with ``test_ds_triton_vs_eager.py``. Distinct from
    the per-kernel DS legs in ``test_forward.py`` / ``test_gradients.py``: those call the
    kernels directly in P1, so nothing there exercises ``SfDS``'s symmetry accumulation or its
    per-call ``force_portable`` argument.

    A non-P1 group on purpose. Both backends share the symmetry algebra, so this does not
    validate the convention -- ``test_forward.py::test_sfds_matches_gemmi_with_symmetry`` does
    that against gemmi. What it validates is that swapping the backend underneath a symmetry
    loop changes nothing.

    Non-vacuity does not depend on being able to *force* Triton: on a CUDA host
    ``test_backend_is_available_where_it_is_expected`` fails if ``ds_triton`` should work and
    does not, and a runtime fallback raises via the degradation warning. Both fire before this
    test could quietly compare the reference against itself.
    """
    from torchref.model.sf_ds import SfDS

    cuda = torch.device("cuda")
    s = scene_fine.to(device=cuda, dtype=torch.float32)
    obs = H.synthetic_obs(H.ds_direct(scene_fine, "eager").detach()).to(cuda, torch.float32)

    def run(force_portable):
        sf = SfDS(
            cell=s.cell, spacegroup="P212121", force_portable=force_portable,
            dtype_float=torch.float32, device=cuda, max_memory_gb=2.0,
        )
        leaves = tuple(t.clone().requires_grad_(True) for t in (s.xyz, s.adp, s.occ))
        xyz, adp, occ = leaves
        F, _ = sf.compute_structure_factors(s.hkl, xyz, adp, occ, s.A, s.B)
        grads = torch.autograd.grad(H.ls_target(F, obs), leaves)
        return F.detach(), grads

    F_t, g_t = run(False)   # default: the Triton DS kernel on CUDA float32
    F_e, g_e = run(True)    # pinned to the checkpointed reference

    rel_F = H.rel_l2(F_t.cpu().to(torch.complex128), F_e.cpu().to(torch.complex128))
    print(f"\n  SfDS P212121 triton vs portable: F relL2 {rel_F:.3e}")
    assert rel_F < RTOL_DS_F32, f"SfDS forward differs by engine: rel {rel_F:.3e}"
    for name, a, b in zip(("xyz", "adp", "occ"), g_t, g_e):
        rel = rel_error(a, b)
        print(f"    {name:4s} grad relL2 {rel:.3e}")
        assert rel < RTOL_DS_F32, f"SfDS {name} gradient differs by engine: rel {rel:.3e}"
