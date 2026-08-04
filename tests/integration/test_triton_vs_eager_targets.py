"""Automatic eager-vs-Triton equivalence sweep for every dispatched target.

For each target registered on the default LossState (xray + geometry +
ADP), we run forward + backward twice on the same model state:

  * once with Triton dispatch disabled (eager / PyTorch fallback)
  * once with Triton dispatch enabled

and assert that loss values and parameter gradients agree to within a
fp32 tolerance accounting for atomic-scatter non-determinism.

Toggling is done with the shared ``torchref.utils.use_portable`` context manager
(``use_portable()`` vs the default). No process restart needed.

Markers: ``@pytest.mark.cuda`` (skipped by default) and
``@pytest.mark.integration``. Run on a CUDA box with
``pytest tests/integration/test_triton_vs_eager_targets.py -m gpu``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import contextlib

import pytest
import torch

from torchref.refinement.targets.xray._specs import XRAY_TARGETS

# Tolerances. Most targets are bit-perfect or within 1e-7 relative;
# atomic-scatter ones (nonbonded, planarity) need a little more slack.
_DEFAULT_ATOL = 1e-2
_DEFAULT_RTOL = 1e-3

# Per-target overrides: targets whose Triton path uses approximations
# (e.g. ML's polynomial Bessel) need looser bounds. The first entry is
# atol, second is rtol.
_TARGET_TOLERANCES: Dict[str, Tuple[float, float]] = {
    # `xray/rice` had a loose tolerance here for the polynomial Bessel approximation. The
    # `rice` mode is no longer selectable (it is a private target with no Triton path), so
    # the row is gone rather than kept as a tolerance for a name nothing can produce.
    "geometry/nonbonded": (5e-3, 5e-4),      # atomic-add scatter on N pairs
    "geometry/planarity": (1e-2, 1e-3),      # SVD/eigh near-degenerate plane normals
    "geometry/ramachandran": (5e-3, 5e-4),   # bilinear interp tolerance
}


#: The x-ray modes whose loss has a Triton kernel: ``nll`` routes to
#: ``torchref.base.targets.triton.xray_nll``, ``ls`` and ``ls_wunit_k1`` to
#: ``triton.xray_ls``. The sigma_A family (``ml``, ``ml_noalpha``, ``ml_full``,
#: ``nll_beta``) is eager-only, so an A/B on those would compare the reference against
#: itself in the loss and add nothing over the density-splat coverage the ``xray`` param
#: below already gives.
_TRITON_XRAY_MODES: Tuple[str, ...] = ("nll", "ls", "ls_wunit_k1")


def _triton_xray_modes() -> Tuple[str, ...]:
    """:data:`_TRITON_XRAY_MODES`, checked against the mode table at collection time.

    Not decoration. This module named ``bhattacharyya``, ``rice`` and ``gaussian`` for as
    long as it took to run the suite on a GPU host after the 2026-08 target refactor
    deleted them, and all twelve of its params then died on a ``ValueError`` raised from
    :func:`~torchref.refinement.targets.xray.factory.create_xray_target` -- a message that
    names the factory rather than the drift. A mode renamed out from under this list now
    fails at collection, quoting the current table.
    """
    missing = sorted(set(_TRITON_XRAY_MODES) - set(XRAY_TARGETS.names))
    if missing:
        raise RuntimeError(
            f"stale x-ray mode(s) {missing} in _TRITON_XRAY_MODES; XRAY_TARGETS now "
            f"offers {', '.join(XRAY_TARGETS.names)}"
        )
    return _TRITON_XRAY_MODES


def _tol_for(name: str) -> Tuple[float, float]:
    if name in _TARGET_TOLERANCES:
        return _TARGET_TOLERANCES[name]
    return _DEFAULT_ATOL, _DEFAULT_RTOL


def _build_state(refinement):
    """Return a LossState identical to the one a refinement would build."""
    from torchref.refinement.loss_state import LossState
    from torchref.refinement.targets.adp.scaler_log_scale import (
        ScalerLogScaleTrendTarget,
    )
    from torchref.refinement.targets.adp.scaler_u import (
        ScalerURegularizationTarget,
    )

    state = LossState(device=refinement.device)
    state.register_target("xray", refinement.xray_target_work)
    state.register_targets(refinement.geometry_target)
    state.register_targets(refinement.adp_target)
    n_ref = int(refinement.reflection_data.hkl.shape[0])
    state.register_target(
        "adp/scaler_U",
        ScalerURegularizationTarget(refinement.scaler, n_reflections=n_ref),
    )
    state.register_target(
        "adp/scaler_log_scale",
        ScalerLogScaleTrendTarget(refinement.scaler, n_reflections=n_ref),
    )
    return state


def _run_target_capture_grads(target, refinement):
    """Call ``target()`` once and capture (loss, {param_name: grad.clone()})."""
    params = [
        (name, p)
        for name, p in refinement.model.named_parameters()
        if p.requires_grad and p.numel() > 0
    ]
    for _, p in params:
        if p.grad is not None:
            p.grad = None

    # Reset the SF cache so model.forward recomputes — keeps the two
    # halves of the comparison from accidentally hitting stale caches.
    refinement.model.reset_cache()
    if hasattr(target, "reset_get_data_cache"):
        target.reset_get_data_cache()

    loss = target()
    if loss.requires_grad:
        loss.backward()
    grads = {
        name: (p.grad.detach().clone() if p.grad is not None else None)
        for name, p in params
    }
    return float(loss.detach().item()), grads


def _assert_close(name: str, eager, triton, atol: float, rtol: float):
    """Compare loss + grads with a target-specific tolerance."""
    e_loss, e_grads = eager
    t_loss, t_grads = triton

    # Loss
    denom = max(abs(e_loss), 1e-9)
    loss_rel = abs(t_loss - e_loss) / denom
    assert loss_rel < rtol or abs(t_loss - e_loss) < atol, (
        f"{name}: loss mismatch eager={e_loss:.6e} triton={t_loss:.6e} "
        f"|Δ|={abs(t_loss-e_loss):.3e} rel={loss_rel:.3e} "
        f"(atol={atol}, rtol={rtol})"
    )

    # Per-param grads
    for k in sorted(set(e_grads) | set(t_grads)):
        ge, gt = e_grads.get(k), t_grads.get(k)
        if ge is None and gt is None:
            continue
        if ge is None or gt is None:
            raise AssertionError(
                f"{name}: grad {k} present in only one path "
                f"(eager={ge is not None}, triton={gt is not None})"
            )
        gmax = ge.abs().max().clamp(min=1e-12).item()
        diff = (ge - gt).abs().max().item()
        rel = diff / gmax
        # The grad comparison passes if EITHER (a) the absolute diff is
        # below atol, or (b) the relative diff is below rtol. This
        # gives a sensible cutoff across grad magnitudes that span
        # many orders of magnitude.
        assert rel < rtol or diff < atol, (
            f"{name}: grad[{k}] mismatch  max|Δ|={diff:.3e}  rel={rel:.3e}  "
            f"(grad max={gmax:.3e}, atol={atol}, rtol={rtol})"
        )


@pytest.fixture(scope="module")
def _1daw_pair(pdb_dir, mtz_dir):
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")
    return pdb, mtz


@pytest.fixture(scope="module")
def gpu_refinement(_1daw_pair):
    """Build a CUDA refinement once for the whole module."""
    from torchref import LBFGSRefinement

    device = torch.device("cuda")
    pdb, mtz = _1daw_pair
    ref = LBFGSRefinement(
        data_file=str(mtz),
        pdb=str(pdb),
        device=device,
        # Any mode with a Triton path will do -- this fixture exercises the geometry/adp
        # targets, and xray is only one row of the comparison. `nll` is the successor of the
        # retired `gaussian` and keeps the fused Triton kernel.
        target_mode="nll",
        verbose=0,
    )
    ref.scaler.initialize()
    ref.scaler.refine_lbfgs()
    return ref


@pytest.fixture(scope="module")
def gpu_state(gpu_refinement):
    return _build_state(gpu_refinement)


def _target_names(state):
    return list(state.targets.keys())


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize(
    "target_name",
    [
        "xray",
        "geometry/bond",
        "geometry/angle",
        "geometry/torsion",
        "geometry/planarity",
        "geometry/chiral",
        "geometry/nonbonded",
        "geometry/ramachandran",
        "adp/simu",
    ],
)
def test_triton_matches_eager_per_target(target_name, gpu_refinement, gpu_state):
    """For each registered target, forward + backward must match between the Triton and
    portable paths (up to the per-target tolerance).

    Four params were removed when this migrated off the ``Engine`` enum:
    ``adp/locality``, ``adp/KL``, ``adp/scaler_U`` and ``adp/scaler_log_scale`` have **no
    Triton implementation at all**, so they were comparing the eager path against itself and
    passing at ``rel == 0``. That was true before the migration too -- forcing an engine had
    never made them non-vacuous, because there was nothing on the other side.

    What keeps the remaining params honest is not the toggle. On a CUDA host
    ``test_backend_is_available_where_it_is_expected`` fails if the Triton kernels *should*
    work and do not, and a runtime fallback raises through the degradation warning. Either
    fires before this test could quietly compare the reference against itself.

    Note the ``xray`` param is broader than a loss-math A/B: it runs through
    ``LBFGSRefinement`` -> ``ModelFT`` -> ``build_electron_density``, so it also exercises the
    density splat. That is deliberate coverage, not an accident of the harness.
    """
    if target_name not in gpu_state.targets:
        pytest.skip(f"target {target_name!r} not in this state")
    target = gpu_state.targets[target_name]

    from torchref.utils import use_portable

    # ----- reference (dispatch pinned to the portable path) -----
    with use_portable():
        eager = _run_target_capture_grads(target, gpu_refinement)

    # ----- Triton (the default on CUDA float32) -----
    triton = _run_target_capture_grads(target, gpu_refinement)

    atol, rtol = _tol_for(target_name)
    _assert_close(target_name, eager, triton, atol, rtol)


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize("target_mode", _triton_xray_modes())
def test_triton_matches_eager_xray_modes(target_mode, _1daw_pair):
    """Same comparison for every xray mode that HAS a Triton path.

    Builds a fresh refinement per mode, since the mode is a constructor argument.

    The list is three of the seven selectable modes, and it is short for a reason: the
    sigma_A family (``ml``, ``ml_noalpha``, ``ml_full``, ``nll_beta``) is eager-only, so
    there is nothing to compare. It previously read
    ``["bhattacharyya", "rice", "ls", "gaussian"]`` -- three names that no longer resolve
    (bhattacharyya and rice were deleted, gaussian was renamed to nll). Because this test is
    CUDA-marked it cannot run on an sm_61 host, so those three sat here erroring at setup and
    invisible until the suite was run on a real GPU node. Hence
    :func:`_triton_xray_modes`, which now fails at collection instead.

    ``ls_wunit_k1`` shares ``ls``'s Triton kernel but not its scale: it is the one target
    overriding ``_scaled_F_calc_full``, recomputing a global scale on every gradient call,
    so it reaches the kernel with different amplitudes than ``ls`` does.

    Measured on 1DAW/CUDA: loss agrees to <= 3e-7 relative and the worst per-parameter
    gradient to <= 3.3e-5, against the default bounds of atol 1e-2 / rtol 1e-3.
    """
    from torchref import LBFGSRefinement
    from torchref.utils import use_portable

    device = torch.device("cuda")
    pdb, mtz = _1daw_pair
    ref = LBFGSRefinement(
        data_file=str(mtz),
        pdb=str(pdb),
        device=device,
        target_mode=target_mode,
        verbose=0,
    )
    ref.scaler.initialize()
    ref.scaler.refine_lbfgs()
    target = ref.xray_target_work

    with use_portable():
        eager = _run_target_capture_grads(target, ref)
    with contextlib.nullcontext():  # the default: Triton on CUDA float32
        triton = _run_target_capture_grads(target, ref)

    name = f"xray/{target_mode}"
    atol, rtol = _tol_for(name)
    _assert_close(name, eager, triton, atol, rtol)


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize("n_atoms", [4, 5, 6])
def test_planarity_triton_per_atom_sigma(n_atoms):
    """Regression: planarity must use PER-ATOM sigmas (P, n_atoms).

    Plane restraint sigmas are stored per-atom. The Triton kernel used to
    read them as one-per-plane (P,), which only happens to agree when all
    atoms in a plane share a sigma (the monomer-library default). With
    non-uniform per-atom sigmas the old kernel diverged badly (>50% loss
    error). This pins eager == Triton for non-uniform sigmas.
    """
    from torchref.base.targets.planarity import (
        _planarity_math_eager,
        planarity_math,
    )
    from torchref.utils import use_portable

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(7)
    P, N = 6, 40
    idx = torch.randint(0, N, (P, n_atoms), generator=g, device=dev)
    # Non-uniform per-atom sigmas — the case the old kernel got wrong.
    sigmas = (torch.rand(P, n_atoms, generator=g, device=dev) * 0.08 + 0.01).float()
    base = (torch.rand(N, 3, generator=g, device=dev) * 10).float()

    xe = base.clone().requires_grad_(True)
    xt = base.clone().requires_grad_(True)
    with use_portable():
        le = _planarity_math_eager(xe, [(idx, sigmas)])
    (ge,) = torch.autograd.grad(le, xe)
    with contextlib.nullcontext():  # the default: Triton on CUDA float32
        lt = planarity_math(xt, [(idx, sigmas)])
    (gt,) = torch.autograd.grad(lt, xt)

    assert torch.allclose(
        le, lt, atol=1e-3, rtol=1e-4
    ), f"loss mismatch: eager={le.item()} triton={lt.item()}"
    assert torch.allclose(ge, gt, atol=1e-4, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.integration
def test_geometry_degenerate_finite_grads():
    """At degenerate geometry, BOTH eager and Triton give finite gradients.

    Regression for the NaN-safety fixes: collinear angles/torsions and
    coincident bonds used to yield NaN gradients (acos / norm / dihedral
    singularities). Both backends are now floored with EPS so the gradient is
    finite (CPU == GPU behavior), avoiding NaN-poisoned refinement steps.
    """
    from torchref.base.targets.angle import angle_math
    from torchref.base.targets.bond import bond_math
    from torchref.base.targets.torsion import torsion_omega_math
    from torchref.utils import use_portable

    dev = torch.device("cuda")
    # Collinear chain of points -> degenerate angle & torsion; plus a
    # coincident pair for a zero-length bond.
    xyz0 = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [1.0, 0, 0]],
        device=dev,
    )
    cases = [
        (
            "angle",
            angle_math,
            (
                torch.tensor([[0, 1, 2]], device=dev),
                torch.tensor([1.9], device=dev),
                torch.tensor([0.05], device=dev),
            ),
        ),
        (
            "bond",
            bond_math,
            (
                torch.tensor([[1, 4]], device=dev),
                torch.tensor([1.5], device=dev),
                torch.tensor([0.02], device=dev),
            ),
        ),
        (
            "torsion",
            torsion_omega_math,
            (
                torch.tensor([[0, 1, 2, 3]], device=dev),
                torch.tensor([3.0], device=dev),
                torch.zeros(1, dtype=torch.bool, device=dev),
            ),
        ),
    ]
    for name, fn, args in cases:
        for pin in (True, False):
            x = xyz0.clone().requires_grad_(True)
            with (use_portable() if pin else contextlib.nullcontext()):
                loss = fn(x, *args)
            (grad,) = torch.autograd.grad(loss, x)
            path = "portable" if pin else "default"
            assert torch.isfinite(grad).all(), f"{name}/{path}: non-finite grad"


def _hvp_vs_fd(model, hkl, eps=1e-5):
    """Return (cosine, rel_err) of the autograd Hessian-vector product vs a
    central finite difference of the gradient, through ``model.forward`` under
    ``use_portable()`` (the genuine pure-torch, double-differentiable
    ED path)."""
    from torchref.utils import use_portable

    x = model.xyz.refinable_params
    x0 = x.detach().clone()
    gen = torch.Generator(device=x.device).manual_seed(1)
    v = torch.randn(x0.shape, generator=gen, device=x.device, dtype=x0.dtype)
    v /= v.norm()

    def grad_at(xv):
        with torch.no_grad():
            x.copy_(xv)
            model.reset_cache()
        with use_portable():
            sf = model(hkl, recalc=True)
        return (
            torch.autograd.grad((sf.real**2 + sf.imag**2).sum(), x)[0].detach().clone()
        )

    with torch.no_grad():
        x.copy_(x0)
        model.reset_cache()
    with use_portable():
        sf = model(hkl, recalc=True)
    (g1,) = torch.autograd.grad((sf.real**2 + sf.imag**2).sum(), x, create_graph=True)
    (hvp,) = torch.autograd.grad((g1 * v).sum(), x)
    hvp = hvp.detach().clone()
    hvp_fd = (grad_at(x0 + eps * v) - grad_at(x0 - eps * v)) / (2 * eps)
    with torch.no_grad():
        x.copy_(x0)
        model.reset_cache()
    cos = torch.nn.functional.cosine_similarity(
        hvp.flatten(), hvp_fd.flatten(), dim=0
    ).item()
    rel = ((hvp - hvp_fd).norm() / hvp_fd.norm()).item()
    assert torch.isfinite(hvp).all(), "non-finite Hessian-vector product"
    return cos, rel


@pytest.mark.cuda
@pytest.mark.integration
def test_eager_gpu_hessian_iso(tmp_path):
    """``use_portable()`` gives correct GPU second derivatives (iso).

    The fast GPU density splat is first-order only (custom-Function backward
    with no grad_fn) and silently drops the second-order term under
    create_graph. Under ``use_portable()`` the iso splat now routes to the pure-
    torch (scatter_add) path, which composes under autograd. A Hessian-vector
    product through ``ModelFT.forward`` must then match finite differences.
    """
    import itertools

    from torchref.config import dtypes
    from torchref.model.model_ft import ModelFT

    f0, c0 = dtypes.float, dtypes.complex
    dtypes.float, dtypes.complex = torch.float64, torch.complex128
    try:
        pdb = tmp_path / "p1.pdb"
        g = torch.Generator().manual_seed(0)
        lines = [
            "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1"
        ]
        for i in range(10):
            x, y, z = (torch.rand(3, generator=g) * 24 + 3).tolist()
            lines.append(
                f"ATOM  {i + 1:5d}  C   GLY A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
        lines.append("END")
        pdb.write_text("\n".join(lines) + "\n")
        m = ModelFT(max_res=1.5, verbose=0, device=torch.device("cuda"))
        m.load_pdb(str(pdb))
        hkl = torch.tensor(
            [h for h in itertools.product(range(-3, 4), repeat=3) if any(h)],
            dtype=torch.float64,
            device="cuda",
        )
        cos, rel = _hvp_vs_fd(m, hkl)
        assert cos > 0.9999 and rel < 1e-4, f"iso HVP cosine={cos:.6f} rel={rel:.2e}"
    finally:
        dtypes.float, dtypes.complex = f0, c0


@pytest.mark.cuda
@pytest.mark.integration
def test_eager_gpu_hessian_aniso(pdb_dir):
    """Same as above but for anisotropic ADPs (real ANISOU structure)."""
    import itertools

    from torchref.config import dtypes
    from torchref.model.model_ft import ModelFT

    pdb = pdb_dir / "7L84.pdb"
    if not pdb.exists():
        pytest.skip("7L84.pdb fixture not present")
    f0, c0 = dtypes.float, dtypes.complex
    dtypes.float, dtypes.complex = torch.float64, torch.complex128
    try:
        m = ModelFT(max_res=1.5, verbose=0, device=torch.device("cuda"))
        m.load_pdb(str(pdb))
        assert m.get_aniso()[0].shape[0] > 0, "expected anisotropic atoms"
        hkl = torch.tensor(
            [h for h in itertools.product(range(-3, 4), repeat=3) if any(h)],
            dtype=torch.float64,
            device="cuda",
        )
        cos, rel = _hvp_vs_fd(m, hkl)
        assert cos > 0.9999 and rel < 1e-4, f"aniso HVP cosine={cos:.6f} rel={rel:.2e}"
    finally:
        dtypes.float, dtypes.complex = f0, c0
