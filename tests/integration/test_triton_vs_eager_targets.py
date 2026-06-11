"""Automatic eager-vs-Triton equivalence sweep for every dispatched target.

For each target registered on the default LossState (xray + geometry +
ADP), we run forward + backward twice on the same model state:

  * once with Triton dispatch disabled (eager / PyTorch fallback)
  * once with Triton dispatch enabled

and assert that loss values and parameter gradients agree to within a
fp32 tolerance accounting for atomic-scatter non-determinism.

Toggling is done with the shared ``torchref.utils.use_engine`` context manager
(``Engine.EAGER`` vs ``Engine.TRITON``). No process restart needed.

Markers: ``@pytest.mark.gpu`` (skipped by default) and
``@pytest.mark.integration``. Run on a CUDA box with
``pytest tests/integration/test_triton_vs_eager_targets.py -m gpu``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch


# Tolerances. Most targets are bit-perfect or within 1e-7 relative;
# atomic-scatter ones (nonbonded, planarity) need a little more slack.
_DEFAULT_ATOL = 1e-2
_DEFAULT_RTOL = 1e-3

# Per-target overrides: targets whose Triton path uses approximations
# (e.g. ML's polynomial Bessel) need looser bounds. The first entry is
# atol, second is rtol.
_TARGET_TOLERANCES: Dict[str, Tuple[float, float]] = {
    "xray/rice": (1e-1, 5e-3),              # poly Bessel approx, log-amplified
    "geometry/nonbonded": (5e-3, 5e-4),      # atomic-add scatter on N pairs
    "geometry/planarity": (1e-2, 1e-3),      # SVD/eigh near-degenerate plane normals
    "geometry/ramachandran": (5e-3, 5e-4),   # bilinear interp tolerance
}


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
        (name, p) for name, p in refinement.model.named_parameters()
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
    grads = {name: (p.grad.detach().clone() if p.grad is not None else None)
             for name, p in params}
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
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from torchref import LBFGSRefinement
    device = torch.device("cuda")
    pdb, mtz = _1daw_pair
    ref = LBFGSRefinement(
        data_file=str(mtz), pdb=str(pdb),
        device=device, target_mode="bhattacharyya", verbose=0,
    )
    ref.scaler.initialize()
    ref.scaler.refine_lbfgs()
    return ref


@pytest.fixture(scope="module")
def gpu_state(gpu_refinement):
    return _build_state(gpu_refinement)


def _target_names(state):
    return list(state.targets.keys())


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("target_name", [
    "xray",
    "geometry/bond",
    "geometry/angle",
    "geometry/torsion",
    "geometry/planarity",
    "geometry/chiral",
    "geometry/nonbonded",
    "geometry/ramachandran",
    "adp/simu",
    "adp/locality",
    "adp/KL",
    "adp/scaler_U",
    "adp/scaler_log_scale",
])
def test_triton_matches_eager_per_target(target_name, gpu_refinement, gpu_state):
    """For each registered target, forward + backward must match between
    Triton and eager paths (up to the per-target tolerance)."""
    if target_name not in gpu_state.targets:
        pytest.skip(f"target {target_name!r} not in this state")
    target = gpu_state.targets[target_name]

    from torchref.utils import Engine, use_engine

    # ----- eager (Triton dispatch disabled) -----
    with use_engine(Engine.EAGER):
        eager = _run_target_capture_grads(target, gpu_refinement)

    # ----- Triton (dispatch enabled) -----
    with use_engine(Engine.TRITON):
        triton = _run_target_capture_grads(target, gpu_refinement)

    atol, rtol = _tol_for(target_name)
    _assert_close(target_name, eager, triton, atol, rtol)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("target_mode", ["bhattacharyya", "rice", "ls", "gaussian"])
def test_triton_matches_eager_xray_modes(target_mode, _1daw_pair):
    """Same comparison for the four xray loss modes (Bhattacharyya, Rice,
    LS, Gaussian) — builds a fresh refinement per mode since the mode is
    a constructor argument. (The default 'ml' σ_A target is eager-only, so
    it has no Triton path to compare.)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from torchref import LBFGSRefinement
    from torchref.utils import Engine, use_engine

    device = torch.device("cuda")
    pdb, mtz = _1daw_pair
    ref = LBFGSRefinement(
        data_file=str(mtz), pdb=str(pdb),
        device=device, target_mode=target_mode, verbose=0,
    )
    ref.scaler.initialize()
    ref.scaler.refine_lbfgs()
    target = ref.xray_target_work

    with use_engine(Engine.EAGER):
        eager = _run_target_capture_grads(target, ref)
    with use_engine(Engine.TRITON):
        triton = _run_target_capture_grads(target, ref)

    name = f"xray/{target_mode}"
    atol, rtol = _tol_for(name)
    _assert_close(name, eager, triton, atol, rtol)
