"""
Molecular replacement: data-prep / FRF stage helpers + the public entry point.

This module hosts the heavy, reusable stage helpers — Lattman-Love / anisotropy
data prep (`_prepare_frf_inputs`), the solvent-aware R-work
(`_external_rwork`), the direct-SF translation evaluator
(`_DirectModelEvaluator`) and the stage
timer (`_StageTimer`) — that are shared by the rotation-ranking benchmarks and
by the orchestrator.

`align_model_to_data` is the public entry point. It delegates the
FRF → FTF(per-candidate) → post-refine control flow to
:class:`torchref.experimental.alignment.pipeline.MolecularReplacementPipeline`,
returning that pipeline's single best `ModelFT`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

from .sh import (
    apply_overall_anisotropy,
    assign_shells,
    equal_count_shell_edges,
    fit_overall_anisotropy,
)

if TYPE_CHECKING:
    from ...io.datasets.reflection_data import ReflectionData
    from ...model.model_ft import ModelFT


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------


class _StageTimer:
    """Lightweight wall-clock accumulator. Gated by ``verbose >= 2``.

    Two interleavable usages:
      * ``with t.stage(name):`` block — records the block's wall time.
      * ``t.start(name)`` / ``t.stop(name)`` — checkpoint pair, no indent.

    The summary table prints stages aggregated by name; per-rotation loop
    stages (translation search etc.) get aggregated counts.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.records: list[tuple[str, float]] = []
        self._open: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.append((name, time.perf_counter() - t0))

    def start(self, name: str) -> None:
        if self.enabled:
            self._open[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        t0 = self._open.pop(name, None)
        if t0 is not None:
            self.records.append((name, time.perf_counter() - t0))

    def summary(self) -> str:
        if not self.records:
            return ""
        # Aggregate repeated stage names (the per-rotation loop visits the
        # translation stages once per candidate rotation).
        agg: dict[str, list[float]] = {}
        for name, dt in self.records:
            agg.setdefault(name, []).append(dt)
        total = sum(sum(v) for v in agg.values())
        lines = [
            f"{'stage':<32s}  {'count':>5s}  {'wall_s':>10s}  {'%':>6s}",
            "-" * 60,
        ]
        for name, vs in agg.items():
            wall = sum(vs)
            lines.append(
                f"{name:<32s}  {len(vs):>5d}  {wall:>10.3f}  "
                f"{100 * wall / total:>5.1f}%"
            )
        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<32s}  {'':>5s}  {total:>10.3f}  100.0%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _external_rwork(model: "ModelFT", data: "ReflectionData") -> float:
    """Full-resolution scaled R-work via the standard Scaler.

    The TF + local refine work in analytical-scale R-factor (which ranks
    candidates correctly but isn't the user-facing R-work). We compute the
    proper Scaler-fit R-work once per finalist.
    """
    from ...base.metrics.rfactor import rfactor_work_free
    from ...scaling import Scaler

    # Build the Scaler on the model's device so that its anisotropy U
    # tensor and per-bin scales land alongside `data.hkl`/`model(hkl)` —
    # otherwise Scaler.forward's `matmul(self.s, U)` mixes CPU/GPU and
    # crashes at refine_lbfgs.
    s = Scaler(model=model, data=data, nbins=20, verbose=0,
               device=model.xyz().device)
    # Detach the model forward — the scaler only needs gradients through its
    # own parameters; leaving `fc` attached to the model's autograd graph
    # keeps SfFFT density-build intermediates alive after this function
    # returns.
    with torch.no_grad():
        fc = model(data.hkl).detach()
    s.initialize(fc)
    s.refine_lbfgs(fcalc=fc)
    with torch.no_grad():
        # rfactor_work_free takes already-scaled amplitudes, not complex F_calc.
        rw, _ = rfactor_work_free(data, torch.abs(s.forward(fc)))
    return rw.item() if hasattr(rw, "item") else float(rw)


class _DirectModelEvaluator:
    """Returns ``F_p1(hkl)`` of a P1-spacegroup model at integer HKL.

    The translation search asks its evaluator for ``F`` at a list of rotated
    Miller indices. The rotation is already baked into the model's coordinates
    by the time this is built, so ``R`` is ignored and every call is a direct
    structure-factor evaluation rather than an interpolation.
    """

    def __init__(self, m: "ModelFT") -> None:
        self._m = m
        self.device = m.xyz().device

    def evaluate(self, R, hkl, real_cell, return_amplitude=False):
        hkl_int = hkl.round().to(torch.int64).to(self.device)
        with torch.no_grad():
            f = self._m(hkl_int)
        return f.abs() if return_amplitude else f


# ---------------------------------------------------------------------------
# FRF input preparation (shared by the live pipeline and the rotation-ranking
# benchmark in tests/integration/alignment/benchmark_rotation_ranking.py)
# ---------------------------------------------------------------------------


@dataclass
class FRFInputs:
    """Prepared reflection arrays shared by the rotation-search stages.

    The resolution-masked, anisotropy-corrected reflection arrays plus the
    overall anisotropy tensor. The rotation search reads ``U_aniso`` and
    ``device``; the rest is there for anything scoring against the same
    observations on the same footing.

    ``sig_F`` carries the same anisotropy correction as ``F_obs``, which is a
    multiplicative factor, so ``F/sigma`` is unchanged by it. It is here because
    the rotation function computes its measurement weight from the sigmas, and
    the earlier code threw them away immediately afterwards. ``None`` when the
    data carry no sigmas.
    """
    F_obs: torch.Tensor              # (N,) anisotropy-corrected amplitudes
    sig_F: Optional[torch.Tensor]    # (N,) their sigmas, same correction
    hkl: torch.Tensor                # (N, 3) integer Miller indices
    s_vec: torch.Tensor              # (N, 3) reciprocal-space Cartesian
    s_mag: torch.Tensor              # (N,) Å⁻¹
    centric: torch.Tensor            # (N,) bool
    U_aniso: torch.Tensor            # (3, 3) Popov-Bourenkov U
    device: torch.device


def _prepare_frf_inputs(
    model: "ModelFT",
    data: "ReflectionData",
    *,
    d_min: float,
    d_max: float,
    n_shells: int,
    verbose: int = 0,
) -> FRFInputs:
    """Prepare the reflection arrays the rotation-search stages share.

    Masks the observations to ``[d_min, d_max]`` and fits and applies the
    overall anisotropy correction, so ``F_obs`` on the returned dataclass is
    anisotropy-corrected.
    """
    device = model.xyz().device

    F_obs = data.F.to(torch.float64).abs()
    hkl_all = data.hkl
    rec_basis = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s_vec_all = hkl_all.to(torch.float64) @ rec_basis
    s_mag_all = s_vec_all.norm(dim=-1)
    keep = (s_mag_all >= 1.0 / d_max) & (s_mag_all <= 1.0 / d_min)
    if keep.sum().item() < n_shells * 5:
        raise ValueError(
            f"Too few reflections ({keep.sum().item()}) in [{d_min},{d_max}] Å "
            f"for {n_shells} shells; widen the resolution range."
        )
    F_obs = F_obs[keep].to(device)
    sig_F = getattr(data, "F_sigma", None)
    if sig_F is not None:
        sig_F = sig_F.to(torch.float64)[keep].to(device)
    hkl = hkl_all[keep].to(device)
    s_vec = s_vec_all[keep].to(device)
    s_mag = s_mag_all[keep].to(device)
    centric = (
        data.centric[keep].to(torch.bool).to(device)
        if hasattr(data, "centric")
        else torch.zeros_like(F_obs, dtype=torch.bool)
    )

    aniso_edges, _ = equal_count_shell_edges(s_mag, n_shells)
    aniso_idx = assign_shells(s_mag, aniso_edges)
    U_aniso = fit_overall_anisotropy(
        F_obs, s_vec, aniso_idx, centric, P=n_shells, min_count=20,
    )
    # Project U onto the point-group-invariant subspace (Phaser
    # RefineANO.cc:116-142, via cctbx `site_symmetry.average_u_star`). An
    # unconstrained six-component fit can return a tensor the lattice forbids,
    # and applying that modulates the observations by a direction-dependent
    # factor the crystal cannot have. After projection: cubic -> U = lambda I
    # (one degree of freedom), tetragonal/trigonal/hexagonal -> diag(l, l, m),
    # orthorhombic -> diag(l, m, n).
    from .sh import hkl_symops_to_cartesian, symmetrize_anisotropy
    _sg_mats = data.spacegroup.matrices.to(torch.float64).to(device)
    _sym_mats_cart = hkl_symops_to_cartesian(_sg_mats, rec_basis.to(device))
    U_aniso = symmetrize_anisotropy(U_aniso, _sym_mats_cart)
    F_obs_aniso = apply_overall_anisotropy(F_obs, s_vec, U_aniso)
    # Same multiplicative factor, so F/sigma survives the correction intact.
    sig_F_aniso = (None if sig_F is None
                   else apply_overall_anisotropy(sig_F, s_vec, U_aniso))

    return FRFInputs(
        F_obs=F_obs_aniso,
        sig_F=sig_F_aniso,
        hkl=hkl,
        s_vec=s_vec,
        s_mag=s_mag,
        centric=centric,
        U_aniso=U_aniso,
        device=device,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def align_model_to_data(
    model: "ModelFT",
    data: "ReflectionData",
    *,
    d_min: float = 4.0,
    d_max: float = 15.0,
    n_shells: int = 20,
    n_rotation_peaks: int = 500,
    verbose: int = 0,
    do_translation: bool = True,
    n_translation_peaks: int = 20,
    n_translation_candidates: int = 3,
    translation_grid_steps: int = 16,
    n_rotation_candidates: int = 15,
    use_llg_tf: bool = False,
    tf_d_min: Optional[float] = None,
    tf_d_max: Optional[float] = None,
    model_error_A: Optional[float] = None,
) -> "ModelFT":
    """Place ``model`` in ``data``'s crystal: rotation search, then translation.

    Returns a new rotated+translated ``ModelFT`` carrying
    ``last_alignment_rotation``, ``last_alignment_translation`` and
    ``last_alignment_rfactor`` provenance attributes. It is a *placement*, not a
    refined structure -- refine it downstream.

    `MolecularReplacementPipeline` is the implementation of record; this
    function returns its single best solution.
    """
    if not model.ctx.initialized:
        raise RuntimeError(
            "Cannot fit an uninitialized ModelFT. Load PDB data first."
        )

    # Imported lazily to avoid an import cycle: `pipeline` imports the stage
    # helpers (`_prepare_frf_inputs`, `_external_rwork`,
    # `_DirectModelEvaluator`, `_StageTimer`) from this module.
    from .pipeline import MolecularReplacementPipeline

    pipeline = MolecularReplacementPipeline(
        data, model,
        device=model.xyz().device,
        verbose=verbose,
        d_min=d_min, d_max=d_max, n_shells=n_shells,
        n_rotation_peaks=n_rotation_peaks,
        model_error_A=model_error_A,
        n_rotation_candidates=n_rotation_candidates,
        n_translation_peaks=n_translation_peaks,
        n_translation_candidates=n_translation_candidates,
        translation_grid_steps=translation_grid_steps,
        use_llg_tf=use_llg_tf,
        tf_d_min=tf_d_min, tf_d_max=tf_d_max,
    )
    solutions = pipeline.run(do_translation=do_translation)
    return solutions[0].model
