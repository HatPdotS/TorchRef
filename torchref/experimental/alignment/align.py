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

from .lattman_love import LattmanLoveInterpolator
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

    Wraps a `ModelFT` to expose the same `.evaluate(R, hkl, cell, ...)` API
    as `LattmanLoveInterpolator`, for use by the translation search.
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

    The resolution-masked, anisotropy-corrected reflection arrays, plus the
    overall anisotropy tensor and the Lattman-Love interpolator. The rotation
    search reads `U_aniso` and `device`; the ML rescore, translation search and
    rigid-body polish read the rest.
    """
    F_obs: torch.Tensor              # (N,) anisotropy-corrected amplitudes
    hkl: torch.Tensor                # (N, 3) integer Miller indices
    s_vec: torch.Tensor              # (N, 3) reciprocal-space Cartesian
    s_mag: torch.Tensor              # (N,) Å⁻¹
    centric: torch.Tensor            # (N,) bool
    ll: "LattmanLoveInterpolator"
    U_aniso: torch.Tensor            # (3, 3) Popov-Bourenkov U
    device: torch.device


def _prepare_frf_inputs(
    model: "ModelFT",
    data: "ReflectionData",
    *,
    d_min: float,
    d_max: float,
    n_shells: int,
    ll_padding_factor: float = 2.0,
    ll_max_res_A: float = 3.0,
    verbose: int = 0,
) -> FRFInputs:
    """Prepare the reflection arrays the rotation-search stages share.

    Masks the observations to ``[d_min, d_max]``, fits and applies the overall
    anisotropy correction, and builds the Lattman-Love interpolator for the
    model. ``F_obs`` on the returned dataclass is anisotropy-corrected.
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

    ll = LattmanLoveInterpolator(
        model, padding_factor=ll_padding_factor, max_res_A=ll_max_res_A,
        verbose=verbose,
    )

    return FRFInputs(
        F_obs=F_obs_aniso,
        hkl=hkl,
        s_vec=s_vec,
        s_mag=s_mag,
        centric=centric,
        ll=ll,
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
    n_ml_refine: int = 20,  # rescore only the top-20 FRF peaks (refinement use case)
    ll_max_res_A: float = 3.0,
    ll_padding_factor: float = 2.0,
    verbose: int = 0,
    auto_variance_weights: bool = True,
    do_translation: bool = True,
    n_translation_peaks: int = 20,
    n_translation_candidates: int = 3,
    translation_grid_steps: int = 16,
    n_rotation_candidates: int = 15,
    do_joint_refine: bool = True,
    joint_refine_max_res_A: float = 4.0,
    joint_refine_expected_rot_error: float = 0.1,
    use_interp_var: bool = False,
    use_llg_tf: bool = False,
    refine_b: bool = False,
    sigma_rot_deg: float = 0.0,
    sigma_trans_ang: float = 0.0,
    sigma_b: float = 0.0,
    model_error_A: Optional[float] = None,
    rescore_engine: str = "m_letf1",
    rescore_scat_mode: str = "legacy",
    subpeak_refine: bool = False,
    subpeak_refine_k: int = -1,
    subpeak_refine_step_deg: float = 1.5,
    subpeak_refine_iters: int = 1,
    subpeak_refine_max_move_deg: Optional[float] = 1.5,
) -> "ModelFT":
    """Run full MR alignment of ``model`` against ``data``.

    Returns a new rotated+translated+refined ``ModelFT`` carrying
    ``last_alignment_rotation``, ``last_alignment_translation`` and
    ``last_alignment_rfactor`` provenance attributes.

    `MolecularReplacementPipeline` is the implementation of record; this
    function returns its single best solution.
    """
    if not model.initialized:
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
        ll_max_res_A=ll_max_res_A, ll_padding_factor=ll_padding_factor,
        n_rotation_peaks=n_rotation_peaks, n_ml_refine=n_ml_refine,
        model_error_A=model_error_A,
        rescore_engine=rescore_engine, rescore_scat_mode=rescore_scat_mode,
        auto_variance_weights=auto_variance_weights,
        use_interp_var=use_interp_var,
        subpeak_refine=subpeak_refine, subpeak_refine_k=subpeak_refine_k,
        subpeak_refine_step_deg=subpeak_refine_step_deg,
        subpeak_refine_iters=subpeak_refine_iters,
        subpeak_refine_max_move_deg=subpeak_refine_max_move_deg,
        n_rotation_candidates=n_rotation_candidates,
        n_translation_peaks=n_translation_peaks,
        n_translation_candidates=n_translation_candidates,
        translation_grid_steps=translation_grid_steps,
        use_llg_tf=use_llg_tf,
        do_joint_refine=do_joint_refine,
        joint_refine_max_res_A=joint_refine_max_res_A,
        joint_refine_expected_rot_error=joint_refine_expected_rot_error,
        refine_b=refine_b,
        sigma_rot_deg=sigma_rot_deg, sigma_trans_ang=sigma_trans_ang,
        sigma_b=sigma_b,
    )
    solutions = pipeline.run(do_translation=do_translation)
    return solutions[0].model
