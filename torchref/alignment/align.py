"""
End-to-end molecular replacement alignment entry point.

This module owns the full MR pipeline:

    LERF1 ball-search rotation
      → Sim MLRF rescore
      → amplitude-correlation translation search
      → local translation refine (analytical-scale R)
      → dense rotation sampling (ML-LLG, multi-pass)
      → LBFGS rigid-body polish
      → final solvent-aware Scaler refit (user-facing R-work)

The single public function `align_model_to_data` is what `ModelFT.fit_to_data`
delegates to; the latter is a thin wrapper. Keep alignment logic in this file
to keep `torchref/model/model_ft.py` focused on the FFT structure-factor model.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from .ball_search import (
    RotationPeak,
    ball_rotation_search,
    edmonds_euler_from_rotation_matrix,
    rotation_matrix_from_edmonds_euler,
)
from .lattman_love import LattmanLoveInterpolator
from .ml_rotation import sim_mlrf_rescore
from .rigid_body import RigidBodyRefinement
from .sh import (
    apply_overall_anisotropy,
    assign_shells,
    equal_count_shell_edges,
    fit_overall_anisotropy,
)
from .translation import (
    amplitude_translation_search,
    local_translation_refine,
    precompute_G_for_rotation,
)

if TYPE_CHECKING:
    from ..io.datasets.reflection_data import ReflectionData
    from ..model.model_ft import ModelFT


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


def _shellbin_norm_etrick(
    F: torch.Tensor, smag: torch.Tensor, P: int
) -> torch.Tensor:
    """Per-shell E-trick normalisation: ``E = F / sqrt(<F²>_shell)``.

    Equal-count shell binning by sorted ``smag``; small-count shells just
    inherit their normaliser from the local mean.
    """
    order = torch.argsort(smag)
    idx = torch.zeros_like(smag, dtype=torch.int64)
    chunk = smag.numel() // P
    for k in range(P):
        a = k * chunk
        b = (k + 1) * chunk if k < P - 1 else smag.numel()
        idx[order[a:b]] = k
    norm = torch.zeros_like(smag, dtype=F.dtype)
    for k in range(P):
        m = idx == k
        norm[m] = (F[m] ** 2).mean().clamp(min=1e-30).sqrt()
    return F / norm


def _external_rwork(model: "ModelFT", data: "ReflectionData") -> float:
    """Full-resolution scaled R-work via the standard Scaler.

    The TF + local refine work in analytical-scale R-factor (which ranks
    candidates correctly but isn't the user-facing R-work). We compute the
    proper Scaler-fit R-work once per finalist.
    """
    from ..scaling import Scaler

    # Build the Scaler on the model's device so that its anisotropy U
    # tensor and per-bin scales land alongside `data.hkl`/`model(hkl)` —
    # otherwise Scaler.forward's `matmul(self.s, U)` mixes CPU/GPU and
    # crashes at refine_lbfgs.
    s = Scaler(model=model, data=data, nbins=20, verbose=0,
               device=model.xyz().device)
    fc = model(data.hkl)
    s.initialize(fc)
    s.refine_lbfgs(fcalc=fc)
    rw, _ = s.rfactor(fc)
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


def _rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues axis-angle → SO(3). `omega = θ · axis` (radians).

    Accepts shape (3,) for a single rotation or (..., 3) for a batched stack
    and returns matching (3, 3) or (..., 3, 3). The small-θ limit is handled
    implicitly: sin(θ)→0 and (1-cos θ)→0 zero out the K and K² contributions
    so R→I as θ→0; `clamp(min=1e-30)` prevents NaN from axis=0/0.
    """
    if omega.dtype != torch.float64:
        omega = omega.to(torch.float64)
    is_single = omega.dim() == 1
    if is_single:
        omega = omega.unsqueeze(0)

    th = omega.norm(dim=-1, keepdim=True)           # (..., 1)
    axis = omega / th.clamp(min=1e-30)              # (..., 3)
    zeros = torch.zeros_like(axis[..., 0])
    K = torch.stack([
        torch.stack([zeros, -axis[..., 2], axis[..., 1]], dim=-1),
        torch.stack([axis[..., 2], zeros, -axis[..., 0]], dim=-1),
        torch.stack([-axis[..., 1], axis[..., 0], zeros], dim=-1),
    ], dim=-2)                                      # (..., 3, 3)

    th_b = th.unsqueeze(-1)                         # (..., 1, 1)
    sin_th = torch.sin(th_b)
    cos_th = torch.cos(th_b)

    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    eye_b = eye.expand(*omega.shape[:-1], 3, 3)
    KK = torch.matmul(K, K)
    R = eye_b + sin_th * K + (1.0 - cos_th) * KK

    if is_single:
        R = R.squeeze(0)
    return R


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def align_model_to_data(
    model: "ModelFT",
    data: "ReflectionData",
    *,
    d_min: float = 4.0,
    d_max: float = 15.0,
    L: int = 48,
    n_shells: int = 20,
    n_rotation_peaks: int = 500,
    n_ml_refine: int = 500,
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
) -> "ModelFT":
    """Run full MR alignment of ``model`` against ``data``.

    Returns a new rotated+translated+refined ``ModelFT`` carrying
    ``last_alignment_rotation``, ``last_alignment_translation`` and
    ``last_alignment_rfactor`` provenance attributes.

    See `ModelFT.fit_to_data` for full kwarg semantics — this function is the
    canonical implementation; `fit_to_data` is a thin wrapper.
    """
    from ..scaling import Scaler  # noqa: F401  (imported by _external_rwork)
    from ..symmetry import SpaceGroup

    if not model.initialized:
        raise RuntimeError(
            "Cannot fit an uninitialized ModelFT. Load PDB data first."
        )

    timer = _StageTimer(enabled=verbose >= 2)

    # Device propagation: align_model_to_data runs on the model's device by
    # default. Data tensors (data.F, data.hkl, etc.) often arrive on CPU and
    # need to be moved to match — otherwise the very first hkl-derived
    # quantity (s_mag) lives on CPU while F_calc from the GPU-resident LL
    # interpolator lives on GPU, and _shellbin_norm_etrick crashes with
    # "Expected all tensors to be on the same device".
    device = model.xyz().device

    # --- Prepare F_obs and shell-normalized E_obs in resolution range ---
    # Do the masking on CPU (data tensors arrive there) then move the
    # resolution-bounded slices to `device`. Avoids GPU-index-into-CPU
    # crashes when `keep` is on a different device than `data.centric`.
    timer.start("0_data_prep")
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
    # Index on CPU, then move slices to `device`. The model device is the
    # canonical destination; downstream stages (LL.evaluate, ball_search,
    # sim_mlrf_rescore) all infer device from their inputs.
    F_obs = F_obs[keep].to(device)
    hkl = hkl_all[keep].to(device)
    s_vec = s_vec_all[keep].to(device)
    s_mag = s_mag_all[keep].to(device)
    centric = (
        data.centric[keep].to(torch.bool).to(device)
        if hasattr(data, "centric")
        else torch.zeros_like(F_obs, dtype=torch.bool)
    )

    timer.stop("0_data_prep")

    # Popov-Bourenkov overall anisotropy correction (full variance fix):
    # Removes the direction-dependent Wilson-falloff in F_obs that otherwise
    # biases the rotation function on anisotropic and high-symmetry crystals
    # (P6522, etc.). Fitted from F_obs alone — no model dependence.
    timer.start("1_anisotropy_fit")
    aniso_edges, _ = equal_count_shell_edges(s_mag, n_shells)
    aniso_idx = assign_shells(s_mag, aniso_edges)
    U_aniso = fit_overall_anisotropy(
        F_obs, s_vec, aniso_idx, P=n_shells, min_count=20,
    )
    if verbose > 0:
        print(
            f"fit_to_data: overall U-aniso diag (Å²) = "
            f"({U_aniso[0, 0].item():+.2f}, {U_aniso[1, 1].item():+.2f}, "
            f"{U_aniso[2, 2].item():+.2f})",
            flush=True,
        )
    F_obs_aniso = apply_overall_anisotropy(F_obs, s_vec, U_aniso)
    timer.stop("1_anisotropy_fit")

    # --- Build LL interpolator from a P1 view of the model ---
    timer.start("2_ll_build")
    if verbose > 0:
        print(
            f"fit_to_data: building Lattman-Love interpolator "
            f"(box={ll_padding_factor}·diam, max_res={ll_max_res_A} Å)…",
            flush=True,
        )
    ll = LattmanLoveInterpolator(
        model, padding_factor=ll_padding_factor, max_res_A=ll_max_res_A,
        verbose=verbose,
    )

    # --- Symmetry-expand reciprocal-space points for the Patterson SH expansion ---
    # The MTZ stores only the spacegroup ASU (1 / n_ops of reciprocal space).
    # The observed Patterson is spacegroup-invariant (|F(S_k h)| = |F(h)|),
    # so the SH expansion of P_obs must sample the full sphere — otherwise
    # the rotation function loses its spacegroup symmetry and the true
    # orientation is no longer a global maximum. On 1AK5 (P432, 24 ops)
    # the un-expanded rotation function picked maxima 5× higher than the
    # value at R_true; expanding F_obs to all 24 symmetry mates makes
    # C(R) = C(S_k R) by construction (and the calc-side LL evaluator
    # is queried at the same expanded HKL set so the cross-correlation is
    # consistent). For low-sym cells (P21, n_ops=2) this is a no-op factor;
    # for P432 it makes 1AK5 / 3K7M find the right basin.
    sg_mats = data.spacegroup.matrices.to(torch.float64).to(device)  # (n_ops, 3, 3)
    n_ops_sg = int(sg_mats.shape[0])
    # h_sym[k, i, :] = S_k · hkl[i]  (rotation part of the symop)
    hkl_sym = torch.einsum("kij,nj->kni", sg_mats, hkl.to(torch.float64))
    hkl_sym_flat = hkl_sym.reshape(-1, 3)                        # (n_ops·N, 3)
    s_vec_sym = hkl_sym_flat @ rec_basis.to(device)              # (n_ops·N, 3)
    s_mag_sym = s_vec_sym.norm(dim=-1)
    # |F_obs| is replicated across symmetry mates (Patterson invariance).
    F_obs_aniso_sym = F_obs_aniso.unsqueeze(0).expand(n_ops_sg, -1).reshape(-1)
    E_obs_sym = _shellbin_norm_etrick(F_obs_aniso_sym, s_mag_sym, n_shells)
    patt_obs = E_obs_sym ** 2 - 1.0

    # F_calc evaluated at the SAME symmetry-expanded HKL set so the cross-
    # correlation between f_obs and f_calc samples the same directions.
    F_calc_sym = ll.evaluate(
        torch.eye(3, dtype=torch.float32), hkl_sym_flat, data.cell,
        return_amplitude=True,
    ).to(torch.float64)
    E_calc_sym = _shellbin_norm_etrick(F_calc_sym, s_mag_sym, n_shells)
    patt_calc = E_calc_sym ** 2 - 1.0
    # Replace the un-expanded s_vec with the expanded one for ball_search.
    s_vec_for_search = s_vec_sym
    timer.stop("2_ll_build")

    # --- Stage 1: fast Patterson ball-search ---
    timer.start("3_ball_search")
    if verbose > 0:
        print(
            f"fit_to_data: ball-search (L={L}, P={n_shells}, "
            f"n_peaks={n_rotation_peaks})…",
            flush=True,
        )
    _, _, _, _, peaks = ball_rotation_search(
        s_vec_for_search, patt_obs, s_vec_for_search, patt_calc,
        L=L, P=n_shells, n_peaks=n_rotation_peaks,
        refine_subvoxel=True, n_refine=min(n_rotation_peaks, 50),
        sigma_threshold=-5.0,
        auto_variance_weights=auto_variance_weights,
    )
    timer.stop("3_ball_search")

    # --- Stage 2: Sim-MLRF rescore (per-shell σA fit per candidate) ---
    timer.start("4_sim_mlrf_rescore")
    if verbose > 0:
        print(
            f"fit_to_data: ML rescoring top "
            f"{min(len(peaks), n_ml_refine)} peaks…",
            flush=True,
        )
    rescored = sim_mlrf_rescore(
        peaks, F_obs, hkl, s_mag, centric, ll, data.cell,
        n_shells=max(n_shells // 2, 8),
        n_refine=min(len(peaks), n_ml_refine),
        batch_size=50,
        verbose=verbose,
        auto_variance_weights=auto_variance_weights,
    )
    timer.stop("4_sim_mlrf_rescore")
    if not rescored:
        raise RuntimeError("Rotation search produced no peaks.")

    n_rot = min(n_rotation_candidates if do_translation else 1, len(rescored))

    # The Patterson rotation function has a centrosymmetric ambiguity, but
    # `sim_mlrf_rescore` ranks Patterson-equivalents adjacent. `n_rot ≥ 3`
    # is enough without explicitly multiplying by spacegroup rotations.
    if verbose > 0 and n_rot > 1:
        print(
            f"fit_to_data: trying top {n_rot} rotation candidates "
            f"(Patterson-equivalents covered by LLG ranking).",
            flush=True,
        )

    def _candidate(k):
        peak = rescored[k]
        R_rec = rotation_matrix_from_edmonds_euler(
            peak.alpha, peak.beta, peak.gamma,
        )
        R_app = R_rec.T.contiguous()
        rot = model.rotate(
            R_app.to(device=model.device, dtype=model.dtype_float),
        )
        rot.last_alignment_rotation = R_rec
        return rot, R_rec, R_app, peak

    if not do_translation:
        rotated, R_recovered, _, top = _candidate(0)
        if verbose > 0:
            print(
                f"fit_to_data: top peak LLG = {top.score:.2f} "
                f"(σ_Z = {top.sigma:.2f}); applying R⁻¹ to coords.",
                flush=True,
            )
        if verbose >= 2:
            print("\n" + timer.summary(), flush=True)
        return rotated

    # --- Stage 3: translation search + analytical-R local refine ---
    hkl_full = data.hkl
    F_obs_full = data.F
    if hasattr(data, "get_valid_mask"):
        tmask = data.get_valid_mask()
    else:
        tmask = torch.ones(
            F_obs_full.shape[0], dtype=torch.bool, device=F_obs_full.device,
        )
    # Index on CPU then move slices to `device` — same pattern as the
    # early `keep`-mask section; the validity mask lives on the data's
    # device, while downstream consumers run on `model.device`.
    F_obs_amp = F_obs_full[tmask].abs().to(torch.float64).to(device)
    hkl_keep = hkl_full[tmask].to(device)

    global_best = None
    for k_rot in range(n_rot):
        rotated_k, R_recovered_k, _, peak_k = _candidate(k_rot)
        if verbose > 0:
            print(
                f"\nfit_to_data: rot{k_rot} "
                f"(LLG={peak_k.score:.2f}, σ_Z={peak_k.sigma:.2f})",
                flush=True,
            )

        if str(rotated_k.spacegroup) != str(data.spacegroup):
            rotated_k.spacegroup = data.spacegroup

        rotated_p1 = rotated_k.copy()
        rotated_p1.spacegroup = SpaceGroup("P 1")
        evaluator = _DirectModelEvaluator(rotated_p1)

        # Pre-compute per-sym F_asu contributions once per rotation; reused
        # by both coarse TF and each local refine.
        timer.start("5_precompute_G")
        G_pre, h_R_pre = precompute_G_for_rotation(
            evaluator, torch.eye(3, dtype=torch.float64),
            hkl_keep, data.spacegroup, data.cell,
        )
        timer.stop("5_precompute_G")

        timer.start("6_amplitude_TF")
        _, _, t_peaks = amplitude_translation_search(
            F_obs=F_obs_amp, interpolator=evaluator,
            R_rotation=torch.eye(3, dtype=torch.float64),
            hkl=hkl_keep,
            spacegroup=data.spacegroup, real_cell=data.cell,
            grid_steps=translation_grid_steps,
            n_peaks=n_translation_peaks,
            cluster_radius=0.05,
            precomputed_G=G_pre, precomputed_h_R=h_R_pre,
        )
        timer.stop("6_amplitude_TF")
        if not t_peaks:
            if verbose > 0:
                print("  no translation peaks; skipping", flush=True)
            continue
        if verbose > 0:
            tt = tuple(round(float(x), 3) for x in t_peaks[0].translation.tolist())
            print(
                f"  top translation t={tt} corr={t_peaks[0].score:.4f}",
                flush=True,
            )

        if not do_joint_refine:
            t_top = torch.as_tensor(
                t_peaks[0].translation,
                dtype=model.dtype_float, device=rotated_k.device,
            )
            translated = rotated_k.translate(t_top, fractional=True)
            translated.last_alignment_rotation = R_recovered_k
            translated.last_alignment_translation = t_top
            return translated

        for k_t, tp in enumerate(t_peaks[:n_translation_candidates]):
            t_init = torch.as_tensor(tp.translation, dtype=torch.float64)
            timer.start("7_local_TF_refine")
            # Single-pass local refine (was 2). Pass-2 zoomed to ~0.0017
            # fractional resolution; the downstream LBFGS rigid-body polish
            # refines to gradient-tolerance anyway, so pass 2 was just
            # paying ~half the local-TF cost for a precision that gets
            # overridden a few steps later.
            t_refined, r_analytic = local_translation_refine(
                F_obs=F_obs_amp, interpolator=evaluator,
                R_rotation=torch.eye(3, dtype=torch.float64),
                hkl=hkl_keep,
                spacegroup=data.spacegroup, real_cell=data.cell,
                t_init=t_init, radius=0.06, grid_steps=13,
                n_refinement_passes=1,
                precomputed_G=G_pre, precomputed_h_R=h_R_pre,
            )
            timer.stop("7_local_TF_refine")
            if verbose > 0:
                print(
                    f"  rot{k_rot} trans{k_t}: "
                    f"R(analytic)={r_analytic:.4f}, "
                    f"t={[round(float(x), 3) for x in t_refined.tolist()]}",
                    flush=True,
                )
            if global_best is None or r_analytic < global_best[0]:
                global_best = (r_analytic, rotated_k, R_recovered_k, t_refined)

    if global_best is None:
        raise RuntimeError("Translation + joint refine produced no candidates.")
    r_analytic_best, rot_best, R_recovered_best, t_refined_best = global_best
    refined = rot_best.translate(
        t_refined_best.to(model.dtype_float), fractional=True,
    )

    # --- Stage 4: dense rotation sampling at the found translation ---
    if do_joint_refine:
        timer.start("8_dense_R_ll_build")
        refined_p1 = refined.copy()
        refined_p1.spacegroup = SpaceGroup("P 1")
        ll_refine = LattmanLoveInterpolator(
            refined_p1, padding_factor=ll_padding_factor,
            max_res_A=ll_max_res_A, verbose=0,
        )
        timer.stop("8_dense_R_ll_build")

        centric_keep = (
            data.centric[tmask].to(torch.bool).to(device) if hasattr(data, "centric")
            else torch.zeros(hkl_keep.shape[0], dtype=torch.bool, device=device)
        )
        rec_basis_keep = data.cell.reciprocal_basis_matrix.to(torch.float64).to(device)
        s_mag_keep = (hkl_keep.to(torch.float64) @ rec_basis_keep).norm(dim=-1)

        # Dense-R sampling: 2-pass zoom on the σ_A-fitted ML LLG which has
        # FWHM ~2–3° (sharper than the Patterson rotation function's ~10°
        # because of log-likelihood curvature and σ_A up-weighting). Pass 1
        # at 9³ × ±5.7° (1.43° spacing) gives 1–2 samples per ML-FWHM and
        # locates the basin; pass 2 at 5³ × ±1.43° (0.71° spacing) zooms in.
        # Single-pass or larger spacing regresses R-work; finer than this
        # is just sampled by the downstream LBFGS polish anyway.
        n_per_axis_pass = [9, 5]
        zoom_factor = 4.0
        radii = [
            float(joint_refine_expected_rot_error),
            float(joint_refine_expected_rot_error) / zoom_factor,
        ]
        R_accumulated = torch.eye(3, dtype=torch.float64)

        for pass_idx, max_perturb_rad in enumerate(radii):
            n_per_axis = n_per_axis_pass[pass_idx]
            coords_r = torch.linspace(
                -max_perturb_rad, max_perturb_rad, n_per_axis,
                dtype=torch.float64,
            )
            wx, wy, wz = torch.meshgrid(
                coords_r, coords_r, coords_r, indexing="ij",
            )
            omegas = torch.stack(
                [wx.flatten(), wy.flatten(), wz.flatten()], dim=-1,
            )
            # Batched Rodrigues + matmul: one (B, 3, 3) build instead of B
            # per-omega calls. Previously _rodrigues had `.item()` × 3 per
            # call, so dense_R cost was dominated by Python dispatch.
            R_perturbs = _rodrigues(omegas)                         # (B, 3, 3)
            R_cand_full = R_perturbs @ R_accumulated                # (B, 3, 3)
            cand_peaks = []
            for R_c in R_cand_full:
                a, b, g = edmonds_euler_from_rotation_matrix(R_c)
                cand_peaks.append(RotationPeak(
                    alpha=a, beta=b, gamma=g, score=0.0, sigma=0.0,
                ))
            if verbose > 0:
                print(
                    f"\nfit_to_data: dense R pass {pass_idx + 1} "
                    f"({n_per_axis}³={omegas.shape[0]} perturbations, "
                    f"±{math.degrees(max_perturb_rad):.2f}°)…",
                    flush=True,
                )
            # batch_size adapted to N_hkl × D_grid (~41) to keep the
            # llg_for_rotation_batch inner tensors bounded.
            rescore_batch = max(
                4, min(100, 1_000_000 // max(hkl_keep.shape[0], 1)),
            )
            timer.start("9_dense_R_rescore")
            # n_D_grid=11 (vs 41 default): we only need relative LLG
            # ranking across a tight rotation neighbourhood; the σA
            # optimum shifts negligibly. 4× fewer Bessel evals.
            rescored_refine = sim_mlrf_rescore(
                cand_peaks, F_obs_amp, hkl_keep, s_mag_keep, centric_keep,
                ll_refine, data.cell,
                n_shells=max(n_shells // 2, 8),
                n_refine=len(cand_peaks), batch_size=rescore_batch,
                verbose=0, n_D_grid=11,
            )
            timer.stop("9_dense_R_rescore")
            top = rescored_refine[0]
            best_idx = next(
                i for i, p in enumerate(cand_peaks)
                if p.alpha == top.alpha and p.beta == top.beta
                and p.gamma == top.gamma
            )
            R_accumulated = R_cand_full[best_idx]
            if verbose > 0:
                print(
                    f"  pass {pass_idx + 1} best LLG={top.score:.2f}, "
                    f"|ω|={omegas[best_idx].norm().item() * 180 / math.pi:.3f}°",
                    flush=True,
                )

        # Trust the LLG. The dense-R rescore picks the perturbation with
        # the highest LLG (scale-invariant ML target); LLG monotonicity
        # implies a non-degraded R-work. Two solvent-aware Scaler refits
        # used to live here as a defensive gate — at ~25 s each on 1DAW
        # they ran twice the wall time of the entire rescore loop they
        # were checking. Cheaper to trust the score.
        refined = refined.rotate(
            R_accumulated.T.to(model.dtype_float).contiguous(),
        )

    # --- Stage 5: joint LBFGS polish on (R, t) ---
    if do_joint_refine:
        timer.start("11_lbfgs_polish")
        rb = RigidBodyRefinement(
            refined, data,
            initial_translation=torch.zeros(
                3, dtype=torch.float32, device=refined.device,
            ),
            expected_rotational_error=joint_refine_expected_rot_error,
            max_res=joint_refine_max_res_A,
            device=refined.device,
            verbose=max(0, verbose - 1),
        )
        rb_result = rb.refine()
        with torch.no_grad():
            R_polish = rb.get_rotation_matrix().detach()
            t_polish = rb.translation_frac.detach()
        polished = refined.rotate(R_polish.to(model.dtype_float))
        polished = polished.translate(
            t_polish.to(model.dtype_float), fractional=True,
        )
        timer.stop("11_lbfgs_polish")
        # Compare initial vs final R-work from the LBFGS's *own* internal
        # scaler (same instance, both numbers no-solvent — apples to
        # apples). Saves two solvent-aware Scaler refits (~25 s each
        # on 1DAW) that previously gated this decision.
        if rb_result.final_r_factor <= rb_result.initial_r_factor:
            refined = polished
            if verbose > 0:
                print(
                    f"\nfit_to_data: joint polish "
                    f"{rb_result.initial_r_factor:.4f} → "
                    f"{rb_result.final_r_factor:.4f} (no-solvent R)",
                    flush=True,
                )
        elif verbose > 0:
            print(
                f"\nfit_to_data: joint polish kept original "
                f"({rb_result.initial_r_factor:.4f} ≤ "
                f"{rb_result.final_r_factor:.4f} no-solvent R)",
                flush=True,
            )

    # Single solvent-aware Scaler refit at the very end on the winner —
    # gives the user-facing R-work without paying the cost on every
    # intermediate gate.
    timer.start("12_final_scaler")
    rwork_final = _external_rwork(refined, data)
    timer.stop("12_final_scaler")

    refined.last_alignment_rotation = R_recovered_best
    refined.last_alignment_translation = t_refined_best
    refined.last_alignment_rfactor = rwork_final
    if verbose > 0:
        print(
            f"fit_to_data: best analytical R={r_analytic_best:.4f}, "
            f"final Scaler-fit R-work={rwork_final:.4f}",
            flush=True,
        )
    if verbose >= 2:
        print("\n" + timer.summary(), flush=True)
    return refined
