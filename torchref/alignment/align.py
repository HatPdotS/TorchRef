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
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from .frf.ball_search import (
    RotationPeak,
    ball_rotation_search,
    edmonds_euler_from_rotation_matrix,
    rotation_matrix_from_edmonds_euler,
)
from .lattman_love import LattmanLoveInterpolator, estimate_interp_var
from .ml_rotation import compute_sigma_a_luzzati, m_letf1_rescore, sim_mlrf_rescore
from .rigid_body import RigidBodyRefinement
from .sh import (
    apply_overall_anisotropy,
    assign_shells,
    compute_patterson_shell_variance,
    equal_count_shell_edges,
    fit_overall_anisotropy,
    get_high_order_axis,
)
from .translation import (
    TranslationPeak,
    amplitude_translation_search,
    llg_translation_rescore,
    local_translation_refine,
    precompute_G_for_rotation,
)
from .ml_rotation import fit_sigma_a_per_shell

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
    # Detach the model forward — the scaler only needs gradients through its
    # own parameters; leaving `fc` attached to the model's autograd graph
    # keeps SfFFT density-build intermediates alive after this function
    # returns.
    with torch.no_grad():
        fc = model(data.hkl).detach()
    s.initialize(fc)
    s.refine_lbfgs(fcalc=fc)
    with torch.no_grad():
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
# FRF input preparation (shared by the live pipeline and the rotation-ranking
# benchmark in tests/integration/alignment/benchmark_rotation_ranking.py)
# ---------------------------------------------------------------------------


@dataclass
class FRFInputs:
    """Container for the spherical-harmonic rotation-search inputs.

    `*_sym` arrays are symmetry-expanded across the spacegroup rotation
    operators (so the Patterson SH expansion samples the full sphere).
    Un-suffixed `F_obs / hkl / s_vec / s_mag / centric` are the resolution-
    masked, anisotropy-corrected reflection arrays — used downstream by
    the MLRF rescore, translation search and rigid-body polish.
    """
    # FRF (symmetry-expanded) inputs
    s_vec_for_search: torch.Tensor   # (n_ops·N, 3)
    s_mag_sym: torch.Tensor          # (n_ops·N,)
    patt_obs: torch.Tensor           # (n_ops·N,) = |E_obs|² − 1
    patt_calc: torch.Tensor          # (n_ops·N,) = |E_calc|² − 1
    # Per-reflection inputs (un-expanded)
    F_obs: torch.Tensor              # (N,) anisotropy-corrected? See `F_obs_aniso` flag
    hkl: torch.Tensor                # (N, 3) integer Miller indices
    s_vec: torch.Tensor              # (N, 3) reciprocal-space Cartesian
    s_mag: torch.Tensor              # (N,) Å⁻¹
    centric: torch.Tensor            # (N,) bool
    # Other state used downstream
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
    """Build the symmetry-expanded SH-rotation-search inputs.

    Encapsulates the data prep / anisotropy / symmetry-expansion logic
    previously inlined in `align_model_to_data`. The returned dataclass
    feeds both the live `ball_rotation_search` call and the benchmark.

    `F_obs` on the returned dataclass is the *anisotropy-corrected* value
    (matches what previously was the `F_obs_aniso` local variable).
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
        F_obs, s_vec, aniso_idx, P=n_shells, min_count=20,
    )
    # Project U onto the spacegroup's point-group-invariant subspace
    # (Phaser RefineANO.cc:116-142 via cctbx `site_symmetry.average_u_star`).
    # Without this constraint a 6-component unconstrained regression can
    # fit physically impossible anisotropy on high-symmetry cells — e.g.
    # 3K7M (cubic) fits eigenvalues (0.8, 17, 70) Å² which then blows up
    # the per-reflection exp(π²·s·U·s) multiplier and destroys the FRF.
    # After projection, cubic → U = λI (1 DOF), tetragonal → diag(λ,λ,μ),
    # orthorhombic → diag(λ,μ,ν), etc.
    from .sh import hkl_symops_to_cartesian, symmetrize_anisotropy
    _sg_mats = data.spacegroup.matrices.to(torch.float64).to(device)
    _sym_mats_cart = hkl_symops_to_cartesian(_sg_mats, rec_basis.to(device))
    U_aniso = symmetrize_anisotropy(U_aniso, _sym_mats_cart)
    F_obs_aniso = apply_overall_anisotropy(F_obs, s_vec, U_aniso)

    ll = LattmanLoveInterpolator(
        model, padding_factor=ll_padding_factor, max_res_A=ll_max_res_A,
        verbose=verbose,
    )

    # Symmetry-expand reciprocal-space points so the Patterson SH expansion
    # samples the full sphere (spacegroup-invariant by construction). See
    # the long comment in `align_model_to_data` for the rationale.
    sg_mats = data.spacegroup.matrices.to(torch.float64).to(device)
    n_ops_sg = int(sg_mats.shape[0])
    hkl_sym = torch.einsum("kij,nj->kni", sg_mats, hkl.to(torch.float64))
    hkl_sym_flat = hkl_sym.reshape(-1, 3)
    s_vec_sym = hkl_sym_flat @ rec_basis.to(device)
    s_mag_sym = s_vec_sym.norm(dim=-1)
    F_obs_aniso_sym = F_obs_aniso.unsqueeze(0).expand(n_ops_sg, -1).reshape(-1)
    E_obs_sym = _shellbin_norm_etrick(F_obs_aniso_sym, s_mag_sym, n_shells)
    patt_obs = E_obs_sym ** 2 - 1.0

    F_calc_sym = ll.evaluate(
        torch.eye(3, dtype=torch.float32), hkl_sym_flat, data.cell,
        return_amplitude=True,
    ).to(torch.float64)
    E_calc_sym = _shellbin_norm_etrick(F_calc_sym, s_mag_sym, n_shells)
    patt_calc = E_calc_sym ** 2 - 1.0

    return FRFInputs(
        s_vec_for_search=s_vec_sym,
        s_mag_sym=s_mag_sym,
        patt_obs=patt_obs,
        patt_calc=patt_calc,
        F_obs=F_obs_aniso,
        hkl=hkl,
        s_vec=s_vec,
        s_mag=s_mag,
        centric=centric,
        ll=ll,
        U_aniso=U_aniso,
        device=device,
    )


def _run_frf_separate_rotation(
    model: "ModelFT",
    data: "ReflectionData",
    frf: "FRFInputs",
    *,
    lmax_cap: int = 48,
    dense_pad: float = 2.0,
    n_peaks: int = 500,
    grid_sampling_deg: float = 3.0,
    delta_vrms_A: float = 0.5,
    verbose: int = 0,
    _orbit_unroll: bool = False,
    # --- Phaser model-prep knobs for the FRF calc side (default OFF) ---
    apply_bulk_solvent: bool = False,
    solvent_fsol: float = 0.95,
    solvent_bsol: float = 300.0,
    vrms_strategy: str = "fixed",
    vrms_identity: float = 1.0,
    apply_wilson_b: bool = False,
):
    """Phaser-faithful (validated) rotation search — the production default.

    Reproduces the v19 benchmark config that solved the high-symmetry cases
    (4BX9 342→4-7, 6G9X 77→1-4; see ``FRF_CONSOLIDATION.md``):

    * obs taken at the **full data resolution** — ``auto_lmax`` coarsens the SH
      bandwidth to ``cap`` internally (the resolution↔bandwidth coupling that
      removes the aliasing background), so we do not pre-restrict resolution;
    * Popov-Bourenkov **anisotropy correction** (reuses ``frf.U_aniso``);
    * obs **symmetry-unroll** to the full reciprocal sphere (critical for
      high-symmetry spacegroups — the SH invariant subspace is otherwise
      under-sampled);
    * **dense P1-box calc** (single molecular transform, not unrolled) at the
      coarsened resolution — fixes high-l SH under-determination on large models;
    * French-Wilson + shell-variance weights; stable Wigner-d; all under no_grad.

    Returns the validated engine's peak list (``frf.types.RotationPeak``, whose
    ``.score`` aliases ``.value`` so it is drop-in for the ball-search peaks).
    """
    from .frf.api import phaser_lmax_resolution, phaser_rotation_search
    from .frf.dense_calc import dense_calc_via_box

    device = frf.device
    with torch.no_grad():
        rec_basis = data.cell.reciprocal_basis_matrix.to(torch.float64).to(device)
        hkl_all = data.hkl.to(device)
        s_vec_all = hkl_all.to(torch.float64) @ rec_basis
        s_mag_all = s_vec_all.norm(dim=-1)
        # Full data resolution window; auto_lmax coarsens d_min to match the cap.
        # d_max ≈ no low-res cutoff (matches the validated config's d_max_mimic).
        d_min_eff = float(1.0 / s_mag_all.max().item())
        d_max_eff = 100.0
        keep = (s_mag_all >= 1.0 / d_max_eff) & (s_mag_all <= 1.0 / d_min_eff)

        s_obs = s_vec_all[keep]
        # Anisotropy correction (reuse the tensor fitted in _prepare_frf_inputs).
        F_obs = apply_overall_anisotropy(
            data.F.to(torch.float64).abs().to(device)[keep], s_obs, frf.U_aniso,
        )
        sigF = (
            data.F_sigma.to(torch.float64).to(device)[keep]
            if getattr(data, "F_sigma", None) is not None
            else None
        )
        centric = (
            data.centric[keep].to(torch.bool).to(device)
            if hasattr(data, "centric")
            else torch.zeros_like(F_obs, dtype=torch.bool)
        )

        # Obs symmetry-unroll → full reciprocal space (each ASU reflection becomes
        # n_ops entries carrying the same |F|², centric, σF — |F(Sh)|=|F(h)|).
        #
        # The `_orbit_unroll=True` path uses `epsilon_aware_unroll` (Phaser
        # DataMR.cc:954-986's `!duplicate(isym, rhkl)` skip — keeps only unique
        # orbit positions). It is OFF by default: as a standalone change it
        # regressed the rebench (job 103409: 3K7M 18->189, 3GR5 47->204, 2DQ6
        # 202->324). The dedup is correct only as part of a coordinated Phaser-
        # faithful preprocessing chain (ε-Wilson + V(h) + σ_A), pending.
        sg_mats = data.spacegroup.matrices.to(torch.float64).to(device)
        if _orbit_unroll:
            from .frf.preprocessing import epsilon_aware_unroll
            hkl_keep_int = hkl_all.to(torch.long).to(device)[keep]
            unrolled_hkl, asu_idx = epsilon_aware_unroll(hkl_keep_int, sg_mats)
            s_obs = unrolled_hkl.to(torch.float64) @ rec_basis
            F_obs = F_obs[asu_idx]
            centric = centric[asu_idx]
            if sigF is not None:
                sigF = sigF[asu_idx]
        else:
            n_ops = int(sg_mats.shape[0])
            hkl_keep = hkl_all.to(torch.float64)[keep]
            hkl_unroll = torch.einsum("kij,nj->kni", sg_mats, hkl_keep).reshape(-1, 3)
            s_obs = hkl_unroll @ rec_basis
            F_obs = F_obs.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()
            centric = centric.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()
            if sigF is not None:
                sigF = sigF.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()

        # Dense P1-box calc on the (un-rotated) search model at the coarsened res.
        model_radius_A = float(
            (model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item()
        )
        dmin_dense = phaser_lmax_resolution(model_radius_A, d_min_eff, lmax_cap)[1]
        s_calc, F_calc = dense_calc_via_box(
            model, d_max_eff, dmin_dense, pad=dense_pad, verbose=verbose > 0,
        )
        s_calc = s_calc.to(device)
        F_calc = F_calc.to(device)

        # Optional Wilson-B match on the dense calc (EnsemblePDB.cc:793-851).
        # Bin obs and calc into the same shells (defined by obs s-distribution),
        # regress log(<F_obs²>/<F_calc²>) vs s², apply DW `exp(-B·s²/4)` to F_calc.
        if apply_wilson_b:
            from .frf.preprocessing import fit_relative_wilson_b
            s_obs_mag = s_obs.norm(dim=-1)
            s_calc_mag = s_calc.norm(dim=-1)
            B_rel = fit_relative_wilson_b(
                F_obs.to(torch.float64), F_calc.to(torch.float64),
                s_obs_mag.to(torch.float64), n_shells=20,
                s_mag_calc=s_calc_mag.to(torch.float64),
            )
            if abs(B_rel) > 1e-6:
                F_calc = F_calc * torch.exp(-B_rel * (s_calc_mag * s_calc_mag) / 4.0)
                if verbose > 0:
                    print(f"  FRF Wilson-B applied: B_rel = {B_rel:+.2f} Å²", flush=True)

        # Optional Oeffner vrms (rms_estimate.cc:37) — depends on n_residues
        # estimated from atom count (≈ 8 heavy atoms / residue).
        delta_vrms_for_frf = delta_vrms_A
        if vrms_strategy == "oeffner":
            from .frf.preprocessing import oeffner_vrms
            n_residues_est = max(1, int(model.xyz().shape[0] / 8))
            delta_vrms_for_frf = oeffner_vrms(n_residues_est, vrms_identity)
            if verbose > 0:
                print(
                    f"  FRF Oeffner vrms = {delta_vrms_for_frf:.3f} Å "
                    f"(n_res≈{n_residues_est}, ident={vrms_identity})",
                    flush=True,
                )
        elif vrms_strategy != "fixed":
            raise ValueError(
                f"vrms_strategy={vrms_strategy!r}; expected 'fixed' or 'oeffner'."
            )

        _arf, peaks = phaser_rotation_search(
            s_obs, F_obs, centric,
            s_calc, F_calc,
            sg_mats,
            d_min=d_min_eff, d_max=d_max_eff, n_peaks=n_peaks,
            delta_vrms_A=delta_vrms_for_frf,
            sigma_threshold=-5.0,
            use_lerf1_intensity=True,
            use_m_symmetry_filter=True,
            sig_F_obs=sigF,
            use_french_wilson=(sigF is not None),
            use_shell_variance_weights=True,
            grid_sampling_deg=grid_sampling_deg,
            model_radius_A=model_radius_A,
            auto_lmax=True,
            lmax_cap=lmax_cap,
            apply_bulk_solvent=apply_bulk_solvent,
            solvent_fsol=solvent_fsol,
            solvent_bsol=solvent_bsol,
        )
    return peaks


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
    use_interp_var: bool = False,
    use_llg_tf: bool = False,
    refine_b: bool = False,
    sigma_rot_deg: float = 0.0,
    sigma_trans_ang: float = 0.0,
    sigma_b: float = 0.0,
    use_sigma_a_frf: bool = False,
    frf_delta_vrms_A: float = 1.0,
    frf_weight_combine: str = "sigma_a_only",
    use_m_symmetry_filter: bool = False,
    use_lerf1_intensity: bool = False,
    use_fitted_delta_vrms: bool = False,
    use_even_l_only: bool = False,
    engine: str = "frf_separate",
    frf_lmax_cap: int = 48,
    frf_dense_pad: float = 2.0,
    rescore_engine: str = "m_letf1",
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

    timer.start("0_data_prep")
    timer.start("1_anisotropy_fit")
    timer.start("2_ll_build")
    frf = _prepare_frf_inputs(
        model, data,
        d_min=d_min, d_max=d_max, n_shells=n_shells,
        ll_padding_factor=ll_padding_factor, ll_max_res_A=ll_max_res_A,
        verbose=verbose,
    )
    timer.stop("0_data_prep")
    timer.stop("1_anisotropy_fit")
    timer.stop("2_ll_build")
    if verbose > 0:
        U_aniso = frf.U_aniso
        print(
            f"fit_to_data: overall U-aniso diag (Å²) = "
            f"({U_aniso[0, 0].item():+.2f}, {U_aniso[1, 1].item():+.2f}, "
            f"{U_aniso[2, 2].item():+.2f})",
            flush=True,
        )
        print(
            f"fit_to_data: built Lattman-Love interpolator "
            f"(box={ll_padding_factor}·diam, max_res={ll_max_res_A} Å)",
            flush=True,
        )

    # Unpack into the local names the rest of this function uses.
    device = frf.device
    F_obs = frf.F_obs
    hkl = frf.hkl
    s_vec = frf.s_vec
    s_mag = frf.s_mag
    centric = frf.centric
    ll = frf.ll
    U_aniso = frf.U_aniso
    s_vec_for_search = frf.s_vec_for_search
    patt_obs = frf.patt_obs
    patt_calc = frf.patt_calc

    # --- Stage 1: fast Patterson ball-search ---
    # F3: Fit ΔVRMS from the model's mean B-factor (runs FIRST so E3/F2
    # downstream use the fitted value). ΔVRMS² = <B> / (8π²) converts
    # Debye-Waller B to a 1-D RMS coordinate displacement.
    effective_delta_vrms = frf_delta_vrms_A
    if use_fitted_delta_vrms:
        with torch.no_grad():
            _, adp_iso, _, _, _ = model.get_iso()
        b_mean = float(adp_iso.mean().item()) if adp_iso.numel() > 0 else 0.0
        effective_delta_vrms = max(
            math.sqrt(max(b_mean, 1e-6) / (8 * math.pi ** 2)), 0.1,
        )
        if verbose > 0:
            print(
                f"fit_to_data: ΔVRMS fitted from <B> = {b_mean:.2f} Å² → "
                f"{effective_delta_vrms:.3f} Å (was {frf_delta_vrms_A} Å).",
                flush=True,
            )

    # Optional Phaser-style σA pre-weighting of the SH input (E3). Off by
    # default — when on, replaces `auto_variance_weights`. The two can be
    # combined explicitly via `frf_weight_combine="sigma_a_x_variance"`.
    rotsearch_weights: Optional[torch.Tensor] = None
    rotsearch_auto_var = auto_variance_weights
    if use_sigma_a_frf:
        shell_edges, _ = equal_count_shell_edges(frf.s_mag_sym, n_shells)
        shell_mid = 0.5 * (shell_edges[:-1] + shell_edges[1:])  # (P,)
        sigma_a_shell = compute_sigma_a_luzzati(
            shell_mid, delta_vrms_A=effective_delta_vrms,
        )
        w = sigma_a_shell ** 2
        if frf_weight_combine == "sigma_a_x_variance":
            shell_idx_sym = assign_shells(frf.s_mag_sym, shell_edges)
            var_shell = compute_patterson_shell_variance(
                patt_obs.to(torch.float64), shell_idx_sym, P=n_shells,
            )
            w = w / var_shell.sqrt().clamp(min=1e-30)
        elif frf_weight_combine != "sigma_a_only":
            raise ValueError(
                f"frf_weight_combine={frf_weight_combine!r}; "
                "expected 'sigma_a_only' or 'sigma_a_x_variance'."
            )
        # Match ball_rotation_search's internal normalisation: sum-to-P.
        w = w * (n_shells / w.sum().clamp(min=1e-30))
        rotsearch_weights = w.to(patt_obs.dtype)
        rotsearch_auto_var = False
        if verbose > 0:
            print(
                f"fit_to_data: σA-weighted FRF (ΔVRMS={effective_delta_vrms}Å, "
                f"combine={frf_weight_combine}, w[0]={rotsearch_weights[0]:.3f}, "
                f"w[-1]={rotsearch_weights[-1]:.3f}).",
                flush=True,
            )

    # F2: LERF1 likelihood intensity (Phaser DataMR.cc:947–951). Replace
    # `E² − 1` on the OBSERVED side with the FRF likelihood intensity
    #   intensity = cweight · (E² − 1) · DFAC²
    # where cweight ∈ {1, 2} for centric/acentric and DFAC is a
    # per-reflection Luzzati factor proxied here as the per-reflection
    # σA(s) (same Luzzati formula as Eterm but evaluated on the actual
    # per-reflection s_mag, not the per-shell mean).
    if use_lerf1_intensity:
        n_ops_sg = int(data.spacegroup.matrices.shape[0])
        cweight_per = torch.where(
            centric, torch.ones_like(F_obs), 2.0 * torch.ones_like(F_obs),
        )
        cweight_sym = cweight_per.unsqueeze(0).expand(n_ops_sg, -1).reshape(-1)
        dfac_sym = compute_sigma_a_luzzati(
            frf.s_mag_sym, delta_vrms_A=effective_delta_vrms,
        ).to(patt_obs.dtype)
        patt_obs = patt_obs * cweight_sym.to(patt_obs.dtype) * (dfac_sym ** 2)
        if verbose > 0:
            print(
                f"fit_to_data: LERF1 intensity ON (mean(cweight·DFAC²) = "
                f"{(cweight_sym * dfac_sym ** 2).mean().item():.3f}).",
                flush=True,
            )

    # F1: m-symmetry filter (Phaser DataMR.cc:1019 / 1117). Compute ZSYMM
    # from the spacegroup. Off-by-default for backwards compat.
    rotsearch_zsymm = 1
    if use_m_symmetry_filter:
        sg_mats_cpu = data.spacegroup.matrices.to(torch.float64).cpu()
        axis, zsymm = get_high_order_axis(sg_mats_cpu)
        if axis != 2 and verbose > 0:
            print(
                f"fit_to_data: WARNING — highest-order axis is {axis} (x/y), "
                f"but m-symmetry filter assumes z. Applying with potentially "
                f"reduced effect; axis permutation not yet implemented.",
                flush=True,
            )
        rotsearch_zsymm = int(zsymm)
        if verbose > 0:
            print(
                f"fit_to_data: m-symmetry filter ON, ZSYMM={rotsearch_zsymm} "
                f"(axis={axis}).",
                flush=True,
            )

    timer.start("3_ball_search")
    if engine not in ("frf_separate", "ball"):
        raise ValueError(
            f"engine={engine!r}; expected 'frf_separate' (default) or 'ball'."
        )
    if engine == "frf_separate":
        # Validated Phaser-faithful default (dense calc + auto_lmax cap +
        # obs-unroll + no_grad); solves the high-symmetry cases. The σA/LERF1/
        # m-filter ball-prep above is ignored on this path.
        if verbose > 0:
            print(
                f"fit_to_data: frf_separate rotation search "
                f"(dense calc + auto_lmax cap={frf_lmax_cap}, "
                f"n_peaks={n_rotation_peaks})…",
                flush=True,
            )
        peaks = _run_frf_separate_rotation(
            model, data, frf,
            lmax_cap=frf_lmax_cap, dense_pad=frf_dense_pad,
            n_peaks=n_rotation_peaks, verbose=verbose,
        )
    else:  # engine == "ball" — legacy ball-harmonic E-value search
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
            weights=rotsearch_weights,
            auto_variance_weights=rotsearch_auto_var,
            zsymm=rotsearch_zsymm,
            skip_odd_l=use_even_l_only,
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
    interp_var_main: Optional[torch.Tensor] = None
    if use_interp_var:
        # Per-reflection interpolation variance (Phaser totvar_search analogue).
        # Inflates the Rice variance budget so a noisy true peak isn't
        # demoted below a noise-free wrong peak by the rescore.
        rescore_n_shells = max(n_shells // 2, 8)
        rescore_edges, _ = equal_count_shell_edges(s_mag, rescore_n_shells)
        rescore_shell_idx = assign_shells(s_mag, rescore_edges)
        interp_var_main = estimate_interp_var(
            ll, hkl, data.cell, rescore_shell_idx, rescore_n_shells,
        ).to(F_obs.dtype)
        if verbose > 0:
            print(
                f"fit_to_data: interp_var enabled (mean={interp_var_main.mean().item():.3f}, "
                f"max={interp_var_main.max().item():.3f}).",
                flush=True,
            )

    if rescore_engine not in ("m_letf1", "sim"):
        raise ValueError(
            f"rescore_engine={rescore_engine!r}; expected 'm_letf1' (default) or 'sim'."
        )
    if rescore_engine == "m_letf1":
        # Phaser-faithful: NSYMP calc sum + V(h) budget + Rice/Woolfson logRel.
        # Cross-rotation case: no fixed model, so totvar_known=0 and the
        # variance budget reduces to ε(h) - σ_A²(s)·n_mol in E-space.
        rescored = m_letf1_rescore(
            peaks, F_obs, hkl, s_mag, centric, ll, data.cell,
            data.spacegroup.matrices.to(torch.float64).to(device),
            n_shells=max(n_shells // 2, 8),
            n_refine=min(len(peaks), n_ml_refine),
            batch_size=50,
            verbose=verbose,
        )
    else:  # rescore_engine == "sim" — legacy Sim/Rice approximation
        rescored = sim_mlrf_rescore(
            peaks, F_obs, hkl, s_mag, centric, ll, data.cell,
            n_shells=max(n_shells // 2, 8),
            n_refine=min(len(peaks), n_ml_refine),
            batch_size=50,
            verbose=verbose,
            auto_variance_weights=auto_variance_weights,
            interp_var=interp_var_main,
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

        # Phase B: re-rank the cheap-correlation peaks by Rice/Woolfson LLG
        # using a shared per-shell σA fitted at the top correlation peak.
        # Mirrors Phaser's FTF — the correlation pre-filter is fast but its
        # ranking is degraded for partial models; the LLG ranks consistently
        # with the rotation rescore.
        if use_llg_tf:
            timer.start("6b_llg_tf_rescore")
            rec_basis_keep = data.cell.reciprocal_basis_matrix.to(torch.float64).to(device)
            s_mag_keep_tf = (hkl_keep.to(torch.float64) @ rec_basis_keep).norm(dim=-1)
            tf_n_shells = max(n_shells // 2, 8)
            tf_edges, _ = equal_count_shell_edges(s_mag_keep_tf, tf_n_shells)
            tf_shell_idx = assign_shells(s_mag_keep_tf, tf_edges)
            centric_keep_tf = (
                data.centric[tmask].to(torch.bool).to(device)
                if hasattr(data, "centric")
                else torch.zeros_like(F_obs_amp, dtype=torch.bool)
            )

            # E_obs normalised on the validity-masked set.
            cnt_tf = torch.bincount(
                tf_shell_idx, minlength=tf_n_shells,
            ).to(torch.float64)
            sum_F2 = torch.zeros(tf_n_shells, dtype=torch.float64, device=device)
            sum_F2.scatter_add_(0, tf_shell_idx, F_obs_amp * F_obs_amp)
            mean_F2 = (sum_F2 / cnt_tf.clamp(min=1.0)).clamp(min=1e-30)
            E_obs_tf = F_obs_amp / mean_F2.sqrt().index_select(0, tf_shell_idx)

            # Compute |F_calc| at the top correlation peak's translation, use
            # that to fit the per-shell σA. One-shot, no per-candidate refit.
            t_top_np = t_peaks[0].translation
            t_top_t = torch.as_tensor(t_top_np, dtype=torch.float64, device=device)
            S_eff, N_eff = G_pre.shape
            phase_top = torch.exp(
                2j * torch.pi * torch.einsum(
                    "ind,d->in",
                    h_R_pre.to(torch.float64), t_top_t,
                ).to(G_pre.dtype),
            )
            Fc_top = (G_pre * phase_top).sum(dim=0).abs().to(torch.float64)
            # Per-shell E normalise
            sum_Fc2 = torch.zeros(tf_n_shells, dtype=torch.float64, device=device)
            sum_Fc2.scatter_add_(0, tf_shell_idx, Fc_top * Fc_top)
            mean_Fc2 = (sum_Fc2 / cnt_tf.clamp(min=1.0)).clamp(min=1e-30)
            E_calc_top = Fc_top / mean_Fc2.sqrt().index_select(0, tf_shell_idx)
            sigma_a_tf = fit_sigma_a_per_shell(
                E_obs_tf, E_calc_top, centric_keep_tf,
                tf_shell_idx, tf_n_shells, n_grid=81,
            )

            # interp_var is only meaningful when the F_calc comes from a
            # trilinear interpolator. The TF stage uses a direct-SF evaluator
            # (`_DirectModelEvaluator`) with no interpolation noise, so the
            # Phaser totvar_search analogue does not apply here.
            interp_var_tf: Optional[torch.Tensor] = None

            t_cands = torch.as_tensor(
                np.stack([p.translation for p in t_peaks]),
                dtype=torch.float64, device=device,
            )
            llg_tf = llg_translation_rescore(
                F_obs=F_obs_amp, hkl=hkl_keep, centric=centric_keep_tf,
                shell_idx=tf_shell_idx, n_shells=tf_n_shells,
                G=G_pre, h_R=h_R_pre, t_candidates=t_cands,
                sigma_a=sigma_a_tf, interp_var=interp_var_tf,
            )
            timer.stop("6b_llg_tf_rescore")
            # Re-rank t_peaks by LLG (descending). Update the score to carry
            # LLG so downstream picks correctly.
            llg_list = llg_tf.detach().cpu().tolist()
            corr_list = [p.score for p in t_peaks]  # original FFT-correlation scores
            order = sorted(
                range(len(t_peaks)), key=lambda i: llg_list[i], reverse=True,
            )
            # Record where the correlation top-1 ended up after LLG re-rank.
            corr_top1_new_rank = order.index(0)
            t_peaks = [
                TranslationPeak(
                    translation=t_peaks[i].translation,
                    score=float(llg_list[i]),
                    sigma=float(llg_list[i]),
                )
                for i in order
            ]
            if verbose > 0:
                tt = tuple(round(float(x), 3) for x in t_peaks[0].translation.tolist())
                llg_top1 = llg_list[order[0]]
                corr_at_llg_top1 = corr_list[order[0]]
                print(
                    f"  LLG-TF rescore: top t={tt} LLG={t_peaks[0].score:.2f}  "
                    f"(corr at LLG-top1: {corr_at_llg_top1:.4f}; "
                    f"corr-top1 demoted to LLG-rank {corr_top1_new_rank}/{len(order)})",
                    flush=True,
                )

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

        # Hoisted out of the per-pass loop: ll_refine is fixed across passes,
        # so interp_var only needs to be estimated once.
        interp_var_dense: Optional[torch.Tensor] = None
        if use_interp_var:
            dense_n_shells = max(n_shells // 2, 8)
            dense_edges, _ = equal_count_shell_edges(s_mag_keep, dense_n_shells)
            dense_shell_idx = assign_shells(s_mag_keep, dense_edges)
            interp_var_dense = estimate_interp_var(
                ll_refine, hkl_keep, data.cell,
                dense_shell_idx, dense_n_shells,
            ).to(F_obs_amp.dtype)

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
            if rescore_engine == "m_letf1":
                rescored_refine = m_letf1_rescore(
                    cand_peaks, F_obs_amp, hkl_keep, s_mag_keep, centric_keep,
                    ll_refine, data.cell,
                    data.spacegroup.matrices.to(torch.float64).to(device),
                    n_shells=max(n_shells // 2, 8),
                    n_refine=len(cand_peaks), batch_size=rescore_batch,
                    verbose=0,
                )
            else:
                rescored_refine = sim_mlrf_rescore(
                    cand_peaks, F_obs_amp, hkl_keep, s_mag_keep, centric_keep,
                    ll_refine, data.cell,
                    n_shells=max(n_shells // 2, 8),
                    n_refine=len(cand_peaks), batch_size=rescore_batch,
                    verbose=0, n_D_grid=11,
                    interp_var=interp_var_dense,
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
            refine_b=refine_b,
            sigma_rot_deg=sigma_rot_deg,
            sigma_trans_ang=sigma_trans_ang,
            sigma_b=sigma_b,
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
