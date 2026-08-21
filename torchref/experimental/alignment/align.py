"""
Molecular replacement: data-prep / FRF stage helpers + the public entry point.

This module hosts the heavy, reusable stage helpers — Lattman-Love / anisotropy
data prep (`_prepare_frf_inputs`), the Phaser-faithful rotation search
(`_run_frf_separate_rotation`), the solvent-aware R-work
(`_external_rwork`), the direct-SF translation evaluator
(`_DirectModelEvaluator`), the Rodrigues helper (`_rodrigues`) and the stage
timer (`_StageTimer`) — that are shared by the rotation-ranking benchmarks and
by the orchestrator.

`align_model_to_data` is the public entry point that `ModelFT.fit_to_data`
delegates to; it in turn delegates the FRF → FTF(per-candidate) → post-refine
control flow to
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
    feeds both the live FRF call and the benchmark scripts.

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
    # h' = h.R, NOT R.h -- reciprocal space transforms with the transpose
    # (SpaceGroup.apply_to_hkl). The two agree only when the symmetry matrices
    # are orthogonal, which they are in orthorhombic/tetragonal/cubic and
    # monoclinic settings but NOT in a hexagonal basis, where S.S^T != I. Using
    # R.h there mixes non-equivalent reflections into one orbit and writes
    # conflicting |F| onto the same Miller index.
    hkl_sym = torch.einsum("kji,nj->kni", sg_mats, hkl.to(torch.float64))
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
    # --- Phaser model-prep knobs (defaults ON post v26 validation: see
    # the SLURM v26 sweep in slurm_logs/rescore_v26_103820_*.csv). ---
    apply_bulk_solvent: bool = True,
    solvent_fsol: float = 0.95,
    solvent_bsol: float = 300.0,
    vrms_strategy: str = "oeffner",
    vrms_identity: float = 1.0,
    apply_wilson_b: bool = True,
    use_epsilon: bool = False,
    # obs-side term toggles (all default ON = production) for knockout bisection
    frf_use_m_filter: bool = True,
    frf_use_shell_variance: bool = True,
    frf_use_french_wilson: bool = True,
    frf_use_lerf1: bool = True,
    frf_acentric_only: bool = False,
    frf_d_max: float = 100.0,
    frf_obs_lmax: Optional[int] = None,
    frf_obs_solid_angle: bool = False,
    frf_patterson_radius_scale: float = 1.0,
    # Run the dominant SH-Bessel expansion (Legendre/Y_lm precompute + radial
    # contraction) in single precision. The contraction is the FRF's bottleneck;
    # FP64 is rate-limited on GPUs and SIMD-narrower on CPUs. The spherical-Bessel
    # downward recurrence keeps its float64 internals and the cross-chunk
    # accumulator stays full-precision, so only the contraction loses precision.
    # `None` (default) → float32 on CUDA, full precision on CPU. Set explicitly
    # to True/False to force single/double precision on either device.
    frf_einsum_float32: Optional[bool] = None,
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
        d_max_eff = float(frf_d_max)  # low-resolution cutoff (default 100 ≈ none)
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
        # NOTE: centered lattices (I/C/F) list each point-group rotation once per
        # centering op, so the raw matrices over-replicate the obs orbit (C2/I422
        # → ×2). Deduping to unique rotations is the correct point group and saves
        # that compute, BUT it is NOT result-neutral: the equal-COUNT Wilson shells
        # rebin when the obs count changes, perturbing the normalisation (3A5V
        # 3→4). Left as-is to keep FRF behaviour stable; tracked in
        # GHOST_INVESTIGATION.md as a follow-up (fix needs count-independent shells).
        # Integer unrolled Miller indices aligned with s_obs — needed for the
        # ε(h) multiplicity correction (use_epsilon), which down-weights the
        # axial/zonal reflections that otherwise over-weight the m=0 SH column
        # and feed high-symmetry rotation-function ghosts (compute_epsilon docstring).
        if _orbit_unroll:
            from .frf.preprocessing import epsilon_aware_unroll
            hkl_keep_int = hkl_all.to(torch.long).to(device)[keep]
            unrolled_hkl, asu_idx = epsilon_aware_unroll(hkl_keep_int, sg_mats)
            s_obs = unrolled_hkl.to(torch.float64) @ rec_basis
            hkl_obs_int = unrolled_hkl.to(torch.float64)
            F_obs = F_obs[asu_idx]
            centric = centric[asu_idx]
            if sigF is not None:
                sigF = sigF[asu_idx]
        else:
            n_ops = int(sg_mats.shape[0])
            hkl_keep = hkl_all.to(torch.float64)[keep]
            # h' = h.R (transpose) -- see the note at the `hkl_sym` unroll.
            hkl_unroll = torch.einsum("kji,nj->kni", sg_mats, hkl_keep).reshape(-1, 3)
            s_obs = hkl_unroll @ rec_basis
            hkl_obs_int = hkl_unroll
            F_obs = F_obs.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()
            centric = centric.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()
            if sigF is not None:
                sigF = sigF.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()

        # Optional: restrict the obs to ACENTRIC reflections before the SH
        # expansion. Centric reflections lie on the reciprocal-space zones
        # perpendicular to symmetry axes and carry concentrated symmetry-axis
        # signal (and a heavier Wilson tail); pooling them into the obs over-
        # weights the symmetry-axis channel that produces high-symmetry ghosts.
        # Dropping them also makes the Wilson normalisation acentric-only.
        if frf_acentric_only:
            acen = ~centric
            s_obs = s_obs[acen]
            F_obs = F_obs[acen]
            centric = centric[acen]
            hkl_obs_int = hkl_obs_int[acen]
            if sigF is not None:
                sigF = sigF[acen]
            if verbose > 0:
                print(f"  FRF acentric-only: kept {int(acen.sum())}/{acen.numel()} obs",
                      flush=True)

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
            use_lerf1_intensity=frf_use_lerf1,
            use_m_symmetry_filter=frf_use_m_filter,
            sig_F_obs=sigF,
            use_french_wilson=(frf_use_french_wilson and (sigF is not None)),
            use_shell_variance_weights=frf_use_shell_variance,
            use_epsilon=use_epsilon,
            hkl_obs=hkl_obs_int,
            grid_sampling_deg=grid_sampling_deg,
            model_radius_A=model_radius_A,
            auto_lmax=True,
            lmax_cap=lmax_cap,
            obs_lmax=frf_obs_lmax,
            obs_solid_angle=frf_obs_solid_angle,
            patterson_radius_scale=frf_patterson_radius_scale,
            apply_bulk_solvent=apply_bulk_solvent,
            solvent_fsol=solvent_fsol,
            solvent_bsol=solvent_bsol,
            compute_dtype=(
                torch.complex64
                if (
                    frf_einsum_float32
                    if frf_einsum_float32 is not None
                    else (device.type == "cuda")  # default: fp32 on GPU only
                )
                else None
            ),
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
    use_sigma_a_frf: bool = False,
    frf_delta_vrms_A: float = 1.0,
    frf_weight_combine: str = "sigma_a_only",
    use_m_symmetry_filter: bool = False,
    use_lerf1_intensity: bool = False,
    use_fitted_delta_vrms: bool = False,
    use_even_l_only: bool = False,
    frf_lmax_cap: int = 48,
    frf_dense_pad: float = 2.0,
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

    See `ModelFT.fit_to_data` for full kwarg semantics — this function is the
    canonical implementation; `fit_to_data` is a thin wrapper.
    """
    if not model.initialized:
        raise RuntimeError(
            "Cannot fit an uninitialized ModelFT. Load PDB data first."
        )

    # `MolecularReplacementPipeline` is the implementation of record. This
    # function preserves the historical kwarg surface (so `ModelFT.fit_to_data`
    # and the benchmark scripts keep working unchanged) and returns the single
    # best `ModelFT`; drive the pipeline directly to get the ranked candidate
    # list. Imported lazily to avoid an import cycle — `pipeline` imports the
    # stage helpers (`_prepare_frf_inputs`, `_run_frf_separate_rotation`,
    # `_external_rwork`, `_DirectModelEvaluator`, `_rodrigues`, `_StageTimer`)
    # from this module.
    from .pipeline import MolecularReplacementPipeline

    pipeline = MolecularReplacementPipeline(
        data, model,
        device=model.xyz().device,
        verbose=verbose,
        d_min=d_min, d_max=d_max, n_shells=n_shells,
        ll_max_res_A=ll_max_res_A, ll_padding_factor=ll_padding_factor,
        n_rotation_peaks=n_rotation_peaks, n_ml_refine=n_ml_refine,
        frf_lmax_cap=frf_lmax_cap, frf_dense_pad=frf_dense_pad,
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
        L=L,
        use_sigma_a_frf=use_sigma_a_frf, frf_delta_vrms_A=frf_delta_vrms_A,
        frf_weight_combine=frf_weight_combine,
        use_m_symmetry_filter=use_m_symmetry_filter,
        use_lerf1_intensity=use_lerf1_intensity,
        use_fitted_delta_vrms=use_fitted_delta_vrms,
        use_even_l_only=use_even_l_only,
    )
    solutions = pipeline.run(do_translation=do_translation)
    return solutions[0].model
