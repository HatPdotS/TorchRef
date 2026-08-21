"""
Molecular replacement pipeline: the single canonical MR orchestrator.

Implements the classic Phaser-style molecular-replacement tree:

1. **Fast Rotation Function (FRF)** — Phaser-faithful Bessel-radial × SH
   expansion (dense P1-box calc + auto_lmax), then ML rescoring
   (``m_letf1_rescore`` / ``sim_mlrf_rescore``) to rank candidate orientations.
2. **Fast Translation Function (FTF)** — for *each* of the top-N rotation
   candidates, an amplitude-correlation translation search (optionally
   re-ranked by a Rice/Woolfson LLG) followed by an analytical-R local refine.
3. **Post-refinement** — optional dense rotation re-sampling on the ML-LLG
   surface, then an LBFGS rigid-body polish on (R, t) (with optional B-factor
   co-refinement and Gaussian restraints).

Each rotation candidate is carried through translation + refinement
independently; the candidates are ranked by their refined R-factor and the
best is returned (a Phaser-style multi-candidate tree, with early-stopping once
a candidate beats ``rfactor_converged``). The user-facing solvent-aware R-work
is computed once, on the winner.

``align_model_to_data`` delegates to
this class — it is the implementation of record. The heavy crystallographic
stage helpers live in :mod:`torchref.experimental.alignment.align`,
:mod:`~torchref.experimental.alignment.translation` and
:mod:`~torchref.experimental.alignment.ml_rotation`; this module owns the
control flow that wires them together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from torchref.config import get_default_device
from torchref.utils.device_mixin import DeviceMixin

from .align import (
    _DirectModelEvaluator,
    _StageTimer,
    _external_rwork,
    _prepare_frf_inputs,
    _rodrigues,
)
from .frf.rotation_utils import (
    edmonds_euler_from_rotation_matrix,
    rotation_matrix_from_edmonds_euler,
)
from .frf.types import RotationPeak
from .rotation_search import search_peaks
from .lattman_love import LattmanLoveInterpolator, estimate_interp_var
from .ml_rotation import (
    fit_sigma_a_per_shell,
    m_letf1_rescore,
    sim_mlrf_rescore,
)
from .rigid_body import RigidBodyRefinement
from .sh import assign_shells, equal_count_shell_edges
from .translation import (
    TranslationPeak,
    amplitude_translation_search,
    llg_translation_rescore,
    local_translation_refine,
    precompute_G_for_rotation,
)

if TYPE_CHECKING:
    from torchref.io.datasets import ReflectionData
    from torchref.model import ModelFT


def rotation_matrix_from_euler_zyz(alpha, beta, gamma) -> np.ndarray:
    """Build R = R_z(α) R_y(β) R_z(γ) (Edmonds active ZYZ) as a NumPy 3×3 matrix.

    Compatibility wrapper around `rotation_matrix_from_edmonds_euler`.
    """
    R = rotation_matrix_from_edmonds_euler(float(alpha), float(beta), float(gamma))
    return R.detach().cpu().numpy()


def rotation_angular_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance between two rotation matrices in degrees.

    The angular distance is the angle of the rotation ``R2 @ R1.T``.
    """
    R_diff = R2 @ R1.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    return np.degrees(np.arccos((trace - 1.0) / 2.0))


def euler_angular_distance(
    euler1: Tuple[float, float, float],
    euler2: Tuple[float, float, float],
) -> float:
    """Angular distance between two ZYZ Euler angle sets (degrees)."""
    R1 = rotation_matrix_from_euler_zyz(*euler1)
    R2 = rotation_matrix_from_euler_zyz(*euler2)
    return rotation_angular_distance(R1, R2)


def cluster_rotation_peaks(
    peaks: list,
    threshold_deg: float = 6.0,
    symmetry_matrices: Optional[np.ndarray] = None,
) -> list:
    """Cluster rotation peaks by angular distance.

    Peaks within ``threshold_deg`` of each other are considered the same
    solution; only the highest-scoring peak from each cluster is kept. Not on
    the default pipeline path (the ML rescore already ranks Patterson-
    equivalents adjacently); retained for callers that want explicit
    de-duplication.

    Parameters
    ----------
    peaks : list
        Rotation peaks as tuples ``(alpha, beta, gamma, score, sigma)``.
    threshold_deg : float
        Angular distance threshold for clustering (degrees).
    symmetry_matrices : np.ndarray, optional
        Point-group symmetry matrices (N, 3, 3) to check symmetry equivalents.
    """
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda p: p[4], reverse=True)
    clustered = []
    used_rotations = []
    for peak in sorted_peaks:
        alpha, beta, gamma, score, sigma = peak
        R = rotation_matrix_from_euler_zyz(alpha, beta, gamma)
        is_new = True
        for R_used in used_rotations:
            if rotation_angular_distance(R, R_used) < threshold_deg:
                is_new = False
                break
            if symmetry_matrices is not None:
                for sym_op in symmetry_matrices:
                    if rotation_angular_distance(sym_op @ R, R_used) < threshold_deg:
                        is_new = False
                        break
                if not is_new:
                    break
        if is_new:
            clustered.append(peak)
            used_rotations.append(R)
    return clustered


@dataclass
class MRSolution:
    """A molecular-replacement placement.

    Attributes
    ----------
    rotation : np.ndarray
        Recovered orientation as a 3×3 rotation matrix (``R_recovered`` — the
        rotation that maps the *search-model* frame onto the *crystal* frame).
    translation : np.ndarray or None
        Fractional translation applied after rotation, shape (3,). ``None`` for
        a rotation-only solution (``do_translation=False``).
    rotation_score : float
        ML-LLG score of the rotation candidate (from the rescore).
    translation_score : float
        Analytical-R of the best translation for this candidate (lower better).
    r_factor : float
        Ranking key. During the candidate loop this is the rigid-body's own
        (no-solvent) R-work; for the returned winner it is replaced by the
        solvent-aware Scaler R-work.
    model : ModelFT
        The rotated (+translated +refined) model for this candidate.
    clash_score : float, optional
        Steric clash score, only populated when ``clash_filter`` is enabled.
    """

    rotation: np.ndarray
    translation: Optional[np.ndarray]
    rotation_score: float
    translation_score: float
    r_factor: float
    model: "ModelFT"
    clash_score: Optional[float] = None


class MolecularReplacementPipeline(DeviceMixin):
    """Canonical MR pipeline: FRF → FTF (per candidate) → post-refine.

    Parameters mirror :func:`align_model_to_data` (which delegates here), so a
    caller can either use ``align_model_to_data`` for the common case or drive this
    class directly for finer control / access to the ranked candidate list.

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data.
    model : ModelFT
        Initialised search model.
    device : torch.device, optional
        Compute device (defaults to the model's device).
    verbose : int
        0 silent, 1 summary, ≥2 adds a per-stage wall-clock table.

    Examples
    --------
    ::

        from torchref.experimental.alignment import MolecularReplacementPipeline

        pipe = MolecularReplacementPipeline(data, model)
        solutions = pipe.run()
        print(f"best R-work: {solutions[0].r_factor:.3f}")
    """

    def __init__(
        self,
        data: "ReflectionData",
        model: "ModelFT",
        *,
        device: Optional[torch.device] = None,
        verbose: int = 0,
        # --- data prep / FRF ---
        d_min: float = 4.0,
        d_max: float = 15.0,
        n_shells: int = 20,
        ll_max_res_A: float = 3.0,
        ll_padding_factor: float = 2.0,
        n_rotation_peaks: int = 500,
        n_ml_refine: int = 20,
        model_error_A: Optional[float] = None,
        # --- rescore ---
        rescore_engine: str = "m_letf1",
        rescore_scat_mode: str = "legacy",
        auto_variance_weights: bool = True,
        use_interp_var: bool = False,
        subpeak_refine: bool = False,
        subpeak_refine_k: int = -1,
        subpeak_refine_step_deg: float = 1.5,
        subpeak_refine_iters: int = 1,
        subpeak_refine_max_move_deg: Optional[float] = 1.5,
        # --- candidate tree ---
        n_rotation_candidates: int = 15,
        n_translation_peaks: int = 20,
        n_translation_candidates: int = 3,
        translation_grid_steps: int = 16,
        use_llg_tf: bool = False,
        # --- post-refine ---
        do_joint_refine: bool = True,
        dense_rotation_refine: bool = True,
        joint_refine_max_res_A: float = 4.0,
        joint_refine_expected_rot_error: float = 0.1,
        refine_b: bool = False,
        sigma_rot_deg: float = 0.0,
        sigma_trans_ang: float = 0.0,
        sigma_b: float = 0.0,
        # --- early stop ---
        min_tries: int = 3,
        max_tries: Optional[int] = None,
        rfactor_converged: float = 0.45,
    ):
        self.data = data
        self.model = model
        self.device = device or get_default_device()
        self.verbose = verbose

        self.d_min = d_min
        self.d_max = d_max
        self.n_shells = n_shells
        self.ll_max_res_A = ll_max_res_A
        self.ll_padding_factor = ll_padding_factor
        self.n_rotation_peaks = n_rotation_peaks
        self.n_ml_refine = n_ml_refine
        # Expected r.m.s. coordinate error of the search model, in Angstrom:
        # it sets the sigma_A fall-off in the rotation function. When the caller
        # does not know it, estimate it from the model's length the way Phaser
        # does (Oeffner et al. 2013), assuming the sequence is the target's --
        # roughly 8 heavy atoms per residue.
        if model_error_A is None:
            from .frf.preprocessing import oeffner_vrms
            n_residues = max(1, int(model.xyz().shape[0] / 8))
            model_error_A = oeffner_vrms(n_residues, 1.0)
        self.model_error_A = float(model_error_A)

        self.rescore_engine = rescore_engine
        self.rescore_scat_mode = rescore_scat_mode
        self.auto_variance_weights = auto_variance_weights
        self.use_interp_var = use_interp_var
        self.subpeak_refine = subpeak_refine
        self.subpeak_refine_k = subpeak_refine_k
        self.subpeak_refine_step_deg = subpeak_refine_step_deg
        self.subpeak_refine_iters = subpeak_refine_iters
        self.subpeak_refine_max_move_deg = subpeak_refine_max_move_deg

        self.n_rotation_candidates = n_rotation_candidates
        self.n_translation_peaks = n_translation_peaks
        self.n_translation_candidates = n_translation_candidates
        self.translation_grid_steps = translation_grid_steps
        self.use_llg_tf = use_llg_tf

        self.do_joint_refine = do_joint_refine
        self.dense_rotation_refine = dense_rotation_refine
        self.joint_refine_max_res_A = joint_refine_max_res_A
        self.joint_refine_expected_rot_error = joint_refine_expected_rot_error
        self.refine_b = refine_b
        self.sigma_rot_deg = sigma_rot_deg
        self.sigma_trans_ang = sigma_trans_ang
        self.sigma_b = sigma_b

        self.min_tries = min_tries
        self.max_tries = max_tries
        self.rfactor_converged = rfactor_converged

        self._timer = _StageTimer(enabled=verbose >= 2)
        # Filled in by run().
        self._frf = None
        self._F_obs_amp = None
        self._hkl_keep = None
        self._tmask = None
        self._eye3 = torch.eye(3, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, do_translation: bool = True) -> List[MRSolution]:
        """Run the MR pipeline and return solutions ranked by R-factor.

        Parameters
        ----------
        do_translation : bool
            If ``False``, stop after rotation rescoring and return a single
            rotation-only solution (the model rotated onto the best
            orientation, no translation or refinement).

        Returns
        -------
        list of MRSolution
            Sorted by ``r_factor`` (ascending). The first element is the best
            placement; its ``r_factor`` is the solvent-aware Scaler R-work.
        """
        if not self.model.initialized:
            raise RuntimeError(
                "Cannot fit an uninitialized ModelFT. Load PDB data first."
            )

        timer = self._timer
        timer.start("0_data_prep")
        frf = _prepare_frf_inputs(
            self.model, self.data,
            d_min=self.d_min, d_max=self.d_max, n_shells=self.n_shells,
            ll_padding_factor=self.ll_padding_factor,
            ll_max_res_A=self.ll_max_res_A, verbose=self.verbose,
        )
        timer.stop("0_data_prep")
        self._frf = frf

        # --- Stage 1+2: FRF rotation search + ML rescore ---
        rescored = self._rotation_candidates(frf)
        if not rescored:
            raise RuntimeError("Rotation search produced no peaks.")

        if not do_translation:
            rotated, R_rec = self._make_rotated(rescored[0])
            top = rescored[0]
            if self.verbose > 0:
                print(
                    f"mr: top peak LLG = {top.score:.2f} "
                    f"(σ_Z = {top.sigma:.2f}); applying R⁻¹ to coords.",
                    flush=True,
                )
            if self.verbose >= 2:
                print("\n" + timer.summary(), flush=True)
            return [
                MRSolution(
                    rotation=R_rec.detach().cpu().numpy(),
                    translation=None,
                    rotation_score=float(top.score),
                    translation_score=float("nan"),
                    r_factor=float("nan"),
                    model=rotated,
                )
            ]

        # --- Stage 3+: per-candidate translation + post-refine tree ---
        self._prepare_translation_arrays()
        n_rot = min(self.n_rotation_candidates, len(rescored))
        max_tries = self.max_tries if self.max_tries is not None else n_rot
        if self.verbose > 0 and n_rot > 1:
            print(
                f"mr: trying up to {n_rot} rotation candidates "
                f"(early-stop after ≥{self.min_tries} once R < "
                f"{self.rfactor_converged}).",
                flush=True,
            )

        solutions: List[MRSolution] = []
        best_r = float("inf")
        for k in range(n_rot):
            peak_k = rescored[k]
            rotated_k, R_rec_k = self._make_rotated(peak_k)
            if self.verbose > 0:
                print(
                    f"\nfit_to_data: rot{k} "
                    f"(LLG={peak_k.score:.2f}, σ_Z={peak_k.sigma:.2f})",
                    flush=True,
                )
            placement = self._placement_for_candidate(rotated_k)
            if placement is None:
                if self.verbose > 0:
                    print("  no translation peaks; skipping", flush=True)
                continue
            r_analytic, t_refined = placement

            refined = rotated_k.copy().translate(
                t_refined.to(self.model.dtype_float), fractional=True,
            )
            r_rank = r_analytic
            if self.do_joint_refine:
                if self.dense_rotation_refine:
                    refined = self._dense_rotation_refine(refined)
                polished, rb_result = self._rigid_body_polish(refined)
                if rb_result.final_r_factor <= rb_result.initial_r_factor:
                    refined = polished
                    r_rank = rb_result.final_r_factor
                    if self.verbose > 0:
                        print(
                            f"  joint polish {rb_result.initial_r_factor:.4f} → "
                            f"{rb_result.final_r_factor:.4f} (no-solvent R)",
                            flush=True,
                        )
                else:
                    r_rank = rb_result.initial_r_factor
                    if self.verbose > 0:
                        print(
                            f"  joint polish kept original "
                            f"({rb_result.initial_r_factor:.4f} ≤ "
                            f"{rb_result.final_r_factor:.4f} no-solvent R)",
                            flush=True,
                        )

            refined.last_alignment_rotation = R_rec_k
            refined.last_alignment_translation = t_refined
            solutions.append(
                MRSolution(
                    rotation=R_rec_k.detach().cpu().numpy(),
                    translation=t_refined.detach().cpu().numpy(),
                    rotation_score=float(peak_k.score),
                    translation_score=float(r_analytic),
                    r_factor=float(r_rank),
                    model=refined,
                )
            )
            best_r = min(best_r, r_rank)

            n_done = k + 1
            if n_done >= self.min_tries and best_r < self.rfactor_converged:
                if self.verbose > 0:
                    print(
                        f"mr: converged (R {best_r:.4f} < "
                        f"{self.rfactor_converged}) after {n_done} candidates.",
                        flush=True,
                    )
                break
            if n_done >= max_tries:
                break

        if not solutions:
            raise RuntimeError("Translation + joint refine produced no candidates.")

        solutions.sort(key=lambda s: s.r_factor)
        winner = solutions[0]

        # Single solvent-aware Scaler refit on the winner for the user-facing R.
        timer.start("12_final_scaler")
        rwork_final = _external_rwork(winner.model, self.data)
        timer.stop("12_final_scaler")
        winner.model.last_alignment_rfactor = rwork_final
        winner.r_factor = rwork_final
        if self.verbose > 0:
            print(
                f"mr: winner analytical-TF R={winner.translation_score:.4f}, "
                f"final Scaler-fit R-work={rwork_final:.4f}",
                flush=True,
            )
        if self.verbose >= 2:
            print("\n" + timer.summary(), flush=True)
        return solutions

    # ------------------------------------------------------------------
    # Stage 1+2: rotation search + ML rescore
    # ------------------------------------------------------------------
    def _rotation_candidates(self, frf) -> list:
        """FRF rotation search followed by ML rescoring of the top peaks."""
        data = self.data
        device = self.device
        timer = self._timer

        timer.start("3_rotation_search")
        if self.verbose > 0:
            print(
                f"mr: rotation search (n_peaks={self.n_rotation_peaks}, "
                f"model error {self.model_error_A:.2f} A)…",
                flush=True,
            )
        peaks, _lmax, _d_min = search_peaks(
            self.model, data, self.model_error_A,
            U_aniso=frf.U_aniso, n_peaks=self.n_rotation_peaks,
            verbose=self.verbose,
        )
        timer.stop("3_rotation_search")

        if self.rescore_engine not in ("m_letf1", "sim", "none"):
            raise ValueError(
                f"rescore_engine={self.rescore_engine!r}; "
                "expected 'm_letf1' (default), 'sim' or 'none'."
            )

        F_obs = frf.F_obs
        hkl = frf.hkl
        s_mag = frf.s_mag
        centric = frf.centric
        ll = frf.ll
        rescore_n_shells = max(self.n_shells // 2, 8)

        # No ML rescore: rank candidates by the raw FRF score and let the
        # multi-candidate tree (FTF + refine + R-ranking) do the discrimination.
        # Sub-peak refinement is still available here: it sharpens each
        # orientation in place and does not reorder, so it is independent of
        # which engine (if any) ranks the candidates.
        if self.rescore_engine == "none":
            if self.verbose > 0:
                print("mr: ML rescore DISABLED — using raw FRF peak "
                      "ranking (RFZ).", flush=True)
            ranked = sorted(peaks, key=lambda p: p.score, reverse=True)
            if self.subpeak_refine:
                ranked = self._subpeak_refine(ranked, F_obs, hkl, s_mag,
                                              centric, ll, rescore_n_shells)
            return ranked

        interp_var_main: Optional[torch.Tensor] = None
        if self.use_interp_var:
            rescore_edges, _ = equal_count_shell_edges(s_mag, rescore_n_shells)
            rescore_shell_idx = assign_shells(s_mag, rescore_edges)
            interp_var_main = estimate_interp_var(
                ll, hkl, data.cell, rescore_shell_idx, rescore_n_shells,
            ).to(F_obs.dtype)

        timer.start("4_ml_rescore")
        if self.verbose > 0:
            print(
                f"mr: ML rescoring top "
                f"{min(len(peaks), self.n_ml_refine)} peaks…",
                flush=True,
            )
        if self.rescore_engine == "m_letf1":
            rescored = m_letf1_rescore(
                peaks, F_obs, hkl, s_mag, centric, ll, data.cell,
                data.spacegroup.matrices.to(torch.float64).to(device),
                n_shells=rescore_n_shells,
                n_refine=min(len(peaks), self.n_ml_refine),
                batch_size=50, verbose=self.verbose,
                scat_mode=self.rescore_scat_mode,
            )
            if self.subpeak_refine:
                rescored = self._subpeak_refine(rescored, F_obs, hkl, s_mag,
                                                centric, ll, rescore_n_shells)
        else:  # legacy Sim/Rice approximation
            rescored = sim_mlrf_rescore(
                peaks, F_obs, hkl, s_mag, centric, ll, data.cell,
                n_shells=rescore_n_shells,
                n_refine=min(len(peaks), self.n_ml_refine),
                batch_size=50, verbose=self.verbose,
                auto_variance_weights=self.auto_variance_weights,
                interp_var=interp_var_main,
            )
        timer.stop("4_ml_rescore")
        return rescored

    def _subpeak_refine(self, rescored, F_obs, hkl, s_mag, centric, ll,
                        rescore_n_shells):
        """Quadratic tangent-space Newton sharpening of the top orientations."""
        from .ml_rotation import _build_llg_context, quadratic_llg_refine

        data = self.data
        device = self.device
        self._timer.start("4b_subpeak_refine")
        ctx = _build_llg_context(
            F_obs, hkl, s_mag, centric, ll, data.cell,
            data.spacegroup.matrices.to(torch.float64).to(device),
            n_shells=rescore_n_shells, batch_size=50,
            scat_mode=self.rescore_scat_mode,
        )
        k = self.subpeak_refine_k if self.subpeak_refine_k > 0 else self.n_rotation_candidates
        k = min(k, len(rescored))
        rescored = quadratic_llg_refine(
            rescored, ctx, k_refine=k,
            step_deg=self.subpeak_refine_step_deg,
            iterations=self.subpeak_refine_iters,
            max_move_deg=self.subpeak_refine_max_move_deg,
            verbose=self.verbose,
        )
        self._timer.stop("4b_subpeak_refine")
        if self.verbose > 0:
            print(
                f"mr: sub-peak refined top {k} orientations "
                f"on the ML-LLG surface (step={self.subpeak_refine_step_deg}°).",
                flush=True,
            )
        return rescored

    def _make_rotated(self, peak: "RotationPeak"):
        """Rotate the search model onto a candidate orientation.

        Returns ``(rotated_model, R_recovered)`` where ``R_recovered`` maps the
        search-model frame onto the crystal frame; the applied coordinate
        rotation is ``R_recovered.T``.
        """
        R_rec = rotation_matrix_from_edmonds_euler(peak.alpha, peak.beta, peak.gamma)
        R_app = R_rec.T.contiguous()
        # .copy() first: Model.rotate mutates in place and returns self, so
        # rotating self.model directly would compound candidate k+1 onto k.
        rot = self.model.copy().rotate(
            R_app.to(device=self.model.device, dtype=self.model.dtype_float),
        )
        rot.last_alignment_rotation = R_rec
        return rot, R_rec

    # ------------------------------------------------------------------
    # Stage 3: per-candidate translation search + local refine
    # ------------------------------------------------------------------
    def _prepare_translation_arrays(self) -> None:
        """Resolution/validity-masked obs amplitudes + Miller indices."""
        data = self.data
        device = self.device
        hkl_full = data.hkl
        F_obs_full = data.F
        if hasattr(data, "get_valid_mask"):
            tmask = data.get_valid_mask()
        else:
            tmask = torch.ones(
                F_obs_full.shape[0], dtype=torch.bool, device=F_obs_full.device,
            )
        self._tmask = tmask
        self._F_obs_amp = F_obs_full[tmask].abs().to(torch.float64).to(device)
        self._hkl_keep = hkl_full[tmask].to(device)

    def _placement_for_candidate(self, rotated_k) -> Optional[tuple]:
        """Translation search + analytical-R local refine for one rotation.

        Returns ``(r_analytic, t_refined)`` for the best translation of this
        rotation candidate, or ``None`` if no translation peaks were found.
        """
        data = self.data
        device = self.device
        timer = self._timer
        eye3 = self._eye3

        if str(rotated_k.spacegroup) != str(data.spacegroup):
            # NOTE: assign the space-group NAME, not a SpaceGroup object. SpaceGroup is an
            # nn.Module, so nn.Module.__setattr__ intercepts object assignment, stores it in
            # _modules and never runs the property setter -- a silent no-op.
            rotated_k.spacegroup = data.spacegroup.hm
        rotated_p1 = rotated_k.copy()
        rotated_p1.spacegroup = "P 1"
        evaluator = _DirectModelEvaluator(rotated_p1)

        timer.start("5_precompute_G")
        G_pre, h_R_pre = precompute_G_for_rotation(
            evaluator, eye3, self._hkl_keep, data.spacegroup, data.cell,
        )
        timer.stop("5_precompute_G")

        timer.start("6_amplitude_TF")
        _, _, t_peaks = amplitude_translation_search(
            F_obs=self._F_obs_amp, interpolator=evaluator,
            R_rotation=eye3, hkl=self._hkl_keep,
            spacegroup=data.spacegroup, real_cell=data.cell,
            grid_steps=self.translation_grid_steps,
            n_peaks=self.n_translation_peaks,
            cluster_radius=0.05,
            precomputed_G=G_pre, precomputed_h_R=h_R_pre,
        )
        timer.stop("6_amplitude_TF")
        if not t_peaks:
            return None

        if self.use_llg_tf:
            t_peaks = self._llg_tf_rescore(t_peaks, G_pre, h_R_pre)

        if self.verbose > 0:
            tt = tuple(round(float(x), 3) for x in t_peaks[0].translation.tolist())
            print(f"  top translation t={tt} score={t_peaks[0].score:.4f}",
                  flush=True)

        # do_joint_refine=False: take the top translation peak directly (no
        # local refine), rank by its correlation score (negated so lower=better
        # like an R-factor).
        if not self.do_joint_refine:
            t_top = torch.as_tensor(t_peaks[0].translation, dtype=torch.float64)
            return -float(t_peaks[0].score), t_top

        best = None
        for k_t, tp in enumerate(t_peaks[:self.n_translation_candidates]):
            t_init = torch.as_tensor(tp.translation, dtype=torch.float64)
            timer.start("7_local_TF_refine")
            t_refined, r_analytic = local_translation_refine(
                F_obs=self._F_obs_amp, interpolator=evaluator,
                R_rotation=eye3, hkl=self._hkl_keep,
                spacegroup=data.spacegroup, real_cell=data.cell,
                t_init=t_init, radius=0.06, grid_steps=13,
                n_refinement_passes=1,
                precomputed_G=G_pre, precomputed_h_R=h_R_pre,
            )
            timer.stop("7_local_TF_refine")
            if self.verbose > 0:
                print(
                    f"    trans{k_t}: R(analytic)={r_analytic:.4f}, "
                    f"t={[round(float(x), 3) for x in t_refined.tolist()]}",
                    flush=True,
                )
            if best is None or r_analytic < best[0]:
                best = (r_analytic, t_refined)
        return best

    def _llg_tf_rescore(self, t_peaks, G_pre, h_R_pre):
        """Re-rank translation peaks by a shared-σA Rice/Woolfson LLG.

        Mirrors Phaser's FTF — the cheap amplitude correlation is a fast
        pre-filter but ranks poorly for partial models; the LLG ranks
        consistently with the rotation rescore.
        """
        data = self.data
        device = self.device
        F_obs_amp = self._F_obs_amp
        hkl_keep = self._hkl_keep
        tmask = self._tmask
        self._timer.start("6b_llg_tf_rescore")

        rec_basis_keep = data.cell.reciprocal_basis_matrix.to(torch.float64).to(device)
        s_mag_keep_tf = (hkl_keep.to(torch.float64) @ rec_basis_keep).norm(dim=-1)
        tf_n_shells = max(self.n_shells // 2, 8)
        tf_edges, _ = equal_count_shell_edges(s_mag_keep_tf, tf_n_shells)
        tf_shell_idx = assign_shells(s_mag_keep_tf, tf_edges)
        centric_keep_tf = (
            data.centric[tmask].to(torch.bool).to(device)
            if hasattr(data, "centric")
            else torch.zeros_like(F_obs_amp, dtype=torch.bool)
        )

        cnt_tf = torch.bincount(tf_shell_idx, minlength=tf_n_shells).to(torch.float64)
        sum_F2 = torch.zeros(tf_n_shells, dtype=torch.float64, device=device)
        sum_F2.scatter_add_(0, tf_shell_idx, F_obs_amp * F_obs_amp)
        mean_F2 = (sum_F2 / cnt_tf.clamp(min=1.0)).clamp(min=1e-30)
        E_obs_tf = F_obs_amp / mean_F2.sqrt().index_select(0, tf_shell_idx)

        t_top_t = torch.as_tensor(
            t_peaks[0].translation, dtype=torch.float64, device=device,
        )
        phase_top = torch.exp(
            2j * torch.pi * torch.einsum(
                "ind,d->in", h_R_pre.to(torch.float64), t_top_t,
            ).to(G_pre.dtype),
        )
        Fc_top = (G_pre * phase_top).sum(dim=0).abs().to(torch.float64)
        sum_Fc2 = torch.zeros(tf_n_shells, dtype=torch.float64, device=device)
        sum_Fc2.scatter_add_(0, tf_shell_idx, Fc_top * Fc_top)
        mean_Fc2 = (sum_Fc2 / cnt_tf.clamp(min=1.0)).clamp(min=1e-30)
        E_calc_top = Fc_top / mean_Fc2.sqrt().index_select(0, tf_shell_idx)
        sigma_a_tf = fit_sigma_a_per_shell(
            E_obs_tf, E_calc_top, centric_keep_tf,
            tf_shell_idx, tf_n_shells, n_grid=81,
        )

        t_cands = torch.as_tensor(
            np.stack([p.translation for p in t_peaks]),
            dtype=torch.float64, device=device,
        )
        llg_tf = llg_translation_rescore(
            F_obs=F_obs_amp, hkl=hkl_keep, centric=centric_keep_tf,
            shell_idx=tf_shell_idx, n_shells=tf_n_shells,
            G=G_pre, h_R=h_R_pre, t_candidates=t_cands,
            sigma_a=sigma_a_tf, interp_var=None,
        )
        self._timer.stop("6b_llg_tf_rescore")

        llg_list = llg_tf.detach().cpu().tolist()
        order = sorted(range(len(t_peaks)), key=lambda i: llg_list[i], reverse=True)
        return [
            TranslationPeak(
                translation=t_peaks[i].translation,
                score=float(llg_list[i]),
                sigma=float(llg_list[i]),
            )
            for i in order
        ]

    # ------------------------------------------------------------------
    # Stage 4+5: dense rotation re-sampling + rigid-body polish
    # ------------------------------------------------------------------
    def _dense_rotation_refine(self, refined):
        """Two-pass dense rotation re-sampling on the ML-LLG surface.

        Zooms the orientation onto the (sharper) ML-LLG basin at the found
        translation before the LBFGS polish. Returns the re-rotated model.
        """
        data = self.data
        device = self.device
        timer = self._timer

        timer.start("8_dense_R_ll_build")
        refined_p1 = refined.copy()
        refined_p1.spacegroup = "P 1"
        ll_refine = LattmanLoveInterpolator(
            refined_p1, padding_factor=self.ll_padding_factor,
            max_res_A=self.ll_max_res_A, verbose=0,
        )
        timer.stop("8_dense_R_ll_build")

        tmask = self._tmask
        hkl_keep = self._hkl_keep
        F_obs_amp = self._F_obs_amp
        centric_keep = (
            data.centric[tmask].to(torch.bool).to(device) if hasattr(data, "centric")
            else torch.zeros(hkl_keep.shape[0], dtype=torch.bool, device=device)
        )
        rec_basis_keep = data.cell.reciprocal_basis_matrix.to(torch.float64).to(device)
        s_mag_keep = (hkl_keep.to(torch.float64) @ rec_basis_keep).norm(dim=-1)
        rescore_n_shells = max(self.n_shells // 2, 8)

        n_per_axis_pass = [9, 5]
        zoom_factor = 4.0
        radii = [
            float(self.joint_refine_expected_rot_error),
            float(self.joint_refine_expected_rot_error) / zoom_factor,
        ]
        R_accumulated = torch.eye(3, dtype=torch.float64)

        interp_var_dense: Optional[torch.Tensor] = None
        if self.use_interp_var:
            dense_edges, _ = equal_count_shell_edges(s_mag_keep, rescore_n_shells)
            dense_shell_idx = assign_shells(s_mag_keep, dense_edges)
            interp_var_dense = estimate_interp_var(
                ll_refine, hkl_keep, data.cell, dense_shell_idx, rescore_n_shells,
            ).to(F_obs_amp.dtype)

        for pass_idx, max_perturb_rad in enumerate(radii):
            n_per_axis = n_per_axis_pass[pass_idx]
            coords_r = torch.linspace(
                -max_perturb_rad, max_perturb_rad, n_per_axis, dtype=torch.float64,
            )
            wx, wy, wz = torch.meshgrid(coords_r, coords_r, coords_r, indexing="ij")
            omegas = torch.stack([wx.flatten(), wy.flatten(), wz.flatten()], dim=-1)
            R_perturbs = _rodrigues(omegas)
            R_cand_full = R_perturbs @ R_accumulated
            cand_peaks = []
            for R_c in R_cand_full:
                a, b, g = edmonds_euler_from_rotation_matrix(R_c)
                cand_peaks.append(RotationPeak(alpha=a, beta=b, gamma=g,
                                               score=0.0, sigma=0.0))
            if self.verbose > 0:
                print(
                    f"  dense R pass {pass_idx + 1} "
                    f"({n_per_axis}³={omegas.shape[0]} perturbations, "
                    f"±{math.degrees(max_perturb_rad):.2f}°)…",
                    flush=True,
                )
            rescore_batch = max(4, min(100, 1_000_000 // max(hkl_keep.shape[0], 1)))
            timer.start("9_dense_R_rescore")
            if self.rescore_engine == "m_letf1":
                rescored_refine = m_letf1_rescore(
                    cand_peaks, F_obs_amp, hkl_keep, s_mag_keep, centric_keep,
                    ll_refine, data.cell,
                    data.spacegroup.matrices.to(torch.float64).to(device),
                    n_shells=rescore_n_shells,
                    n_refine=len(cand_peaks), batch_size=rescore_batch, verbose=0,
                )
            else:
                rescored_refine = sim_mlrf_rescore(
                    cand_peaks, F_obs_amp, hkl_keep, s_mag_keep, centric_keep,
                    ll_refine, data.cell,
                    n_shells=rescore_n_shells,
                    n_refine=len(cand_peaks), batch_size=rescore_batch,
                    verbose=0, n_D_grid=11, interp_var=interp_var_dense,
                )
            timer.stop("9_dense_R_rescore")
            top = rescored_refine[0]
            best_idx = next(
                i for i, p in enumerate(cand_peaks)
                if p.alpha == top.alpha and p.beta == top.beta
                and p.gamma == top.gamma
            )
            R_accumulated = R_cand_full[best_idx]
            if self.verbose > 0:
                print(
                    f"    pass {pass_idx + 1} best LLG={top.score:.2f}, "
                    f"|ω|={omegas[best_idx].norm().item() * 180 / math.pi:.3f}°",
                    flush=True,
                )

        return refined.copy().rotate(
            R_accumulated.T.to(self.model.dtype_float).contiguous(),
        )

    def _rigid_body_polish(self, refined):
        """LBFGS rigid-body polish on (R, t) — returns ``(polished, result)``.

        The model is pre-rotated/translated; the refinement optimises a small
        delta with ``initial_translation=0``. ``result`` carries the no-solvent
        initial/final R-work used by the caller's accept/reject gate.
        """
        timer = self._timer
        timer.start("11_lbfgs_polish")
        rb = RigidBodyRefinement(
            refined, self.data,
            initial_translation=torch.zeros(
                3, dtype=torch.float32, device=refined.device,
            ),
            expected_rotational_error=self.joint_refine_expected_rot_error,
            max_res=self.joint_refine_max_res_A,
            device=refined.device,
            verbose=max(0, self.verbose - 1),
            refine_b=self.refine_b,
            sigma_rot_deg=self.sigma_rot_deg,
            sigma_trans_ang=self.sigma_trans_ang,
            sigma_b=self.sigma_b,
        )
        rb_result = rb.refine()
        with torch.no_grad():
            R_polish = rb.get_rotation_matrix().detach()
            t_polish = rb.translation_frac.detach()
        # .copy() first: the caller keeps `refined` when the polish does not
        # improve R, so `polished` must not be the same object.
        polished = refined.copy().rotate(R_polish.to(self.model.dtype_float))
        polished = polished.translate(
            t_polish.to(self.model.dtype_float), fractional=True,
        )
        timer.stop("11_lbfgs_polish")
        return polished, rb_result
