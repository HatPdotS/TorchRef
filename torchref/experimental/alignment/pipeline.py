"""Molecular replacement: the FRF hands a shortlist to the FTF.

Two stages, and the division of labour between them is the design:

1. **Fast Rotation Function** — Phaser-faithful Bessel-radial × SH expansion
   against a dense P1-box calc. It is a *shortlist generator*. It does not have
   to rank well, and measurably does not: over 30 seeded cells it puts truth at
   rank 0 in 6 of them. What it does reliably is put truth somewhere in the top
   twenty.
2. **Fast Translation Function** — for *each* of the top-N orientations, a
   Crowther-Blow amplitude-correlation search over the fractional cell,
   optionally re-ranked by a Rice/Woolfson LLG, then an analytical-R local
   refine. On the same 30 cells it puts truth at rank 0 in 27. Rotation ghosts
   are morphologically identical to truth in a rotation function by
   construction; they are not identical once the crystal is involved.

There is deliberately nothing between them. An ML re-ranking of the FRF peaks
used to sit there and was removed: it reorders a shortlist that already contains
truth, and end-to-end pose recovery was 18/30 with it against 24/30 without
(McNemar p = 0.031, 6-0 discordant).

Every candidate is placed and then the best is taken, with no early stopping.
Stopping early made the pipeline's answer depend on the order the rotation
function happened to produce -- it walked the list until one placement beat an
R-factor threshold and returned that, so it could accept the third candidate
without ever scoring the tenth. On a structure where several orientations place
plausibly that is not a choice between them, and it made the selection rule
impossible to reason about or to measure against a ranking harness.

The candidates are ranked by the translation search's analytical R. The
user-facing solvent-aware R-work is computed once, on the winner. The pipeline
returns a *placement* -- refining it is the caller's job, and downstream
refinement does it better than a bolted-on polish did.

``align_model_to_data`` delegates here; this class is the implementation of
record. The crystallographic stages live in
:mod:`torchref.experimental.alignment.align` and
:mod:`~torchref.experimental.alignment.translation`; this module owns the
control flow that wires them together.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from torchref.config import get_default_device
from torchref.utils.device_mixin import DeviceMixin

from .frf.rotation_utils import rotation_matrix_from_edmonds_euler
from .frf.types import RotationPeak
from .rotation_search import prepare_frf_inputs, search_peaks
from .translation import (
    DirectModelEvaluator,
    TranslationObs,
    TranslationPeak,
    amplitude_translation_search,
    correlation_at,
    fit_sigma_a_per_shell,
    llg_translation_rescore,
    local_translation_refine,
    normalise_calc,
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


# ---------------------------------------------------------------------------
# Stage timing and the user-facing R-work
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


def _external_rwork(model: "ModelFT", data: "ReflectionData") -> float:
    """Full-resolution scaled R-work via the standard Scaler.

    The TF + local refine work in analytical-scale R-factor (which ranks
    candidates correctly but isn't the user-facing R-work). We compute the
    proper Scaler-fit R-work once per finalist.
    """
    from ...base.metrics.rfactor import rfactor_work_free
    from ...scaling import Scaler

    # No device override: the Scaler takes the configured default, which is
    # the one place a device is decided. Reading it off whichever tensor is
    # nearest is what puts a run on two devices at once.
    s = Scaler(model=model, data=data, nbins=20, verbose=0)
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
        The rotation function's score for this candidate.
    translation_score : float
        The translation function's correlation at the chosen translation, higher
        better. Reported, not ranked -- see the sort in :meth:`run`.
    r_factor : float
        **The ranking key**: the analytical-scale R at that placement, lower
        better. For the returned winner it is replaced by the solvent-aware
        Scaler R-work.
    model : ModelFT
        The rotated (+translated +refined) model for this candidate.
    """

    rotation: np.ndarray
    translation: Optional[np.ndarray]
    rotation_score: float
    translation_score: float
    r_factor: float
    model: "ModelFT"


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
        How much the run says about itself. Each level is a superset of the one
        below, and the boundaries are chosen so that a level is useful on its
        own rather than being "a bit more of the same":

        0
            Silent.
        1
            What happened: the search settings, one line per stage, and the
            winner. Enough to see that a run did the expected work.
        2
            **Why it chose what it chose.** One ``CAND`` line per rotation
            candidate carrying every score the selection could have used, plus
            the per-stage wall-clock table. This is the level that makes the
            pipeline diagnosable without a second implementation of its own
            scoring -- see :meth:`_log_candidate`.
        3
            Per-translation-peak detail inside each candidate.

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
        n_rotation_peaks: int = 500,
        model_error_A: Optional[float] = None,
        # --- candidate tree ---
        n_rotation_candidates: int = 25,
        n_translation_peaks: int = 20,
        n_translation_candidates: int = 3,
        translation_grid_steps: int = 16,
        use_llg_tf: bool = False,
        # Resolution window for the TRANSLATION set only, independent of the
        # rotation search's [d_max, d_min]. None means no cut, which is what
        # this stage has always done -- see `_prepare_translation_arrays`.
        tf_d_min: Optional[float] = None,
        tf_d_max: Optional[float] = None,
    ):
        self.data = data
        self.model = model
        self.device = device or get_default_device()
        self.verbose = verbose

        self.d_min = d_min
        self.d_max = d_max
        self.n_shells = n_shells
        self.n_rotation_peaks = n_rotation_peaks
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

        self.n_rotation_candidates = n_rotation_candidates
        self.n_translation_peaks = n_translation_peaks
        self.n_translation_candidates = n_translation_candidates
        self.translation_grid_steps = translation_grid_steps
        self.use_llg_tf = use_llg_tf
        self.tf_d_min = tf_d_min
        self.tf_d_max = tf_d_max

        self._timer = _StageTimer(enabled=verbose >= 2)
        # Filled in by run().
        self._frf = None
        self._obs = None
        self._tmask = None
        self._eye3 = torch.eye(3, dtype=torch.float64)

    #: Levels are documented on the class. They are a contract, not a dial:
    #: level 2 is specifically "one machine-readable line per candidate", and
    #: anything added at that level should preserve that.
    def _log(self, level: int, msg: str) -> None:
        """Emit ``msg`` if the run is at least this verbose.

        One emitter rather than ``if self.verbose > 0: print(...)`` at every
        site. The scattered form is how levels drift -- the same stage ends up
        reporting at 1 in one place and 2 in another, and nothing enforces that
        a level means the same thing twice.
        """
        if self.verbose >= level:
            print(msg, flush=True)

    def _log_candidate(self, k: int, peak, r_analytic, t_frac,
                       tf_score=None) -> None:
        """One line per rotation candidate, with every score behind the choice.

        Machine-readable on purpose. Diagnosing a wrong placement means asking
        which candidate won and on what, and the only alternative to emitting it
        here is a harness that re-implements the placement loop -- which drifts
        from the pipeline and then disagrees with it about which candidate the
        pipeline picked. A caller that knows the true orientation (a benchmark)
        can join these lines against it; the pipeline cannot, and does not try.

        Fields are ``key=value`` so a reader does not depend on column order:
        ``k`` candidate index in rotation-function order, ``rf``/``rfz`` its
        score and z, ``tf`` the translation correlation at the chosen peak,
        ``r`` the analytical-scale R that ranks it, ``t`` the fractional
        translation.
        """
        tf = "nan" if tf_score is None else f"{float(tf_score):.5f}"
        t = ",".join(f"{float(x):.4f}" for x in t_frac)
        self._log(2, f"CAND k={k} rf={float(peak.score):.4f} "
                     f"rfz={float(peak.sigma):.3f} tf={tf} "
                     f"r={float(r_analytic):.5f} t={t}")


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
        if not self.model.ctx.initialized:
            raise RuntimeError(
                "Cannot fit an uninitialized ModelFT. Load PDB data first."
            )

        timer = self._timer
        timer.start("0_data_prep")
        frf = prepare_frf_inputs(
            self.model, self.data,
            d_min=self.d_min, d_max=self.d_max, n_shells=self.n_shells,
            verbose=self.verbose,
        )
        timer.stop("0_data_prep")
        self._frf = frf

        # --- Stage 1: FRF rotation search ---
        candidates = self._rotation_candidates(frf)
        if not candidates:
            raise RuntimeError("Rotation search produced no peaks.")

        if not do_translation:
            rotated, R_rec = self._make_rotated(candidates[0])
            top = candidates[0]
            self._log(1, f"mr: top peak RF = {top.score:.2f} "
                         f"(σ_Z = {top.sigma:.2f}); applying R⁻¹ to coords.")
            self._log(2, "\n" + timer.summary())
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

        # --- Stage 2: per-candidate translation search ---
        self._prepare_translation_arrays()
        n_rot = min(self.n_rotation_candidates, len(candidates))
        if n_rot > 1:
            self._log(1, f"mr: placing all {n_rot} rotation candidates…")

        solutions: List[MRSolution] = []
        for k in range(n_rot):
            peak_k = candidates[k]
            rotated_k, R_rec_k = self._make_rotated(peak_k)
            self._log(3, f"\nfit_to_data: rot{k} "
                         f"(RF={peak_k.score:.2f}, σ_Z={peak_k.sigma:.2f})")
            placement = self._placement_for_candidate(rotated_k)
            if placement is None:
                self._log(2, f"CAND k={k} rf={float(peak_k.score):.4f} "
                             f"rfz={float(peak_k.sigma):.3f} tf=nan r=nan "
                             f"t=none  # no translation peaks")
                continue
            r_analytic, t_refined, tf_score = placement
            self._log_candidate(k, peak_k, r_analytic, t_refined, tf_score)

            placed = rotated_k.copy().translate(
                t_refined.to(self.model.dtype_float), fractional=True,
            )
            placed.last_alignment_rotation = R_rec_k
            placed.last_alignment_translation = t_refined
            solutions.append(
                MRSolution(
                    rotation=R_rec_k.detach().cpu().numpy(),
                    translation=t_refined.detach().cpu().numpy(),
                    rotation_score=float(peak_k.score),
                    translation_score=float(tf_score),
                    r_factor=float(r_analytic),
                    model=placed,
                )
            )

        if not solutions:
            raise RuntimeError("Translation + joint refine produced no candidates.")

        # Lowest analytical-scale R. Ranking by the translation correlation was
        # tried and is worse end to end -- 31/40 against 36/40 over four
        # structures x ten seeds, losing on both 2DQ6 (6/10 -> 3/10) and 6G9X
        # (10/10 -> 8/10). A rank-level harness had predicted the opposite by a
        # wide margin, which is why the correlation is still reported on every
        # candidate at verbose >= 2: the two orderings disagree and the
        # disagreement is not yet understood.
        solutions.sort(key=lambda s: s.r_factor)
        winner = solutions[0]

        # Single solvent-aware Scaler refit on the winner for the user-facing R.
        timer.start("12_final_scaler")
        rwork_final = _external_rwork(winner.model, self.data)
        timer.stop("12_final_scaler")
        winner.model.last_alignment_rfactor = rwork_final
        winner.r_factor = rwork_final
        self._log(1, f"mr: winner TF correlation={winner.translation_score:.5f}, "
                     f"analytic R={winner.r_factor:.4f}, "
                     f"final Scaler-fit R-work={rwork_final:.4f}")
        self._log(2, "\n" + timer.summary())
        return solutions

    # ------------------------------------------------------------------
    # Stage 1: rotation search
    # ------------------------------------------------------------------
    def _rotation_candidates(self, frf) -> list:
        """FRF rotation search; the peaks it returns, ranked by its own score."""
        timer = self._timer

        timer.start("3_rotation_search")
        self._log(1, f"mr: rotation search (n_peaks={self.n_rotation_peaks}, "
                     f"model error {self.model_error_A:.2f} A)…")
        peaks, _lmax, _d_min = search_peaks(
            self.model, self.data, self.model_error_A,
            U_aniso=frf.U_aniso, n_peaks=self.n_rotation_peaks,
            verbose=self.verbose,
        )
        timer.stop("3_rotation_search")

        # Rank by the FRF's own score and hand the shortlist to the
        # translation search. There is no rescore here by design: an ML
        # re-ranking of these peaks was measured to lower end-to-end pose
        # recovery from 24/30 to 18/30 (McNemar p = 0.031), because it reorders
        # a shortlist that already contains truth and sometimes pushes truth
        # out of it. The translation function does the discrimination.
        return sorted(peaks, key=lambda p: p.score, reverse=True)

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
    # Stage 2: per-candidate translation search + local refine
    # ------------------------------------------------------------------
    def _prepare_translation_arrays(self) -> None:
        """Mask the observations for the translation search and normalise them once.

        The window is ``[tf_d_max, tf_d_min]`` on top of the dataset's own
        validity mask. Both default to ``None``, meaning **no resolution cut** --
        which is what this stage has always done, though it used to claim
        otherwise. So the translation search sees the data's full resolution
        while the rotation search runs at ``[d_max, d_min]`` = [15, 4] A. That
        asymmetry is deliberate on one side and unexamined on the other: the
        rotation function is bandwidth-limited and cannot use high-resolution
        terms, and nobody has measured what the translation function wants. The
        parameter exists so that choosing is possible; the default does not
        choose.
        """
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
        if self.tf_d_min is not None or self.tf_d_max is not None:
            rec_basis = data.cell.reciprocal_basis_matrix.to(torch.float64)
            s_all = (hkl_full.to(torch.float64) @ rec_basis.to(hkl_full.device)
                     ).norm(dim=-1)
            if self.tf_d_min is not None:
                tmask = tmask & (s_all <= 1.0 / float(self.tf_d_min))
            if self.tf_d_max is not None:
                tmask = tmask & (s_all >= 1.0 / float(self.tf_d_max))
        self._tmask = tmask

        sig_F_full = getattr(data, "F_sigma", None)
        self._obs = TranslationObs.build(
            F_obs_full[tmask], hkl_full[tmask],
            data.spacegroup, data.cell,
            sig_F=None if sig_F_full is None else sig_F_full[tmask],
            delta_vrms_A=self.model_error_A,
            n_shells=max(self.n_shells // 2, 8),
            device=device,
        )
        if self.verbose >= 1:
            d_hi = 1.0 / float(self._obs.s_mag.max())
            d_lo = 1.0 / float(self._obs.s_mag.min().clamp(min=1e-9))
            self._log(1, f"mr: translation set {self._obs.F_obs.numel()} "
                         f"reflections, {d_lo:.1f}-{d_hi:.2f} A"
                         + ("" if sig_F_full is not None
                            else " (no sigmas: unit weight)"))

    def _placement_for_candidate(self, rotated_k) -> Optional[tuple]:
        """Translation search + analytical-R local refine for one rotation.

        Returns ``(r_analytic, t_refined, tf_score)`` for the best translation
        of this rotation candidate, or ``None`` if no translation peaks were
        found.

        ``tf_score`` is the translation function's own score at its top peak.
        It does not select anything -- ``r_analytic`` does -- but it is carried
        out so ``verbose >= 2`` can report both. Which of the two a wrong
        placement disagreed on is the first thing anyone diagnosing one asks,
        and it is not recoverable afterwards from the winner alone.
        """
        data = self.data
        timer = self._timer
        eye3 = self._eye3

        if str(rotated_k.spacegroup) != str(data.spacegroup):
            # NOTE: assign the space-group NAME, not a SpaceGroup object. SpaceGroup is an
            # nn.Module, so nn.Module.__setattr__ intercepts object assignment, stores it in
            # _modules and never runs the property setter -- a silent no-op.
            rotated_k.spacegroup = data.spacegroup.hm
        rotated_p1 = rotated_k.copy()
        rotated_p1.spacegroup = "P 1"
        evaluator = DirectModelEvaluator(rotated_p1)

        timer.start("5_precompute_G")
        G_pre, h_R_pre = precompute_G_for_rotation(
            evaluator, eye3, self._obs.hkl, data.spacegroup, data.cell,
        )
        timer.stop("5_precompute_G")

        timer.start("6_amplitude_TF")
        _, _, t_peaks = amplitude_translation_search(
            obs=self._obs, interpolator=evaluator, R_rotation=eye3,
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

        tf_top = float(t_peaks[0].score)
        if self.verbose >= 3:
            tt = tuple(round(float(x), 3) for x in t_peaks[0].translation.tolist())
            self._log(3, f"  top translation t={tt} score={tf_top:.4f}")

        best = None
        for k_t, tp in enumerate(t_peaks[:self.n_translation_candidates]):
            t_init = torch.as_tensor(tp.translation, dtype=torch.float64)
            timer.start("7_local_TF_refine")
            t_refined, r_analytic = local_translation_refine(
                obs=self._obs, interpolator=evaluator, R_rotation=eye3,
                spacegroup=data.spacegroup, real_cell=data.cell,
                t_init=t_init, radius=0.06, grid_steps=13,
                n_refinement_passes=1,
                precomputed_G=G_pre, precomputed_h_R=h_R_pre,
            )
            timer.stop("7_local_TF_refine")
            # Both scores at the REFINED position, so the reported correlation
            # belongs to the translation that was actually chosen. Selection is
            # by R: ranking candidates by the correlation instead was measured
            # end to end and is WORSE (31/40 against 36/40 over four structures
            # x ten seeds), despite a rank-level harness predicting the reverse.
            tf_ref = correlation_at(self._obs, G_pre, h_R_pre, t_refined)
            self._log(3, f"    trans{k_t}: tf={tf_ref:.5f} "
                         f"R(analytic)={r_analytic:.4f}, "
                         f"t={[round(float(x), 3) for x in t_refined.tolist()]}")
            if best is None or r_analytic < best[0]:
                best = (r_analytic, t_refined, tf_ref)
        return best

    def _llg_tf_rescore(self, t_peaks, G_pre, h_R_pre):
        """Re-rank translation peaks by a shared-σA Rice/Woolfson LLG.

        Mirrors Phaser's FTF: the amplitude correlation is a cheap pre-filter,
        and this is the likelihood that ranks its peaks. It reuses the run's
        single Wilson normalisation and its shell binning, so ``E_obs`` here is
        the same ``E_obs`` the correlation maximised.

        Off by default. It is the strongest discriminator at rank level, and
        end-to-end it changes nothing: 27/30 against 28/30 with one discordant
        cell in 30, which the correlation wins.
        """
        device = self.device
        obs = self._obs
        self._timer.start("6b_llg_tf_rescore")

        # sigma_A is fitted against the top translation only, and reused for
        # every candidate. It is a per-shell model-reliability curve, not a
        # per-candidate score: refitting it per t would let each candidate
        # choose the D that flatters it, which is scoring a model against a
        # likelihood tuned to that model.
        t_top_t = torch.as_tensor(
            t_peaks[0].translation, dtype=torch.float64, device=device,
        )
        phase_top = torch.exp(
            2j * torch.pi * torch.einsum(
                "ind,d->in", h_R_pre.to(torch.float64), t_top_t,
            ).to(G_pre.dtype),
        )
        Fc_top = (G_pre * phase_top).sum(dim=0).abs().to(torch.float64)
        E_calc_top = normalise_calc(Fc_top, obs)
        sigma_a_tf = fit_sigma_a_per_shell(
            obs.E_obs, E_calc_top, obs.centric,
            obs.shell_idx, obs.n_shells, n_grid=81,
        )

        t_cands = torch.as_tensor(
            np.stack([p.translation for p in t_peaks]),
            dtype=torch.float64, device=device,
        )
        llg_tf = llg_translation_rescore(
            obs=obs, G=G_pre, h_R=h_R_pre, t_candidates=t_cands,
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
    n_rotation_candidates: int = 25,
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

    pipeline = MolecularReplacementPipeline(
        data, model,
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
