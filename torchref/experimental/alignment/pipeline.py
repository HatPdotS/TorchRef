"""Molecular replacement: the FRF hands a shortlist to the FTF.

Two stages, and the division of labour between them is the design:

1. **Fast Rotation Function** — Phaser-faithful Bessel-radial × SH expansion
   against a dense P1-box calc. It is a *shortlist generator*. It does not have
   to rank well, and it does not: over the panel its own ordering puts truth
   first in a minority of cells. What it does reliably is put the true
   orientation somewhere in the top twenty-five -- in every cell of every
   panel run on record -- and its peaks are one per orientation, symmetry
   mates suppressed.
2. **Fast Translation Function** — for *each* of the top-N orientations, one
   Crowther-Blow FFT over the fractional cell on a resolution-sized grid, with
   the rotation function's own normalised score equation as its coefficients,
   then the Rice/Woolfson likelihood at the best few peaks. Rotation ghosts
   are morphologically identical to truth in a rotation function by
   construction; they are not identical once the crystal is involved.

There is deliberately nothing between them. An ML re-ranking of the FRF peaks
used to sit there and was removed: it reorders a shortlist that already contains
truth, and rotation recovery was 18/30 with it against 24/30 without (McNemar
p = 0.031, 6-0 discordant). Those figures, like every figure on this pipeline
before September 2026, gated on the rotation alone; see ``rank_by`` for what
the pose-gated panel measures.

Every candidate is placed and then the best is taken, with no early stopping.
Stopping early made the pipeline's answer depend on the order the rotation
function happened to produce -- it walked the list until one placement beat an
R-factor threshold and returned that, so it could accept the third candidate
without ever scoring the tenth. On a structure where several orientations place
plausibly that is not a choice between them, and it made the selection rule
impossible to reason about or to measure against a ranking harness.

The candidates are ranked by the translation search's likelihood -- see
``rank_by``, and the sort in :meth:`MolecularReplacementPipeline.run` for what
the three available scores measured against each other. The user-facing
solvent-aware R-work is computed once, on the winner. The pipeline returns a
*placement* -- refining it is the caller's job, and downstream refinement does
it better than a bolted-on polish did.

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
from typing import List, Optional, TYPE_CHECKING

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
    analytic_r_at,
    fast_translation_function,
    llg_at_translations,
    prepare_candidate,
)

if TYPE_CHECKING:
    from torchref.io.datasets import ReflectionData
    from torchref.model import ModelFT


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
        The fast translation function's score at the chosen peak, higher
        better. Reported, not ranked -- see the sort in :meth:`run`.
    r_factor : float
        The analytical-scale R at that placement, lower better. Reported, not
        ranked. A single global scale, so it is not the number a full Scaler
        would return -- build one on the returned model if that is wanted.
    llg_score : float
        **The ranking key**: the translation likelihood at that placement,
        higher better.
    model : ModelFT or None
        The rotated and translated model. Built for the winner only -- copying
        and moving a 20k-atom model 25 times was a quarter of the run on the
        large structures, to produce 24 models nobody reads.
        :meth:`MolecularReplacementPipeline.place` builds it for any other
        solution on request.
    """

    rotation: np.ndarray
    translation: Optional[np.ndarray]
    rotation_score: float
    translation_score: float
    r_factor: float
    model: Optional["ModelFT"] = None
    llg_score: float = float("nan")


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
        # Peaks of the fast translation function re-scored by the likelihood
        # for each orientation. The fast map only has to get the true peak
        # into this many; the likelihood picks.
        n_translation_candidates: int = 3,
        # Which score picks the winner among placed candidates. "llg" is the
        # translation likelihood; "r" the analytical-scale R-factor; "corr" the
        # fast translation function's own score. Not a tuning knob -- it exists
        # because the three can disagree and a rank-level proxy once got the
        # ordering wrong, so the comparison has to be made end to end on poses.
        # See the sort in `run` for what that measured.
        rank_by: str = "llg",
        # Resolution window for the translation set. None means the rotation
        # search's own [d_max, d_min], so one window and one normalisation
        # serve both stages. Pass 0.0 / inf to remove a cut -- and see
        # `_prepare_translation_arrays` for what the uncut set does.
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
        self.n_translation_candidates = n_translation_candidates
        if rank_by not in ("r", "corr", "llg"):
            raise ValueError(
                f"rank_by={rank_by!r}; expected 'r', 'corr' or 'llg'.")
        self.rank_by = rank_by
        self.tf_d_min = float(d_min if tf_d_min is None else tf_d_min)
        self.tf_d_max = float(d_max if tf_d_max is None else tf_d_max)

        self._timer = _StageTimer(enabled=verbose >= 2)
        # Filled in by run().
        self._frf = None
        self._obs = None
        self._tmask = None
        # One P1 copy of the search model, re-oriented in place per candidate
        # -- see `_prepare_translation_arrays`.
        self._p1 = None
        self._p1_xyz0 = None
        self._p1_center = None
        self._evaluator = None

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
                       tf_score=None, llg_score=None) -> None:
        """One line per rotation candidate, with every score behind the choice.

        Machine-readable on purpose. Diagnosing a wrong placement means asking
        which candidate won and on what, and the only alternative to emitting it
        here is a harness that re-implements the placement loop -- which drifts
        from the pipeline and then disagrees with it about which candidate the
        pipeline picked. A caller that knows the true orientation (a benchmark)
        can join these lines against it; the pipeline cannot, and does not try.

        Fields are ``key=value`` so a reader does not depend on column order:
        ``k`` candidate index in rotation-function order, ``rf``/``rfz`` its
        score and z, ``tf`` the fast translation function's score, ``llg`` the
        translation likelihood, ``r`` the analytical-scale R, ``t`` the
        fractional translation. All three placement scores are reported whichever one
        ranks, because which of them a wrong placement disagreed on is the
        question, and they do disagree.
        """
        tf = "nan" if tf_score is None else f"{float(tf_score):.5f}"
        llg = "nan" if llg_score is None else f"{float(llg_score):.1f}"
        t = ",".join(f"{float(x):.4f}" for x in t_frac)
        self._log(2, f"CAND k={k} rf={float(peak.score):.4f} "
                     f"rfz={float(peak.sigma):.3f} tf={tf} llg={llg} "
                     f"r={float(r_analytic):.5f} t={t}")


    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, do_translation: bool = True) -> List[MRSolution]:
        """Run the MR pipeline and return solutions, best first.

        Ranked by ``rank_by`` -- the translation likelihood by default.

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
            R_rec_k = rotation_matrix_from_edmonds_euler(
                peak_k.alpha, peak_k.beta, peak_k.gamma)
            self._orient_template(R_rec_k)
            self._log(3, f"\nfit_to_data: rot{k} "
                         f"(RF={peak_k.score:.2f}, σ_Z={peak_k.sigma:.2f})")
            placement = self._placement_for_candidate()
            if placement is None:
                self._log(2, f"CAND k={k} rf={float(peak_k.score):.4f} "
                             f"rfz={float(peak_k.sigma):.3f} tf=nan r=nan "
                             f"t=none  # no translation peaks")
                continue
            r_analytic, t_refined, tf_score, llg_score = placement
            self._log_candidate(k, peak_k, r_analytic, t_refined, tf_score,
                                llg_score)
            solutions.append(
                MRSolution(
                    rotation=R_rec_k.detach().cpu().numpy(),
                    translation=t_refined.detach().cpu().numpy(),
                    rotation_score=float(peak_k.score),
                    translation_score=float(tf_score),
                    r_factor=float(r_analytic),
                    llg_score=float(llg_score),
                )
            )

        if not solutions:
            raise RuntimeError("Translation + joint refine produced no candidates.")

        # Highest translation likelihood. The three scores are measured end to
        # end on POSES -- rotation and translation, against Cartesian symmetry
        # mates -- over six structures x ten seeds (the four the translation
        # search used to mis-place, plus two controls; job 544953):
        #
        #     llg    60/60
        #     r      60/60
        #     corr   60/60
        #
        # and they do not merely tie: in every one of the 60 cells the three
        # pick the SAME candidate, so the residual distributions are identical
        # arm for arm. Once the translation objective was normalised there was
        # nothing left for the selection rule to decide on this panel.
        #
        # The likelihood stays the default because it is the right object for
        # the question -- an R-factor on a partial model at this resolution has
        # little to distinguish with, and the fast score is an expansion of the
        # likelihood rather than the likelihood -- and because the arm is
        # selectable if a structure ever separates them. Every earlier figure
        # for these arms (37/40, 36/40, 32/40) gated on the rotation alone, with
        # a metric that miscounted trigonal mates; none of them stands.
        if self.rank_by == "r":
            solutions.sort(key=lambda s: s.r_factor)
        elif self.rank_by == "corr":
            solutions.sort(key=lambda s: -s.translation_score)
        else:
            solutions.sort(key=lambda s: -s.llg_score)
        winner = solutions[0]

        # No solvent-aware Scaler refit. It used to run here on the winner to
        # report an R-work, and cost about a third of the whole alignment -- 8.6
        # of 28 seconds on 2DQ6 -- to fit sixteen scaling parameters that change
        # nothing about which placement is returned. The pipeline's contract is
        # a placement; downstream refinement fits its own scaler properly, and
        # doing a worse version of that here to print a number is not worth a
        # third of the runtime. A caller that wants an R-work can build a
        # `Scaler` on the returned model.
        winner.model = self.place(winner)
        self._log(1, f"mr: winner ({self.rank_by}) "
                     f"LLG={winner.llg_score:.1f} "
                     f"TF={winner.translation_score:.5f} "
                     f"analytic R={winner.r_factor:.4f}")
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

    def place(self, solution: MRSolution) -> "ModelFT":
        """Build the placed model for ``solution``: a copy of the search model,
        rotated and translated, carrying the alignment provenance attributes."""
        R_rec = torch.as_tensor(solution.rotation, dtype=torch.float64)
        placed = self.model.copy().rotate(
            R_rec.T.contiguous().to(device=self.model.device,
                                    dtype=self.model.dtype_float),
        )
        if str(placed.spacegroup) != str(self.data.spacegroup):
            placed.spacegroup = self.data.spacegroup.hm
        if solution.translation is not None:
            t = torch.as_tensor(solution.translation, dtype=self.model.dtype_float)
            placed.translate(t, fractional=True)
            placed.last_alignment_translation = t
        placed.last_alignment_rotation = R_rec
        placed.last_alignment_rfactor = solution.r_factor
        return placed

    def _orient_template(self, R_rec: torch.Tensor) -> None:
        """Write the candidate orientation into the shared P1 copy.

        ``xyz = R_rec^T (xyz0 - c) + c`` about the search model's centroid, the
        same rotation ``Model.rotate`` would apply. The forward cache
        fingerprints parameters by pointer and version, so the next
        structure-factor call recomputes.
        """
        p1 = self._p1
        R_app = R_rec.T.to(device=self._p1_xyz0.device, dtype=self._p1_xyz0.dtype)
        p1.xyz[:] = (self._p1_xyz0 - self._p1_center) @ R_app.T + self._p1_center

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
        validity mask, and by default it is the rotation search's ``[d_max,
        d_min]``: one resolution window, one Wilson normalisation, both stages.

        The uncut set is not a safe default. With all data -- 228k reflections
        to 1.5 A on 2DQ6 -- the translation search places the four largest panel
        structures (2DQ6, 3VRJ, 4BX9, 6G9X) at the right orientation and 20-56 A
        from the true position, on every trial, while its own score is HIGHER
        at the wrong place than at the deposited pose (0.665 against 0.350 on
        2DQ6, where the likelihood is 1616 against 157865). At 15-4 A the same
        search recovers all thirty poses to within 0.32 A. The objective's
        calc side is raw ``|F_calc|^2``, so at high resolution it is dominated
        by whatever reflections happen to carry the largest calculated
        intensity rather than by the fit; the window is the first line of
        defence and the normalisation of that objective is the second.
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
        rec_basis = data.cell.reciprocal_basis_matrix.to(torch.float64)
        s_all = (hkl_full.to(torch.float64) @ rec_basis.to(hkl_full.device)
                 ).norm(dim=-1)
        if self.tf_d_min > 0.0:
            tmask = tmask & (s_all <= 1.0 / self.tf_d_min)
        if np.isfinite(self.tf_d_max):
            tmask = tmask & (s_all >= 1.0 / self.tf_d_max)
        self._tmask = tmask

        sig_F_full = getattr(data, "F_sigma", None)
        self._obs = TranslationObs.build(
            F_obs_full[tmask], hkl_full[tmask],
            data.spacegroup, data.cell,
            sig_F=None if sig_F_full is None else sig_F_full[tmask],
            delta_vrms_A=self.model_error_A,
            device=device,
        )
        if self.verbose >= 1:
            d_hi = 1.0 / float(self._obs.s_mag.max())
            d_lo = 1.0 / float(self._obs.s_mag.min().clamp(min=1e-9))
            self._log(1, f"mr: translation set {self._obs.F_obs.numel()} "
                         f"reflections, {d_lo:.1f}-{d_hi:.2f} A"
                         + ("" if sig_F_full is not None
                            else " (no sigmas: unit weight)"))

        # One P1 copy of the search model for the whole run, re-oriented in
        # place per candidate. Two copies per candidate -- one to rotate, one to
        # set P1 on -- were a quarter of the run on the large structures.
        #
        # Its FFT grid is sized to the translation set, not to the model's
        # default 1.0 A: |s| is invariant under the symmetry rotations, so every
        # rotated index the evaluator is asked for lies inside 1/tf_d_min. Two
        # thirds of the window's resolution, not the resolution itself: at
        # max_res = tf_d_min the transform's coherence with the 1.0 A grid over
        # the 15-4 A set is 0.987 on 2DQ6 (0.9987-0.9999 on 1DAW, 3K7M, 4BX9);
        # at tf_d_min/1.5 it is 0.9995-1.0000 everywhere, at 10-38 ms against
        # 200-860 ms. max_res first -- the space-group setter rebuilds the FFT
        # and reads it.
        p1 = self.model.copy()
        if self.tf_d_min > 0.0:
            p1.max_res = self.tf_d_min / 1.5
        p1.spacegroup = "P 1"
        self._p1 = p1
        self._p1_xyz0 = p1.xyz().detach().clone()
        self._p1_center = self._p1_xyz0.mean(dim=0)
        self._evaluator = DirectModelEvaluator(p1)

    def _placement_for_candidate(self) -> Optional[tuple]:
        """Translation search for the orientation currently in the P1 template.

        Returns ``(r_analytic, t, tf_score, llg)`` for the translation the
        likelihood prefers among the fast search's top peaks, or ``None`` if the
        map had no peaks. All three scores are at the same ``t``, so the
        reported numbers belong to the placement that was actually chosen.
        """
        data = self.data
        timer = self._timer
        obs = self._obs

        timer.start("5_candidate_transform")
        cand = prepare_candidate(self._evaluator, obs, data.spacegroup, data.cell)
        timer.stop("5_candidate_transform")

        # One FFT on a grid a third of the set's resolution apart: dense enough
        # that the parabolic peak refinement lands within a fraction of a step,
        # and no coarse-then-refine pair whose coarse half could miss the peak.
        d_min_set = 1.0 / float(obs.s_mag.max())
        timer.start("6_translation_function")
        _, t_peaks = fast_translation_function(
            obs, cand, data.cell,
            grid_spacing_A=d_min_set / 3.0,
            n_peaks=self.n_translation_candidates,
            cluster_radius_A=d_min_set,
        )
        timer.stop("6_translation_function")
        if not t_peaks:
            return None

        timer.start("7_translation_llg")
        t_cands = torch.as_tensor(
            np.stack([p.translation for p in t_peaks]), dtype=torch.float64,
        )
        llg = llg_at_translations(obs, cand, t_cands)
        k_best = int(llg.argmax())
        t_best = t_cands[k_best]
        r_analytic = analytic_r_at(obs, cand, t_best)
        timer.stop("7_translation_llg")
        for k_t, tp in enumerate(t_peaks):
            self._log(3, f"    trans{k_t}: tf={tp.score:.4f} z={tp.sigma:.2f} "
                         f"llg={float(llg[k_t]):.1f} "
                         f"t={[round(float(x), 3) for x in tp.translation]}")
        return (r_analytic, t_best, float(t_peaks[k_best].score), float(llg[k_best]))


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
    n_translation_candidates: int = 3,
    n_rotation_candidates: int = 25,
    rank_by: str = "llg",
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
        n_translation_candidates=n_translation_candidates,
        rank_by=rank_by,
        tf_d_min=tf_d_min, tf_d_max=tf_d_max,
    )
    solutions = pipeline.run(do_translation=do_translation)
    return solutions[0].model
