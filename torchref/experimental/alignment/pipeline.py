"""Molecular replacement: the FRF hands a shortlist to the FTF.

1. **Fast Rotation Function** (:mod:`~torchref.experimental.alignment.rotation_search`
   over :mod:`~torchref.experimental.alignment.frf`) -- a Phaser-style
   Bessel-radial x spherical-harmonic expansion against a dense P1-box calc. A
   shortlist generator: one peak per orientation, symmetry mates suppressed.
2. **Fast Translation Function** (:mod:`~torchref.experimental.alignment.translation`)
   -- per orientation, one Crowther-Blow FFT over the cell, then the Rice/Woolfson
   likelihood at the best few peaks.

Every candidate is placed, with no early stopping, and the placements are ranked
by ``rank_by`` (the translation likelihood by default). The result is a
*placement*, not a refined model. :func:`align_model_to_data` delegates to
:class:`MolecularReplacementPipeline`; this module owns the control flow.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch

from torchref.config import get_default_device, get_float_dtype
from torchref.utils.device_mixin import DeviceMixin

from .frf.rotation_utils import rotation_matrix_from_edmonds_euler
from .frf.types import RotationPeak
from .rotation_search import _valid_mask, fit_anisotropy, search_peaks
from .translation import (
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
    candidate_index : int
        Position of this orientation in the rotation function's own ordering,
        so the depth of shortlist a solution came from can be read off.
    model : ModelFT or None
        The rotated and translated model. Built for the winner only, since
        copying a large model per candidate is costly;
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
    candidate_index: int = -1


class MolecularReplacementPipeline(DeviceMixin):
    """Canonical MR pipeline: FRF → FTF per candidate → rank.

    Parameters mirror :func:`align_model_to_data` (which delegates here), so a
    caller can either use ``align_model_to_data`` for the common case or drive this
    class directly for finer control / access to the ranked candidate list.

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data.
    model : ModelFT
        Initialised search model. Its own cell and space group are ignored:
        every model the pipeline returns is in the data's cell and space group.
    device : torch.device, optional
        Compute device (defaults to torchref's configured default device).
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
    d_min, d_max : float
        High- and low-resolution limits in Å of the overall-anisotropy fit, and
        the default translation window. The rotation search sets its own window
        from the bandwidth coupling. Defaults 4.0 and 15.0.
    n_shells : int
        Resolution shells of the anisotropy fit. Default 20.
    n_rotation_peaks : int
        Peaks the rotation function returns. Default 500.
    model_error_A : float, optional
        Expected r.m.s. coordinate error of the search model in Å; sets the
        sigma_A fall-off. Default ``None`` estimates it from the atom count
        (about 8 atoms per residue) with Oeffner et al. (2013), assuming the
        sequence is the target's.
    n_rotation_candidates : int
        Distinct orientations carried into the translation search, best first.
        Each costs a structure-factor evaluation and one FFT. A margin: with
        deposited models as search models the first peak was the true
        orientation on every panel cell. Raise it for poorer models. Default 10.
    n_translation_candidates : int
        Peaks of the fast translation map re-scored by the likelihood per
        orientation. Default 3.
    rank_by : {"llg", "r", "corr"}
        Score that ranks the placed candidates: the translation likelihood
        (default), the analytic R, or the fast translation score.
    tf_d_min, tf_d_max : float, optional
        Resolution window of the translation set in Å. ``None`` takes ``d_min``
        / ``d_max``; ``0.0`` / ``inf`` removes a cut, which is not safe: on the
        uncut set the fast score peaks away from the true position.

    Raises
    ------
    ValueError
        If ``rank_by`` is not one of the three scores.

    Examples
    --------
    ::

        from torchref.experimental.alignment import MolecularReplacementPipeline

        pipe = MolecularReplacementPipeline(data, model)
        solutions = pipe.run()
        print(f"best analytic R: {solutions[0].r_factor:.3f}")
    """

    def __init__(
        self,
        data: "ReflectionData",
        model: "ModelFT",
        *,
        device: Optional[torch.device] = None,
        verbose: int = 0,
        d_min: float = 4.0,
        d_max: float = 15.0,
        n_shells: int = 20,
        n_rotation_peaks: int = 500,
        model_error_A: Optional[float] = None,
        n_rotation_candidates: int = 10,
        n_translation_candidates: int = 3,
        rank_by: str = "llg",
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
        # Phaser's estimate from the model's length; ~8 heavy atoms per residue.
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
        self._obs = None
        self._tmask = None
        # One P1 copy of the search model, re-oriented in place per candidate
        # -- see `_prepare_translation_arrays`.
        self._p1 = None
        self._p1_xyz0 = None
        self._p1_center = None

    def _log(self, level: int, msg: str) -> None:
        """Print ``msg`` if ``verbose >= level`` (levels as on the class)."""
        if self.verbose >= level:
            print(msg, flush=True)

    def _log_candidate(self, k: int, peak, r_analytic, t_frac,
                       tf_score=None, llg_score=None) -> None:
        """Log one machine-readable ``CAND`` line per rotation candidate (level 2).

        ``key=value`` fields: ``k`` index in rotation-function order, ``rf``/``rfz``
        its score and z, ``tf`` the fast translation score, ``llg`` the translation
        likelihood, ``r`` the analytic R, ``t`` the fractional translation. All
        three placement scores are logged whichever one ranks.
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
            If ``False``, stop after the rotation search and return a single
            rotation-only solution (the model rotated onto the best
            orientation, no translation or refinement).

        Returns
        -------
        list of MRSolution
            Sorted by ``rank_by`` (``llg``/``corr`` descending, ``r`` ascending).
            ``r_factor`` is the analytic single-scale R, not a Scaler R-work.
        """
        if not self.model.ctx.initialized:
            raise RuntimeError(
                "Cannot fit an uninitialized ModelFT. Load PDB data first."
            )

        timer = self._timer
        timer.start("0_data_prep")
        U_aniso = fit_anisotropy(
            self.data, d_min=self.d_min, d_max=self.d_max,
            n_shells=self.n_shells, device=get_default_device(),
        )
        timer.stop("0_data_prep")

        # --- Stage 1: FRF rotation search ---
        candidates = self._rotation_candidates(U_aniso)
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
                    candidate_index=k,
                )
            )

        if not solutions:
            raise RuntimeError("Translation search produced no candidates.")

        # The three scores pick the same candidate on every pose-gated panel
        # cell measured. The likelihood is the default because it is the right
        # object: an R-factor on a partial model at this resolution has little
        # to distinguish with, and the fast score only expands the likelihood.
        if self.rank_by == "r":
            solutions.sort(key=lambda s: s.r_factor)
        elif self.rank_by == "corr":
            solutions.sort(key=lambda s: -s.translation_score)
        else:
            solutions.sort(key=lambda s: -s.llg_score)
        winner = solutions[0]

        # No Scaler refit for an R-work: it changes nothing about which placement
        # is returned. A caller that wants one builds a `Scaler` on the model.
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
    def _rotation_candidates(self, U_aniso: torch.Tensor) -> list:
        """FRF rotation search; the peaks it returns, ranked by its own score."""
        timer = self._timer

        timer.start("3_rotation_search")
        self._log(1, f"mr: rotation search (n_peaks={self.n_rotation_peaks}, "
                     f"model error {self.model_error_A:.2f} A)…")
        peaks, _lmax, _d_min = search_peaks(
            self.model, self.data, self.model_error_A,
            U_aniso=U_aniso, n_peaks=self.n_rotation_peaks,
            verbose=self.verbose,
        )
        timer.stop("3_rotation_search")

        # No rescore between the stages: re-ranking a shortlist that already
        # contains truth can push truth out of it, and on the panel it lowered
        # pose recovery. The translation function does the discrimination.
        return sorted(peaks, key=lambda p: p.score, reverse=True)

    def place(self, solution: MRSolution) -> "ModelFT":
        """Build the placed model for ``solution``: a copy of the search model,
        rotated and translated, carrying the alignment provenance attributes."""
        R_rec = torch.as_tensor(solution.rotation, dtype=torch.float64)  # dtype-ok: 3x3 rotation algebra in double on the host
        placed = self.model.copy().rotate(
            R_rec.T.contiguous().to(device=self.model.device,
                                    dtype=self.model.dtype_float),
        )
        self._into_crystal(placed)
        if solution.translation is not None:
            t = torch.as_tensor(solution.translation, dtype=self.model.dtype_float)
            placed.translate(t, fractional=True)
            placed.last_alignment_translation = t
        placed.last_alignment_rotation = R_rec
        placed.last_alignment_rfactor = solution.r_factor
        return placed

    def _into_crystal(self, m: "ModelFT", spacegroup=None) -> None:
        """Set ``m``'s cell and space group to the data's (or ``spacegroup``), in place.

        The search model's CRYST1 belongs to another crystal, or is a placeholder,
        and both fractional placement and the template's structure factors must
        be in the data's cell.
        """
        m.cell = self.data.cell.clone().to(device=m.device, dtype=m.dtype_float)
        m.spacegroup = self.data.spacegroup if spacegroup is None else spacegroup

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
        self._into_crystal(rot)
        rot.last_alignment_rotation = R_rec
        return rot, R_rec

    # ------------------------------------------------------------------
    # Stage 2: per-candidate translation search
    # ------------------------------------------------------------------
    def _prepare_translation_arrays(self) -> None:
        """Mask the observations for the translation search and normalise them once.

        The set is ``[tf_d_max, tf_d_min]`` and the data's validity mask, with
        its own Wilson fit. Do not remove the cut: on the uncut set the
        high-``|F_calc|^2`` reflections dominate the fast score, which then peaks
        away from the true position on the large structures.
        """
        data = self.data
        device = self.device
        hkl_full = data.hkl
        F_obs_full = data.F
        tmask = _valid_mask(data, F_obs_full.device)
        real = get_float_dtype()
        rec_basis = data.cell.reciprocal_basis_matrix.to(real)
        s_all = (hkl_full.to(real) @ rec_basis.to(hkl_full.device)).norm(dim=-1)
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
        # place per candidate rather than copied per candidate.
        #
        # Its FFT grid is sized to the translation set (|s| is invariant under
        # the symmetry rotations, so every rotated index lies inside 1/tf_d_min).
        # tf_d_min/1.5 rather than tf_d_min: at tf_d_min the transform loses
        # visible coherence with a fine grid, at /1.5 it matches it at a
        # fraction of the cost. max_res first -- the space-group setter rebuilds
        # the FFT and reads it.
        p1 = self.model.copy()
        if self.tf_d_min > 0.0:
            p1.max_res = self.tf_d_min / 1.5
        self._into_crystal(p1, "P 1")
        self._p1 = p1
        self._p1_xyz0 = p1.xyz().detach().clone()
        self._p1_center = self._p1_xyz0.mean(dim=0)

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
        cand = prepare_candidate(self._p1, obs, data.spacegroup, data.cell)
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
            np.stack([p.translation for p in t_peaks]), dtype=get_float_dtype(),
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
    n_rotation_candidates: int = 10,
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
