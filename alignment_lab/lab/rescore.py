"""Run the ML rescore over FRF peaks, and measure what it did to the ranking.

The rescore's job is narrow: take the top ~20 FRF peaks -- which on this
benchmark reliably contain the true orientation -- and promote the true one to
the front. It is **not** a global search, so feeding it a peak list that does
not contain truth measures nothing.

That makes the only honest metric a **paired** one: the rank of truth in the
list going in, versus its rank in the list coming out, on the same peaks. An
absolute post-rescore rank cannot distinguish "the rescore worked" from "the
FRF handed it an easy list".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

#: Rescore engines. ``none`` keeps the FRF's own ordering and is the control
#: arm -- without it, a rescore that merely preserves a good input ranking is
#: indistinguishable from one that improves it.
ENGINES = ("none", "m_letf1", "sim")


@dataclass
class RescoreResult:
    """Outcome of one rescore.

    Attributes
    ----------
    peaks : list
        Re-ordered ``RotationPeak`` list.
    engine : str
        Engine used.
    seconds : float
        Wall time.
    n_input : int
        Peaks handed to the engine.
    """

    peaks: list
    engine: str
    seconds: float
    n_input: int


def run_rescore(
    peaks: Sequence,
    data,
    frf_inputs,
    *,
    engine: str = "m_letf1",
    n_refine: int = 20,
    n_shells: Optional[int] = None,
    batch_size: int = 50,
    subpeak_refine: bool = False,
    verbose: int = 0,
    **engine_kwargs: Any,
) -> RescoreResult:
    """Rescore the top ``n_refine`` FRF peaks.

    Parameters
    ----------
    peaks : sequence
        FRF peaks, descending score.
    data : ReflectionData
        Dataset the peaks were scored against.
    frf_inputs : FRFInputs
        Prepared observations from the FRF run (``FRFResult.inputs``).
    engine : {'none', 'm_letf1', 'sim'}, optional
        ``'none'`` returns the input order unchanged -- the control arm.
    n_refine : int, optional
        How many leading peaks to rescore. Default 20, the intended use case.
    n_shells : int, optional
        Resolution shells for the rescore. Defaults to the pipeline's own rule,
        ``max(n_shells // 2, 8)`` with ``n_shells = 20``.
    batch_size : int, optional
        Orientations evaluated per batch. Default 50.
    subpeak_refine : bool, optional
        Apply the quadratic tangent-space refinement after rescoring.
    verbose : int, optional
        Engine verbosity.
    **engine_kwargs
        Passed through to the engine (e.g. ``e_convention``).

    Returns
    -------
    RescoreResult
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")

    subset = list(peaks)[: max(int(n_refine), 0)]
    if engine == "none" or not subset:
        return RescoreResult(peaks=subset, engine=engine, seconds=0.0,
                             n_input=len(subset))

    from torchref.experimental.alignment.ml_rotation import (
        m_letf1_rescore, sim_mlrf_rescore,
    )

    n_shells = n_shells if n_shells is not None else max(20 // 2, 8)
    device = frf_inputs.F_obs.device
    common = dict(
        n_shells=n_shells, n_refine=len(subset), batch_size=batch_size,
        verbose=verbose,
    )

    t0 = time.time()
    if engine == "m_letf1":
        # The sigmas now reach the rescore. They did not before: the FRF
        # computed the French-Wilson posterior from them and then discarded
        # them, leaving a likelihood with no measurement-error information.
        # An explicit `sig_F_obs` in `engine_kwargs` still wins, so an arm can
        # withhold them as a control.
        kw = dict(engine_kwargs)
        kw.setdefault("sig_F_obs", frf_inputs.sig_F)
        out = m_letf1_rescore(
            subset, frf_inputs.F_obs, frf_inputs.hkl, frf_inputs.s_mag,
            frf_inputs.centric, frf_inputs.ll, data.cell,
            data.spacegroup,
            **common, **kw,
        )
    else:
        out = sim_mlrf_rescore(
            subset, frf_inputs.F_obs, frf_inputs.hkl, frf_inputs.s_mag,
            frf_inputs.centric, frf_inputs.ll, data.cell,
            **common, **engine_kwargs,
        )
    if subpeak_refine:
        from torchref.experimental.alignment.ml_rotation import (
            _build_llg_context, quadratic_llg_refine,
        )

        ctx = _build_llg_context(
            frf_inputs.F_obs, frf_inputs.hkl, frf_inputs.s_mag,
            frf_inputs.centric, frf_inputs.ll, data.cell, n_shells=n_shells,
        )
        out = quadratic_llg_refine(out, ctx)
    seconds = time.time() - t0
    return RescoreResult(peaks=list(out), engine=engine, seconds=seconds,
                         n_input=len(subset))


def paired_ranks(
    frf_peaks: Sequence,
    rescored: Sequence,
    R_true: torch.Tensor,
    symops: torch.Tensor,
    *,
    n_refine: int,
    thr_deg: float = 5.0,
    **orbit_kw: Any,
) -> Dict[str, Any]:
    """Rank of truth before and after rescoring, on the same peak subset.

    Parameters
    ----------
    frf_peaks : sequence
        Full FRF peak list.
    rescored : sequence
        Output of :func:`run_rescore`.
    R_true : torch.Tensor
        True rotation.
    symops : torch.Tensor
        Symmetry rotation parts.
    n_refine : int
        Size of the subset handed to the rescore -- the comparison window.
    thr_deg : float, optional
        Orbit match threshold.
    **orbit_kw
        Orbit convention (``side``/``frame``/``reciprocal_basis``).

    Returns
    -------
    dict
        ``rank_frf`` (within the subset), ``rank_rescored``, ``delta``
        (positive = the rescore made it worse), ``truth_in_window`` and
        ``rank_frf_full``. ``delta`` is ``None`` when truth was never in the
        window, because then the rescore was never given the chance.
    """
    from .truth import orbit_rank

    subset = list(frf_peaks)[:n_refine]
    rank_full, _ = orbit_rank(frf_peaks, R_true, symops, thr_deg=thr_deg, **orbit_kw)
    rank_in, ang_in = orbit_rank(subset, R_true, symops, thr_deg=thr_deg, **orbit_kw)
    rank_out, ang_out = orbit_rank(rescored, R_true, symops, thr_deg=thr_deg, **orbit_kw)

    in_window = rank_in >= 0
    delta = (rank_out - rank_in) if (in_window and rank_out >= 0) else None
    return {
        "rank_frf_full": rank_full,
        "rank_frf": rank_in,
        "rank_rescored": rank_out,
        "delta": delta,
        "truth_in_window": in_window,
        "angle_frf": ang_in,
        "angle_rescored": ang_out,
    }
