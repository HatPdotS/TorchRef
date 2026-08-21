"""Run the FRF and capture the full rotation function, not just the peak list.

Every rank/ghost diagnostic needs the dense adaptive sample list as well as the
peaks, and the engine only returns the peaks. The capture below wraps the
engine's search entry point for the duration of one call; nine scripts each
carried their own copy of this monkeypatch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class FRFConfig:
    """Engine settings for one FRF evaluation.

    Collected into one object so a diagnostic passes a single config around and
    the settings can be written into the result row verbatim.
    """

    d_min: float = 4.0
    d_max: float = 15.0
    n_shells: int = 20
    n_peaks: int = 500
    lmax_cap: int = 48
    dense_pad: float = 2.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """Config fields for a result row (``extra`` flattened out)."""
        d = asdict(self)
        d.pop("extra")
        d.update(self.extra)
        return d


@contextmanager
def patched(module: Any, name: str, replacement: Any):
    """Temporarily replace ``module.name``, restoring it on exit.

    The "swap one engine internal and re-measure the rank" pattern -- used for
    the dense-grid and box-construction experiments -- always needs the original
    restored even when the body raises.

    Parameters
    ----------
    module : module or object
        Namespace holding the attribute.
    name : str
        Attribute name.
    replacement : Any
        Temporary value.
    """
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield original
    finally:
        setattr(module, name, original)


@dataclass
class FRFResult:
    """Outcome of one FRF evaluation.

    Attributes
    ----------
    peaks : list
        ``RotationPeak`` list, descending score.
    arf : AdaptiveRotationFunction or None
        The full adaptive sample list, when captured.
    sigma : torch.Tensor or None
        ``arf.values`` standardised to zero mean / unit sd -- the scale peak
        heights are quoted in.
    seconds : float
        Wall time of the search call.
    inputs : FRFInputs or None
        The prepared observations (``F_obs``/``hkl``/``s_mag``/``centric``/``ll``).
        The rescore consumes these, so keeping them lets a rescore run reuse one
        FRF evaluation instead of recomputing it.
    """

    peaks: list
    arf: Optional[Any]
    sigma: Optional[torch.Tensor]
    seconds: float
    inputs: Optional[Any] = None

    @property
    def map_max_sigma(self) -> float:
        """Largest value of the standardised rotation function."""
        return float(self.sigma.max()) if self.sigma is not None else float("nan")


def merge_peak_lists(peak_lists, *, n_peaks: int, nms_radius_deg: float):
    """Merge several peak lists into one, ranked by z-score.

    Used for the Patterson-radius union: the same obs expanded to two different
    integration radii give two rotation functions whose absolute values are not
    comparable, but whose per-run standardised heights (``RotationPeak.sigma``)
    are. Peaks are pooled, sorted by sigma, and greedily suppressed by SO(3)
    angular distance so the same orientation found by both radii appears once.

    Parameters
    ----------
    peak_lists : sequence of list of RotationPeak
        One list per run.
    n_peaks : int
        Cap on the merged list.
    nms_radius_deg : float
        Suppression radius, in degrees of SO(3) geodesic distance.

    Returns
    -------
    list of RotationPeak
    """
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_angular_distance_deg,
        rotation_matrix_from_edmonds_euler,
    )

    pooled = [p for pl in peak_lists for p in pl]
    pooled.sort(key=lambda p: p.sigma, reverse=True)
    kept, kept_R = [], []
    for p in pooled:
        R = rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma)
        if any(rotation_angular_distance_deg(R, Rk) < nms_radius_deg
               for Rk in kept_R):
            continue
        kept.append(p)
        kept_R.append(R)
        if len(kept) >= n_peaks:
            break
    return kept


def run_frf(
    model,
    data,
    cfg: Optional[FRFConfig] = None,
    *,
    capture_arf: bool = True,
    verbose: int = 0,
) -> FRFResult:
    """Run the separated FRF on an already-rotated search model.

    Parameters
    ----------
    model : ModelFT
        Search model, already in the orientation to be scored.
    data : ReflectionData
        Observed reflections.
    cfg : FRFConfig, optional
        Engine settings. Defaults to :class:`FRFConfig`.
    capture_arf : bool, optional
        Also return the dense adaptive sample list. Default True.
    verbose : int, optional
        Engine verbosity. Default 0.

    Returns
    -------
    FRFResult
    """
    import time

    from torchref.experimental.alignment import align as _align
    from torchref.experimental.alignment.frf import api as _api

    cfg = cfg or FRFConfig()
    captured: Dict[str, Any] = {}

    def _wrapped(*args, **kwargs):
        arf, peaks = _original(*args, **kwargs)
        captured["arf"] = arf
        return arf, peaks

    frf_inputs = _align._prepare_frf_inputs(
        model, data,
        d_min=cfg.d_min, d_max=cfg.d_max, n_shells=cfg.n_shells, verbose=verbose,
    )

    t0 = time.time()
    if capture_arf:
        _original = _api.phaser_rotation_search
        with patched(_api, "phaser_rotation_search", _wrapped):
            peaks = _align._run_frf_separate_rotation(
                model, data, frf_inputs, n_peaks=cfg.n_peaks, verbose=verbose,
                lmax_cap=cfg.lmax_cap, dense_pad=cfg.dense_pad, **cfg.extra,
            )
    else:
        peaks = _align._run_frf_separate_rotation(
            model, data, frf_inputs, n_peaks=cfg.n_peaks, verbose=verbose,
            lmax_cap=cfg.lmax_cap, dense_pad=cfg.dense_pad, **cfg.extra,
        )
    seconds = time.time() - t0

    arf = captured.get("arf")
    sigma = None
    if arf is not None:
        vals = arf.values.to(torch.float64)
        sigma = (vals - vals.mean()) / vals.std().clamp(min=1e-30)
    return FRFResult(peaks=peaks, arf=arf, sigma=sigma, seconds=seconds,
                     inputs=frf_inputs)
