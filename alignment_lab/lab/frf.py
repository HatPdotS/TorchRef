"""Run the FRF and capture the full rotation function, not just the peak list.

Every rank/ghost diagnostic needs the dense adaptive sample list as well as the
peaks, and the engine only returns the peaks. The capture below wraps the
engine's scoring method for the duration of one call; nine scripts each
carried their own copy of this monkeypatch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


def e_convention_name(conv) -> str:
    """Display name for a convention class, a ``partial`` of one, or ``None``."""
    if conv is None:
        return "default"
    inner = getattr(conv, "func", conv)
    name = getattr(inner, "__name__", str(inner))
    kw = getattr(conv, "keywords", None)
    if kw:
        name += "(" + ",".join(f"{k}={v}" for k, v in sorted(kw.items())) + ")"
    return name


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
    grid_sampling_deg: float = 3.0
    #: Expected r.m.s. coordinate error, in Angstrom. ``None`` uses the Oeffner
    #: estimate from the model's length, which is what the pipeline does.
    model_error_A: Optional[float] = None
    #: E-value convention, as the CLASS the engine instantiates once per side.
    #: ``None`` leaves the production default in place; a class (or a
    #: ``functools.partial`` of one) sweeps it. Unlike the deleted ``extra``
    #: knobs this is a real production parameter, so the lab passes it through
    #: rather than patching a constant.
    e_convention: Optional[type] = None
    #: Weighting, the other half of the split. ``None`` leaves the production
    #: default. These are separate arms on purpose: the design changes three
    #: things at once -- the observed-side weight, the calculated-side weight
    #: and whether the per-shell reweight runs -- and a panel that moves all
    #: three cannot say which one did anything.
    obs_weight: Optional[str] = None
    shell_variance_weights: Optional[bool] = None
    snr_cap: Optional[float] = None
    trust_cap: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """Config fields for a result row (``extra`` flattened out)."""
        d = asdict(self)
        d.pop("extra")
        # `asdict` cannot render a class or a partial; name it instead.
        d["e_convention"] = e_convention_name(self.e_convention)
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

    import importlib

    from torchref.experimental.alignment import align as _align
    from torchref.experimental.alignment.frf import api as _api

    # `from ...alignment import rotation_search` gives the FUNCTION, which the
    # package re-exports under the module's own name. Patching constants needs
    # the module object.
    _rs = importlib.import_module(
        "torchref.experimental.alignment.rotation_search")
    from torchref.experimental.alignment.frf.preprocessing import oeffner_vrms

    cfg = cfg or FRFConfig()
    captured: Dict[str, Any] = {}

    def _wrapped(self, *args, **kwargs):
        arf, peaks = _original(self, *args, **kwargs)
        captured["arf"] = arf
        return arf, peaks

    # The engine takes no tuning arguments any more: `lmax_cap`, `dense_pad` and
    # the SO(3) sampling are module constants of `rotation_search`. The lab
    # sweeps them by rebinding those constants for the duration of one call, so
    # the production API stays switch-free while the measurements that chose the
    # values remain reproducible.
    frf_inputs = _align._prepare_frf_inputs(
        model, data,
        d_min=cfg.d_min, d_max=cfg.d_max, n_shells=cfg.n_shells, verbose=verbose,
    )
    model_error_A = cfg.model_error_A
    if model_error_A is None:
        model_error_A = oeffner_vrms(max(1, int(model.xyz().shape[0] / 8)), 1.0)
    if cfg.extra:
        raise ValueError(
            f"FRFConfig.extra is no longer plumbed anywhere: {sorted(cfg.extra)}. "
            f"The engine knobs it reached were deleted; patch the constants in "
            f"torchref.experimental.alignment.rotation_search instead."
        )

    # Omitted rather than passed as None, so an unset convention takes the
    # production default from the signature instead of overriding it with one.
    conv_kw = {} if cfg.e_convention is None else {
        "e_convention": cfg.e_convention}
    # Engine knobs are omitted when unset so the production default applies,
    # rather than being passed as None and overriding it with nothing.
    for _name in ("obs_weight", "shell_variance_weights",
                  "snr_cap", "trust_cap"):
        _v = getattr(cfg, _name)
        if _v is not None:
            conv_kw[_name] = _v

    t0 = time.time()
    with patched(_rs, "LMAX_CAP", int(cfg.lmax_cap)), \
         patched(_rs, "DENSE_CALC_PAD", float(cfg.dense_pad)), \
         patched(_rs, "GRID_SAMPLING_DEG", float(cfg.grid_sampling_deg)):
        if capture_arf:
            _original = _api.FastRotationFunction.score_model
            with patched(_api.FastRotationFunction, "score_model", _wrapped):
                peaks, _lmax, _dmin = _rs.search_peaks(
                    model, data, model_error_A, U_aniso=frf_inputs.U_aniso,
                    n_peaks=cfg.n_peaks, verbose=verbose, **conv_kw,
                )
        else:
            peaks, _lmax, _dmin = _rs.search_peaks(
                model, data, model_error_A, U_aniso=frf_inputs.U_aniso,
                n_peaks=cfg.n_peaks, verbose=verbose, **conv_kw,
            )
    seconds = time.time() - t0

    arf = captured.get("arf")
    sigma = None
    if arf is not None:
        vals = arf.values.to(torch.float64)
        sigma = (vals - vals.mean()) / vals.std().clamp(min=1e-30)
    return FRFResult(peaks=peaks, arf=arf, sigma=sigma, seconds=seconds,
                     inputs=frf_inputs)
