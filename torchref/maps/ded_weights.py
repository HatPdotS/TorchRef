"""Registered per-reflection weights for light-minus-dark difference coefficients.

A weight scheme turns the observed differences and their uncertainties into one weight
per reflection, normalised to mean one so that maps built from different schemes sit on
comparable scales. Three schemes are registered:

``none``
    Every reflection weighted equally.
``inverse_variance``
    ``1 / sigma_diff**2``. Weights by precision alone; the right rule for averaging
    estimates of one quantity, and the default for difference maps.
``sigma_d``
    The Wiener weight ``S / (S + sigma_diff**2)`` with ``S`` the expected true difference
    power from :mod:`torchref.refinement.model_error_estimation.sigma_d`. Weights by the
    signal fraction of each coefficient, so strong reflections whose expected difference
    is large keep their weight. ``S`` is ``mean(dF**2) - mean(sigma**2)`` per shell, so
    it inherits any miscalibration of ``sigma_diff``: where the reported sigmas are too
    large the estimate finds no power and the weight vanishes, which turns the scheme
    into a resolution cut. The count of such shells is reported as ``n_s2_clamped``;
    a large fraction means the sigmas, not the data, are deciding the map.

Plain tensors in and out. The weights live on the device of ``delta_obs``. The sigma_D
estimator is imported inside the scheme that needs it so that :mod:`torchref.maps` does
not import :mod:`torchref.refinement` at module load.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from torchref.base.reciprocal.basis import get_scattering_vectors
from torchref.base.targets.xray_likelihoods import SIGMA_FLOOR_ABS, SIGMA_FLOOR_FRAC

if TYPE_CHECKING:
    from torchref.refinement.model_error_estimation.sigma_d import SigmaDConfig

#: The selectable schemes, in the order they are reported.
SCHEMES = ("none", "inverse_variance", "sigma_d")
#: Scheme applied when none is named. Inverse variance, because ``sigma_d`` depends on
#: calibrated sigmas: on the 15 Sep campaign TorchSX's TD1 sigmas were ~1.5x too large at
#: high resolution, ``sigma_d`` zeroed 60-90 % of the shells there and the map agreement
#: rose in the bulk solvent as much as in the region of interest.
DEFAULT_SCHEME = "inverse_variance"
#: MTZ column carrying each scheme's weight (type ``W``); ``none`` writes no column.
WEIGHT_COLUMNS = {"inverse_variance": "W_IVW", "sigma_d": "W_SD"}


class DedWeightFallbackWarning(UserWarning):
    """A requested weight scheme could not be evaluated and another was applied."""


@dataclass(frozen=True)
class DedWeights:
    """One scheme's weights and how they came about.

    Attributes
    ----------
    scheme
        The scheme requested.
    applied
        The scheme whose weights are in ``weights``; differs from ``scheme`` only after a
        fallback, which ``diagnostics["fallback_reason"]`` then names.
    weights
        Per-reflection weights, shape ``(N,)``, mean one over finite positive entries.
    diagnostics
        Scheme-specific record: for ``sigma_d`` the fitted exponent, shrinkage sd,
        clamp counters and the per-shell table.
    """

    scheme: str
    applied: str
    weights: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def normalise_mean_one(w: torch.Tensor) -> torch.Tensor:
    """Divide by the mean over all entries so the column averages one; unchanged when
    that mean is not positive.

    Non-finite entries become zero, so a coefficient without a usable uncertainty drops
    out of the map rather than poisoning it. Zero weights (shells without difference
    power) stay zero and count in the mean, so the column mean is one whatever fraction
    of the reflections carries weight.
    """
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    if w.numel() == 0 or not bool((w > 0).any()):
        return w
    return w / w.mean()


def reflection_geometry(hkl, cell, spacegroup, device, dtype):
    """``(epsilon, d_star_sq)`` for ``hkl``: the reflection multiplicity from
    ``spacegroup`` (ones when ``None``) and ``1/d**2`` in A^-2 from ``cell``, both on
    ``device`` in ``dtype``."""
    from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl

    hkl_t = torch.as_tensor(hkl, device=device)
    cell_t = cell if torch.is_tensor(cell) else cell.data
    cell_t = torch.as_tensor(cell_t, device=device, dtype=dtype)
    s = get_scattering_vectors(hkl_t, cell_t)
    dss = (s * s).sum(dim=1).to(dtype)
    eps = epsilon_from_hkl(hkl_t, spacegroup).to(device=device, dtype=dtype)
    return eps, dss


def _inverse_variance(sigma_diff: torch.Tensor) -> torch.Tensor:
    """``1 / sigma**2`` with sigma floored at a tenth of its median, so a reported zero
    uncertainty gives a large finite weight rather than an infinite one."""
    finite = torch.isfinite(sigma_diff) & (sigma_diff >= 0)
    positive = finite & (sigma_diff > 0)
    if not bool(positive.any()):
        return torch.zeros_like(sigma_diff)
    floor = (sigma_diff[positive].median() * SIGMA_FLOOR_FRAC).clamp_min(
        SIGMA_FLOOR_ABS
    )
    sig = torch.where(finite, sigma_diff, torch.full_like(sigma_diff, float("inf")))
    return 1.0 / sig.clamp(min=floor) ** 2


def compute_ded_weights(
    scheme: str,
    *,
    delta_obs: torch.Tensor,
    sigma_diff: torch.Tensor,
    hkl: torch.Tensor,
    cell,
    spacegroup,
    f_dark: torch.Tensor | None = None,
    fit_mask: torch.Tensor | None = None,
    sigma_d_config: "SigmaDConfig | None" = None,
) -> DedWeights:
    """Per-reflection weights for one scheme.

    Parameters
    ----------
    scheme : str
        One of :data:`SCHEMES`.
    delta_obs, sigma_diff : torch.Tensor
        Signed observed differences and their propagated uncertainty, shape ``(N,)``, on
        one common amplitude scale.
    hkl : torch.Tensor
        Miller indices, shape ``(N, 3)``.
    cell : Cell or torch.Tensor
        Unit cell, as a :class:`~torchref.symmetry.Cell` or its six parameters in A and
        degrees.
    spacegroup : SpaceGroup or None
        For the reflection multiplicity; ``None`` means ones.
    f_dark : torch.Tensor, optional
        Dark amplitudes, shape ``(N,)``, for the sigma_D amplitude power law.
    fit_mask : torch.Tensor, optional
        Reflections entering the sigma_D fit; default every finite one.
    sigma_d_config : SigmaDConfig, optional
        Exponent and shrinkage settings for ``sigma_d``.

    Returns
    -------
    DedWeights
        Mean-one weights on ``delta_obs.device``. When ``sigma_d`` finds no difference
        power in any shell, the inverse-variance weights are returned with
        ``applied="inverse_variance"`` and a :class:`DedWeightFallbackWarning`.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown DED weight scheme {scheme!r}; choose from {SCHEMES}")
    sigma_diff = sigma_diff.reshape(-1).to(delta_obs.device, delta_obs.dtype)
    if scheme == "none":
        return DedWeights(scheme, scheme, torch.ones_like(sigma_diff))
    if scheme == "inverse_variance":
        return DedWeights(
            scheme, scheme, normalise_mean_one(_inverse_variance(sigma_diff))
        )

    from torchref.refinement.model_error_estimation.sigma_d import (
        SigmaDConfig,
        estimate_sigma_d,
        sigma_d_per_reflection,
    )

    config = sigma_d_config if sigma_d_config is not None else SigmaDConfig()
    d = delta_obs.reshape(-1)
    eps, dss = reflection_geometry(hkl, cell, spacegroup, d.device, d.dtype)
    f = f_dark.reshape(-1).to(d.device, d.dtype) if f_dark is not None else None
    mask = (
        fit_mask.reshape(-1).to(d.device, torch.bool)
        if fit_mask is not None
        else torch.isfinite(d) & torch.isfinite(sigma_diff)
    )
    shells = estimate_sigma_d(
        d, sigma_diff, eps, dss, f, mask, gamma=config.gamma, shrink=config.shrink
    )
    est = sigma_d_per_reflection(shells, dss, eps, f, sigma_diff)
    diagnostics = {
        "gamma": shells.gamma,
        "gamma_fitted": shells.gamma_fitted,
        "tau": shells.tau,
        "curve_a": shells.curve_a,
        "curve_b": shells.curve_b,
        "degenerate": shells.degenerate,
        "all_zero": shells.all_zero,
        **shells.diagnostics,
        "shells": {
            "d_star_sq": shells.bin_dss.detach().cpu().tolist(),
            "counts": shells.counts.detach().cpu().tolist(),
            "B": shells.B.detach().cpu().tolist(),
            "S2": shells.S2.detach().cpu().tolist(),
            "Sigma_N_raw": shells.Sigma_N_raw.detach().cpu().tolist(),
            "Sigma_N": shells.Sigma_N.detach().cpu().tolist(),
        },
        "weight_sigma_d_raw": est.w.detach(),
    }
    if shells.all_zero or shells.degenerate:
        reason = (
            "no difference power above the measurement variance in any shell "
            f"(n_s2_clamped={shells.diagnostics['n_s2_clamped']})"
            if shells.all_zero
            else "fewer than two usable reflections"
        )
        warnings.warn(
            f"sigma_d weights: {reason}; applying inverse-variance weights instead",
            DedWeightFallbackWarning,
            stacklevel=2,
        )
        diagnostics["fallback_reason"] = reason
        return DedWeights(
            scheme,
            "inverse_variance",
            normalise_mean_one(_inverse_variance(sigma_diff)),
            diagnostics,
        )
    return DedWeights(scheme, scheme, normalise_mean_one(est.w), diagnostics)


def all_ded_weights(**kwargs) -> dict[str, DedWeights]:
    """Every registered scheme on the same inputs, keyed by scheme name.

    Takes the keyword arguments of :func:`compute_ded_weights` except ``scheme``. Used for
    side-by-side reporting and for writing every weight column at once.
    """
    return {scheme: compute_ded_weights(scheme, **kwargs) for scheme in SCHEMES}


__all__ = [
    "DEFAULT_SCHEME",
    "SCHEMES",
    "WEIGHT_COLUMNS",
    "DedWeightFallbackWarning",
    "DedWeights",
    "all_ded_weights",
    "compute_ded_weights",
    "normalise_mean_one",
    "reflection_geometry",
]
