"""Registered per-reflection weights for light-minus-dark difference coefficients.

A weight scheme turns the observed differences and their uncertainties into one weight
per reflection, normalised to mean one so that maps built from different schemes sit on
comparable scales. Three schemes are registered:

``none``
    Every reflection weighted equally.
``inverse_variance``
    ``1 / sigma_diff**2``. Weights by precision alone; the right rule for averaging
    estimates of one quantity. On French-Wilson amplitudes it *up*-weights the weak
    high-resolution reflections, whose posterior sigma the prior keeps small.
``q``
    The q-weight: a Wiener weight ``(snr + b) / (snr + 1 + b)`` with
    ``snr = S / (k sigma_diff)**2``, the default. ``S`` and the sigma scale ``k`` come
    from :func:`~torchref.refinement.model_error_estimation.difference_power.
    fit_difference_power`, a per-reflection maximum-likelihood fit of ``log S`` as a
    Chebyshev series in resolution plus ``gamma log F_dark``: no resolution shells. The
    floor ``b`` keeps a reflection without signal at ``b / (1 + b)`` of the full weight
    (one third at the default), so a noisy resolution range is down-weighted, never
    removed. ``k`` absorbs a uniform miscalibration of the sigmas; French-Wilson sigmas
    of two datasets overstate the error of their difference, so ``k < 1`` is normal.

Plain tensors in and out. The weights live on the device of ``delta_obs``. The
difference-power fit is imported inside the scheme that needs it so that
:mod:`torchref.maps` does not import :mod:`torchref.refinement` at module load.
"""

import warnings
from dataclasses import dataclass, field

import torch

from torchref.base.reciprocal.basis import get_scattering_vectors
from torchref.base.targets.xray_likelihoods import SIGMA_FLOOR_ABS, SIGMA_FLOOR_FRAC

#: The selectable schemes, in the order they are reported.
SCHEMES = ("none", "inverse_variance", "q")
#: Scheme applied when none is named. On simulated dark/light half datasets of 1DAW
#: (ligand moved at 0.3 and 0.6 occupancy, calibrated and 1.5x-inflated sigmas) ``q``
#: recovered the most contour-level information about the true difference density;
#: ``inverse_variance`` recovered less than no weighting.
DEFAULT_SCHEME = "q"
#: MTZ column carrying each scheme's weight (type ``W``); ``none`` writes no column.
WEIGHT_COLUMNS = {"inverse_variance": "W_InVa", "q": "W_Q"}


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
        Scheme-specific record: for ``q`` the fitted exponent, sigma scale, centric
        factor, Chebyshev coefficients and their standard errors, and the weight range.
    """

    scheme: str
    applied: str
    weights: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def normalise_mean_one(w: torch.Tensor) -> torch.Tensor:
    """Divide by the mean over all entries so the column averages one; unchanged when
    that mean is not positive.

    Non-finite entries become zero, so a coefficient without a usable uncertainty drops
    out of the map rather than poisoning it. Zero weights stay zero and count in the
    mean, so the column mean is one whatever fraction of the reflections carries weight.
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
    gamma: float | None = None,
    snr_floor: float | None = None,
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
        For the reflection multiplicity and centric flags; ``None`` means P1.
    f_dark : torch.Tensor, optional
        Dark amplitudes, shape ``(N,)``, for the ``q`` power law in ``F_dark``.
    fit_mask : torch.Tensor, optional
        Reflections entering the ``q`` fit; default every finite one.
    gamma : float, optional
        Fix the ``F_dark`` exponent of the ``q`` fit instead of fitting it.
    snr_floor : float, optional
        Signal-to-noise floor of the ``q`` weight; default
        :data:`~torchref.refinement.model_error_estimation.difference_power.
        DEFAULT_SNR_FLOOR`.

    Returns
    -------
    DedWeights
        Mean-one weights on ``delta_obs.device``. When the ``q`` fit has too few
        usable reflections, the inverse-variance weights are returned with
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

    from torchref.refinement.model_error_estimation.difference_power import (
        DEFAULT_SNR_FLOOR,
        bounded_wiener_weight,
        fit_difference_power,
    )

    floor = DEFAULT_SNR_FLOOR if snr_floor is None else float(snr_floor)
    d = delta_obs.reshape(-1)
    eps, dss = reflection_geometry(hkl, cell, spacegroup, d.device, d.dtype)
    f = f_dark.reshape(-1).to(d.device, d.dtype) if f_dark is not None else None
    centric = (
        spacegroup.is_centric(torch.as_tensor(hkl, device=d.device)).to(d.device)
        if spacegroup is not None
        else None
    )
    try:
        fit = fit_difference_power(
            d,
            sigma_diff,
            dss,
            epsilon=eps,
            f_dark=f,
            centric=centric,
            fit_mask=fit_mask,
            gamma=gamma,
        )
    except ValueError as err:
        reason = str(err)
        warnings.warn(
            f"q weights: {reason}; applying inverse-variance weights instead",
            DedWeightFallbackWarning,
            stacklevel=2,
        )
        return DedWeights(
            scheme,
            "inverse_variance",
            normalise_mean_one(_inverse_variance(sigma_diff)),
            {"fallback_reason": reason},
        )
    snr = fit.snr(sigma_diff, d_star_sq=dss, epsilon=eps, f_dark=f, centric=centric)
    w = bounded_wiener_weight(snr, floor)
    ok = torch.isfinite(w)
    diagnostics = {
        "gamma": fit.gamma,
        "gamma_fitted": gamma is None and f is not None,
        "sigma_scale": fit.sigma_scale,
        "centric_factor": fit.centric_factor,
        "order": len(fit.coeffs) - 1,
        "coeffs": fit.coeffs.detach().cpu().tolist(),
        "stderr": fit.stderr.detach().cpu().tolist(),
        "stol_range": list(fit.stol_range),
        "converged": fit.converged,
        "n_fit": fit.n_fit,
        "snr_floor": floor,
        "weight_min": float(w[ok].min()) if bool(ok.any()) else float("nan"),
        "weight_max": float(w[ok].max()) if bool(ok.any()) else float("nan"),
    }
    return DedWeights(scheme, scheme, normalise_mean_one(w), diagnostics)


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
