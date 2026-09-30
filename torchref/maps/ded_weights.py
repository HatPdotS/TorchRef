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
    The q-weight: a Wiener weight ``(snr + b) / (snr + 1 + b)`` on the per-reflection
    signal-to-noise ratio of :func:`difference_snr`, the default. The floor ``b`` keeps
    a reflection without signal at ``b / (1 + b)`` of the full weight (one third at the
    default), so a noisy resolution range is down-weighted, never removed.

:func:`difference_snr` is the one estimate of how much of each observed difference is
signal; the extrapolated amplitudes of ``torchref.difference-map`` shrink by the same
ratio. It prefers intensities: French-Wilson amplitude sigmas describe the posterior of
one amplitude, not the noise of a difference between two, and overstate it increasingly
toward high resolution, where intensity sigmas are the measurement noise itself.

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


@dataclass(frozen=True)
class DifferenceSNR:
    """Per-reflection signal-to-noise ratio of observed differences.

    Attributes
    ----------
    snr : torch.Tensor
        ``S / noise**2``, shape ``(N,)``, the same in amplitude and intensity terms.
    noise : torch.Tensor
        Calibrated noise ``k * sigma`` of the amplitude difference, shape ``(N,)``, in
        the amplitude units of ``delta_obs``.
    fit : DifferencePowerFit
        The fit ``snr`` came from.
    source : str
        ``"intensity"`` or ``"amplitude"``: which observations were fitted.
    """

    snr: torch.Tensor
    noise: torch.Tensor
    fit: object
    source: str


def difference_snr(
    *,
    delta_obs: torch.Tensor,
    sigma_diff: torch.Tensor,
    hkl: torch.Tensor,
    cell,
    spacegroup,
    f_dark: torch.Tensor | None = None,
    delta_intensity: torch.Tensor | None = None,
    sigma_delta_intensity: torch.Tensor | None = None,
    fit_mask: torch.Tensor | None = None,
    gamma: float | None = None,
) -> DifferenceSNR:
    """Fit the expected difference power and return each reflection's SNR.

    Given intensity differences and ``f_dark``, the fit runs on
    ``delta_intensity / (2 F_dark)`` with sigma ``sigma_delta_intensity / (2 F_dark)``:
    the ratio of an intensity difference to its measurement noise, carried on the
    amplitude scale so the likelihood is as well conditioned as an amplitude fit. The
    ``F_dark`` exponent then defaults to 0, because ``S`` and ``F_dark`` enter the
    divided difference together and the fitted exponent is not identified. Otherwise
    the amplitude differences and their sigmas are fitted.

    Parameters
    ----------
    delta_obs, sigma_diff : torch.Tensor
        Signed amplitude differences and their propagated sigma, shape ``(N,)``.
    hkl : torch.Tensor
        Miller indices, shape ``(N, 3)``.
    cell : Cell or torch.Tensor
        Unit cell, or its six parameters in A and degrees.
    spacegroup : SpaceGroup or None
        For the multiplicity and centric flags; ``None`` means P1.
    f_dark : torch.Tensor, optional
        Dark amplitudes, shape ``(N,)``; required for the intensity path.
    delta_intensity, sigma_delta_intensity : torch.Tensor, optional
        Intensity differences ``I_light - I_dark`` and their propagated sigma, shape
        ``(N,)``, on the scale whose square root is the amplitude scale of
        ``delta_obs``.
    fit_mask : torch.Tensor, optional
        Reflections entering the fit; default every finite one.
    gamma : float, optional
        Fix the ``F_dark`` exponent.

    Returns
    -------
    DifferenceSNR

    Raises
    ------
    ValueError
        If too few reflections are usable for the fit.
    """
    from torchref.refinement.model_error_estimation.difference_power import (
        fit_difference_power,
    )

    d = delta_obs.reshape(-1)
    sig = sigma_diff.reshape(-1).to(d.device, d.dtype)
    eps, dss = reflection_geometry(hkl, cell, spacegroup, d.device, d.dtype)
    f = f_dark.reshape(-1).to(d.device, d.dtype) if f_dark is not None else None
    centric = (
        spacegroup.is_centric(torch.as_tensor(hkl, device=d.device)).to(d.device)
        if spacegroup is not None
        else None
    )
    use_intensity = (
        delta_intensity is not None
        and sigma_delta_intensity is not None
        and f is not None
    )
    if use_intensity:
        # A floor on the divisor: a near-zero dark amplitude would send both terms to
        # infinity. Their ratio, the SNR, does not depend on it.
        ok_f = torch.isfinite(f) & (f > 0)
        f_floor = 0.1 * float(f[ok_f].median()) if bool(ok_f.any()) else 1.0
        two_f = 2.0 * f.clamp(min=f_floor)
        values = delta_intensity.reshape(-1).to(d.device, d.dtype) / two_f
        sigma = sigma_delta_intensity.reshape(-1).to(d.device, d.dtype) / two_f
        g = 0.0 if gamma is None else gamma
    else:
        values, sigma, g = d, sig, gamma
    fit = fit_difference_power(
        values,
        sigma,
        dss,
        epsilon=eps,
        f_dark=f,
        centric=centric,
        fit_mask=fit_mask,
        gamma=g,
    )
    snr = fit.snr(sigma, d_star_sq=dss, epsilon=eps, f_dark=f, centric=centric)
    return DifferenceSNR(
        snr=snr,
        noise=fit.sigma_scale * sigma,
        fit=fit,
        source="intensity" if use_intensity else "amplitude",
    )


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
    delta_intensity: torch.Tensor | None = None,
    sigma_delta_intensity: torch.Tensor | None = None,
    fit_mask: torch.Tensor | None = None,
    gamma: float | None = None,
    snr_floor: float | None = None,
    snr_estimate: DifferenceSNR | ValueError | None = None,
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
    delta_intensity, sigma_delta_intensity : torch.Tensor, optional
        Intensity differences and their sigma; given with ``f_dark``, the ``q`` SNR is
        fitted on them (see :func:`difference_snr`).
    fit_mask : torch.Tensor, optional
        Reflections entering the ``q`` fit; default every finite one.
    gamma : float, optional
        Fix the ``F_dark`` exponent of the ``q`` fit instead of fitting it.
    snr_floor : float, optional
        Signal-to-noise floor of the ``q`` weight; default
        :data:`~torchref.refinement.model_error_estimation.difference_power.
        DEFAULT_SNR_FLOOR`.
    snr_estimate : DifferenceSNR or ValueError, optional
        A :func:`difference_snr` result on the same inputs, reused instead of fitting
        again, so a caller that also needs the SNR fits once; or the ``ValueError`` that
        call raised, taken as the failure without retrying.

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
    )

    floor = DEFAULT_SNR_FLOOR if snr_floor is None else float(snr_floor)
    try:
        if isinstance(snr_estimate, ValueError):
            raise snr_estimate
        est = snr_estimate or difference_snr(
            delta_obs=delta_obs,
            sigma_diff=sigma_diff,
            hkl=hkl,
            cell=cell,
            spacegroup=spacegroup,
            f_dark=f_dark,
            delta_intensity=delta_intensity,
            sigma_delta_intensity=sigma_delta_intensity,
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
    fit = est.fit
    w = bounded_wiener_weight(est.snr, floor)
    ok = torch.isfinite(w)
    diagnostics = {
        "source": est.source,
        "gamma": fit.gamma,
        "gamma_fitted": (
            gamma is None and est.source == "amplitude" and f_dark is not None
        ),
        "sigma_scale": fit.sigma_scale,
        "sigma_scale_at_bound": fit.sigma_scale_at_bound,
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
    "DifferenceSNR",
    "all_ded_weights",
    "compute_ded_weights",
    "difference_snr",
    "normalise_mean_one",
    "reflection_geometry",
]
