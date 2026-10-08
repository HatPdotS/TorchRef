"""Per-reflection weights: how much each reflection should count.

:mod:`torchref.scaling.wilson` decides *what* is compared (normalised amplitudes with
``<E^2> = 1``); this module decides how much each comparison counts. Two error sources
set the weight: measurement error, per reflection, from ``I/sigma_I``, and model error,
smooth in resolution, through ``sigma_A``. They enter one inverse-variance denominator,
``w = 1/(sigma_meas^2 + eps - sigma_A^2)`` (:func:`inverse_variance_weight`), because
they do not factorise: alone, the ``sigma_A`` term is dominated by its singularity at
``sigma_A -> 1`` and the measurement term is what bounds it. :func:`information_weight`
is the measurement half alone, for when no ``sigma_A`` is available.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "information_weight",
    "inverse_variance_weight",
    "snr_from_amplitude",
    "normalise_weight",
    "empirical_sigma_a",
]

#: ``I/sigma_I`` at which measurement error stops being the limiting term.
#: Beyond it a reflection is no better determined for the purpose of comparing
#: against a model, because the model is what limits. Free parameter in
#: practice: the crossover really sits wherever ``sigma_A`` puts it, and
#: ``sigma_A`` before placement is assumed rather than fitted.
DEFAULT_SNR_CAP = 5.0

#: Backstop on the inverse-variance weight, for the case where measurement and
#: model variance both vanish. Not the working mechanism: the measurement term
#: is what bounds the weight at low resolution, where ``sigma_A -> 1`` would
#: otherwise send it to infinity on the strongest reflections. If this binds on
#: real data, ``sigma_A`` is wrong rather than the cap being too low.
DEFAULT_TRUST_CAP = 100.0


def snr_from_amplitude(
    F: torch.Tensor, sig_F: torch.Tensor, floor: float = 1e-12,
) -> torch.Tensor:
    """``I/sigma_I`` from an amplitude and its error.

    With ``I = F^2`` the error propagates as ``sigma_I = 2 F sigma_F``, so the
    intensity signal-to-noise is ``F / (2 sigma_F)`` -- half the amplitude's.
    The factor is worth being explicit about: it only rescales the cap, but
    quoting a cap against the wrong one silently doubles it.
    """
    return (F.abs() / (2.0 * sig_F.abs().clamp(min=floor))).clamp(min=0.0)


def information_weight(
    snr: torch.Tensor, *, cap: float = DEFAULT_SNR_CAP,
) -> torch.Tensor:
    """Saturating measurement-information weight, ``snr^2 / (snr^2 + cap^2)``.

    Equal to ``1 / (1 + sigma_meas^2/sigma_model^2)`` with ``cap`` the signal-to-noise at
    which the two variances are equal, so it rises as ``(snr/cap)^2`` and saturates at 1.

    Parameters
    ----------
    snr : torch.Tensor
        ``(N,)`` ``I/sigma_I``. Values ``<= 0`` give weight 0.
    cap : float, optional
        Signal-to-noise at which the weight reaches 1/2.
    """
    s2 = snr.clamp(min=0.0) ** 2
    return s2 / (s2 + float(cap) ** 2)


def inverse_variance_weight(
    snr: torch.Tensor,
    sigma_a: torch.Tensor,
    *,
    eps: Optional[torch.Tensor] = None,
    cap: float = DEFAULT_TRUST_CAP,
) -> torch.Tensor:
    """``1 / (1/snr^2 + eps - sigma_A^2)``: measurement and model error in one variance.

    ``1/snr^2`` is the measurement variance in units of a normalised ``<E^2> = 1``, the
    units of ``eps - sigma_A^2``. ``cap`` is a backstop for both variances vanishing; if
    it binds on real data, ``sigma_A`` is wrong.

    Parameters
    ----------
    snr : torch.Tensor
        ``(N,)`` ``I/sigma_I``. Zero gives zero weight.
    sigma_a : torch.Tensor
        ``(N,)`` model reliability in ``[0, 1)``, evaluated at each reflection.
    eps : torch.Tensor, optional
        ``(N,)`` multiplicity; ``None`` means 1.
    cap : float, optional
        Ceiling on the weight before normalisation.

    Returns
    -------
    torch.Tensor
        ``(N,)`` weights in ``[0, cap]``.
    """
    sa = sigma_a.clamp(min=0.0, max=1.0 - 1e-6)
    e = torch.ones_like(sa) if eps is None else eps.to(sa.dtype).clamp(min=1.0)
    v_meas = 1.0 / (snr.clamp(min=1e-8) ** 2)
    v_model = (e - sa * sa).clamp(min=0.0)
    w = 1.0 / (v_meas + v_model).clamp(min=1e-12)
    return w.clamp(max=float(cap))


def normalise_weight(w: torch.Tensor) -> torch.Tensor:
    """Scale a weight to mean 1.

    Cosmetic for a correlation, where an overall factor cancels, and not
    cosmetic for anything that compares scores across runs or reads a sigma
    level off them. Doing it here means the cap is a number about *relative*
    weighting rather than one entangled with whatever scale the inputs had.
    """
    return w / w.mean().clamp(min=1e-30)


def empirical_sigma_a(
    sigma_obs: torch.Tensor,
    sigma_calc: torch.Tensor,
    *,
    floor: float = 1e-3,
) -> torch.Tensor:
    """Model reliability ``sigma_A`` from the data and model Wilson curves.

    With ``R = Sigma_obs / Sigma_calc``, each curve first divided by its geometric mean
    over the points supplied, ``sigma_A = sqrt(min(R, 1/R))``, clamped to
    ``[floor, 1)``. Total scattering per shell is rotation-invariant, so this needs no
    placed model and shifts every orientation's score alike. The unit-mean normalisation
    makes a uniform completeness deficit invisible: only the resolution tilt survives.

    Parameters
    ----------
    sigma_obs, sigma_calc : torch.Tensor
        ``(N,)`` fitted Wilson curves evaluated at the same ``|s|``. They must come from
        fits sharing an abscissa, or each is frozen flat outside its own range and the
        ratio is meaningless there.
    floor : float, optional
        Lower bound on the returned ``sigma_A``.

    Returns
    -------
    torch.Tensor
        ``(N,)`` ``sigma_A`` in ``[floor, 1)``.
    """
    log_r = (sigma_obs.clamp(min=1e-30).log()
             - sigma_calc.clamp(min=1e-30).log())
    log_r = log_r - log_r.mean()           # unit geometric mean: scale-free
    shared = torch.exp(-log_r.abs())       # min(R, 1/R)
    return shared.sqrt().clamp(min=float(floor), max=1.0 - 1e-6)
