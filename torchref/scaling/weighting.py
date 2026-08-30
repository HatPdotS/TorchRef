"""How much should a reflection count? The other half of the scaling/weighting split.

:mod:`torchref.scaling.wilson` answers *what* we compare -- it removes the
resolution trend and leaves ``<E^2> = 1``. That question turns out to be gauge
for a correlation: any per-shell scaling is absorbed downstream, which is why
twelve normalisation conventions moved a rotation function's truth rank by
nothing. This module answers the question that is not gauge.

The weight has two sources:

* **Measurement error**, per reflection, from ``I/sigma_I``. Two reflections at
  the same resolution can differ enormously in how well they were measured, and
  this is the only part that varies *within* a shell. That matters more than it
  sounds -- a weight constant within a shell is a per-shell weight, and those
  are exactly what a correlation absorbs.
* **Model error**, per resolution, through ``sigma_A``, which is smooth in
  ``|s|`` and has no per-reflection content at all.

Both come out of weighting by inverse variance,
``w = 1/(sigma_meas^2 + sigma_model^2)`` with ``sigma_model^2 = eps -
sigma_A^2``, the standard MLHL budget. The saturation usually added by hand is
already in there: once model error dominates, extra measurement precision buys
nothing, because what is wrong is the model and not the data.

**They do not separate, and that was measured rather than assumed.** Writing the
weight as a product of a ``snr`` term and a ``sigma_A`` term looks natural and
fails: the ``sigma_A`` half is then dominated by its own singularity as
``sigma_A -> 1`` and stops depending on the model error it is named after. The
measurement term is what regularises it, so the two belong in one denominator.
:func:`inverse_variance_weight` is the form that works;
:func:`information_weight` is the pure measurement half, kept because it is the
right answer when no ``sigma_A`` is available and because it is the control that
says what the coupling is worth.
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

    Rises as ``(snr/cap)^2`` while measurement error dominates and flattens to 1
    once it does not. This is not a sigmoid chosen for its shape -- it is
    ``1 / (1 + sigma_meas^2/sigma_model^2)`` rewritten, with ``cap`` the
    signal-to-noise at which the two are equal. The saturation is a consequence
    of the variance budget rather than a clip applied on top of one.

    Structurally this is what Phaser's ``DFAC`` already does: a monotone
    function of signal-to-noise, in ``(0, 1)``, tending to 1 for well-measured
    reflections. The difference is that its saturation point falls out of a Rice
    moment calculation and cannot be moved, and this one is a number that can be
    screened.

    Parameters
    ----------
    snr : torch.Tensor
        ``(N,)`` ``I/sigma_I``. Negative or zero values give weight 0, which is
        the right answer for a measurement consistent with nothing.
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
    """``1 / (1/snr^2 + eps - sigma_A^2)`` -- the two error sources, together.

    The obvious design is to factorise: a per-reflection term in ``snr`` times a
    resolution term in ``sigma_A``. It does not work, and the reason is worth
    keeping.

    Taken alone, ``sigma_A/(eps - sigma_A^2)`` is dominated by its own
    singularity. On a realistic Luzzati falloff it runs 10.1 at the lowest
    resolution shell against 1.0 at the next, and it comes out *identical* for
    a 0.5 A and a 1.0 A coordinate error -- because as ``sigma_A -> 1`` the
    shape is set entirely by ``1/(1 - sigma_A^2)`` and the model error it was
    supposed to encode drops out. A weight carrying no information about the
    thing it is named after is not a weight worth having.

    What regularises it is the term the factorisation threw away. Those
    low-resolution reflections are strong but not infinitely well measured, so
    ``1/snr^2`` is what stops the variance reaching zero. The two sources have
    to sit in one denominator; they do not separate.

    ``1/snr^2`` is the measurement variance expressed in the same units as
    ``eps - sigma_A^2``, i.e. relative to a normalised ``<E^2> = 1``. The cap
    is a backstop for the pathological case where both terms vanish, not the
    working mechanism -- if it binds on real data, ``sigma_A`` is wrong.

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
    """Model reliability measured, rather than assumed, from two Wilson curves.

    ``sigma_A`` in a rotation search is normally a *prior*: a Luzzati falloff
    from a coordinate error guessed off the residue count, patched at low
    resolution by Babinet's two universal constants. It never sees a residual.

    It does not have to. Total scattering per shell is **rotation-invariant**,
    so the resolution-dependent disagreement between model and data is
    measurable before the molecule is placed, even though the per-reflection
    disagreement is not. Normalise both sides to ``<E^2> = 1`` and their fitted
    curves' ratio is exactly that disagreement:

        R(s) = Sigma_obs(s) / Sigma_calc(s)

    ``R < 1`` means the model predicts more scattering than is there, which at
    low resolution is the bulk solvent it does not have; ``R > 1`` means it
    predicts less. Either way the shared fraction is bounded by
    ``min(R, 1/R)``, and ``sigma_A`` is its square root because ``sigma_A^2`` is
    the fraction of intensity the model accounts for.

    **This is safe to estimate from the data being scored**, which normally it
    would not be: the quantity is identical for every candidate orientation, so
    it shifts all scores together and cannot bias the ranking toward any of
    them.

    What it conflates -- solvent, an overall B mismatch, missing atoms, genuine
    coordinate error -- it conflates deliberately. For deciding how far to trust
    a resolution range the cause does not matter, only the size. What it cannot
    see is *completeness*: forcing both sides to unit mean absorbs a uniform
    factor, so a model that is half the asymmetric unit looks like a model that
    is all of it, and only the tilt survives.

    Parameters
    ----------
    sigma_obs, sigma_calc : torch.Tensor
        ``(N,)`` fitted Wilson curves evaluated at the same ``|s|``. They must
        come from fits sharing an abscissa, or each is frozen flat outside its
        own range and the ratio is meaningless there.
    floor : float, optional
        Lower bound on the returned ``sigma_A``.
    """
    r = (sigma_obs / sigma_calc.clamp(min=1e-30)).clamp(min=1e-30)
    shared = torch.minimum(r, 1.0 / r).clamp(min=0.0, max=1.0)
    return shared.sqrt().clamp(min=float(floor), max=1.0 - 1e-6)
