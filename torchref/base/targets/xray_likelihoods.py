"""The three X-ray likelihood shapes, and the variance as a separate concern.

Five selectable X-ray targets, but only **three** likelihoods:

===================  =======================  ================================
primitive            distribution             consumed by
===================  =======================  ================================
:func:`nll_math`     Gaussian on ``|F|``      ``nll``, ``nll_beta``
:func:`rice_math`    Rice / folded normal     ``ml``, ``ml_noalpha``
:func:`rice_marginal_math`  Rice (x) Gaussian, marginalised   ``ml_full``
===================  =======================  ================================

What distinguishes ``nll`` from ``nll_beta``, and ``ml`` from ``ml_noalpha``, is not the
likelihood -- it is where the variance comes from and where the mean is centred. The
variance is therefore built by an explicit function and passed in.

**The two variance conventions differ, on purpose.** :func:`rice_math` takes the
**complex** variance ``Sigma`` (the Rice denominator); :func:`nll_math` takes the
**amplitude** variance ``var``. Converting between them is
:func:`amplitude_var_from_complex` -- ``Sigma/2`` acentric, ``Sigma`` centric, from the
large-signal Rice limit. Getting that factor wrong rescales the whole x-ray gradient by 2,
indistinguishable from a change of x-ray weight. ``sigma_obs**2`` needs **no** conversion:
it is already an amplitude variance, a 1-DOF error on a measured amplitude.

Do not pair ``sigma_obs`` with a Rice ``Sigma``: that asserts an isotropic *complex* error
where ``sigma_obs`` carries no phase at all, and no regime makes it correct (it was tried,
and was the worst of every target). Model error -- ``beta`` -- is what belongs in a Rice
``Sigma``. :func:`inflate_with_sigma_obs` approximates the correct treatment;
:func:`rice_marginal_math` is it.
"""

import math

import torch

#: ``0.5 * log(2*pi)``, the Gaussian normaliser's constant term.
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

#: Floor on any variance before it reaches a division or a log.
VAR_FLOOR = 1e-10


# =====================================================================
# Variance builders -- the axis that distinguishes the five targets
# =====================================================================


def amplitude_var_from_sigma_obs(sigma: torch.Tensor) -> torch.Tensor:
    """Amplitude variance from the experimental sigma: ``clamp(sigma)**2``.

    The floor is ``median(sigma) * 1e-1`` -- **data-dependent**, not an absolute
    epsilon, so it shifts results on datasets with near-zero sigmas. The
    beta-derived builders below floor only at :data:`VAR_FLOOR`; the two
    conventions are deliberately not reconciled.
    """
    floor = torch.median(sigma) * 1e-1
    return torch.clamp(sigma, min=floor) ** 2


def amplitude_var_from_complex(
    Sigma: torch.Tensor, centric_flags: torch.Tensor
) -> torch.Tensor:
    """Amplitude variance from a complex variance: ``Sigma/2`` acentric, ``Sigma`` centric.

    The large-signal limit of the Rice -- see the module docstring. This is a *lossy*
    conversion (it is an asymptotic limit, not an identity), which is why ``nll_beta`` is
    documented as a diagnostic rather than a competitor to ``ml``.
    """
    parity = torch.where(centric_flags, 1.0, 0.5).to(Sigma.dtype)
    return torch.clamp(Sigma * parity, min=VAR_FLOOR)


def complex_var_from_beta(
    beta: torch.Tensor, epsilon: torch.Tensor = None
) -> torch.Tensor:
    """Complex variance ``Sigma = epsilon * beta`` from a model-error variance.

    ``epsilon`` is the reflection multiplicity; ``None`` means 1. ``beta`` comes from
    :mod:`torchref.refinement.model_error_estimation.sigma_a` and is the *absolute*
    model-error variance in F**2 units.
    """
    beta = torch.clamp(beta, min=VAR_FLOOR)
    if epsilon is None:
        return beta
    return torch.clamp(epsilon.to(beta.dtype) * beta, min=VAR_FLOOR)


def inflate_with_sigma_obs(
    Sigma: torch.Tensor, sigma_obs: torch.Tensor, centric_flags: torch.Tensor
) -> torch.Tensor:
    """Refmac's ``ll_amp`` variance inflation: ``Sigma + (3 - c) * sigma_obs**2``.

    Two conventions here are easy to get backwards and both are load-bearing:

    * the parity factor is ``(3 - c)`` -- **2** for acentrics, **1** for centrics, i.e. the
      opposite way round from most epsilon-like factors;
    * ``sigma_obs**2`` is **not** scaled by ``epsilon``. Refmac folds epsilon into ``S``
      only ("S: must include epsilon"), so the measurement term is added flat.

    This is the Green (1979) shortcut: it *inflates* the variance instead of marginalising
    over the amplitude-error/phase-error annulus, which is what :func:`rice_marginal_math`
    does by quadrature.

    ``Sigma`` must be the MODEL-error variance alone (``SigmaAEstimate.beta_model`` scaled by
    epsilon), or the measurement variance is counted twice -- and note the removal is not a
    per-reflection inverse of this addition: ``beta_model = beta - S2`` subtracts an
    ``epsilon``-divided, parity-weighted *shell mean* while this adds flat.

    No production target uses this; it is retained so the equivalence with
    ``servalcat/src/amplitude.cpp::ll_amp`` stays reproducible, which two tests check.
    """
    parity = torch.where(
        centric_flags, torch.ones_like(Sigma), torch.full_like(Sigma, 2.0)
    )
    return torch.clamp(
        Sigma + parity * sigma_obs.reshape(-1).to(Sigma.dtype) ** 2, min=VAR_FLOOR
    )


# =====================================================================
# The three likelihoods
# =====================================================================


def _masked_sum(loss: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Sum, with non-finite entries replaced by a large finite penalty -- one NaN
    would poison the whole gradient, whereas ``1e6`` lets the line search reject
    the step. ``mask=None`` means all reflections (compact inputs need none).
    """
    loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))
    if mask is None:
        return loss.sum()
    return (loss * mask).sum()


def nll_per_refl(
    F_obs: torch.Tensor, F_calc: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """Per-reflection Gaussian NLL on the amplitude (NOT masked or summed).

        0.5 * (F_obs - |F_calc|)**2 / var + 0.5 * log(var) + 0.5 * log(2*pi)

    ``var`` is the **amplitude** variance. Build it with
    :func:`amplitude_var_from_sigma_obs` (``nll``) or :func:`amplitude_var_from_complex`
    (``nll_beta``) -- see the module docstring on why those are not interchangeable.
    """
    diff = F_obs - torch.abs(F_calc)
    var = torch.clamp(var, min=VAR_FLOOR)
    return 0.5 * diff**2 / var + 0.5 * torch.log(var) + HALF_LOG_2PI


def nll_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    var: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Masked sum of :func:`nll_per_refl`. The form a target's ``forward`` returns."""
    return _masked_sum(nll_per_refl(F_obs, F_calc, var), mask)


def rice_per_refl(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    Sigma: torch.Tensor,
    centric_flags: torch.Tensor,
) -> torch.Tensor:
    """Per-reflection Read-MLF (NOT masked or summed). See :func:`rice_math`."""
    return _rice_body(F_obs, F_calc, Sigma, centric_flags)


def rice_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    Sigma: torch.Tensor,
    centric_flags: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Read-MLF: Rice for acentrics, folded normal for centrics, at **complex** ``Sigma``.

    Build ``Sigma`` with :func:`complex_var_from_beta`. To centre on ``alpha*|F_calc|``,
    scale ``F_calc`` before calling -- the mean coupling enters the likelihood only as
    ``alpha*Fc`` (the model term carries ``(alpha*Fc)**2`` and ``2 t (alpha*Fc)``), so
    folding it in is exact rather than an approximation.
    """
    return _masked_sum(_rice_body(F_obs, F_calc, Sigma, centric_flags), mask)


def _rice_body(F_obs, F_calc, Sigma, centric_flags):
    """The per-reflection Rice / folded-normal NLL. One implementation, two entry points."""
    if centric_flags is None:
        centric_flags = torch.zeros_like(F_obs, dtype=torch.bool)
    Fc = torch.abs(F_calc)
    Sigma = torch.clamp(Sigma, min=VAR_FLOOR)

    # --- acentric -----------------------------------------------------------
    term1 = -torch.log(2 * F_obs / Sigma + 1e-12)
    term2 = (F_obs**2) / Sigma
    term3 = Fc**2 / Sigma
    arg_bessel = torch.clamp(2 * Fc * F_obs / Sigma, max=1e6)
    # i0e is the exp-SCALED Bessel, so the +arg restores log I0 without overflowing.
    term4 = -(torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel)
    loss_acentric = term1 + term2 + term3 + term4

    # --- centric ------------------------------------------------------------
    term1_c = -0.5 * torch.log(2 / (math.pi * Sigma) + 1e-12)
    term2_c = (F_obs**2) / (2 * Sigma)
    term3_c = Fc**2 / (2 * Sigma)
    term4_c = -(Fc * F_obs) / Sigma
    # log cosh in shifted form: log cosh(z) = |z| + log((1+exp(-2|z|))/2).
    arg_exp = torch.clamp(-2 * Fc * F_obs / Sigma, min=-80.0, max=80.0)
    term5_c = -torch.log((1 + torch.exp(arg_exp)) / 2 + 1e-12)
    loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

    return torch.where(centric_flags, loss_centric, loss_acentric)


def rice_marginal_per_refl(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    Sigma: torch.Tensor,
    sigma_obs: torch.Tensor,
    centric_flags: torch.Tensor,
    idx=None,
    n_quad: int = None,
    n_sigma: float = None,
    li0=None,
) -> torch.Tensor:
    """Per-reflection full-form MLF (NOT masked or summed). See :func:`rice_marginal_math`."""
    from .xray_ml_full import log_i0, ml_full_nll_per_refl

    # `beta=Sigma, epsilon=None` because Sigma is ALREADY epsilon*beta_model: the callee
    # would otherwise multiply epsilon in a second time. `alpha=None` for the same reason
    # the Rice above takes a pre-scaled F_calc -- the caller centres the mean.
    return ml_full_nll_per_refl(
        F_obs,
        sigma_obs,
        F_calc,
        Sigma,
        centric_flags,
        epsilon=None,
        alpha=None,
        n_quad=n_quad,
        n_sigma=n_sigma,
        li0=log_i0 if li0 is None else li0,
        idx=idx,
    )


def rice_marginal_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    Sigma: torch.Tensor,
    sigma_obs: torch.Tensor,
    centric_flags: torch.Tensor,
    mask: torch.Tensor = None,
    idx=None,
    n_quad: int = None,
    n_sigma: float = None,
    li0=None,
) -> torch.Tensor:
    """Full-form MLF: the Rice above, convolved with the measurement Gaussian.

    The unknown error-free amplitude is marginalised out, so the observation error enters as
    an amplitude-only Gaussian while ``Sigma`` keeps the phase component it physically has
    -- rather than the two error kinds being folded into one variance, which is what
    :func:`inflate_with_sigma_obs` does. Acentrics by 32-node Gauss-Legendre quadrature,
    centrics in closed form.

    ``Sigma`` must be the MODEL-error variance (``epsilon * beta_model``): this likelihood
    accounts for ``sigma_obs`` explicitly, so a ``Sigma`` that already contains the
    measurement variance counts it twice.

    Pass ``idx`` from :func:`~torchref.base.targets.xray_ml_full.parity_indices` to avoid a
    device sync per call. ``li0`` overrides the log-Bessel implementation (default: the fast
    piecewise one) -- tests pass ``log_i0_exact`` to separate the quadrature's own error from
    the Bessel approximation's. The quadrature internals live in
    :mod:`torchref.base.targets.xray_ml_full`; this is their single public entry point.
    """
    return _masked_sum(
        rice_marginal_per_refl(
            F_obs,
            F_calc,
            Sigma,
            sigma_obs,
            centric_flags,
            idx=idx,
            n_quad=n_quad,
            n_sigma=n_sigma,
            li0=li0,
        ),
        mask,
    )
