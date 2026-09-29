"""Expected difference power by maximum likelihood over reflections, without shells.

A light-minus-dark amplitude difference ``dF_obs = dF_true + noise`` is modelled as

.. math::

    dF_h \\sim N(0, V_h), \\qquad V_h = \\epsilon_h c_h S_h + k^2 \\sigma_h^2,

    \\log S_h = \\sum_{j=0}^{n} a_j T_j(x_h) + \\gamma \\log F_{dark,h},

with :math:`T_j` the Chebyshev polynomials of :func:`torchref.scaling.basis.
chebyshev_design` in :math:`x_h = \\sin\\theta/\\lambda` over the fitted range (the
abscissa every smooth resolution curve in the package uses), :math:`k` a single scale on
the reported sigmas and :math:`c_h` a free factor on centric reflections. Every
parameter is fitted jointly by Newton's method on the per-reflection negative
log-likelihood, so there are no resolution shells, no clamped moments and no
interpolation: ``S`` is positive and smooth by construction.

The ``gamma * log F_dark`` term needs no per-resolution normalisation of ``F_dark``:
``log <F_dark>(d*^2)`` is itself a smooth function of resolution, so the polynomial
absorbs it. The sigma scale ``k`` is identifiable because the reported sigmas vary
many-fold between reflections at one resolution while ``S`` does not; it is what keeps
inflated sigmas from reading as an absence of signal.

Given a model difference ``dF_calc``, the mean becomes :math:`\\alpha(x_h) dF_{calc,h}`
with :math:`\\alpha` a low-order Chebyshev series in the same abscissa, and ``S`` is the
power the model does not explain -- the coupling and unexplained power a difference
likelihood needs, fitted in one pass instead of from per-shell cross moments.

:func:`bounded_wiener_weight` turns a fit into a weight that down-weights noisy
reflections but never removes one, the resolution-continuous counterpart of the
q-weight's floor.

Plain tensors in and out; every result lives on the device of ``delta_obs``.
"""

import math
from dataclasses import dataclass

import torch

from torchref.config import get_float_dtype
from torchref.scaling.basis import chebyshev_design

#: Chebyshev order of ``log S`` in ``sin(theta)/lambda``. Four covers a Wilson-like
#: fall-off (quadratic in this abscissa) plus low-resolution curvature; the fit
#: carries the standard errors to judge whether a higher order is supported.
DEFAULT_ORDER = 4
#: Signal-to-noise floor of :func:`bounded_wiener_weight`. One half reproduces the
#: q-weight's noise-only limit (``S`` floored at half the raw difference power gives
#: ``w = 1/3``), so a reflection without signal keeps a third of the full weight.
DEFAULT_SNR_FLOOR = 0.5
#: Chebyshev order of the model coupling ``alpha``. Quadratic follows the fall of the
#: coupling with resolution, which is smooth and far less structured than ``S``.
DEFAULT_ALPHA_ORDER = 2
#: Degrees of freedom of the Student-t likelihood when ``robust=True``.
DEFAULT_NU = 4.0
#: Newton iterations; the problem has at most eight parameters and converges in ~10.
MAX_ITER = 60
_GAMMA_BOUNDS = (-1.0, 3.0)
#: Bounds on the sigma scale ``k``. No merge misreports its sigmas tenfold; outside
#: these the data hold no noise to calibrate against (identical datasets drive ``k``
#: to zero), and a zero noise would give an infinite SNR and zero sigmas downstream.
SIGMA_SCALE_BOUNDS = (0.1, 10.0)
_LOG_K_BOUNDS = tuple(math.log(b) for b in SIGMA_SCALE_BOUNDS)
# Bounds on the log centric factor, so a fit without signal cannot underflow it to zero.
_LOG_CENTRIC_BOUNDS = (-7.0, 7.0)


@dataclass(frozen=True)
class DifferencePowerFit:
    """A fitted difference-power model; evaluate it with :meth:`signal_power`.

    Attributes
    ----------
    coeffs : torch.Tensor
        Chebyshev coefficients of ``log S`` in the standardised amplitude units, shape
        ``(order + 1,)``.
    gamma : float
        Exponent on ``F_dark``; ``0`` when no dark amplitude was used.
    sigma_scale : float
        The factor ``k`` on the reported sigmas, within :data:`SIGMA_SCALE_BOUNDS`;
        ``1`` when not fitted.
    sigma_scale_at_bound : bool
        Whether ``k`` stopped at a bound: the reported sigmas and the scatter of the
        differences disagree beyond any plausible miscalibration.
    centric_factor : float
        Power of a centric reflection relative to an acentric one at equal resolution;
        ``1`` when no centric flags were given.
    stol_range : tuple of float
        The ``sin(theta)/lambda`` range (A^-1) the polynomial is defined on. Outside it
        the curve is held at its endpoint value, because a Chebyshev series diverges
        beyond ``[-1, 1]``.
    amp_scale, log_f_ref : float
        The amplitude unit the fit was done in and the mean ``log F_dark`` it was
        centred on.
    alpha_coeffs : torch.Tensor
        Chebyshev coefficients of the model coupling ``alpha``; empty when no model
        difference was fitted. Evaluate with :meth:`alpha_at`.
    stderr : torch.Tensor
        Standard errors of ``(coeffs, gamma, log k, log centric_factor, alpha_coeffs)``
        from the inverse Hessian; NaN where a parameter was fixed.
    nll : float
        Mean negative log-likelihood per reflection at the optimum.
    converged : bool
        Whether the Newton step fell below tolerance.
    n_fit : int
        Reflections in the fit.
    """

    coeffs: torch.Tensor
    gamma: float
    sigma_scale: float
    sigma_scale_at_bound: bool
    centric_factor: float
    stol_range: tuple
    alpha_coeffs: torch.Tensor
    amp_scale: float
    log_f_ref: float
    stderr: torch.Tensor
    nll: float
    converged: bool
    n_fit: int

    def signal_power(
        self,
        d_star_sq: torch.Tensor,
        epsilon: torch.Tensor | None = None,
        f_dark: torch.Tensor | None = None,
        centric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expected true difference power ``epsilon * c * S`` per reflection.

        Parameters
        ----------
        d_star_sq : torch.Tensor
            ``1/d**2`` in A^-2, shape ``(N,)``.
        epsilon : torch.Tensor, optional
            Reflection multiplicity, shape ``(N,)``; ones when omitted.
        f_dark : torch.Tensor, optional
            Dark amplitudes, shape ``(N,)``, in the units of the fitted differences.
            Required when the fit used them (``gamma != 0``).
        centric : torch.Tensor, optional
            Boolean centric flags, shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Power in squared units of the fitted differences, shape ``(N,)``.
        """
        dss = d_star_sq.to(self.coeffs)
        log_s = _design(dss, len(self.coeffs), self.stol_range) @ self.coeffs
        if self.gamma != 0.0:
            if f_dark is None:
                raise ValueError("this fit uses F_dark; pass f_dark")
            log_s = log_s + self.gamma * (_log_amp(f_dark.to(dss), self.amp_scale)
                                          - self.log_f_ref)
        if centric is not None and self.centric_factor != 1.0:
            log_s = log_s + math.log(self.centric_factor) * centric.to(dss)
        power = torch.exp(log_s) * self.amp_scale**2
        if epsilon is not None:
            power = power * epsilon.to(dss)
        return power

    def alpha_at(self, d_star_sq: torch.Tensor) -> torch.Tensor:
        """Model coupling ``alpha`` at each ``1/d**2`` (A^-2); ones without a model."""
        dss = d_star_sq.to(self.coeffs)
        if len(self.alpha_coeffs) == 0:
            return torch.ones_like(dss)
        n = len(self.alpha_coeffs)
        return _design(dss, n, self.stol_range) @ self.alpha_coeffs

    def snr(self, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """``signal_power / (k * sigma)**2`` per reflection; keywords as
        :meth:`signal_power`."""
        s = self.signal_power(**kwargs)
        noise = (self.sigma_scale * sigma.to(s)) ** 2
        return s / noise.clamp(min=torch.finfo(s.dtype).tiny)


def _design(dss: torch.Tensor, n_coeff: int, stol_range: tuple) -> torch.Tensor:
    """Chebyshev design in ``sin(theta)/lambda = sqrt(d*^2) / 2``."""
    lo, hi = stol_range
    return chebyshev_design(dss.clamp(min=0.0).sqrt() / 2.0, n_coeff, lo=lo, hi=hi)


def _log_amp(f: torch.Tensor, amp_scale: float) -> torch.Tensor:
    # A zero dark amplitude would send log S to -inf and delete the reflection's power;
    # a floor at 1 % of the amplitude unit keeps it finite without moving real values.
    return torch.log((f / amp_scale).clamp(min=1e-2))


# The Newton steps take the Hessian by autograd, so the fit enables gradients itself:
# its callers (map writers) run under ``torch.no_grad()``.
@torch.enable_grad()
def fit_difference_power(
    delta_obs: torch.Tensor,
    sigma_diff: torch.Tensor,
    d_star_sq: torch.Tensor,
    *,
    epsilon: torch.Tensor | None = None,
    f_dark: torch.Tensor | None = None,
    centric: torch.Tensor | None = None,
    fit_mask: torch.Tensor | None = None,
    delta_calc: torch.Tensor | None = None,
    order: int = DEFAULT_ORDER,
    alpha_order: int = DEFAULT_ALPHA_ORDER,
    gamma: float | None = None,
    fit_sigma_scale: bool = True,
    robust: bool = False,
    nu: float = DEFAULT_NU,
) -> DifferencePowerFit:
    """Fit the expected difference power by per-reflection maximum likelihood.

    Parameters
    ----------
    delta_obs, sigma_diff : torch.Tensor
        Signed observed differences and their reported uncertainty, shape ``(N,)``, on
        one amplitude scale.
    d_star_sq : torch.Tensor
        ``1/d**2`` in A^-2, shape ``(N,)``.
    epsilon : torch.Tensor, optional
        Reflection multiplicity; ones when omitted.
    f_dark : torch.Tensor, optional
        Dark amplitudes for the ``F_dark**gamma`` term; without them ``gamma`` is 0.
    centric : torch.Tensor, optional
        Boolean centric flags; given, a centric power factor is fitted.
    fit_mask : torch.Tensor, optional
        Reflections entering the fit; default every finite one with positive sigma.
    delta_calc : torch.Tensor, optional
        Model difference, shape ``(N,)``, on the scale of ``delta_obs``. Given, the
        mean is ``alpha * delta_calc`` and ``S`` is the unexplained power.
    order : int
        Chebyshev order of ``log S`` in ``sin(theta)/lambda``.
    alpha_order : int
        Chebyshev order of ``alpha``; used only with ``delta_calc``.
    gamma : float, optional
        Fix the ``F_dark`` exponent instead of fitting it.
    fit_sigma_scale : bool
        Fit the scale ``k`` on the reported sigmas. Off, the sigmas are taken as
        calibrated.
    robust : bool
        Use a Student-t likelihood with ``nu`` degrees of freedom, so a large
        difference loses influence on the fit smoothly instead of dominating it.
    nu : float
        Student-t degrees of freedom.

    Returns
    -------
    DifferencePowerFit
        The fitted model, detached from the inputs. Standard errors are NaN for fixed
        parameters. Runs with gradients enabled internally, so it works under
        ``torch.no_grad()``.

    Raises
    ------
    ValueError
        If fewer than ``order + 4`` reflections are usable.
    """
    dtype = torch.promote_types(get_float_dtype(), delta_obs.dtype)
    dev = delta_obs.device
    d = delta_obs.detach().reshape(-1).to(dev, dtype)
    sig = sigma_diff.detach().reshape(-1).to(dev, dtype)
    dss = d_star_sq.detach().reshape(-1).to(dev, dtype)
    ok = torch.isfinite(d) & torch.isfinite(sig) & (sig > 0) & torch.isfinite(dss)
    if fit_mask is not None:
        ok = ok & fit_mask.reshape(-1).to(dev, torch.bool)
    use_f = f_dark is not None and gamma != 0.0
    if use_f:
        f = f_dark.detach().reshape(-1).to(dev, dtype)
        ok = ok & torch.isfinite(f)
    if delta_calc is not None:
        c_all = delta_calc.detach().reshape(-1).to(dev, dtype)
        ok = ok & torch.isfinite(c_all)
    n_fit = int(ok.sum())
    if n_fit < order + 4:
        raise ValueError(f"need at least {order + 4} usable reflections, got {n_fit}")

    d, sig, dss = d[ok], sig[ok], dss[ok]
    # Work in units of the rms difference so every term of the likelihood is O(1) and
    # the Hessian is well conditioned in float32.
    amp_scale = float(d.square().mean().sqrt().clamp(min=1e-12))
    d_std = d / amp_scale
    d2 = d_std.square()
    log_sig2 = 2.0 * torch.log(sig / amp_scale)
    log_eps = (
        torch.log(epsilon.reshape(-1).to(dev, dtype)[ok])
        if epsilon is not None
        else torch.zeros_like(d)
    )
    stol = dss.clamp(min=0.0).sqrt() / 2.0
    stol_range = (float(stol.min()), float(stol.max()))
    basis = _design(dss, order + 1, stol_range)
    if use_f:
        log_f = _log_amp(f[ok], amp_scale)
        log_f_ref = float(log_f.mean())
        log_f = log_f - log_f_ref
    else:
        log_f, log_f_ref = torch.zeros_like(d), 0.0
    has_centric = centric is not None and bool(centric.reshape(-1)[ok].any())
    cen = centric.reshape(-1).to(dev)[ok].to(dtype) if has_centric else None

    n_c = order + 1
    n_a = alpha_order + 1 if delta_calc is not None else 0
    i_a = n_c + 3
    if n_a:
        c_std = c_all[ok] / amp_scale
        basis_a = _design(dss, n_a, stol_range)
    fit_gamma = use_f and gamma is None
    free = torch.zeros(i_a + n_a, dtype=torch.bool, device=dev)
    free[:n_c] = True
    free[n_c] = fit_gamma
    free[n_c + 1] = fit_sigma_scale
    free[n_c + 2] = has_centric
    free[i_a:] = True

    theta = torch.zeros(i_a + n_a, dtype=dtype, device=dev)
    if n_a:
        # Start from the global least-squares coupling, so the power starts from the
        # residual rather than from the whole difference.
        cc = float(c_std.square().mean())
        theta[i_a] = float((d_std * c_std).mean()) / cc if cc > 0 else 1.0
        d2 = (d_std - theta[i_a] * c_std).square()
    excess = float((d2.mean() - log_sig2.exp().mean()))
    theta[0] = math.log(max(excess, 0.1 * float(d2.mean())))
    theta[n_c] = float(gamma) if (use_f and gamma is not None) else (1.0 if use_f else 0.0)

    def nll(t):
        log_s = basis @ t[:n_c] + t[n_c] * log_f
        if cen is not None:
            log_s = log_s + t[n_c + 2] * cen
        # log V = log(eps S + k^2 sigma^2), formed in log space so neither term can
        # underflow the sum.
        log_v = torch.logaddexp(log_eps + log_s, 2.0 * t[n_c + 1] + log_sig2)
        if n_a:
            resid2 = (d_std - (basis_a @ t[i_a:]) * c_std).square()
        else:
            resid2 = d2
        z = resid2 * torch.exp(-log_v)
        if robust:
            per = 0.5 * log_v + 0.5 * (nu + 1.0) * torch.log1p(z / nu)
        else:
            per = 0.5 * (log_v + z)
        return per.mean()

    idx = torch.nonzero(free, as_tuple=True)[0]
    current = float(nll(theta))
    converged = False
    lam = 1e-3
    for _ in range(MAX_ITER):
        t = theta.detach().requires_grad_(True)
        g_full = torch.autograd.grad(nll(t), t, create_graph=True)[0]
        h_full = torch.stack(
            [torch.autograd.grad(g_full[i], t, retain_graph=True)[0] for i in idx]
        )
        g = g_full[idx].detach()
        h = h_full[:, idx].detach()
        eye = torch.eye(len(idx), dtype=dtype, device=dev)
        improved = False
        for _ in range(20):
            # Levenberg damping on the diagonal: the Hessian can be indefinite far from
            # the optimum, where log V is a log-sum-exp of two linear forms.
            step = torch.linalg.solve(h + lam * eye * h.diagonal().abs().max(), g)
            trial = theta.clone()
            trial[idx] = trial[idx] - step
            trial[n_c] = trial[n_c].clamp(*_GAMMA_BOUNDS)
            trial[n_c + 1] = trial[n_c + 1].clamp(*_LOG_K_BOUNDS)
            trial[n_c + 2] = trial[n_c + 2].clamp(*_LOG_CENTRIC_BOUNDS)
            new = float(nll(trial))
            if math.isfinite(new) and new <= current:
                theta, improved = trial, True
                lam = max(lam / 10.0, 1e-9)
                break
            lam *= 10.0
        if not improved:
            break
        delta = current - new
        current = new
        if float(step.abs().max()) < 1e-6 or delta < 1e-10:
            converged = True
            break

    t = theta.detach().requires_grad_(True)
    g_full = torch.autograd.grad(nll(t), t, create_graph=True)[0]
    h = torch.stack(
        [torch.autograd.grad(g_full[i], t, retain_graph=True)[0][idx] for i in idx]
    ).detach()
    stderr = torch.full_like(theta, float("nan"))
    try:
        # The objective is a mean, so the per-reflection Hessian is n_fit times it.
        cov = torch.linalg.inv(h) / n_fit
        stderr[idx] = cov.diagonal().clamp(min=0.0).sqrt()
    except RuntimeError:
        pass

    theta = theta.detach()
    return DifferencePowerFit(
        coeffs=theta[:n_c].clone(),
        gamma=float(theta[n_c]) if use_f else 0.0,
        sigma_scale=float(theta[n_c + 1].exp()),
        sigma_scale_at_bound=bool(
            fit_sigma_scale
            and min(abs(float(theta[n_c + 1]) - b) for b in _LOG_K_BOUNDS) < 1e-4
        ),
        centric_factor=float(theta[n_c + 2].exp()) if has_centric else 1.0,
        stol_range=stol_range,
        alpha_coeffs=theta[i_a:].clone(),
        amp_scale=amp_scale,
        log_f_ref=log_f_ref,
        stderr=stderr,
        nll=current,
        converged=converged,
        n_fit=n_fit,
    )


def bounded_wiener_weight(
    snr: torch.Tensor, snr_floor: float = DEFAULT_SNR_FLOOR
) -> torch.Tensor:
    """Wiener weight with the signal-to-noise ratio floored smoothly at ``snr_floor``.

    ``w = (snr + snr_floor) / (snr + 1 + snr_floor)``, which lies in
    ``[snr_floor / (1 + snr_floor), 1)``: a reflection is down-weighted by its noise
    fraction but never removed. ``snr_floor = 0`` is the plain Wiener weight.

    Parameters
    ----------
    snr : torch.Tensor
        Per-reflection ``S / (k sigma)**2``, e.g. from :meth:`DifferencePowerFit.snr`.
    snr_floor : float
        Non-negative floor.

    Returns
    -------
    torch.Tensor
        Weights, same shape as ``snr``, not normalised.
    """
    if snr_floor < 0:
        raise ValueError("snr_floor must be non-negative")
    return (snr + snr_floor) / (snr + 1.0 + snr_floor)


__all__ = [
    "DEFAULT_ALPHA_ORDER",
    "DEFAULT_ORDER",
    "DEFAULT_SNR_FLOOR",
    "SIGMA_SCALE_BOUNDS",
    "DifferencePowerFit",
    "bounded_wiener_weight",
    "fit_difference_power",
]
