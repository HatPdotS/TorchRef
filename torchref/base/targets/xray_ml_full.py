"""Full-form maximum-likelihood (MLF) X-ray math: model *and* observation error.

Model/data mismatch (``beta``, the Luzzati sigma_A variance) is an error in the
*complex* structure factor, so it carries a phase -- marginalising that phase is
what produces the Rice / ``I0`` form, and it is why this variance is scaled by
the multiplicity ``epsilon``. Observation error (``sigma_obs``) is an error in
the measured *amplitude only*: a 1-D real Gaussian on ``|F|``, and it must
**never** be multiplied by ``epsilon``. Folding one into the other -- the
variance-inflation shortcut ``Sigma_tot = epsilon*beta + sigma^2`` (Green, 1979)
-- conflates them; instead the unknown error-free amplitude ``t`` is
marginalised (MLF of Pannu & Read, 1996)::

    p(F_obs | F_calc) = int_0^inf dt  p_m(t)  *  N(F_obs; t, sigma_obs)

Centric reflections have an exact closed form (:func:`centric_nll`); acentric
ones have none and go through 1-D Gauss-Legendre quadrature
(:func:`acentric_nll`), whose node count ``N_QUAD`` and window ``N_SIGMA`` are
empirical; the comment on them states the accuracy they were chosen for.
"""

import math

import numpy as np
import torch

from torchref.config import get_compile_targets

LOG_2PI = math.log(2.0 * math.pi)

# Gauss-Legendre nodes and window half-width in Laplace widths. Against a float64
# reference (max|dNLL| < 1e-6, |bias| < 1e-8, relative gradient error < 1e-5) the
# smallest passing setting is 24/6; 32/8 adds a grid level and avoids n_sigma=6's
# 1.5e-8 truncation floor. tests/unit/refinement/test_ml_full.py pins the result.
N_QUAD = 32
N_SIGMA = 8.0

_GL_CACHE: dict = {}
_SIGMA_FLOOR = 1e-6
_VAR_FLOOR = 1e-10


def _gl_nodes(n: int, dtype: torch.dtype, device) -> tuple:
    """Cached Gauss-Legendre nodes/weights on [-1, 1]; they are constants."""
    key = (n, dtype, str(device))
    if key not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[key] = (
            torch.as_tensor(x, dtype=dtype, device=device),
            torch.as_tensor(w, dtype=dtype, device=device),
        )
    return _GL_CACHE[key]


# =====================================================================
# log I0
# =====================================================================


def log_i0_exact(z: torch.Tensor) -> torch.Tensor:
    """``log I0(z)`` via ATen's exp-scaled Bessel. Accurate but ~25x ``exp``.

    ``i0e(z) = exp(-|z|) * I0(z)``, so the undo factor is ``+|z|``, **not** ``+z``.
    Production only sees ``z = 2 t Fc / Sigma >= 0`` where the two agree, but with
    ``+z`` the extension to negative argument silently returns ``log I0(z) - 2|z|``,
    which destroys the evenness of ``I0`` and hence of ``NLL(F_calc)``. Any
    finite-difference check that steps ``F_calc`` negative lands there and reports a
    spurious non-zero derivative at ``F_calc = 0``, where the true one is exactly 0.
    """
    return torch.log(torch.special.i0e(z)) + torch.abs(z)


def log_i0(z: torch.Tensor) -> torch.Tensor:
    """Branchless two-region ``log I0(z)`` (Abramowitz & Stegun 9.8.1 / 9.8.2).

    Replaces ``i0e`` + ``log``, which at 25x ``exp`` per element would dominate
    this kernel (~2x measured per forward). Accuracy is max ``|dlog I0|`` = 4.7e-7
    against :func:`log_i0_exact` over ``z`` in [0, 1e4] -- under the target's
    float32 floor, but *above* the float64 quadrature error, which is why
    :func:`acentric_nll` takes :func:`log_i0_exact` for float64 inputs.

    Both branches run everywhere and are selected with ``where``, each input first
    clamped into its own valid domain so the unused branch cannot emit a NaN that
    would poison the gradient. Takes ``|z|`` because ``I0`` is even; clamping
    negatives to zero would return ``log I0(0)`` and break that symmetry.
    """
    zc = torch.abs(z)

    t = torch.clamp(zc, max=3.75) / 3.75
    t2 = t * t
    i0_small = 1.0 + t2 * (
        3.5156229
        + t2
        * (
            3.0899424
            + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))
        )
    )
    small = torch.log(i0_small)

    zl = torch.clamp(zc, min=3.75)
    u = 3.75 / zl
    poly = 0.39894228 + u * (
        0.01328592
        + u
        * (
            0.00225319
            + u
            * (
                -0.00157565
                + u
                * (
                    0.00916281
                    + u * (-0.02057706 + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377)))
                )
            )
        )
    )
    large = zl - 0.5 * torch.log(zl) + torch.log(poly)

    return torch.where(zc <= 3.75, small, large)


# =====================================================================
# analytic Laplace centre of the log-integrand
# =====================================================================


def _laplace_centre_acentric(F_obs, sigma, Fc, Sigma):
    """Closed-form Laplace centre ``t0`` and width ``s`` -- no root-find, no Bessel.

    From ``log I0(x) ~ x`` at ``x = 2 t Fc / Sigma``, so ``a`` is set by whichever
    density is narrower and the window auto-scales onto the peak. Acentric only:
    the centric integrand's asymptote has half this slope, and its closed form
    needs no window.
    """
    a = 1.0 / Sigma + 1.0 / (2.0 * sigma**2)
    t0 = (F_obs / sigma**2 + 2.0 * Fc / Sigma) / (2.0 * a)
    s = 1.0 / torch.sqrt(2.0 * a)
    return t0, s


# =====================================================================
# acentric: Gauss-Legendre on a detached, peak-centred window
# =====================================================================


def acentric_nll(F_obs, sigma, Fc, Sigma, n_quad=None, n_sigma=None, li0=None):
    """Per-reflection acentric NLL by Gauss-Legendre + log-sum-exp.

    The window is **detached**: it need only *cover* the mass, and not
    differentiating the (negligible) truncation error keeps the graph simple.
    ``t = 0`` is an interval *endpoint*, not an interior kink, so the Rice support
    boundary is handled exactly -- this is why Gauss-Legendre is used rather than
    Gauss-Hermite in the measurement variable, which straddles ``t < 0`` for weak
    reflections. The integrand is a product of two log-concave densities, hence
    log-concave and unimodal, so one window suffices and there is no second mode.

    ``li0`` is the ``log I0`` implementation; ``None`` takes :func:`log_i0_exact` for
    float64 inputs and the fast :func:`log_i0` otherwise.

    Routes through a ``torch.compile(dynamic=True)`` build when
    ``torchref.config.compile_targets`` is on (off by default;
    ``TORCHREF_COMPILE_TARGETS``) and the standard configuration is in use -- eager
    costs ~20 array passes per node, so fusing is worth an order of magnitude. See
    :class:`torchref.config.CompileTargetsConfig`.
    """
    n_quad = N_QUAD if n_quad is None else n_quad
    n_sigma = N_SIGMA if n_sigma is None else n_sigma
    if li0 is None:
        # dtype-ok: selects the reference log-Bessel for float64, not an allocation
        li0 = log_i0_exact if F_obs.dtype is torch.float64 else log_i0

    if (
        li0 is log_i0
        # dtype-ok: validation guard (compile eligibility), not an allocation
        and F_obs.dtype is not torch.float64
        and F_obs.numel() > 1  # 0/1-specialisation would force a 2nd compile
        and get_compile_targets()
    ):
        fn = _compiled_acentric(n_quad, n_sigma)
        if fn is not None:
            return fn(F_obs, sigma, Fc, Sigma)

    return _acentric_nll_eager(F_obs, sigma, Fc, Sigma, n_quad, n_sigma, li0)


_COMPILED: dict = {}


def _compiled_acentric(n_quad: int, n_sigma: float):
    """Lazily-built compiled kernel, one per ``(n_quad, n_sigma)``.

    Both are closed over rather than passed, so the unrolled node loop is a
    compile-time constant; only the leading dimension varies, and ``dynamic=True``
    then gives one compilation for every dataset size.
    """
    key = (n_quad, float(n_sigma))
    if key not in _COMPILED:
        def worker(F_obs, sigma, Fc, Sigma):
            return _acentric_nll_eager(F_obs, sigma, Fc, Sigma, n_quad, n_sigma, log_i0)

        try:
            _COMPILED[key] = torch.compile(worker, dynamic=True)
        except Exception:  # pragma: no cover - no inductor/triton available
            _COMPILED[key] = None
    return _COMPILED[key]


def _acentric_nll_eager(F_obs, sigma, Fc, Sigma, n_quad, n_sigma, li0):
    """Eager reference. Keep everything independent of ``t`` hoisted out of the
    node loop: at ``n_quad = 32`` one stray ``log`` or divide inside is 32 extra
    passes over the whole reflection array (~2x measured).
    """
    inv_S = 1.0 / Sigma
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    two_Fc_invS = 2.0 * Fc * inv_S
    # log(2 t / Sigma) = log(2/Sigma) + log(t); only log(t) depends on t.
    log_2_invS = torch.log(2.0 * inv_S)
    # Constant-in-t part of the integrand's log: -Fc^2/Sigma, log(2/Sigma) and the
    # Gaussian normaliser. Added ONCE at the end instead of inside every node.
    const = log_2_invS - (Fc * Fc) * inv_S - 0.5 * (LOG_2PI + 2.0 * torch.log(sigma))

    def log_h_var(t):
        """The t-dependent part only; ``const`` is restored outside the loop."""
        return (
            torch.log(t)
            - t * t * inv_S
            + li0(t * two_Fc_invS)
            - (F_obs - t) ** 2 * inv_2s2
        )

    with torch.no_grad():
        t0, s = _laplace_centre_acentric(F_obs, sigma, Fc, Sigma)
        # Laplace window ONLY. Do not union this with F_obs +- n*sigma: `s` already
        # accounts for both densities, so a union can only widen the interval away
        # from the peak and wreck the accuracy (max|dNLL| ~ 1e1..1e2, worsening with
        # n_sigma). The integrand is a *product*; the factors' own supports are
        # irrelevant to where its mass sits.
        lo = torch.clamp(t0 - n_sigma * s, min=0.0)
        hi = t0 + n_sigma * s
        half = (hi - lo) * 0.5
        mid = (hi + lo) * 0.5
        # Shift computed on the t-dependent part alone, so `const` cancels out of
        # the loop entirely rather than being added and subtracted 32 times. All
        # three probes are needed: t0 alone underestimates the peak when the two
        # densities are far apart, and exp(h - shift) then overflows.
        shift = torch.maximum(
            log_h_var(torch.clamp(t0, min=1e-30)),
            torch.maximum(
                log_h_var(torch.clamp(F_obs, min=1e-30)),
                log_h_var(torch.clamp(Fc, min=1e-30)),
            ),
        )

    x, w = _gl_nodes(n_quad, F_obs.dtype, F_obs.device)
    acc = torch.zeros_like(F_obs)
    for k in range(n_quad):
        t = torch.clamp(mid + half * x[k], min=1e-30)
        acc = acc + w[k] * torch.exp(log_h_var(t) - shift)
    return -(torch.log(acc) + torch.log(half) + shift + const)


# =====================================================================
# centric: exact closed form
# =====================================================================


def centric_nll(F_obs, sigma, Fc, Sigma):
    """Per-reflection centric NLL -- exact, no quadrature.

    ``p_m`` is the folded normal ``N(t;+Fc,Sigma) + N(t;-Fc,Sigma)`` (which is
    exactly the ``cosh`` form used by the ``ml`` target), so each branch against
    the measurement Gaussian is a half-line Gaussian x Gaussian integral::

        p = sum_{s=+-1} N(F_obs; s*Fc, Sigma + sigma^2) * Phi(m_s / s_w)
        1/s_w^2 = 1/Sigma + 1/sigma^2
        m_s     = s_w^2 * (s*Fc/Sigma + F_obs/sigma^2)

    Note the variance **does** add here (``Sigma + sigma^2``) -- but gated by the
    ``Phi`` factor, which is exactly what the variance-inflation shortcut omits.

    ``torch.special.log_ndtr`` is required, not ``log(ndtr(.))``: the ``s = -1``
    branch drives the argument deep negative and the naive form underflows to
    ``-inf``.
    """
    var = Sigma + sigma**2
    sw2 = 1.0 / (1.0 / Sigma + 1.0 / sigma**2)
    sw = torch.sqrt(sw2)
    terms = []
    for s in (1.0, -1.0):
        m = sw2 * (s * Fc / Sigma + F_obs / sigma**2)
        log_n = -0.5 * (LOG_2PI + torch.log(var)) - (F_obs - s * Fc) ** 2 / (2.0 * var)
        terms.append(log_n + torch.special.log_ndtr(m / sw))
    return -torch.logsumexp(torch.stack(terms, dim=0), dim=0)


# =====================================================================
# per-reflection dispatch and masked sum
# =====================================================================


def parity_indices(centric_flags: torch.Tensor) -> tuple:
    """``(idx_acentric, idx_centric)``. Cache this per dataset; it never changes."""
    cen = centric_flags.reshape(-1).to(torch.bool)
    return (
        torch.nonzero(~cen, as_tuple=True)[0],
        torch.nonzero(cen, as_tuple=True)[0],
    )


def ml_full_nll_per_refl(
    F_obs,
    sigma_obs,
    F_calc,
    beta,
    centric_flags,
    epsilon=None,
    alpha=None,
    n_quad=None,
    n_sigma=None,
    li0=None,
    idx=None,
):
    """Per-reflection full-form NLL (NOT masked/summed).

    ``Sigma = epsilon * beta`` -- ``epsilon`` multiplies the **model** variance
    only; ``sigma_obs`` is an amplitude error and is never scaled by it.

    The two parities are **gathered and scattered back** rather than both being
    evaluated everywhere and selected with ``torch.where``. The ``ml`` target can
    afford ``where`` because all its branches are cheap arithmetic; here each branch
    is expensive (``log_ndtr`` alone costs ~2 ms per 200K reflections, comparable to
    ``i0e``), so evaluating the unused one would roughly double the target's cost.
    Gathering also sidesteps the ``torch.where`` NaN-gradient trap: a branch that is
    never evaluated cannot poison anything.

    Pass ``idx=(idx_acentric, idx_centric)`` from :func:`parity_indices` to avoid a
    per-call ``nonzero`` (and the device sync it implies). ``li0`` is passed to
    :func:`acentric_nll`, which resolves ``None`` by dtype.
    """
    F_obs = F_obs.reshape(-1)
    Fc = torch.abs(F_calc).reshape(-1)
    if alpha is not None:
        # Exact, not an approximation: the Luzzati coupling enters only as the
        # Rice/folded-normal mean and only ever as alpha*Fc, so absorbing it into Fc
        # leaves quadrature, window, shift and the centric closed form untouched.
        Fc = alpha.reshape(-1).to(Fc.dtype) * Fc
    sig = torch.clamp(sigma_obs.reshape(-1), min=_SIGMA_FLOOR)
    beta = torch.clamp(beta.reshape(-1), min=_VAR_FLOOR)
    if epsilon is None:
        Sigma = beta
    else:
        Sigma = torch.clamp(epsilon.reshape(-1).to(beta.dtype) * beta, min=_VAR_FLOOR)

    if idx is None:
        idx = parity_indices(centric_flags)
    ia, ic = idx

    out = torch.zeros_like(F_obs)
    if ia.numel():
        out = out.index_copy(
            0,
            ia,
            acentric_nll(
                F_obs.index_select(0, ia),
                sig.index_select(0, ia),
                Fc.index_select(0, ia),
                Sigma.index_select(0, ia),
                n_quad,
                n_sigma,
                li0,
            ),
        )
    if ic.numel():
        out = out.index_copy(
            0,
            ic,
            centric_nll(
                F_obs.index_select(0, ic),
                sig.index_select(0, ic),
                Fc.index_select(0, ic),
                Sigma.index_select(0, ic),
            ),
        )
    return out
