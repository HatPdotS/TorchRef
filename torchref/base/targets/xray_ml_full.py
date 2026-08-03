"""Full-form maximum-likelihood (MLF) X-ray math: model error *and* observation
error, each treated according to its own nature.

The ``ml`` target (:func:`torchref.base.targets.xray_likelihoods.rice_math`) uses the
conditional variance ``Sigma = epsilon * beta`` and nothing else -- the
experimental sigma never enters. This module adds it, and adds it *correctly*:

* **Model/data mismatch** (``beta``, the Luzzati sigma_A variance) is an error in
  the *complex* structure factor: it has a phase component. Marginalising that
  unknown phase is what produces the Rice / ``I0`` form, and it is why this
  variance carries the multiplicity ``epsilon`` (symmetry-enhanced expected
  intensity).
* **Observation error** (``sigma_obs``) is an error in the measured *amplitude
  only*. It is a 1-D real Gaussian on ``|F|``, carries no phase, and must **not**
  be multiplied by ``epsilon``.

Folding a 1-D amplitude error into a 2-D phase-carrying variance -- the
variance-inflation shortcut ``Sigma_tot = epsilon*beta + sigma^2`` (Green, 1979) --
conflates the two. The correct treatment marginalises the unknown error-free
amplitude ``t`` (the MLF target of Pannu & Read, 1996)::

    p(F_obs | F_calc) = int_0^inf dt  p_m(t)  *  N(F_obs; t, sigma_obs)

with ``p_m`` the phase-marginalised model-error density. Geometrically: the
observation is an annulus in the complex plane (radius ``F_obs +- sigma_obs``,
phase unconstrained), the model is a 2-D Gaussian blob at ``F_calc``, and the
target is their overlap.

**The two parities are evaluated differently, because one is exactly integrable.**

* *Centric* -- the ``cosh`` form of ``p_m`` *is* a folded normal,
  ``N(t;+Fc,Sigma) + N(t;-Fc,Sigma)``, so each branch against the measurement
  Gaussian is a half-line Gaussian x Gaussian integral with a closed form in
  ``Phi``. Exact, no quadrature.
* *Acentric* -- no elementary closed form exists. The double integral (radial x
  model phase) is elementary in *either* order but only one of the two: phase-first
  gives ``I0`` and leaves a non-elementary radial integral, radial-first gives
  ``erfc`` and leaves a non-elementary periodic phase integral. So: quadrature,
  on the reduced 1-D integral.

Note that this is a **1-D** quadrature. The angular half of the complex-plane
integral is done analytically and exactly -- ``I0`` *is* the phase integral,
``I0(x) = (1/2pi) int_0^2pi exp(x cos theta) dtheta`` -- so the Rice density
already carries the exact angular result at zero quadrature cost. ``N_QUAD``
nodes are ``N_QUAD`` nodes on the single remaining dimension, not a product grid;
spending nodes on the phase would throw away a closed form.

Quadrature parameters
---------------------
``N_QUAD`` and ``N_SIGMA`` are **empirical**, from the screening study in
``sigma_a_rework/quad_screen.py`` (Gauss-Legendre beat Gauss-Hermite by ~10 orders
of magnitude head-to-head). Achieved accuracy and the worst-case grid point are
recorded beside their definitions below -- do not change them without re-running
that screen.
"""

import math

import numpy as np
import torch

from torchref.config import get_compile_targets

LOG_2PI = math.log(2.0 * math.pi)

# --- screened quadrature parameters ----------------------------------------
# Do NOT change these without re-running sigma_a_rework/quad_screen.py.
N_QUAD = 32
N_SIGMA = 8.0

QUAD_PROVENANCE = """
Gauss-Legendre, N_QUAD=32 nodes, window +-8 Laplace widths. From
sigma_a_rework/quad_screen.py over 990 dimensionless grid points x both parities
(sigma_obs/sqrt(Sigma) 1e-3..1e2, Fc/sqrt(Sigma) 0..50, F_obs/sigma_obs 0..200),
in float64 against an adaptive-quadrature reference that is itself validated
against the exact centric closed form to 3.6e-12.

Achieved at N=32, n_sigma=8 (acentric):
    max |dNLL|            7.3e-12   (the reference's own floor)
    signed bias           1.2e-12   -> 1.2e-7 summed over 1e5 reflections
    max rel. grad error   1.0e-8
Accept criterion, fixed in advance: max|dNLL| < 1e-6, |bias| < 1e-8,
max relative gradient error < 1e-5.

Why these values and not smaller:
  * Smallest passing config was N=24, n_sigma=6; N=32 is one grid level of margin.
  * n_sigma=6 imposes a TRUNCATION FLOOR at 1.5e-8 that no node count removes
    (identical from N=24 through N=128). n_sigma=8 converges instead. Cost is
    linear in N and independent of n_sigma, so the wider window is free.
  * n_sigma=3 would be badly wrong: truncating a Gaussian at 3 sigma discards
    2.7e-3 of the mass, which lands directly in the NLL.
  * Gauss-Hermite (nodes symmetric about the peak) was screened head-to-head and
    lost by ~10 orders of magnitude: 1.3e-2 at N=32 vs 7.3e-12 for GL, with far
    worse gradients. It has to drop nodes at t<0, which is exactly the weak-data
    regime this target exists to handle.

Worst grid points at the adopted config:
    value    F_obs=0, sigma_obs=1,  Fc=0,   Sigma=1
    gradient F_obs=0, sigma_obs=10, Fc=0.1, Sigma=1

float32 (the production dtype) is the binding limit, not the quadrature:
max|f32-f64| = 2.3e-3, set by the magnitude of the per-reflection NLL rather than
by the integration. That is not a regression -- it is far BETTER conditioned than
the current `ml` target, which reaches max|NLL| ~ 4e8 (f32 error 13.1) on the same
grid because with no measurement error a large residual is charged entirely to
model error. ml_full caps max|NLL| at ~2e4.
"""

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

    ``torch.special.i0e`` costs ~10.5 ns/element -- 25x ``exp`` -- and would
    dominate this kernel on its own (measured: 50 ms vs 23 ms per forward at 200K
    reflections and 16 nodes). Since ``log I0`` is what is needed, not ``I0``, there
    is no reason to pay for ``i0e`` and then a ``log``.

    Screened accuracy: max ``|dlog I0|`` = 4.7e-7 against :func:`log_i0_exact` over
    ``z`` in [0, 1e4] (A&S quote ~2e-7 for the underlying rational forms). That is
    below the float32 floor of the target and so does not limit it, but it *is*
    larger than the float64 quadrature error -- use :func:`log_i0_exact` on the
    float64 / EAGER reference path.

    Both branches are evaluated everywhere and selected with ``where``, with each
    branch's input first clamped into its own valid domain so the unused branch can
    never emit a NaN that would poison the gradient. Uses ``|z|`` because ``I0`` is
    even; clamping negatives to zero instead would return ``log I0(0) = 0`` rather
    than ``log I0(|z|)`` and break that symmetry.
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


def _log_cosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically safe ``log cosh(x) = |x| + log1p(exp(-2|x|)) - log 2``."""
    ax = torch.abs(x)
    return ax + torch.log1p(torch.exp(-2.0 * ax)) - math.log(2.0)


# =====================================================================
# log-integrand and its analytic Laplace centre
# =====================================================================


def _log_h_acentric(t, F_obs, sigma, Fc, Sigma, li0):
    """``log[ Rice(t; Fc, Sigma) * N(F_obs; t, sigma) ]``."""
    inv_S = 1.0 / Sigma
    return (
        torch.log(2.0 * t * inv_S)
        - (t * t + Fc * Fc) * inv_S
        + li0(2.0 * t * Fc * inv_S)
        - 0.5 * (LOG_2PI + 2.0 * torch.log(sigma))
        - (F_obs - t) ** 2 / (2.0 * sigma**2)
    )


def _laplace_centre_acentric(F_obs, sigma, Fc, Sigma):
    """Closed-form Laplace centre ``t0`` and width ``s`` -- no root-find, no Bessel.

    From ``log I0(x) ~ x`` with ``x = 2 t Fc / Sigma`` (slope ``2Fc/Sigma``,
    curvature ``2/Sigma``). ``a`` is dominated by whichever distribution is
    *narrower*, which is what makes the window auto-scale onto the peak:
    ``sigma << sqrt(Sigma)`` gives ``s -> sqrt(2) sigma`` (measurement spike),
    the reverse gives ``s -> sqrt(Sigma/2)`` (model density).

    The centric integrand has a *different* asymptote (``log cosh(x) ~ |x|`` with
    ``x = t Fc / Sigma``, so half the slope); it is handled by the closed form
    below and never needs a window.
    """
    a = 1.0 / Sigma + 1.0 / (2.0 * sigma**2)
    t0 = (F_obs / sigma**2 + 2.0 * Fc / Sigma) / (2.0 * a)
    s = 1.0 / torch.sqrt(2.0 * a)
    return t0, s


# =====================================================================
# acentric: Gauss-Legendre on a detached, peak-centred window
# =====================================================================


def acentric_nll(F_obs, sigma, Fc, Sigma, n_quad=None, n_sigma=None, li0=log_i0):
    """Per-reflection acentric NLL by Gauss-Legendre + log-sum-exp.

    The window is **detached**: it need only *cover* the mass, and not
    differentiating the (negligible) truncation error keeps the graph simple.
    ``t = 0`` is an interval *endpoint*, not an interior kink, so the Rice support
    boundary is handled exactly -- this is why Gauss-Legendre is used rather than
    Gauss-Hermite in the measurement variable, which straddles ``t < 0`` for weak
    reflections. The integrand is a product of two log-concave densities, hence
    log-concave and unimodal, so one window suffices and there is no second mode.

    Routes through a ``torch.compile(dynamic=True)`` build when
    ``torchref.compile_targets`` is on (the default) and the standard configuration
    is in use. Eager costs ~20 array passes *per node*; fusing is worth 3.7-16x on
    CPU and 13-29x on GPU. See :class:`torchref.config.CompileTargetsConfig`.
    """
    n_quad = N_QUAD if n_quad is None else n_quad
    n_sigma = N_SIGMA if n_sigma is None else n_sigma

    if (
        li0 is log_i0
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

    ``n_quad`` and ``n_sigma`` are closed over rather than passed as arguments so
    the unrolled node loop is a compile-time constant. Only the leading dimension
    varies, so ``dynamic=True`` yields **one** compilation for every dataset size --
    verified across 13 reflection counts (2 to 739 272), separate datasets,
    work/free subsets and gathered tensors, with ``unique_graphs`` staying at 1.
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
    """Eager reference. Everything independent of ``t`` is hoisted out of the node
    loop -- with ``n_quad = 32`` a single un-hoisted ``log`` or divide is 32 extra
    passes over the whole reflection array, and leaving the measurement normaliser
    and ``1/Sigma`` inside cost ~2x measured.
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
        # accounts for both distributions, so a union can only widen the interval
        # away from the peak. At sigma=100, sqrt(Sigma)=1 the peak is 0.7 wide while
        # F_obs +- 8 sigma spans ~800 -- measured max|dNLL| ~ 1e1..1e2, worsening
        # with n_sigma and flat in n_quad. The integrand is a *product*, so its mass
        # sits at the compromise peak; the factors' own supports are irrelevant.
        lo = torch.clamp(t0 - n_sigma * s, min=0.0)
        hi = t0 + n_sigma * s
        half = (hi - lo) * 0.5
        mid = (hi + lo) * 0.5
        # Shift computed on the t-dependent part alone, so `const` cancels out of
        # the loop entirely rather than being added and subtracted 32 times.
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


def _log_shift(F_obs, sigma, Fc, Sigma, t0, li0):
    """Log-sum-exp shift: ``max h`` over three cheap candidate peak locations.

    A *fixed* analytic shift (rather than a running max) is what lets the node loop
    use one ``exp`` per node with no second pass and no ``(n, n_quad)`` intermediate.
    But ``t0`` *under*estimates the true peak when the measurement spike and the
    model density are well separated (e.g. ``F_obs=200, Fc=50, sqrt(Sigma)=1`` --
    150 sigma apart), and an under-estimated shift makes ``exp(h - shift)``
    overflow. In that regime the peak is pinned near one of the two individual
    maxima, so probing ``t0``, ``F_obs`` and ``Fc`` covers it at O(1) cost,
    independent of ``n_quad``.
    """
    out = None
    for t in (
        torch.clamp(t0, min=1e-30),
        torch.clamp(F_obs, min=1e-30),
        torch.clamp(Fc, min=1e-30),
    ):
        h = _log_h_acentric(t, F_obs, sigma, Fc, Sigma, li0)
        out = h if out is None else torch.maximum(out, h)
    return out


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
    li0=log_i0,
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
    per-call ``nonzero`` (and the device sync it implies).
    """
    F_obs = F_obs.reshape(-1)
    Fc = torch.abs(F_calc).reshape(-1)
    if alpha is not None:
        # The Luzzati mean coupling enters ONLY as the Rice/folded-normal mean, and
        # it appears there solely as alpha*Fc: the model term carries
        # -(t^2 + alpha^2 Fc^2)/Sigma and log I0(2 t alpha Fc / Sigma), i.e.
        # (alpha*Fc)^2 and 2 t (alpha*Fc). So folding it into Fc here is exact and
        # leaves the quadrature, window, shift and centric closed form untouched.
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
