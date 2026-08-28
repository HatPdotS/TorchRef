"""Data-driven model-error estimation: the per-shell Luzzati ``sigma_A``.

``sigma_A`` is fitted per resolution shell on the FREE set and
``alpha``/``beta``/``beta_model`` are algebraic consequences of it, which is what makes
the second-moment identity ``alpha**2 * Sigma_P + beta_model + S2 == B`` hold exactly
rather than approximately. ``beta`` -- the absolute model-error variance in F**2 units --
is the overfit-controlling ingredient. :func:`estimate_beta` fits by *joint*
``(alpha, beta)`` ML even for likelihoods that later pin the mean coupling at 1, because
fitting with the mean pinned at ``|F_calc|`` biases ``sigma_A`` high.

Two traps. **Do not move :func:`estimate_beta` out of this module**: the out-of-repo
estimator lab monkeypatches it as a same-module global that :meth:`SigmaAEstimator.get`
resolves, and nothing asserts the patch took. And keep it plain-tensor in/out (no
``ReflectionData``/``Scaler`` coupling), so :mod:`torchref.scaling` -- which must import
it *inside* the method that uses it -- stays free of an import cycle.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch

from torchref.config import get_float_dtype


def epsilon_from_hkl(hkl: torch.Tensor, spacegroup) -> torch.Tensor:
    """Per-reflection epsilon, tolerating a missing space group.

    Thin adapter over :meth:`~torchref.symmetry.symmetry.Symmetry.epsilon`, which owns
    the multiplicity count. It exists because reflection data may carry no space group
    at all, and every consumer here would otherwise repeat the same guard.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices, shape ``(N, 3)``.
    spacegroup : Symmetry or None
        The group. ``None`` means no symmetry information, which yields ones -- the
        same answer P1 gives.

    Returns
    -------
    torch.Tensor
        Multiplicities, shape ``(N,)``, at the configured float dtype, on ``hkl``'s
        device.
    """
    if spacegroup is None or not hasattr(spacegroup, "epsilon"):
        return torch.ones(hkl.shape[0], device=hkl.device, dtype=get_float_dtype())
    return spacegroup.epsilon(hkl)

# --- sigma_A estimator constants -------------------------------------------------
#: Upper bound on the per-shell ``sigma_A``, i.e. the floor on the model-error variance at
#: ``(1 - SIGMA_A_MAX**2) * Sigma_N``.
SIGMA_A_MAX = 0.99
#: Hard floor on the returned ``alpha``. A BACKSTOP, not a model choice: ``sigma_A = 0``
#: gives ``alpha = 0``, which deletes a whole shell's model contribution from a likelihood
#: centred on ``alpha*|F_calc|``. The stability shrinkage normally prevents it; this
#: catches "every shell collapsed".
ALPHA_FLOOR = 0.1
#: Whether to run the stability shrinkage (:func:`_shrink_to_curve`). There is no pass
#: count: the shrinkage target is a fixed fitted curve, so one shot is exact.
SHRINK_ENABLED = True
#: Bounded-grid solve: candidates per stage and number of nested-zoom stages. 17 x 3 steps
#: ``beta`` down 27.7% -> 3.1% -> 0.38%, comfortably under the 12-20% sampling noise.
N_GRID = 17
N_STAGES = 3
#: Floor on ``Sigma_N / B``. Fires only where a shell's observed power is at or below its
#: own measurement noise, and is counted when it does.
ETA = 1e-3
#: Cap on ``Sigma_N / Sigma_P``, so ``alpha <= sqrt(RATIO_MAX) * sigma_A``. Measured
#: ``sqrt(Sigma_N/Sigma_P)`` is 1.02-1.6, so this can only fire on a shell with
#: essentially no model amplitude. Counted when it does.
RATIO_MAX = 25.0

@dataclass(frozen=True)
class SigmaAConfig:
    """The estimator's knobs, as one value.

    Packed by the target factory and handed to every taxonomy row, so construction needs
    no ``needs_estimator`` conditional; only the ``sigma_A``-family classes read it.
    ``shrink=None`` means "the module default", normalised here so consumers never have to
    handle ``None``. Frozen, so two targets sharing a config cannot drift apart.
    """

    sigma_a_max: float = SIGMA_A_MAX
    shrink: Optional[bool] = None

    def __post_init__(self):
        object.__setattr__(self, "sigma_a_max", float(self.sigma_a_max))
        object.__setattr__(
            self, "shrink", bool(SHRINK_ENABLED if self.shrink is None else self.shrink)
        )


@dataclass(frozen=True)
class SigmaAShells:
    """Per-shell output of :func:`estimate_beta`: one bounded parameter plus derivations.

    ``sigma_A`` is the only thing estimated; ``alpha``, ``beta`` and ``beta_model`` are
    algebraic consequences, so ``alpha**2 * Sigma_P + beta_model + S2 == B`` holds exactly.

    Attributes
    ----------
    sigma_a
        The estimate actually used, after the stability shrinkage.
    sigma_a_raw
        Before shrinkage, so the shrinkage's effect stays attributable and a collapsed
        shell (``sigma_a_raw == 0``) stays visible after being rescued.
    alpha
        ``sigma_A * sqrt(Sigma_N/Sigma_P)``, floored at ``ALPHA_FLOOR``.
    beta
        Total conditional variance, ``beta_model + S2``.
    beta_model
        Model error alone -- the sigma_Fobs-free variance. Equals ``beta`` when no
        ``sigma_obs`` was supplied (then ``S2 == 0``).
    Sigma_N, Sigma_P, S2, B
        Expected true intensity, model intensity, mean measurement variance, and the raw
        second moment. All ``epsilon``-reduced and weighted identically, so the identity
        above is exact in these units.
    counts, bin_dss
        Reflections per shell and its mean ``d*^2`` (the interpolation abscissa).
    shrink_w, tau
        Per-shell shrinkage weight and the DerSimonian-Laird between-shell sd **about the
        fitted curve**: ``w -> 0`` means the shell's departure from the curve is real,
        ``w -> 1`` means it is noise and the curve is used.
    curve_a, curve_b
        Fitted ``-ln sigma_A = a + b*d*^2`` coefficients (``b >= 0``). Diagnostics only.
        NaN when no curve was fitted: shrinkage off, fewer than 4 shells, or one ``d*^2``.
    degenerate
        True when no usable free set existed and the fields are the conservative fallback.
    diagnostics
        Counters for every clamp and filter: ``n_dropped``, ``n_free``, ``n_shell``,
        ``n_s2_clamped``, ``n_ratio_clamped``, ``n_alpha_floored``, ``n_sigma_a_zero``,
        ``rho_min``.
    """

    sigma_a: torch.Tensor
    sigma_a_raw: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    beta_model: torch.Tensor
    Sigma_N: torch.Tensor
    Sigma_P: torch.Tensor
    S2: torch.Tensor
    B: torch.Tensor
    counts: torch.Tensor
    bin_dss: torch.Tensor
    shrink_w: torch.Tensor
    tau: float
    curve_a: float
    curve_b: float
    degenerate: bool
    diagnostics: dict


@dataclass(frozen=True)
class SigmaAEstimate:
    """Everything a target needs from one model-error estimate, per reflection.

    All tensors are per-reflection, detached (the estimate is a nuisance quantity;
    gradients must never flow through it) and share one length, so a consumer reading
    several fields cannot hit a shape mismatch.

    Attributes
    ----------
    sigma_a
        The bounded correlation coefficient, ``sqrt(1 - beta/Sigma_N)`` in ``[0, 1]``.
        Zeros on the degenerate path, where no shell fit exists.
    alpha
        Luzzati mean coupling, ``sigma_A * sqrt(Sigma_N/Sigma_P)``. **Not** bounded by 1:
        ``Sigma_N/Sigma_P > 1`` whenever the model explains less scattering than the data
        contain, which is the normal state (measured 1.02-1.6 on refined models).
    beta
        TOTAL conditional variance -- model error plus the measurement variance the raw
        second moment carries. What a likelihood consumes when it does not account for
        ``sigma_obs`` itself (``ml``, ``nll_beta``).
    beta_model
        Model error alone, ``beta`` less the per-shell mean measurement variance. What a
        likelihood consumes when it DOES account for ``sigma_obs`` explicitly
        (``ml_full``), so the measurement error is not counted twice. Equal to ``beta``
        when no ``sigma_obs`` was supplied.
    epsilon
        The multiplicity actually applied, so the caller need not re-derive it.
    shells
        The :class:`SigmaAShells` this was interpolated from -- per-shell diagnostics and
        clamp counters.
    """

    sigma_a: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    beta_model: torch.Tensor
    epsilon: torch.Tensor
    shells: Optional["SigmaAShells"] = None


# =====================================================================
# Phenix-style ML alpha/beta estimator (Lunin-Skovoroda)
# =====================================================================


def _rice_nll_reduced(
    fo: torch.Tensor,
    fc: torch.Tensor,
    Sigma: torch.Tensor,
    centric: torch.Tensor,
) -> torch.Tensor:
    """Per-reflection Read-MLF NLL, ``Sigma``-independent terms dropped for conditioning.

    The kernel the sigma_A grid minimises. Same Rice/folded-normal form as ``ml``'s
    likelihood up to a per-shell constant, but evaluated at the mean ``alpha*|F_c|``
    rather than ``|F_c|`` -- so it is NOT the same objective. ``i0e`` is the exp-scaled
    Bessel: ``log I0(z) = log i0e(z) + z`` for ``z >= 0``.

    **Written to avoid a catastrophic cancellation.** Both branches would otherwise
    subtract two nearly-equal large terms: the ``+z`` inside ``log I0`` cancels against
    ``(Fo^2 + Fc^2)/Sigma`` down to ``(Fo - Fc)^2/Sigma``. On a well-refined model those
    operands are ~20.0 and ~19.99 while the answer is ~0.026 -- 1.3e-3 of their magnitude,
    so roughly three decimal digits are gone before the result is formed. Folding the
    cancellation in by hand is an exact algebraic identity (verified bit-identical in
    float64) and is what lets the fit run in float32. Same collapse in the centric branch
    via ``log cosh y = y + log1p(exp(-2y)) - log 2``, using ``Fo, Fc >= 0`` (both are
    amplitudes, and ``alpha >= 0``) so that ``2|Fo Fc| = 2 Fo Fc``.
    """
    Sigma = Sigma.clamp(min=1e-30)
    q = fo * fo + fc * fc
    # (|Fo| - |Fc|)**2 == Fo^2 + Fc^2 - 2|Fo Fc| is the cancelled form of both quadratics.
    dif = fo.abs() - fc.abs()
    d2 = dif * dif

    # Each branch folds its own linear term, and each guards on its OWN clamp: `z` and `y`
    # saturate at different raw magnitudes, so they cannot share one folded quantity. Where
    # a guard is active the fold would change the value, so the literal form is kept -- such
    # a candidate is at an absurd magnitude and loses either way, but this keeps the rewrite
    # exact everywhere rather than exact-on-the-data-that-was-tested.
    raw_z = 2.0 * fo * fc / Sigma
    z = raw_z.clamp(min=0.0, max=1e8)
    acen_quad = torch.where(raw_z == z, d2 / Sigma, q / Sigma - z)
    log_i0e = torch.log(torch.special.i0e(z).clamp(min=1e-300))
    acen = torch.log(Sigma) + acen_quad - log_i0e

    # centric: 0.5 log Sigma + (Fo^2 + Fc^2)/(2 Sigma) - log cosh(Fo Fc / Sigma),
    # with log cosh in the overflow-safe shifted form y + log1p(exp(-2y)) - log 2.
    raw_y = (fo * fc / Sigma).abs()
    y = raw_y.clamp(max=1e8)
    cen_quad = torch.where(raw_y == y, d2 / (2.0 * Sigma), q / (2.0 * Sigma) - y)
    cen = (
        0.5 * torch.log(Sigma)
        + cen_quad
        - torch.log1p(torch.exp(-2.0 * y))
        + math.log(2.0)
    )
    return torch.where(centric, cen, acen)


@lru_cache(maxsize=8)
def _segment_layout(lengths: Tuple[int, ...], device_str: str):
    """``(index, mask)`` placing contiguous segments on a padded ``(n_seg, max_len)`` grid.

    Cached: ``_solve_sigma_a`` reduces ``n_grid * n_stages`` times over one layout.
    ``lengths`` is a tuple so it can be a cache key.
    """
    device = torch.device(device_str)
    L = torch.tensor(lengths, dtype=torch.long, device=device)
    total = int(L.sum())
    max_len = int(L.max()) if L.numel() else 0
    zero = torch.zeros(1, dtype=torch.long, device=device)
    starts = torch.cat([zero, L.cumsum(0)[:-1]])
    ar = torch.arange(max_len, device=device).reshape(1, max_len)
    # Clamp keeps the gather in bounds for the padding slots; `mask` zeroes them anyway.
    index = (starts.reshape(-1, 1) + ar).clamp(max=max(total - 1, 0))
    mask = ar < L.reshape(-1, 1)
    return index, mask


def _segsum(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Sum ``x`` over contiguous segments, reducing along a padded trailing axis.

    Replaces ``torch.segment_reduce``, which is unimplemented on MPS. Keeps the properties
    that op was chosen for: atomic-free, one fixed reduction order per segment, so the
    result is bit-stable run to run and does not depend on ``scatter_add``'s CUDA atomicAdd
    accumulation order (the original GPU non-determinism bug -- see
    ``tests/unit/refinement/test_estimate_beta_determinism.py``).

    Deliberately NOT ``cumsum[end] - cumsum[start]``, the usual contiguous-segment trick:
    that recovers each shell sum by subtracting two running totals of the whole array,
    reintroducing the large-minus-large this module is written to avoid.

    ``x`` reduces over its last axis, so a leading batch dimension (the grid candidates)
    is handled in one call. Segments here differ in length by at most one element, so the
    padding overhead is at most ``n_seg`` slots.
    """
    index, mask = _segment_layout(tuple(int(v) for v in lengths), str(x.device))
    return (x[..., index] * mask.to(x.dtype)).sum(dim=-1)


def _grid_ladder(n: int, ratio: float, device, dtype) -> torch.Tensor:
    """``ratio ** (k - (n-1)/2)`` for ``k`` in ``[0, n)``, built from Python floats.

    Built on the host and moved, not with a device-side ``exp``/``linspace``, so the
    stages that multiply it stay bit-identical on CPU and GPU.
    """
    mid = (n - 1) / 2.0
    return torch.tensor(
        [ratio ** (k - mid) for k in range(n)], device=device, dtype=dtype
    )


def _solve_sigma_a(
    fo, fc, cen, eps, seg_lengths, seg, Sigma_N, S2, ratio, v_min, n_grid, n_stages
):
    """Per-shell ML ``v = 1 - sigma_A**2`` by a bounded geometric grid with nested zoom.

    Returns ``(v, objective_at_v)``, both per shell.

    **This is the JOINT (alpha, beta) maximum-likelihood fit.** Each candidate ``v`` fixes
    both the variance ``beta = v*Sigma_N + S2`` and the mean coupling
    ``alpha = sqrt(1-v) * sqrt(Sigma_N/Sigma_P)``, and the objective is evaluated at the
    mean ``alpha*|F_c|`` -- not at ``|F_c|``, which would bias ``sigma_A`` high. The
    ``dL/dbeta = 0`` condition *is* the moment identity, so a 1-D search over ``sigma_A``
    scored with the true joint likelihood lands on the joint ML optimum (the same problem
    Phenix's ``funcgm(t) = 0`` root solves).

    A grid rather than a root-find, so there is **no data-dependent control flow at all**
    -- a fixed ``n_stages * n_grid`` evaluations, deterministic across devices and
    processes, where sign-triggered gates on float32 made ``beta`` differ between GPU
    processes. Non-finite candidates map to ``+inf``, so the worst outcome is ``v = 1``
    (``sigma_A = 0``, ``beta = B``): conservative for the *variance* but NOT for the mean,
    which is why ``alpha`` is guarded separately. Stage 1 spans ``[v_min, 1]``; each later
    stage re-centres on the winner with the previous stage's step as its full span.
    Searching ``v`` rather than ``sigma_A`` is deliberate: ``1 - sigma_A**2`` is pure
    round-off once ``sigma_A`` approaches 1, so a grid in ``sigma_A`` could not resolve
    the small-variance end.
    """
    dtype, device = Sigma_N.dtype, Sigma_N.device
    n_bins = Sigma_N.numel()
    # Hoisted: `ratio` is candidate-independent, so the sqrt is paid once, not 51 times.
    ratio_sqrt = ratio.clamp(max=RATIO_MAX).sqrt()

    span = 1.0 / max(v_min, 1e-12)
    ratio1 = span ** (1.0 / (n_grid - 1))
    # stage 1: absolute ladder covering [v_min, 1] with both ends hit exactly
    v_cand = v_min * _grid_ladder(n_grid, ratio1, device, dtype) * (span ** 0.5)
    v_cand = v_cand.clamp(min=v_min, max=1.0).reshape(n_grid, 1).expand(n_grid, n_bins)

    ratio = ratio1
    best_v = torch.full((n_bins,), 1.0, device=device, dtype=dtype)
    best_f = torch.full((n_bins,), float("inf"), device=device, dtype=dtype)

    for stage in range(n_stages):
        if stage > 0:
            # re-centre on the winner; the previous step becomes this stage's full span
            ratio = ratio ** (2.0 / (n_grid - 1))
            ladder = _grid_ladder(n_grid, ratio, device, dtype).reshape(n_grid, 1)
            v_cand = (best_v.reshape(1, n_bins) * ladder).clamp(min=v_min, max=1.0)

        # objective for every candidate at once: (n_grid, n_free) -> (n_grid, n_bins)
        beta_c = v_cand * Sigma_N.reshape(1, n_bins) + S2.reshape(1, n_bins)
        Sigma = eps.reshape(1, -1) * beta_c[:, seg]
        # alpha from the SAME candidate, so the pair stays on the identity surface and
        # the objective is the joint one (see the docstring).
        alpha_c = (1.0 - v_cand).clamp(min=0.0, max=1.0).sqrt() * ratio_sqrt.reshape(
            1, n_bins
        )
        nll = _rice_nll_reduced(
            fo.reshape(1, -1),
            alpha_c[:, seg] * fc.reshape(1, -1),
            Sigma,
            cen.reshape(1, -1),
        )
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e30))
        # All candidates reduce in one call: _segsum reduces over the trailing axis, so
        # the (n_grid, n_free) block collapses to (n_grid, n_bins) without a Python loop.
        f_cand = _segsum(nll, seg_lengths)
        f_cand = torch.where(
            torch.isfinite(f_cand), f_cand, torch.full_like(f_cand, float("inf"))
        )

        # `where`-tournament rather than argmin: argmin's tie-breaking is not documented
        # as stable across backends, and this runs on both. Strict `<` makes the FIRST
        # (lowest-v) candidate win a tie, on every device.
        for i in range(n_grid):
            take = f_cand[i] < best_f
            best_f = torch.where(take, f_cand[i], best_f)
            best_v = torch.where(take, v_cand[i], best_v)

    return best_v, best_f


def _sigma_a_sampling_var(sigma_a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Sampling variance of a per-shell ``sigma_A`` estimate.

    Two terms::

        var_ml = ((1 - sA^2) / (sA * sqrt(2n)))**2      sigma_A's own asymptotic ML term
        var_SN = ((1 - sA^2) / (2 sA))**2 / n           propagated from Sigma_N's own
                                                        sampling error, relsd ~ 1/sqrt(n)

    ``var_SN`` is **exactly** ``var_ml / 2`` for every ``sigma_A`` and ``n``, so this is
    ``1.5 * var_ml``; the terms are split only to record where the factor comes from --
    do not read the second as "the one that matters when the model is good".
    Deliberately uncalibrated, and it overestimates, which is the safe direction (slightly
    more shrinkage than warranted).
    """
    sa = sigma_a.clamp(min=1e-4)
    one_m = 1.0 - sa * sa
    var_ml = (one_m / (sa * torch.sqrt(2.0 * n))) ** 2
    var_sn = (one_m / (2.0 * sa)) ** 2 / n
    return (var_ml + var_sn).clamp(min=1e-30)


def _shrink_to_curve(sigma_a, var, bin_dss):
    """Shrink each shell's ``sigma_A`` toward a 2-parameter FITTED CURVE by its instability.

    DerSimonian-Laird shrinkage toward a curve fitted across ALL shells, which the two
    parameters determine far better than any one shell is determined::

        fit    -ln sigma_A = a + b*d*^2, weights 1/var(-ln sigma_A) = sigma_A^2/var
        tau^2  = DL between-shell variance ABOUT THE CURVE, weights 1/var
        w_i    = var_i / (var_i + tau^2)
        sigma_A_i <- (1 - w_i)*sigma_A_i + w_i*curve_i

    ``tau^2`` is the size of the dataset-specific residual (ice rings, anisotropy, local
    incompleteness), so ``w_i -> 0`` where that residual is real and large and ``w_i -> 1``
    where the shell is badly determined. One shot, no iteration: the target is a fixed
    curve. ``b`` is clamped at ``>= 0`` to enforce the physically required monotone decay
    that noisy per-shell fits can violate, with ``a`` refitted after the clamp so the curve
    still passes through the weighted centroid. Weights are ``1/var``, never counts --
    count weighting lets high-``var`` shells dominate ``Q`` and veto shrinkage entirely.

    Only ``sigma_A`` is pooled; ``Sigma_N`` and ``S2`` stay per shell, so
    ``beta_i = (1 - sigma_A_pooled^2) * Sigma_N,i`` still falls exactly as ``Sigma_N``
    does. Smoothing ``beta`` itself smears that falloff and was measured harmful.
    ``tau^2`` is a single GLOBAL number, so a curve that fits most shells well shrinks a
    genuinely badly-fit shell too hard; a per-shell residual scale is not estimable from
    one observation per shell.

    Returns ``(sigma_a_shrunk, w, tau_sq, a, b)``; ``a``/``b`` are NaN scalars when no
    curve was fitted.
    """
    nan = float("nan")
    k = sigma_a.numel()
    # Two fitted parameters need at least two residual degrees of freedom.
    if k < 4:
        return sigma_a, torch.zeros_like(sigma_a), sigma_a.new_zeros(()), nan, nan

    sa = sigma_a.clamp(1e-6, 1.0 - 1e-9)
    y = -torch.log(sa)
    # weights are 1/var(y); var(-ln s) = var(s)/s^2 by propagation
    wt = (sa * sa / var).clamp(min=1e-30)
    x = bin_dss
    S = wt.sum()
    Sx = (wt * x).sum()
    Sxx = (wt * x * x).sum()
    Sy = (wt * y).sum()
    Sxy = (wt * x * y).sum()
    det = S * Sxx - Sx * Sx
    # Relative, not absolute: `det` is a difference of two ~`S**2 * x**2` terms, so on a
    # degenerate input it lands at the cancellation floor, not near zero, and an absolute
    # threshold would never fire while the fit proceeded on noise.
    if float(det.abs()) <= 1e-12 * float((S * Sxx).abs()):
        # All shells at one d*^2: the slope is unidentifiable.
        return sigma_a, torch.zeros_like(sigma_a), sigma_a.new_zeros(()), nan, nan
    b = ((S * Sxy - Sx * Sy) / det).clamp(min=0.0)
    a = (wt * (y - b * x)).sum() / S.clamp(min=1e-30)
    curve = torch.exp(-(a + b * x)).clamp(1e-6, 1.0)

    prec = 1.0 / var
    resid = sigma_a - curve
    Q = (prec * resid * resid).sum()
    dof = float(k - 2)  # two parameters were fitted
    c = (prec.sum() - (prec * prec).sum() / prec.sum().clamp(min=1e-30)).clamp(min=1e-30)
    # Q < k-2 means the scatter about the curve is SMALLER than the noise alone predicts,
    # i.e. no evidence of structure the curve is missing -> tau^2 = 0 -> w = 1 -> take
    # the curve outright.
    tau_sq = ((Q - dof) / c).clamp(min=0.0)
    w = var / (var + tau_sq).clamp(min=1e-30)
    out = ((1.0 - w) * sigma_a + w * curve).clamp(min=0.0, max=1.0)
    return out, w, tau_sq, float(a), float(b)


def estimate_beta(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    centric: torch.Tensor,
    epsilon: torch.Tensor,
    d_star_sq: torch.Tensor,
    free_mask: torch.Tensor,
    sigma_obs: torch.Tensor = None,
    per_bin: int = 140,
    sigma_a_max: float = SIGMA_A_MAX,
    alpha_floor: float = ALPHA_FLOOR,
    shrink: bool = SHRINK_ENABLED,
    n_grid: int = N_GRID,
    n_stages: int = N_STAGES,
    min_bins: int = 5,
    min_per_bin: int = 40,
) -> "SigmaAShells":
    """Per-shell Luzzati ``sigma_A``, and ``alpha``/``beta``/``beta_model`` derived from it.

    One bounded parameter is estimated per equal-count resolution shell on the FREE set;
    everything else follows algebraically, so the second-moment identity

        alpha**2 * Sigma_P + beta_model + S2 == B

    holds **exactly** rather than approximately -- to ~1e-16 relative in float64 and ~1e-7
    in float32, which is that dtype's floor for an algebraic identity.

    Runs under ``torch.no_grad()``. The working dtype is the wider of the configured float
    dtype and ``F_obs.dtype`` (float32 on MPS, which has no float64); results are cast back
    to ``F_obs.dtype``. It used to force float64 unconditionally, which made the fit
    unrunnable on MPS -- see the dtype note in the body and ``_rice_nll_reduced``.

    Parameters
    ----------
    F_obs, F_calc, centric, epsilon, d_star_sq, free_mask : torch.Tensor
        1-D length-N tensors. ``F_calc`` is the **scaled** amplitude.
    sigma_obs : torch.Tensor, optional
        Scaled experimental sigmas, on the same scale as ``F_obs``. Supplying them is
        what makes ``beta_model`` differ from ``beta``; with ``None`` the shell-mean
        measurement variance ``S2`` is zero and the two coincide.
    per_bin : int, optional
        Target reflections per shell. A *count* target, so the ``sigma_A`` precision it
        delivers varies with model quality; the shrinkage exists to absorb that.
    sigma_a_max : float, optional
        Upper bound on ``sigma_A``, i.e. the floor on the model-error variance
        ``(1 - sigma_a_max**2) * Sigma_N``.
    alpha_floor : float, optional
        Hard floor on the returned ``alpha``. A backstop only: ``sigma_A = 0`` gives
        ``alpha = 0``, which would delete a whole shell's model contribution from a mean
        centred on ``alpha*|F_calc|``.
    shrink : bool, optional
        Run the stability shrinkage toward the fitted ``sigma_A(d*^2)`` curve
        (:func:`_shrink_to_curve`).
    n_grid, n_stages : int, optional
        Grid resolution of the bounded solve.
    min_bins, min_per_bin : int, optional
        Floor the shell count for sparse free sets.

    Returns
    -------
    SigmaAShells
        One frozen record of per-shell quantities plus clamp counters.
    """
    device = F_obs.device
    out_dtype = F_obs.dtype

    dtype = torch.promote_types(get_float_dtype(), out_dtype)
    if dtype == torch.float64 and device.type == "mps":
        raise RuntimeError(
            "MPS has no float64; set the defaults float dtype to float32 or use CPU"
        )

    fo_all = F_obs.reshape(-1).to(dtype)
    fc_all = torch.abs(F_calc).reshape(-1).to(dtype)
    cen_all = (
        centric.reshape(-1).to(torch.bool)
        if centric is not None
        else torch.zeros_like(fo_all, dtype=torch.bool)
    )
    eps_all = (
        epsilon.reshape(-1).to(dtype) if epsilon is not None else torch.ones_like(fo_all)
    )
    dss_all = d_star_sq.reshape(-1).to(dtype)
    sig_all = (
        sigma_obs.reshape(-1).to(dtype)
        if sigma_obs is not None
        else torch.zeros_like(fo_all)
    )

    # --- usable estimation set -------------------------------------------------
    # Filter rather than clamp: a non-finite amplitude or a non-positive sigma carries no
    # information, and leaving it in poisons a whole shell's moments. Counted, not hidden.
    finite = (
        torch.isfinite(fo_all)
        & torch.isfinite(fc_all)
        & torch.isfinite(sig_all)
        & torch.isfinite(dss_all)
        & (sig_all >= 0.0)
    )
    usable = free_mask.reshape(-1).to(torch.bool) & finite
    n_dropped = int((free_mask.reshape(-1).to(torch.bool) & ~finite).sum())
    free_idx = torch.nonzero(usable, as_tuple=True)[0]
    n_free = int(free_idx.numel())

    if n_free < 2:
        # Degenerate: no usable free set. beta = <Fo^2>, sigma_A = 0, alpha floored.
        ok = torch.isfinite(fo_all)
        b = (
            (fo_all[ok] ** 2).mean()
            if bool(ok.any())
            else torch.ones((), device=device, dtype=dtype)
        )
        one = torch.ones(1, device=device, dtype=dtype)
        return SigmaAShells(
            sigma_a=torch.zeros(1, device=device, dtype=out_dtype),
            sigma_a_raw=torch.zeros(1, device=device, dtype=out_dtype),
            alpha=(one * alpha_floor).to(out_dtype),
            beta=(one * b).to(out_dtype),
            beta_model=(one * b).to(out_dtype),
            Sigma_N=(one * b).to(out_dtype),
            Sigma_P=one.to(out_dtype),
            S2=torch.zeros(1, device=device, dtype=out_dtype),
            B=(one * b).to(out_dtype),
            counts=torch.zeros(1, device=device, dtype=out_dtype),
            bin_dss=torch.zeros(1, device=device, dtype=out_dtype),
            shrink_w=torch.zeros(1, device=device, dtype=out_dtype),
            tau=0.0,
            curve_a=float("nan"),
            curve_b=float("nan"),
            degenerate=True,
            diagnostics=dict(n_dropped=n_dropped, n_free=n_free, n_shell=0,
                             n_s2_clamped=0, n_ratio_clamped=0, n_alpha_floored=1,
                             n_sigma_a_zero=1),
        )

    # --- equal-count resolution shells ---------------------------------------
    # stable=True so tied d_star_sq break identically on CPU and GPU: a non-stable CUDA
    # argsort reshuffles ties per process, which reshuffles shell membership.
    order = torch.argsort(dss_all[free_idx], stable=True)
    sel = free_idx[order]
    fo, fc, cen = fo_all[sel], fc_all[sel], cen_all[sel]
    eps, dss, sig = eps_all[sel], dss_all[sel], sig_all[sel]

    n_by_count = max(1, n_free // per_bin)
    n_cap = max(1, n_free // min_per_bin)
    n_bins = max(n_by_count, min(min_bins, n_cap))
    seg = (torch.arange(n_free, device=device) * n_bins) // n_free
    seg_lengths = torch.bincount(seg, minlength=n_bins)

    def segsum(x):
        # Contiguous segments (data sorted by resolution, `seg` a non-decreasing ramp),
        # so this is atomic-free with one fixed reduction order per segment -- bit-stable
        # run to run, unlike scatter_add's CUDA atomicAdd accumulation order. See _segsum.
        return _segsum(x, seg_lengths)

    # Lunin-Skovoroda moment weighting. The identity below holds under any consistent
    # positive weighting, so the exact choice is immaterial.
    w = torch.where(cen, torch.ones_like(fo), 2.0 * torch.ones_like(fo))
    sum_w = segsum(w).clamp(min=1e-30)
    counts = segsum(torch.ones_like(fo))

    B = segsum(w * fo * fo / eps) / sum_w
    Sigma_P = segsum(w * fc * fc / eps) / sum_w
    S2 = segsum(w * sig * sig / eps) / sum_w
    # Diagnostic only (the shell correlation); nothing branches on it.
    C = segsum(w * fo * fc / eps) / sum_w

    # S2 > B means the shell's observed power is at or below its own measurement noise.
    # Clamp so Sigma_N stays positive and beta <= B; counted, because a shell in this
    # state is telling you the resolution cut is too generous.
    s2_cap = (1.0 - ETA) * B
    n_s2_clamped = int((S2 > s2_cap).sum())
    S2 = torch.minimum(S2, s2_cap).clamp(min=0.0)
    Sigma_N = (B - S2).clamp(min=1e-30)

    # alpha = sigma_A * sqrt(Sigma_N/Sigma_P). NOT bounded by 1: in Read's decomposition
    # Sigma_N = Sigma_P + Sigma_Q the unmodelled part Sigma_Q > 0 for any real structure,
    # so Sigma_N/Sigma_P > 1 is the normal state and alpha > 1 is legitimate. The ratio is
    # capped only for the degenerate case of a shell with no model amplitude, and computed
    # BEFORE the solve because the fit is joint in (alpha, beta) and so needs it.
    ratio = Sigma_N / Sigma_P.clamp(min=1e-30)
    n_ratio_clamped = int((ratio > RATIO_MAX).sum())

    # --- the one estimated parameter -----------------------------------------
    v_min = max(1.0 - float(sigma_a_max) ** 2, 1e-6)
    v, _obj = _solve_sigma_a(
        fo, fc, cen, eps, seg_lengths, seg, Sigma_N, S2, ratio, v_min, n_grid, n_stages
    )
    sigma_a_raw = (1.0 - v).clamp(min=0.0, max=1.0).sqrt()

    # --- stability shrinkage toward the fitted sigma_A(d*^2) curve -----------
    # `bin_dss` is the shrinkage abscissa as well as the interpolation one, so it is
    # computed here rather than just before the return.
    bin_dss = segsum(dss) / counts.clamp(min=1.0)
    var_sa = _sigma_a_sampling_var(sigma_a_raw, counts.clamp(min=1.0))
    if shrink:
        sigma_a, shrink_w, tau_sq, curve_a, curve_b = _shrink_to_curve(
            sigma_a_raw, var_sa, bin_dss
        )
    else:
        sigma_a = sigma_a_raw
        shrink_w = torch.zeros_like(sigma_a_raw)
        tau_sq = sigma_a_raw.new_zeros(())
        curve_a = curve_b = float("nan")

    # --- everything else, derived --------------------------------------------
    # The alpha backstop is applied to ``sigma_A``, NOT to ``alpha``: clamping ``alpha``
    # after deriving it breaks the moment identity for precisely the shells the backstop
    # rescues. Since ``alpha = sigma_A * sqrt(ratio)`` is monotone in ``sigma_A``, flooring
    # ``sigma_A`` at ``alpha_floor / sqrt(ratio)`` gives the same guarantee with everything
    # still derived from one ``sigma_A``, so the identity stays exact.
    ratio_sqrt = ratio.clamp(max=RATIO_MAX).sqrt()
    sa_floor = (alpha_floor / ratio_sqrt.clamp(min=1e-30)).clamp(
        max=float(sigma_a_max)
    )
    n_alpha_floored = int((sigma_a < sa_floor).sum())
    sigma_a = torch.maximum(sigma_a, sa_floor)

    sa2 = (sigma_a * sigma_a).clamp(min=0.0, max=1.0)
    beta_model = ((1.0 - sa2) * Sigma_N).clamp(min=1e-30)
    beta = beta_model + S2
    alpha = sigma_a * ratio_sqrt

    to = lambda t: t.to(out_dtype)  # noqa: E731
    return SigmaAShells(
        sigma_a=to(sigma_a),
        sigma_a_raw=to(sigma_a_raw),
        alpha=to(alpha),
        beta=to(beta),
        beta_model=to(beta_model),
        Sigma_N=to(Sigma_N),
        Sigma_P=to(Sigma_P),
        S2=to(S2),
        B=to(B),
        counts=to(counts),
        bin_dss=to(bin_dss),
        shrink_w=to(shrink_w),
        tau=float(tau_sq.clamp(min=0.0).sqrt()),
        curve_a=curve_a,
        curve_b=curve_b,
        degenerate=False,
        diagnostics=dict(
            n_dropped=n_dropped,
            n_free=n_free,
            n_shell=int(n_bins),
            n_s2_clamped=n_s2_clamped,
            n_ratio_clamped=n_ratio_clamped,
            n_alpha_floored=n_alpha_floored,
            n_sigma_a_zero=int((sigma_a_raw <= 0.0).sum()),
            rho_min=float((C / (Sigma_P * B).clamp(min=1e-30).sqrt()).min()),
        ),
    )


def _interp_in_dss(dss_all, bin_dss, vals):
    """Linear interpolation of per-bin ``vals`` (at ``bin_dss``) to all
    reflections by their ``d_star_sq``; clamp-to-edge outside the range."""
    n_bins = bin_dss.numel()
    if n_bins == 1:
        return torch.full_like(dss_all, float(vals[0]))
    idx = torch.searchsorted(bin_dss, dss_all).clamp(1, n_bins - 1)
    x0 = bin_dss[idx - 1]
    x1 = bin_dss[idx]
    wlin = ((dss_all - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
    return (1 - wlin) * vals[idx - 1] + wlin * vals[idx]


# =====================================================================
# Stateful estimator (owned by the consuming target, not the scaler)
# =====================================================================


class SigmaAEstimator:
    """Lazy, cached free-set model-error variance ``beta`` (Luzzati σ_A).

    Thin stateful wrapper around :func:`estimate_beta`: caches the detached estimate and
    re-estimates only after :meth:`reset`. **The owning target must call :meth:`reset`
    from its ``maintenance()`` hook**, otherwise ``beta`` is frozen for the whole run.
    Owned by the consuming target, not the scaler (which owns scaling only), and every
    caller must pass the same ``sigma_obs``/``shrink`` settings so comparing two
    likelihoods does not also compare two estimators.
    """

    def __init__(self):
        self._cache = None  # (beta_per_refl, epsilon) detached
        self._alpha = None  # alpha_per_refl, detached
        self._beta_per_bin = None  # diagnostics
        self._alpha_per_bin = None  # diagnostics

    def reset(self) -> None:
        """Invalidate the cache so the next :meth:`get` re-estimates ``beta``."""
        self._cache = None
        self._alpha = None

    def alpha_per_bin(self):
        """Last-estimated per-bin Luzzati ``alpha`` (diagnostics)."""
        return self._alpha_per_bin

    @property
    def beta_per_bin(self):
        """Last-estimated per-bin ``beta`` (diagnostics); ``None`` until first call."""
        return self._beta_per_bin

    def get(
        self,
        F_obs: torch.Tensor,
        F_calc_scaled: torch.Tensor,
        centric: torch.Tensor,
        epsilon: torch.Tensor,
        d_star_sq: torch.Tensor,
        free_mask: torch.Tensor,
        out_epsilon: torch.Tensor = None,
        target_dss: torch.Tensor = None,
        sigma_obs: torch.Tensor = None,
        **kwargs,
    ) -> SigmaAEstimate:
        """Return the cached-or-recomputed :class:`SigmaAEstimate`, all fields detached.

        Estimated on the **free** set under ``no_grad``; gradients never flow through it.
        Four shell curves are interpolated (``sigma_A``, ``log Sigma_N``, ``log Sigma_P``,
        ``S2``) and the rest derived per reflection, so the identity
        ``alpha**2 Sigma_P + beta_model + S2 == B`` holds at every reflection rather than
        merely per shell -- interpolating ``beta`` directly can yield a value consistent
        with no ``sigma_A <= 1`` at all.

        Parameters
        ----------
        F_obs, F_calc_scaled, centric, epsilon, d_star_sq, free_mask
            Length-N tensors passed to :func:`estimate_beta`. ``F_calc_scaled`` must
            already carry the scaler's scaling.
        out_epsilon : torch.Tensor, optional
            Multiplicity to return/cache if it differs from the estimation ``epsilon``
            (the collection case pools several datasets for the fit but applies one
            common ``epsilon``). Defaults to ``epsilon``.
        target_dss : torch.Tensor, optional
            Interpolate onto this ``d_star_sq`` grid instead of the input one (used to map
            a pooled multi-dataset fit back onto the common HKL).
        sigma_obs : torch.Tensor, optional
            Scaled experimental sigmas. Supplying them is what makes ``beta_model``
            differ from ``beta``; with ``None`` the two coincide.
        **kwargs
            Forwarded to :func:`estimate_beta` (``sigma_a_max``, ``shrink``, ...).
        """
        if self._cache is not None:
            return self._cache
        with torch.no_grad():
            sh = estimate_beta(
                F_obs, F_calc_scaled, centric, epsilon, d_star_sq, free_mask,
                sigma_obs=sigma_obs, **kwargs,
            )
            self._shells = sh
            self._beta_per_bin = sh.beta
            self._alpha_per_bin = sh.alpha

            grid = (
                target_dss.reshape(-1)
                if target_dss is not None
                else d_star_sq.reshape(-1)
            )
            grid = grid.to(sh.beta.dtype)
            if sh.degenerate or sh.bin_dss.numel() == 0:
                # One conservative value everywhere; no curve to interpolate.
                sigma_a = torch.zeros_like(grid)
                alpha = torch.full_like(grid, float(sh.alpha[0]))
                beta = torch.full_like(grid, float(sh.beta[0]))
                beta_model = beta
            else:
                sigma_a = _interp_in_dss(grid, sh.bin_dss, sh.sigma_a).clamp(0.0, 1.0)
                log_sn = _interp_in_dss(
                    grid, sh.bin_dss, torch.log(sh.Sigma_N.clamp(min=1e-30))
                )
                log_sp = _interp_in_dss(
                    grid, sh.bin_dss, torch.log(sh.Sigma_P.clamp(min=1e-30))
                )
                s2 = _interp_in_dss(grid, sh.bin_dss, sh.S2).clamp(min=0.0)
                Sigma_N = torch.exp(log_sn)
                beta_model = ((1.0 - sigma_a * sigma_a) * Sigma_N).clamp(min=1e-30)
                beta = beta_model + s2
                ratio = torch.exp(log_sn - log_sp).clamp(max=RATIO_MAX)
                alpha = (sigma_a * ratio.sqrt()).clamp(min=ALPHA_FLOOR)

            self._alpha = alpha.detach()
            eps_ret = out_epsilon if out_epsilon is not None else epsilon
            eps_ret = eps_ret.detach() if torch.is_tensor(eps_ret) else eps_ret
            self._cache = SigmaAEstimate(
                sigma_a=sigma_a.detach(),
                alpha=alpha.detach(),
                beta=beta.detach(),
                beta_model=beta_model.detach(),
                epsilon=eps_ret,
                shells=sh,
            )
        return self._cache
