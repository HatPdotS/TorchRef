"""Data-driven model-error estimation: the per-shell Luzzati ``sigma_A``.

The first of two model-error estimators (see the package docstring). ``sigma_A`` is
inferred from data-model *disagreement* per resolution shell on the FREE set, and
``alpha``/``beta``/``beta_model`` are algebraic consequences of it -- which is what makes
the second-moment identity ``alpha**2 * Sigma_P + beta_model + S2 == B`` hold exactly
rather than approximately. ``beta`` is the absolute model-error variance in F**2 units and
is the overfit-controlling ingredient.

**The estimator and the likelihood use alpha differently, deliberately.**
:func:`estimate_beta` fits ``sigma_A`` by *joint* ``(alpha, beta)`` maximum likelihood --
the mean is ``alpha*|F_calc|`` during the fit, the same problem Phenix's ``funcgm(t) = 0``
root solves. Which target then *consumes* ``alpha`` is a separate question: ``ml`` and
``ml_full`` centre on ``alpha*|F_calc|``, while ``ml_noalpha`` and ``nll_beta`` fix the mean
coupling at 1 (the scaler owning that gauge). Fitting with the mean pinned at ``|F_calc|``
-- which this code used to do -- biases ``sigma_A`` **+0.035 high** on synthetic data with
known truth, so the estimator must use alpha even where the likelihood does not.

Mechanically the fit is a bounded 3-stage geometric grid over ``v = 1 - sigma_A**2`` (no
root-find, no sign-triggered gates, hence deterministic across devices and processes),
followed by an instability-weighted shrinkage toward a fitted ``sigma_A(d*^2)`` curve
(:func:`_shrink_to_curve`), and finally interpolation of
``sigma_A``/``Sigma_N``/``Sigma_P``/``S2`` to every reflection with
``alpha``/``beta``/``beta_model`` derived per reflection from those.

The likelihoods that consume ``beta`` live in
:mod:`torchref.base.targets.xray_likelihoods`; nothing here imports them.

**Do not split :func:`estimate_beta` from :class:`SigmaAEstimator`.** The out-of-repo
estimator lab (``sigma_a_rework/estimator_lab/install.py``) swaps estimator variants by
monkeypatching this module's ``estimate_beta`` global, and :meth:`SigmaAEstimator.get`
resolves it as a same-module global. Moving one without the other makes that patch a
silent no-op, and the lab's own assertion checks the wrong symbol so it would not catch it.

Plain tensors in, plain tensors out -- no ``ReflectionData``/``Scaler`` coupling -- so this
is usable from both :mod:`torchref.scaling` and :mod:`torchref.refinement.targets` without
an import cycle. ``scaling`` must import it *inside* the method that uses it: see the note
in ``ScalerBase.refine_lbfgs``.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from torchref.config import get_float_dtype

# --- sigma_A estimator constants -------------------------------------------------
#: Upper bound on the per-shell ``sigma_A``, i.e. the floor on the model-error variance at
#: ``(1 - SIGMA_A_MAX**2) * Sigma_N``. Matches the effective floor the previous fit had.
SIGMA_A_MAX = 0.99
#: Hard floor on the returned ``alpha``. A BACKSTOP, not a model choice: ``sigma_A = 0``
#: gives ``alpha = 0``, which deletes a whole shell's model contribution from a likelihood
#: centred on ``alpha*|F_calc|``. Observed on a real structure (6OHI shell 2). The
#: stability shrinkage normally prevents it; this catches "every shell collapsed".
ALPHA_FLOOR = 0.1
#: Whether to run the stability shrinkage (:func:`_shrink_to_curve`). There is no pass
#: count: the shrinkage target is a fixed fitted curve, so one shot is exact. The retired
#: ``SHRINK_PASSES = 3`` counted Jacobi passes of the neighbour-shrinkage this replaced.
SHRINK_ENABLED = True
#: Bounded-grid solve: candidates per stage and number of nested-zoom stages. 17 x 3 gives
#: steps in ``beta`` of 27.7% -> 3.1% -> 0.38%, against 12-20% sampling noise.
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

    Exists so target construction is a single call for every taxonomy row: the factory packs
    this and hands it to all seven, and only the ``sigma_A``-family classes read it. The
    alternative -- passing the knobs only to the rows that consume them -- puts a
    ``needs_estimator`` conditional back in the factory, which is the dispatch the 2026-08
    refactor removed.

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


# =====================================================================
# Per-reflection epsilon (multiplicity)
# =====================================================================


def epsilon_from_hkl(hkl: torch.Tensor, spacegroup) -> torch.Tensor:
    """Per-reflection epsilon: number of rotation symops mapping h -> +/-h.

    Mirrors ``ReciprocalSymmetry.get_epsilon`` (Friedel-aware) but works directly
    on the scattered HKL list. Returns ones if ``spacegroup`` is None or lacks
    ``apply_to_hkl``.

    Always returns on ``hkl.device``, whatever device the space group's symmetry
    matrices live on: the caller multiplies this against per-reflection data
    sitting beside ``hkl``.
    """
    n = hkl.shape[0]
    float_dtype = get_float_dtype()
    if spacegroup is None or not hasattr(spacegroup, "apply_to_hkl"):
        return torch.ones(n, device=hkl.device, dtype=float_dtype)

    with torch.no_grad():
        # Configured float dtype, not float64: MPS has no float64 and casting
        # there raises. Symmetry arithmetic on Miller indices is exact in
        # float32 (integer-valued rotation matrices, small indices), so the
        # exact `==` comparisons below remain valid.
        #
        # ``apply_to_hkl`` moves its input onto the matrices' device, so build
        # ``h`` there too -- otherwise ``Hs`` and ``h0`` land on different
        # devices and the comparisons below raise. The space group wins for the
        # arithmetic; the result is handed back on the caller's device.
        sym_device = getattr(spacegroup, "matrices", hkl).device
        h = hkl.to(device=sym_device, dtype=float_dtype)
        Hs = spacegroup.apply_to_hkl(h)  # (N,3,ops)
        h0 = h.unsqueeze(-1)  # (N,3,1)
        same = (Hs == h0).all(dim=1)
        friedel = (Hs == -h0).all(dim=1)
        eps = (same | friedel).sum(dim=1).clamp(min=1).to(float_dtype)
    return eps.to(hkl.device)


@dataclass(frozen=True)
class SigmaAShells:
    """Per-shell output of :func:`estimate_beta`: one bounded parameter plus derivations.

    ``sigma_A`` is the only thing estimated. ``alpha``, ``beta`` and ``beta_model`` are
    algebraic consequences, which is what makes the second-moment identity
    ``alpha**2 * Sigma_P + beta_model + S2 == B`` hold exactly instead of approximately.

    Attributes
    ----------
    sigma_a
        The estimate actually used, after the stability shrinkage.
    sigma_a_raw
        Before shrinkage. Kept so the shrinkage's effect is always attributable, and so a
        collapsed shell (``sigma_a_raw == 0``) stays visible after being rescued.
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
        fitted curve**. ``w -> 0`` means "this shell's departure from the curve is real";
        ``w -> 1`` means "it is noise, use the curve".
    curve_a, curve_b
        The fitted ``-ln sigma_A = a + b*d*^2`` coefficients the shells were shrunk toward
        (``b >= 0`` by construction). Diagnostics only -- nothing in the estimate is
        derived from them. NaN when no curve was fitted: shrinkage disabled, fewer than 4
        shells, or every shell at one ``d*^2``.
    degenerate
        True when no usable free set existed and the fields are the conservative fallback.
    diagnostics
        Counters for every clamp and filter, so a badly-behaved shell is visible rather
        than merely finite: ``n_dropped``, ``n_free``, ``n_shell``, ``n_s2_clamped``,
        ``n_ratio_clamped``, ``n_alpha_floored``, ``n_sigma_a_zero``, ``rho_min``.
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

    Replaces ``(beta, epsilon)`` / ``(alpha, beta, epsilon)`` tuples, and the eight return
    shapes the old flag matrix produced (four flag combinations x two degenerate paths) in
    which slot 4 was either ``shells`` or ``alpha_refl`` depending on the flags -- so
    callers had to branch on flags rather than on arity.

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
        TOTAL conditional variance -- model error plus the measurement variance that the
        raw second moment carries. What a likelihood consumes when it does not account
        for ``sigma_obs`` itself (``ml``, ``nll_beta``).
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
    """Per-reflection Read-MLF NLL with every ``Sigma``-independent term dropped.

    This is the kernel the sigma_A grid minimises. It is the same Rice/folded-normal form
    as ``ml``'s own likelihood (:func:`_ml_beta_nll_per_refl`) up to a per-shell additive
    constant, but the *solve* evaluates it at the mean ``alpha*|F_c|`` while ``ml``
    evaluates it at ``|F_c|`` -- so it is not the same objective, and must not be
    described as one. That difference is the point: see :func:`_solve_sigma_a` for why
    fitting at ``alpha == 1`` biases ``sigma_A`` high.

    The dropped terms -- ``-log(2 F_o)`` acentric and
    ``-0.5 log(2/pi)`` centric -- do not depend on the parameter being fitted, so they
    only inflate the magnitude that the differences between candidates have to survive.
    The naive full objective reaches ~4e4 while the difference to resolve is ~5e-3,
    i.e. 1.25e-7 relative = float32 epsilon; that is why the old fit had a recorded
    f32-vs-f64 discrepancy of 13.1. (The solve additionally runs in float64, which is
    what actually retires the problem; this is margin.)

    ``i0e`` is the exp-scaled Bessel, so ``log I0(z) = log i0e(z) + z`` for ``z >= 0``.
    """
    Sigma = Sigma.clamp(min=1e-30)
    z = (2.0 * fo * fc / Sigma).clamp(min=0.0, max=1e8)
    log_i0 = torch.log(torch.special.i0e(z).clamp(min=1e-300)) + z
    acen = torch.log(Sigma) + (fo * fo + fc * fc) / Sigma - log_i0
    # centric: 0.5 log Sigma + (Fo^2 + Fc^2)/(2 Sigma) - log cosh(Fo Fc / Sigma),
    # with log cosh written in the overflow-safe shifted form |y| + log1p(exp(-2|y|)).
    y = (fo * fc / Sigma).abs().clamp(max=1e8)
    log_cosh = y + torch.log1p(torch.exp(-2.0 * y)) - math.log(2.0)
    cen = 0.5 * torch.log(Sigma) + (fo * fo + fc * fc) / (2.0 * Sigma) - log_cosh
    return torch.where(centric, cen, acen)


def _grid_ladder(n: int, ratio: float, device, dtype) -> torch.Tensor:
    """``ratio ** (k - (n-1)/2)`` for ``k`` in ``[0, n)``, built from Python floats.

    Built on the host and moved, rather than computed with ``exp``/``linspace`` on the
    device: the stages below are then pure multiplications of this constant by the
    running winner, which is bit-identical on CPU and GPU. A device-side ``exp`` is not
    guaranteed to be.
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
    mean ``alpha*|F_c|`` -- not at ``|F_c|``.

    That distinction is the whole point, so it is worth stating why. The Rice likelihood
    has two stationarity conditions::

        dL/dalpha = 0  <=>  alpha*Sigma_P = <b_j * I1/I0(2 t b_j)>,  b_j = Fo Fc/eps
        dL/dbeta  = 0  <=>  beta = B - alpha**2 * Sigma_P

    The second **is** the moment identity -- it is a consequence of the unconstrained
    two-parameter fit, not a constraint imposed on it. So the joint optimum necessarily
    lies on the identity surface, and that surface is exactly what ``sigma_A``
    parameterises (``alpha**2 Sigma_P + beta = sigma_A**2 Sigma_N + (1-sigma_A**2) Sigma_N
    + S2 = B`` for every candidate). A 1-D search over ``sigma_A``, scored with the true
    joint likelihood, therefore lands on the joint ML optimum -- which is precisely what
    Phenix's ``funcgm(t) = 0`` root solves (``mmtbx/max_lik/max_lik.h``), reproduced here
    with a bounded deterministic grid instead of a bracket-and-regula-falsi that needed
    three sign-triggered gates to survive.

    A previous version scored candidates with the mean pinned at ``|F_c|`` (alpha == 1) and
    then back-derived ``alpha`` from the identity. That fits the wrong likelihood, and the
    error is not academic: on synthetic data with known per-shell ``sigma_A`` it biases the
    estimate **+0.035 high** (100 shells x 400 reflections, shrinkage off), which this form
    removes (-0.006). See ``tests/.../test_sigma_a_solve.py`` for the pinned regression.

    Why a grid and not a root-find. The previous solve bracketed and regula-falsi'd an
    unbounded parametrisation ``t in (0, inf)`` and needed three *sign-triggered* gates
    to cope (``OMEGA <= 0``, ``wiAB <= 3e-7``, and a "never bracketed" fallback), plus
    the cancellation-prone ``wi = A*B - C**2``. Discrete gates on a float32 quantity are
    what made ``beta`` differ across GPU processes. This has:

    * **no data-dependent control flow at all** -- a fixed ``n_stages * n_grid``
      evaluations, hence deterministic by construction;
    * **global character** -- interior unimodality is *not* guaranteed (a heavy-tailed
      within-shell amplitude distribution can give two local minima), and every
      candidate including both boundaries is evaluated;
    * a **hard invariant** -- the result minimises the objective over all evaluated
      candidates, with non-finite mapped to ``+inf``, so the worst possible outcome is
      ``v = 1`` (``sigma_A = 0``, ``beta = B``): the conservative "this shell's model
      carries no information" answer for the *variance*. Note it is NOT conservative
      for the mean, which is why ``alpha`` is guarded separately.

    Stage 1 spans ``[v_min, 1]``; each later stage re-centres on the winner with the
    previous stage's step as its full span. With 17 points and 3 stages the step in
    ``beta`` is 27.7% -> 3.1% -> 0.38%, against a measured within-shell sampling noise
    of 12-20% -- three orders of margin.

    Searching ``v`` rather than ``sigma_A`` is deliberate: ``1 - sigma_A**2`` is pure
    round-off once ``sigma_A`` approaches 1, so the small-variance end would be
    unresolvable if the grid lived in ``sigma_A``.
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
        f_cand = torch.stack(
            [
                torch.segment_reduce(nll[i], "sum", lengths=seg_lengths, unsafe=True)
                for i in range(n_grid)
            ]
        )
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

    ``var_SN`` is **exactly** ``var_ml / 2`` for every ``sigma_A`` and ``n``, since
    ``var_SN/var_ml = [1/(4 sA^2 n)] / [1/(2 sA^2 n)] = 1/2``. So this is
    ``1.5 * var_ml``, and the two terms are kept separate only to record where the factor
    comes from -- do not read the second as "the one that matters when the model is good".

    (An earlier comment here claimed the second term dominates at the good end. That is
    true of **alpha's** relative sd, not sigma_A's: ``alpha = sigma_A sqrt(Sigma_N/Sigma_P)``
    picks up ``Sigma_N`` at ``0.5/sqrt(n)`` relative, which does *not* vanish as
    ``sigma_A -> 1`` whereas the ML term does. That is what makes alpha's bootstrapped
    relative sd 1.8-3.3x the ML-only prediction on well-fitting structures. Shrinkage
    here is applied to ``sigma_A``, so ``sigma_A``'s variance is the right input.)

    Against a 400-rep bootstrap of alpha through the production solve, propagating this
    form **overestimates** by 1.3-1.7x. Overestimating is the safe direction: slightly
    more shrinkage than strictly warranted. Left uncalibrated on purpose -- a correction
    fitted to five structures would not generalise.
    """
    sa = sigma_a.clamp(min=1e-4)
    one_m = 1.0 - sa * sa
    var_ml = (one_m / (sa * torch.sqrt(2.0 * n))) ** 2
    var_sn = (one_m / (2.0 * sa)) ** 2 / n
    return (var_ml + var_sn).clamp(min=1e-30)


def _shrink_to_curve(sigma_a, var, bin_dss):
    """Shrink each shell's ``sigma_A`` toward a 2-parameter FITTED CURVE by its instability.

    Same DerSimonian-Laird machinery as the neighbour shrinkage this replaces; only the
    target changes, and that is the whole point. Shrinking toward neighbours borrows from
    shells that are themselves as noisy as the shell being corrected. Shrinking toward
    ``exp(-(a + b*d*^2))`` fitted across ALL shells borrows from something 10-80x better
    determined, because the two parameters see every free reflection::

        fit    -ln sigma_A = a + b*d*^2, weights 1/var(-ln sigma_A) = sigma_A^2/var
        tau^2  = DL between-shell variance ABOUT THE CURVE, weights 1/var
        w_i    = var_i / (var_i + tau^2)
        sigma_A_i <- (1 - w_i)*sigma_A_i + w_i*curve_i

    **Why a hybrid and not either extreme.** Measured on logged refinement trajectories,
    the residual (per-shell minus curve) is 100% reproducible within a structure
    (correlation between consecutive refinement steps 0.998-1.000) and only ~61% shared
    between structures. So the curve captures a real common trend, while ~39% of the
    residual is dataset-specific (ice rings, anisotropy, local incompleteness) and no
    function of resolution can express it. Five separate attempts to *remove* per-shell
    freedom all refined worse -- global curve, fixed boxcar, Gaussian, an unbinned
    2-parameter ML fit, and injected noise; the noise control is what proved the per-shell
    scatter is information rather than stochastic regularisation. This *adds* a
    well-determined fallback instead of taking information away, and lets DL decide per
    shell how much of each to use. Measured against the neighbour shrinkage on 765 of the
    767-structure AF-start benchmark: a tie overall (R_free p=0.083, R_work p=0.817) and
    -0.00060 (p=0.022) on the data-poor quartile, which is what it is here for.

    ``tau^2`` is the between-shell variance about the curve, i.e. the size of that
    dataset-specific residual, so ``w_i -> 0`` where the residual is real and large (keep
    the shell) and ``w_i -> 1`` where the shell is badly determined (use the curve). No
    iteration: unlike the Jacobi neighbour passes the target is fixed, so one shot is exact
    -- which is why there is no pass count any more.

    ``b`` is clamped at ``>= 0`` to enforce monotone decay, the physically required
    direction that noisy per-shell fits violate (measured: 2 of 10 structures produced a
    slightly negative slope). ``a`` is refitted after the clamp so the curve still passes
    through the weighted centroid.

    Only ``sigma_A`` is pooled; ``Sigma_N`` and ``S2`` stay per shell. That is what makes
    this safe where smoothing ``beta`` was measured harmful: ``beta = (1-sigma_A^2)*Sigma_N``
    multiplies a dimensionless factor that *should* vary smoothly with resolution by
    ``Sigma_N``, which falls steeply and is well determined (relsd ~ 1/sqrt(n)). Smoothing
    the product smears the falloff; pooling only ``sigma_A`` leaves
    ``beta_i = (1 - sigma_A_pooled^2) * Sigma_N,i`` falling exactly as ``Sigma_N`` does.

    The DL weights are ``1/var``, never counts. Count weighting lets shells with huge
    ``var`` dominate ``Q`` and veto shrinkage entirely.

    **Known weakness, deliberately left in.** ``tau^2`` is a single GLOBAL number, so a
    curve that fits most shells well shrinks a badly-fit shell hard too. On synthetic data
    ``w`` reached 0.36 on the top-resolution shell even at 4000 reflections/shell, pulling
    ``sigma_A`` 0.609 -> 0.653 against a truth of 0.60. A per-shell residual scale would
    fix it and is not estimable from one observation per shell -- but the trajectory data
    above shows the residual is reproducible within a structure, so a *previous cycle's*
    residual could supply it. That is the next step, and it is not taken here.

    Returns ``(sigma_a_shrunk, w, tau_sq, a, b)``; ``a``/``b`` are NaN scalars when no
    curve was fitted.
    """
    nan = float("nan")
    k = sigma_a.numel()
    # Two fitted parameters need at least two residual degrees of freedom to be
    # meaningful. Note this is stricter than the neighbour shrinkage's `k < 3`: a
    # 3-shell fit now gets no shrinkage where it previously got some.
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
    # degenerate input it lands at the cancellation floor (~1e-10 for S~1e4, x~0.2), not
    # near zero. An absolute 1e-30 threshold therefore never fires and the fit proceeds on
    # noise. Fires only when every shell shares one d*^2, which equal-count resolution
    # shells built by sorting on d*^2 cannot produce -- so this guard is unreachable on
    # real data and exists for synthetic/1-shell inputs.
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

    holds **exactly** rather than approximately. The previous implementation fitted
    ``alpha`` on the raw moment ``B`` and subtracted the measurement variance from
    ``beta`` afterwards, which left the shipped pair implying
    ``sigma_A**2 = (B - beta_raw)/(B - S2) > 1`` whenever ``beta_raw < S2``. (Measured
    frequency of that on 24 real structures x ~170 shells: zero. It is a latent
    inconsistency, not an observed one -- do not cite it as a performance motivation.)

    Runs under ``torch.no_grad()`` in float64 internally; ~3 ms, against a 145 s
    refinement.

    Parameters
    ----------
    F_obs, F_calc, centric, epsilon, d_star_sq, free_mask : torch.Tensor
        1-D length-N tensors. ``F_calc`` is the **scaled** amplitude.
    sigma_obs : torch.Tensor, optional
        Scaled experimental sigmas, on the same scale as ``F_obs``. Supplying them is
        what makes ``beta_model`` differ from ``beta``; with ``None`` the shell-mean
        measurement variance ``S2`` is zero and the two coincide (documented, and the
        only difference between the two paths -- there is no branch).
    per_bin : int, optional
        Target reflections per shell. Default 140. Note this is a *count* target, so the
        precision of ``sigma_A`` it delivers varies enormously with the model quality
        (measured: 0.3% relative on a good structure, 114% on a bad one at the same
        count). The stability shrinkage exists to absorb that.
    sigma_a_max : float, optional
        Upper bound on ``sigma_A``, i.e. the floor on the model-error variance
        ``(1 - sigma_a_max**2) * Sigma_N``. Default matches the historical effective floor.
    alpha_floor : float, optional
        Hard floor on the returned ``alpha``. A backstop only: ``sigma_A = 0`` gives
        ``alpha = 0``, which would delete a whole shell's model contribution from a mean
        centred on ``alpha*|F_calc|``. Observed on one real structure. The shrinkage
        normally prevents it; this catches the case where *every* shell collapses.
    shrink : bool, optional
        Run the stability shrinkage toward the fitted ``sigma_A(d*^2)`` curve
        (:func:`_shrink_to_curve`). One shot, no pass count -- the target is a fixed
        curve, so iterating would change nothing.
    n_grid, n_stages : int, optional
        Grid resolution of the bounded solve.
    min_bins, min_per_bin : int, optional
        Floor the shell count for sparse free sets.

    Returns
    -------
    SigmaAShells
        One frozen record of per-shell quantities plus clamp counters. This replaces the
        eight return shapes the flag matrix used to produce, in which slot 4 was either
        ``shells`` or ``alpha_refl`` depending on the flags.
    """
    device = F_obs.device
    out_dtype = F_obs.dtype
    # float64 throughout the fit. The objective's magnitude (~4e4) against the
    # differences that decide the winner (~5e-3) is 1.25e-7 relative -- float32 epsilon.
    dtype = torch.float64

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
        # so this is atomic-free and one program per segment -- bit-stable run to run and
        # identical CPU/GPU, unlike scatter_add's CUDA atomicAdd accumulation order.
        return torch.segment_reduce(x, "sum", lengths=seg_lengths, unsafe=True)

    # Lunin-Skovoroda moment weighting, kept for continuity with the previous fit. The
    # identity below holds under any consistent positive weighting, and centrics are a
    # few percent of reflections, so the choice is immaterial.
    w = torch.where(cen, torch.ones_like(fo), 2.0 * torch.ones_like(fo))
    sum_w = segsum(w).clamp(min=1e-30)
    counts = segsum(torch.ones_like(fo))

    B = segsum(w * fo * fo / eps) / sum_w
    Sigma_P = segsum(w * fc * fc / eps) / sum_w
    S2 = segsum(w * sig * sig / eps) / sum_w
    # Diagnostic only: the shell correlation. Nothing branches on it any more -- the old
    # `wi = A*B - C**2` and `OMEGA` reformulations existed solely to make two sign gates
    # survive float32, and both gates are gone.
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
    # so Sigma_N/Sigma_P > 1 is the normal state and alpha > 1 is legitimate (measured
    # sqrt(Sigma_N/Sigma_P) 1.02 at low resolution to 1.6 at high). It is *not* a
    # low-resolution bulk-solvent artefact -- that hypothesis was tested and refuted.
    # The ratio is capped only for the degenerate case of a shell with no model amplitude.
    # Computed BEFORE the solve because the fit is joint in (alpha, beta) and so needs it.
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
    # The alpha backstop is applied to ``sigma_A``, NOT to ``alpha``. Clamping ``alpha``
    # after deriving it would break the moment identity
    # ``alpha**2 Sigma_P + beta_model + S2 == B`` for precisely the shells the backstop
    # exists to rescue -- reintroducing the mutual inconsistency between the triple that
    # this estimator was rewritten to remove. (Measured before the fix: one collapsed
    # shell in 100 gave a 9.2e-3 relative violation while every other shell was exact.)
    # Since ``alpha = sigma_A * sqrt(ratio)`` is monotone in ``sigma_A``, the identical
    # guarantee ``alpha >= alpha_floor`` is obtained by flooring ``sigma_A`` at
    # ``alpha_floor / sqrt(ratio)``, and then EVERYTHING is still derived from one
    # ``sigma_A``, so the identity stays exact.
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

    Thin stateful wrapper around :func:`estimate_beta`: it caches the detached
    ``(beta, epsilon)`` from the last estimate and re-estimates only after
    :meth:`reset`. The owning target calls :meth:`reset` from its
    ``maintenance()`` hook so ``beta`` refreshes once per optimizer-step block
    (the same cadence the scaler used previously).

    Ownership note
    --------------
    ``beta`` (the conditional variance ``epsilon*beta``) is the overfit-controlling
    ingredient, and ``alpha`` is produced alongside it: the fit is joint in the pair, and
    a target that centres on ``alpha*|F_calc|`` (``ml``, ``ml_full``) reads it.
    ``ml_noalpha`` and ``nll_beta`` fix the mean coupling at 1 in their *likelihood* — the
    scaler owns that gauge — but that is a property of the likelihood, not of the estimate.
    This estimator therefore belongs to the *target* that consumes it, not to the scaler
    (which owns scaling only). Plain tensor in/out — no ``ReflectionData``/``Scaler``
    coupling — so it is usable from both ``scaling`` and ``refinement.targets`` without an
    import cycle.

    Every caller passes ``sigma_obs`` and the same ``shrink`` setting. There is exactly one
    estimator behaviour in the codebase; a configuration that varied by consumer meant
    ``ml`` and ``ml_full`` were fitted differently, so comparing them measured the
    estimator as much as the likelihood.
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

        Four shell curves are interpolated and everything is then derived per reflection,
        rather than interpolating ``beta`` directly: ``sigma_A`` (bounded, so linear
        interpolation cannot leave ``[0, 1]``), ``log Sigma_N`` and ``log Sigma_P``
        (positive by construction under log-linear interpolation) and ``S2``. The
        identity ``alpha**2 Sigma_P + beta_model + S2 == B`` therefore holds at every
        reflection, not merely per shell -- interpolating ``beta`` could yield a
        per-reflection value consistent with no ``sigma_A <= 1`` at all.

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
