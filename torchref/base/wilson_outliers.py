"""Model-free outlier detection from Wilson statistics.

Asks one question per reflection: *how improbable is this observation if the
crystal is a random arrangement of atoms?* No model, no ``F_calc`` -- only the
intensity, its sigma, and what Wilson statistics predict for its resolution
shell, multiplicity and centricity.

The observation, not the true intensity, is what gets tested. Convolving the
acentric Wilson prior ``P(J) = (1/S)exp(-J/S)`` with the Gaussian measurement
error ``I|J ~ N(J, sigma^2)`` gives a predictive density whose upper tail has a
closed form,

    P(I' > I) = Q(I/sigma) + exp(sigma^2/(2 S^2) - I/S) Phi(I/sigma - sigma/S)

whose ``Phi`` argument is exactly :func:`~torchref.base.french_wilson.french_wilson_h`.
That function's ``h >= -4`` cut is the *lower* tail of the same density -- French
and Wilson's guard against intensities too negative to be noise. Real outliers --
zingers, ice rings, overlapped or mis-integrated spots -- are on the strong side,
which is what the upper tail above supplies.

Two things decide whether the test means anything, and both are about ``S``:

- ``S`` must be robust. A zinger inflates the arithmetic mean of its own
  resolution shell and hides itself, so the shell estimate here is a median.
- ``S`` must be anisotropic. Diffraction is routinely 3-5x stronger along one
  reciprocal axis than another; a shell average over-normalises the weak
  direction and under-normalises the strong one, so an isotropic ``S`` flags the
  strong direction of every anisotropic dataset while missing the weak one.

The threshold scales with the number of reflections tested: a fixed per-reflection
p-value rejects a fixed *fraction* of good data, which on 10^5 simultaneous tests
is not a criterion at all.

Assumes the intensity distribution really is Wilson's. Pseudo-translational
symmetry (systematic strong/weak classes) and twinning both violate that and will
show up as an implausible rejection rate rather than as an error.
"""

import math

import torch

from torchref.base.french_wilson import french_wilson_h
from torchref.base.math_torch import U_to_matrix
from torchref.base.reciprocal.basis import get_scattering_vectors

#: median of Exp(1) is ln 2, so ``Sigma = median(I) / ln 2`` for acentrics.
_ACENTRIC_MEDIAN = math.log(2.0)
#: median of chi^2_1 is 0.4549, the centric equivalent.
_CENTRIC_MEDIAN = 0.4549364231195730
#: Below this, log Phi switches from ``erfc`` to a Mills-ratio continued fraction.
#: Set by ``erfc``, not by the fraction: MPS's ``erfc`` is already 4% wrong at
#: argument 3.5 and returns exactly zero beyond 4, so the tail cannot be reached
#: through it there. The fraction is accurate to 2e-10 at this boundary and to
#: machine precision below it, so nothing is lost by switching early.
_LOG_PHI_CF_BELOW = -2.0
#: Continued-fraction depth. 40 terms is machine precision at the boundary above;
#: the cost is 40 divisions on a tensor evaluated once per dataset.
_LOG_PHI_CF_TERMS = 40
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def log_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Log of the standard normal CDF, accurate into the far negative tail.

    ``torch.special.log_ndtr`` is not implemented on MPS: it raises, and only
    survives at all through the process-wide ``PYTORCH_ENABLE_MPS_FALLBACK``,
    which silently moves the work to the CPU. This is the portable replacement --
    ``log(erfc(-x/sqrt 2)/2)`` near zero, and a Mills-ratio continued fraction
    below ``x = -2``, which needs nothing but division.

    Parameters
    ----------
    x : torch.Tensor
        Any shape.

    Returns
    -------
    torch.Tensor
        ``log Phi(x)``, same shape. Agrees with ``torch.special.log_ndtr`` to
        better than 1e-6 absolute over ``[-40, 10]`` in float32, on every device.
    """
    far = x < _LOG_PHI_CF_BELOW
    # Both branches are evaluated everywhere, so each input is first clamped into
    # its own branch's valid range: the fraction needs a positive tail argument
    # and erfc must not be pushed into the range where MPS zeroes it.
    safe_near = torch.clamp(x, min=_LOG_PHI_CF_BELOW)
    near = torch.log(0.5 * torch.erfc(-safe_near * (0.5**0.5)))

    tail = torch.clamp(-x, min=-_LOG_PHI_CF_BELOW)
    fraction = torch.zeros_like(tail)
    for term in range(_LOG_PHI_CF_TERMS, 0, -1):
        fraction = term / (tail + fraction)
    cf = -0.5 * tail * tail - _LOG_SQRT_2PI - torch.log(tail + fraction)

    return torch.where(far, cf, near)


def wilson_log_upper_tail(
    I: torch.Tensor,
    sigma: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor = None,
) -> torch.Tensor:
    """Log probability of observing an intensity at least this large.

    The upper tail of the Wilson prior convolved with Gaussian measurement error,
    evaluated in log space because ``exp(sigma^2/(2 S^2) - I/S)`` overflows
    float32 long before the tail probability underflows.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities, any shape.
    sigma : torch.Tensor
        Their standard deviations, same shape. Must be positive.
    mean_intensity : torch.Tensor
        Wilson ``Sigma`` for each reflection -- the shell mean already divided by
        the multiplicity ``epsilon``. Same shape, must be positive.
    is_centric : torch.Tensor or bool, optional
        As for :func:`~torchref.base.french_wilson.french_wilson_h`, which
        supplies the centric factor of two. None treats everything as acentric.

    Returns
    -------
    torch.Tensor
        ``log P(I' > I)``, same shape. Zero (probability one) is approached from
        below for very negative ``I``.
    """
    z = I / sigma
    # h is the French-Wilson parameter, so sigma/S_eff falls out of it rather
    # than being recomputed -- the two tails then cannot disagree about the
    # centric convention.
    h = french_wilson_h(I, sigma, mean_intensity, is_centric)
    a = z - h
    # I/S_eff == z*a, so S_eff itself is never needed here.
    return torch.logaddexp(
        log_normal_cdf(-z), 0.5 * a * a - z * a + log_normal_cdf(h)
    )


def robust_mean_intensity(
    I: torch.Tensor,
    d_spacings: torch.Tensor,
    assign: torch.Tensor,
    estimate: torch.Tensor = None,
    is_centric: torch.Tensor = None,
    per_shell: int = 250,
) -> torch.Tensor:
    """Wilson ``Sigma`` per reflection, from resolution-shell medians.

    The median, not the mean: an outlier raises the mean of its own shell, which
    raises its own ``Sigma``, which is exactly how a contaminated shell hides its
    contamination. ``median(Exp(1)) = ln 2`` and ``median(chi^2_1) = 0.4549``
    convert back to ``Sigma`` in closed form, so nothing is fitted.

    Centrics and acentrics are binned separately -- they follow different
    distributions, so one pooled median calibrates neither.

    Parameters
    ----------
    I : torch.Tensor
        Intensities of shape ``(n,)``, already divided by the multiplicity
        ``epsilon`` and by any anisotropic correction.
    d_spacings : torch.Tensor
        Resolution in Å, shape ``(n,)``.
    assign : torch.Tensor
        Boolean mask of reflections that receive a ``Sigma``.
    estimate : torch.Tensor, optional
        Boolean mask of reflections that *contribute* to the shell medians,
        a subset of ``assign``. Defaults to ``assign``. The two are separate so
        that a reflection can be held out of the estimate on a later pass and
        still be given the ``Sigma`` it will be tested against -- dropping it
        from ``assign`` instead would quietly stop testing it, which is how a
        previously flagged outlier gets un-flagged.
    is_centric : torch.Tensor, optional
        Boolean mask, shape ``(n,)``. None treats everything as acentric.
    per_shell : int, optional
        Target reflections per shell. Default 250. ``Sigma`` is piecewise
        constant across shells; at this width the within-shell variation of the
        Wilson fall-off is well below the width of the tail being tested.

    Returns
    -------
    torch.Tensor
        ``Sigma`` per reflection, shape ``(n,)``. NaN outside ``assign`` and
        wherever no shell estimate could be formed.
    """
    Sigma = torch.full_like(I, float("nan"))
    if estimate is None:
        estimate = assign
    if is_centric is None:
        is_centric = torch.zeros_like(assign)

    for centric, median_factor in (
        (False, _ACENTRIC_MEDIAN),
        (True, _CENTRIC_MEDIAN),
    ):
        group = assign & (is_centric == centric)
        n_group = int(group.sum())
        if n_group < per_shell:
            continue
        index = torch.nonzero(group, as_tuple=True)[0]
        # Descending d: shell 0 is the lowest-resolution one.
        index = index[torch.argsort(d_spacings[index], descending=True)]

        # Held-out members keep their place in the shell but not their vote.
        member = I[index]
        values = torch.where(
            estimate[index], member, torch.full_like(member, float("nan"))
        )

        n_shells = n_group // per_shell
        n_full = n_shells * per_shell
        head = values[:n_full].reshape(n_shells, per_shell)
        shell_sigma = torch.nanmedian(head, dim=1).values / median_factor

        Sigma[index[:n_full]] = torch.repeat_interleave(shell_sigma, per_shell)
        # The remainder is finer than one shell; it joins the last one rather
        # than forming an under-populated shell of its own.
        Sigma[index[n_full:]] = shell_sigma[-1]

    return Sigma


def fit_anisotropic_scale(
    I: torch.Tensor,
    sigma: torch.Tensor,
    mean_intensity: torch.Tensor,
    s_vectors: torch.Tensor,
    fittable: torch.Tensor,
    min_i_over_sigma: float = 3.0,
) -> torch.Tensor:
    """Fit the anisotropic part of ``Sigma`` to the observations alone.

    Models ``Sigma(h) = Sigma_iso(d) * exp(-2 pi^2 s^T U s)`` and returns ``U``.
    Least squares on ``log(I / Sigma_iso)``: a Wilson variate contributes a
    constant ``-gamma`` in the mean and a fixed spread, both absorbed by the
    intercept, so the quadratic terms see only the direction dependence.

    Weak reflections are excluded because ``log I`` of a near-zero measurement is
    dominated by noise rather than by the fall-off being fitted.

    Parameters
    ----------
    I, sigma : torch.Tensor
        Intensities and their sigmas, shape ``(n,)``.
    mean_intensity : torch.Tensor
        Current isotropic ``Sigma`` per reflection, shape ``(n,)``.
    s_vectors : torch.Tensor
        Scattering vectors of shape ``(n, 3)`` in Å⁻¹, from
        :func:`~torchref.base.reciprocal.basis.get_scattering_vectors`.
    fittable : torch.Tensor
        Boolean mask of reflections allowed into the fit.
    min_i_over_sigma : float, optional
        Weak-data cutoff on ``I/sigma``. Default 3.0.

    Returns
    -------
    torch.Tensor
        ``U`` of shape ``(6,)`` in the ``[u11, u22, u33, u12, u13, u23]`` order
        :func:`~torchref.base.math_torch.U_to_matrix` expects. All zeros when
        there is too little data to fit.
    """
    zero = torch.zeros(6, dtype=I.dtype, device=I.device)
    usable = (
        fittable
        & torch.isfinite(mean_intensity)
        & (mean_intensity > 0)
        & (I > 0)
        & (I > min_i_over_sigma * sigma)
    )
    if int(usable.sum()) < 200:
        return zero

    s = s_vectors[usable]
    # Scaled to unit maximum before forming the normal equations: raw s^2 terms
    # span ~1e-4 to 1e-1 in Å^-2, and their squares condition a float32 solve
    # badly. The scale is undone on the coefficients below.
    scale = torch.linalg.vector_norm(s, dim=1).max()
    if not torch.isfinite(scale) or scale <= 0:
        return zero
    q = s / scale

    y = torch.log(I[usable] / mean_intensity[usable])
    design = torch.stack(
        [
            torch.ones_like(y),
            q[:, 0] * q[:, 0],
            q[:, 1] * q[:, 1],
            q[:, 2] * q[:, 2],
            2.0 * q[:, 0] * q[:, 1],
            2.0 * q[:, 0] * q[:, 2],
            2.0 * q[:, 1] * q[:, 2],
        ],
        dim=1,
    )

    gram = design.T @ design
    # A ridge proportional to the trace keeps the solve well posed when a
    # direction is barely sampled (thin resolution wedges, low-symmetry cells).
    ridge = 1e-6 * torch.diagonal(gram).mean()
    gram = gram + ridge * torch.eye(7, dtype=gram.dtype, device=gram.device)
    try:
        coefficients = torch.linalg.solve(gram, design.T @ y)
    except RuntimeError:
        return zero
    if not bool(torch.isfinite(coefficients).all()):
        return zero

    # exp(-2 pi^2 s^T U s) == exp(q^T beta) with s = scale * q.
    beta = coefficients[1:] / (scale * scale)
    return -beta / (2.0 * math.pi**2)


def anisotropic_correction(s_vectors: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """``exp(-2 pi^2 s^T U s)`` per reflection, normalised to unit mean.

    Renormalising leaves the isotropic shell medians carrying the absolute scale
    and this factor carrying only the direction dependence, so the two estimates
    do not fight over it.

    Parameters
    ----------
    s_vectors : torch.Tensor
        Scattering vectors of shape ``(n, 3)`` in Å⁻¹.
    U : torch.Tensor
        Six anisotropic parameters, as returned by :func:`fit_anisotropic_scale`.

    Returns
    -------
    torch.Tensor
        Positive multiplicative correction of shape ``(n,)``, mean 1.
    """
    matrix = U_to_matrix(U)
    exponent = -2.0 * math.pi**2 * ((s_vectors @ matrix) * s_vectors).sum(dim=1)
    correction = torch.exp(exponent.clamp(min=-10.0, max=10.0))
    mean = correction.mean()
    return correction / mean if mean > 0 else correction


def wilson_outlier_mask(
    I: torch.Tensor,
    sigma: torch.Tensor,
    hkl: torch.Tensor,
    d_spacings: torch.Tensor,
    cell: torch.Tensor,
    epsilon: torch.Tensor = None,
    is_centric: torch.Tensor = None,
    usable: torch.Tensor = None,
    alpha: float = 0.01,
    d_max: float = 4.0,
    per_shell: int = 250,
    n_passes: int = 2,
) -> tuple:
    """Flag observations Wilson statistics cannot explain, in either tail.

    Parameters
    ----------
    I, sigma : torch.Tensor
        Intensities and their standard deviations, shape ``(n,)``.
    hkl : torch.Tensor
        Miller indices of shape ``(n, 3)``.
    d_spacings : torch.Tensor
        Resolution in Å, shape ``(n,)``.
    cell : torch.Tensor
        Cell parameters ``[a, b, c, alpha, beta, gamma]``, lengths in Å and
        angles in degrees.
    epsilon : torch.Tensor, optional
        Per-reflection multiplicity. None means all ones, which mis-normalises
        the reflections lying on symmetry axes.
    is_centric : torch.Tensor, optional
        Boolean mask, shape ``(n,)``. None treats everything as acentric.
    usable : torch.Tensor, optional
        Boolean mask of rows carrying a real measurement. Rows outside it are
        neither tested nor used to estimate ``Sigma``, and come back kept -- an
        unmeasured reflection is not an outlier, and conflating the two makes
        the reported rate meaningless.
    alpha : float, optional
        Family-wise error rate. Default 0.01: on clean data roughly one
        false rejection per hundred datasets, not per hundred reflections.
    d_max : float, optional
        Only reflections finer than this are tested, in Å. Default 4.0. Below it
        bulk solvent dominates and the shells hold too few reflections, so the
        Wilson distribution is not what the data follow and the test produces
        false positives rather than findings.
    per_shell : int, optional
        Reflections per resolution shell for the ``Sigma`` estimate. Default 250.
    n_passes : int, optional
        Estimate/flag rounds. Default 2: the first pass fits ``Sigma`` and ``U``
        against contaminated data, the second refits with the first pass's
        outliers held out.

    Returns
    -------
    keep : torch.Tensor
        Boolean keep-mask of shape ``(n,)``. True for kept.
    info : dict
        ``n_tested``, ``n_strong``, ``n_weak``, ``log_p_threshold``, ``h_min``
        and the fitted ``U``, for reporting and diagnostics.
    """
    n = I.shape[0]
    # One working dtype for everything. The caller's tensors legitimately differ:
    # observations carry whatever dtype they were read in, while epsilon and the
    # resolutions come from the configured float dtype, and mixing the two makes
    # the anisotropy fit fail on a matmul rather than on anything meaningful.
    dtype, device = I.dtype, I.device
    sigma = sigma.to(dtype=dtype, device=device)
    d_spacings = d_spacings.to(dtype=dtype, device=device)

    if usable is None:
        usable = torch.ones(n, dtype=torch.bool, device=device)
    usable = (
        usable.to(device)
        & torch.isfinite(I)
        & torch.isfinite(sigma)
        & (sigma > 0)
    )
    epsilon = (
        torch.ones_like(I)
        if epsilon is None
        else epsilon.to(dtype=dtype, device=device)
    )
    if is_centric is not None:
        is_centric = is_centric.to(device)

    testable = usable & torch.isfinite(d_spacings) & (d_spacings < d_max)
    keep = torch.ones(n, dtype=torch.bool, device=I.device)
    info = {
        "n_tested": 0,
        "n_strong": 0,
        "n_weak": 0,
        "log_p_threshold": float("-inf"),
        "h_min": float("-inf"),
        "U": torch.zeros(6, dtype=I.dtype, device=I.device),
    }
    n_testable = int(testable.sum())
    if n_testable < per_shell:
        return keep, info

    s_vectors = get_scattering_vectors(
        hkl.to(dtype=dtype, device=device), cell.to(dtype=dtype, device=device)
    )
    log_threshold = math.log(alpha) - math.log(n_testable)
    # Same family-wise rate on the negative side: h is the standardized argument
    # of the same density, so its cut is the corresponding normal quantile.
    h_min = _normal_quantile(alpha / n_testable)

    I_reduced = I / epsilon
    correction = torch.ones_like(I)
    for _ in range(max(1, n_passes)):
        # Every testable reflection is assigned a Sigma; only the ones still kept
        # get a vote in the shell medians and the anisotropy fit, so the second
        # pass estimates against data the first pass has already cleaned.
        estimating = testable & keep
        Sigma_iso = robust_mean_intensity(
            I_reduced / correction, d_spacings, testable, estimating,
            is_centric, per_shell,
        )
        U = fit_anisotropic_scale(
            I_reduced, sigma, Sigma_iso, s_vectors, estimating
        )
        correction = anisotropic_correction(s_vectors, U)
        Sigma_iso = robust_mean_intensity(
            I_reduced / correction, d_spacings, testable, estimating,
            is_centric, per_shell,
        )

        # Wilson predicts E[I(h)] = epsilon(h) * Sigma(d), so the multiplicity
        # divided out for the shell estimate has to come back for the test. Any
        # constant factor in epsilon cancels -- the shell median calibrates
        # Sigma against whatever convention it is on -- but the *relative*
        # enhancement of reflections on symmetry axes does not.
        Sigma = Sigma_iso * correction * epsilon
        valid = testable & torch.isfinite(Sigma) & (Sigma > 0)
        if not bool(valid.any()):
            return keep, info

        log_p = torch.full_like(I, 0.0)
        h = torch.full_like(I, float("inf"))
        log_p[valid] = wilson_log_upper_tail(
            I[valid],
            sigma[valid],
            Sigma[valid],
            is_centric[valid] if is_centric is not None else None,
        )
        h[valid] = french_wilson_h(
            I[valid],
            sigma[valid],
            Sigma[valid],
            is_centric[valid] if is_centric is not None else None,
        )

        strong = valid & (log_p < log_threshold)
        weak = valid & (h < h_min)
        keep = ~(strong | weak)
        info = {
            "n_tested": int(valid.sum()),
            "n_strong": int(strong.sum()),
            "n_weak": int(weak.sum()),
            "log_p_threshold": log_threshold,
            "h_min": h_min,
            "U": U,
        }

    # Untested rows -- unmeasured, or coarser than d_max -- are not outliers.
    return keep | ~testable, info


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF for a scalar probability, via bisection.

    A plain scalar helper: ``torch.special.ndtri`` is unavailable on MPS and this
    is called once per dataset on a Python float, so there is nothing to
    vectorise or accelerate.
    """
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0.0 else float("inf")
    target = math.log(p)
    low, high = -40.0, 10.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        # dtype-ok: deliberate float64 for a scalar CDF; extracted via float()
        value = float(log_normal_cdf(torch.tensor(mid, dtype=torch.float64)))
        if value < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


__all__ = [
    "anisotropic_correction",
    "fit_anisotropic_scale",
    "log_normal_cdf",
    "robust_mean_intensity",
    "wilson_log_upper_tail",
    "wilson_outlier_mask",
]
