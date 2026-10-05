"""
French-Wilson conversion of merged intensities to amplitudes.

Reference: French, S. & Wilson, K. (1978). Acta Cryst. A34, 517-525.

The conversion has two parts:

- **The prior.** :func:`fit_mean_intensity` fits the expected intensity of every
  reflection, ``epsilon Sigma(h)``: a smooth radial curve in resolution, an
  ellipsoidal anisotropy restricted to what the Laue class allows, and the
  multiplicity ``epsilon``. It is a fit rather than a shell average, so it is
  positive everywhere and has no shell edges.
- **The posterior.** :func:`french_wilson` turns ``I``, ``sigma_I`` and the
  prior into posterior amplitudes: French and Wilson's tables, their expansion
  for large ``h``, and the corresponding series for ``h`` below the tables, so
  every reflection with a prior has a posterior. A weak reflection in a
  region where the prior is far below the noise is shrunk towards the prior,
  not discarded. :func:`french_wilson_valid_mask` rejects only intensities too
  negative for their own sigma.

:func:`french_wilson_auto` does both from ``hkl``, ``d`` and a space group.
:func:`french_wilson_h` is the standardized argument shared by the posterior and
the Wilson outlier test in :mod:`torchref.base.wilson_outliers`.

Usage::

        from torchref.base.french_wilson import french_wilson_auto

        F, sigma_F, valid = french_wilson_auto(
            I, sigma_I, hkl, d_spacings, space_group='P212121'
        )

These are plain functions on purpose: the conversion runs once per dataset, and
a cached estimator holding per-row buffers goes stale the moment the rows are
reordered (as ``ReflectionData`` canonicalization does).
"""

import math

import torch

from torchref.config import get_float_dtype, get_int_dtype
from torchref.symmetry import SpaceGroup, SpaceGroupLike

#: B-spline coefficients in ``log Sigma``. Enough to follow the low-resolution
#: solvent deficit and the shoulder near 4-5 Å to within the scatter of shell
#: means, which fewer smooth away; more start to follow that scatter.
DEFAULT_N_COEFF = 16
#: Fitted reflections per coefficient; a small dataset gets a stiffer curve
#: rather than a noisy one.
_ROWS_PER_COEFF = 400
#: A reflection whose sigma exceeds this multiple of the typical sigma at its
#: resolution is weighted by its own sigma. Below it every reflection counts the
#: same; the spread of sigmas within a resolution range, which follows the
#: intensity, stays well below it.
_NOISY_SIGMA_RATIO = 10.0
#: Step halvings allowed per Fisher-scoring iteration before the fit is taken
#: as converged, and the iteration cap, a runaway guard that should never bind.
_MAX_HALVINGS = 30
_MAX_ITER = 200
#: The fit has converged once no step moves ``log Sigma`` by more than this at
#: any reflection where ``Sigma`` still matters (see :func:`_fisher_scoring`).
_CURVE_TOL = 1e-4
#: Floor on ``Sigma`` below the typical measurement error, as a log. Far below
#: anything that changes a French-Wilson result -- every reflection is rejected
#: long before it -- and it keeps ``Sigma`` representable and positive where a
#: pure-noise region drives the fit towards zero.
_MAX_LOG_SIGMA_RATIO = 30.0

#: The posterior tables start at h = -4 and step by 0.1. Above the asymptote
#: for its class French and Wilson's expansion replaces the table; below the
#: origin, the series in 1/h^2 of :func:`_posterior_below_tables`.
_TABLE_ORIGIN = -4.0
_ACENTRIC_ASYMPTOTE = 3.0
_CENTRIC_ASYMPTOTE = 4.0
#: Terms of that series. Four keep it within 0.2% of the posterior mean and
#: 1% of its standard deviation at h = -4, and it improves fast below.
_SERIES_TERMS = 4
#: Default lower cut on I/sigma_I: no true intensity J >= 0 makes a measurement
#: this far below zero plausible under its own sigma.
DEFAULT_MIN_I_OVER_SIGMA = -3.7

# Posterior mean and standard deviation of F / sqrt(sigma_I) as a function of h,
# from the French-Wilson (1978) supplement.
_ACENTRIC_F = torch.tensor(
    [
        0.423,
        0.428,
        0.432,
        0.437,
        0.442,
        0.447,
        0.453,
        0.458,
        0.464,
        0.469,
        0.475,
        0.482,
        0.488,
        0.495,
        0.502,
        0.509,
        0.516,
        0.524,
        0.532,
        0.540,
        0.549,
        0.557,
        0.567,
        0.576,
        0.586,
        0.597,
        0.608,
        0.619,
        0.631,
        0.643,
        0.656,
        0.670,
        0.684,
        0.699,
        0.714,
        0.730,
        0.747,
        0.765,
        0.783,
        0.802,
        0.822,
        0.843,
        0.865,
        0.887,
        0.911,
        0.935,
        0.960,
        0.987,
        1.014,
        1.042,
        1.070,
        1.100,
        1.130,
        1.161,
        1.192,
        1.224,
        1.257,
        1.289,
        1.322,
        1.355,
        1.388,
        1.421,
        1.454,
        1.487,
        1.519,
        1.551,
        1.583,
        1.615,
        1.646,
        1.676,
        1.706,
    ],
    dtype=get_float_dtype(),
)

_ACENTRIC_SIGMA_F = torch.tensor(
    [
        0.216,
        0.218,
        0.220,
        0.222,
        0.224,
        0.226,
        0.229,
        0.231,
        0.234,
        0.236,
        0.239,
        0.241,
        0.244,
        0.247,
        0.250,
        0.253,
        0.256,
        0.259,
        0.262,
        0.266,
        0.269,
        0.272,
        0.276,
        0.279,
        0.283,
        0.287,
        0.291,
        0.295,
        0.298,
        0.302,
        0.307,
        0.311,
        0.315,
        0.319,
        0.324,
        0.328,
        0.332,
        0.337,
        0.341,
        0.345,
        0.349,
        0.353,
        0.357,
        0.360,
        0.364,
        0.367,
        0.369,
        0.372,
        0.374,
        0.375,
        0.376,
        0.377,
        0.377,
        0.377,
        0.376,
        0.374,
        0.372,
        0.369,
        0.366,
        0.362,
        0.358,
        0.353,
        0.348,
        0.343,
        0.338,
        0.332,
        0.327,
        0.321,
        0.315,
        0.310,
        0.304,
    ],
    dtype=get_float_dtype(),
)

_CENTRIC_F = torch.tensor(
    [
        0.269,
        0.272,
        0.276,
        0.279,
        0.282,
        0.286,
        0.289,
        0.293,
        0.297,
        0.301,
        0.305,
        0.309,
        0.314,
        0.318,
        0.323,
        0.328,
        0.333,
        0.339,
        0.344,
        0.350,
        0.356,
        0.363,
        0.370,
        0.377,
        0.384,
        0.392,
        0.400,
        0.409,
        0.418,
        0.427,
        0.438,
        0.448,
        0.460,
        0.471,
        0.484,
        0.498,
        0.512,
        0.527,
        0.543,
        0.560,
        0.578,
        0.597,
        0.618,
        0.639,
        0.662,
        0.687,
        0.713,
        0.740,
        0.769,
        0.800,
        0.832,
        0.866,
        0.901,
        0.938,
        0.976,
        1.016,
        1.057,
        1.098,
        1.140,
        1.183,
        1.227,
        1.270,
        1.313,
        1.356,
        1.398,
        1.439,
        1.480,
        1.519,
        1.558,
        1.595,
        1.632,
        1.667,
        1.701,
        1.735,
        1.767,
        1.799,
        1.829,
        1.859,
        1.889,
        1.917,
        1.945,
    ],
    dtype=get_float_dtype(),
)

_CENTRIC_SIGMA_F = torch.tensor(
    [
        0.203,
        0.205,
        0.207,
        0.209,
        0.211,
        0.214,
        0.216,
        0.219,
        0.222,
        0.224,
        0.227,
        0.230,
        0.233,
        0.236,
        0.239,
        0.243,
        0.246,
        0.250,
        0.253,
        0.257,
        0.261,
        0.265,
        0.269,
        0.273,
        0.278,
        0.283,
        0.288,
        0.293,
        0.298,
        0.303,
        0.309,
        0.314,
        0.320,
        0.327,
        0.333,
        0.340,
        0.346,
        0.353,
        0.361,
        0.368,
        0.375,
        0.383,
        0.390,
        0.398,
        0.405,
        0.413,
        0.420,
        0.427,
        0.433,
        0.440,
        0.445,
        0.450,
        0.454,
        0.457,
        0.459,
        0.460,
        0.460,
        0.458,
        0.455,
        0.451,
        0.445,
        0.438,
        0.431,
        0.422,
        0.412,
        0.402,
        0.392,
        0.381,
        0.370,
        0.360,
        0.349,
        0.339,
        0.330,
        0.321,
        0.312,
        0.304,
        0.297,
        0.290,
        0.284,
        0.278,
        0.272,
    ],
    dtype=get_float_dtype(),
)


def _interpolate(h: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Linear interpolation in a posterior table at ``h``, clamped to its range."""
    table = table.to(device=h.device, dtype=h.dtype)
    position = torch.clamp(10.0 * (h - _TABLE_ORIGIN), 0.0, len(table) - 1.001)
    lower = position.floor()
    weight = position - lower
    index = lower.to(get_int_dtype())
    return (1.0 - weight) * table[index] + weight * table[index + 1]


def french_wilson_h(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor | bool | None = None,
) -> torch.Tensor:
    """
    The French-Wilson normalized parameter ``h``.

    ``h`` is not merely an interpolation coordinate for the lookup tables: it is
    the standardized argument of the Wilson-predictive density of the
    *observation*. Convolving the acentric Wilson prior
    ``P(J) = (1/S)exp(-J/S)`` with the Gaussian measurement error
    ``I|J ~ N(J, sigma^2)`` gives

        p(I) = (1/S) exp(sigma^2/(2 S^2) - I/S) Phi(I/sigma - sigma/S)

    whose ``Phi`` argument is exactly the acentric ``h`` below. The centric prior
    ``J^(-1/2) exp(-J/2S)`` yields the factor of two. ``h`` is not itself a
    tail probability: where ``sigma/S`` is large the exponential factor
    compensates for ``Phi``, and an observation that is plain noise has a very
    negative ``h`` and an ordinary ``p(I)``. That is why nothing here is
    rejected on ``h``.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities (any shape).
    sigma_I : torch.Tensor
        Standard deviations of intensities (same shape as I).
    mean_intensity : torch.Tensor
        Expected intensity of each reflection under the Wilson prior (same
        shape as I). Must be positive: there is no Wilson prior with a mean at
        or below zero.
    is_centric : torch.Tensor or bool, optional
        Boolean mask of centric reflections, or a plain ``bool`` for an input
        that is all one or the other. If None, all are treated as acentric.

    Returns
    -------
    torch.Tensor
        ``h`` for each reflection (same shape as I). NaN wherever
        ``mean_intensity`` is not positive: such a row has no posterior.
    """
    # A centric reflection's prior has twice the variance per degree of freedom,
    # which halves the sigma/S penalty.
    if is_centric is None or is_centric is False:
        denom = mean_intensity
    elif is_centric is True:
        denom = 2.0 * mean_intensity
    else:
        denom = torch.where(is_centric, 2.0 * mean_intensity, mean_intensity)
    h = (I / sigma_I) - (sigma_I / denom)
    # With S < 0 the -sigma/S term changes sign and turns an observation that no
    # Wilson reflection could explain into an apparently strong one.
    return torch.where(denom > 0, h, torch.full_like(h, float("nan")))


def _keep(
    h: torch.Tensor, I: torch.Tensor, sigma_I: torch.Tensor, min_i_over_sigma: float
) -> torch.Tensor:
    """The rejection rule on an already computed ``h``."""
    # A non-finite h means sigma_I was degenerate or the prior mean was not a
    # positive number; such a reflection has no posterior and must not be kept
    # on the strength of a NaN comparison (which is False anyway, but not by
    # intent).
    return torch.isfinite(h) & (I / sigma_I >= min_i_over_sigma)


def french_wilson_valid_mask(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor | bool | None = None,
    min_i_over_sigma: float = DEFAULT_MIN_I_OVER_SIGMA,
) -> torch.Tensor:
    """
    Which reflections French-Wilson converts, as a keep-mask.

    A reflection is kept when it has a posterior -- a positive prior mean and
    a usable sigma -- and its intensity is not too negative for its own sigma:
    ``I/sigma_I >= min_i_over_sigma``. That cut asks whether *any* true
    intensity ``J >= 0`` could have produced the measurement, so it does not
    depend on the prior. A weak reflection whose prior lies far below the
    noise is kept; its posterior is shrunk towards the prior.

    Parameters
    ----------
    I, sigma_I, mean_intensity, is_centric
        As for :func:`french_wilson_h`.
    min_i_over_sigma : float, optional
        Lower cut on ``I/sigma_I``. Default -3.7, a one-sided probability of
        about 1e-4 for a correct sigma.

    Returns
    -------
    torch.Tensor
        Boolean keep-mask (same shape as I).
    """
    h = french_wilson_h(I, sigma_I, mean_intensity, is_centric)
    return _keep(h, I, sigma_I, min_i_over_sigma)


def _series_coefficients(shift: float) -> dict[float, list[float]]:
    """Coefficients of ``N_nu(h) = sum_k c_k / h^(2k)`` for nu = 0, 1/2, 1.

    ``N_nu`` is the asymptotic series of ``int_0^inf u^(nu + shift) e^-u
    e^(-u^2/(2h^2)) du``; ``shift`` is 0 acentric and -1/2 centric.
    """
    return {
        nu: [
            (-1.0) ** k
            * math.gamma(nu + shift + 2 * k + 1)
            / (math.factorial(k) * 2.0**k)
            for k in range(_SERIES_TERMS + 1)
        ]
        for nu in (0.0, 0.5, 1.0)
    }


_ACENTRIC_SERIES = _series_coefficients(0.0)
_CENTRIC_SERIES = _series_coefficients(-0.5)


def _posterior_below_tables(
    h: torch.Tensor, centric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior mean and standard deviation of ``F / sqrt(sigma_I)`` for h < -4.

    The posterior of ``t = J / sigma_I`` is ``exp(-(t - h)^2 / 2)`` on ``t >= 0``
    (centric: times ``t^(-1/2)``). With ``t = u / |h|`` its moments are
    ``|h|^-nu N_nu / N_0``, and the factor ``e^(-u^2/(2h^2))`` expands in
    ``1/h^2``; the leading term is the exponential posterior of a reflection
    whose prior lies far below the noise. ``h`` must be at most -4.
    """
    coefficients = _CENTRIC_SERIES if centric else _ACENTRIC_SERIES
    x = 1.0 / (h * h)

    def N(nu):
        total = torch.zeros_like(h)
        for c in reversed(coefficients[nu]):
            total = total * x + c
        return total

    root = torch.rsqrt(-h)
    zero, half, one = N(0.0), N(0.5), N(1.0)
    mean = half / zero
    variance = one / zero - mean * mean
    return root * mean, root * torch.sqrt(variance)


def _posterior_amplitude(
    h: torch.Tensor, sigma_I: torch.Tensor, centric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior mean and standard deviation of F for one centricity class.

    ``h`` must be finite: the series below the tables, the tables, or French
    and Wilson's expansion above them, by ``h``.
    """
    if centric:
        mean_table, sd_table, asymptote = (
            _CENTRIC_F,
            _CENTRIC_SIGMA_F,
            _CENTRIC_ASYMPTOTE,
        )
    else:
        mean_table, sd_table, asymptote = (
            _ACENTRIC_F,
            _ACENTRIC_SIGMA_F,
            _ACENTRIC_ASYMPTOTE,
        )
    root = torch.sqrt(sigma_I)
    F = torch.empty_like(h)
    sigma_F = torch.empty_like(h)

    below = h < _TABLE_ORIGIN
    if bool(below.any()):
        mean, sd = _posterior_below_tables(h[below], centric)
        F[below] = mean * root[below]
        sigma_F[below] = sd * root[below]

    tabulated = ~below & (h < asymptote)
    if bool(tabulated.any()):
        F[tabulated] = _interpolate(h[tabulated], mean_table) * root[tabulated]
        sigma_F[tabulated] = _interpolate(h[tabulated], sd_table) * root[tabulated]

    large = h >= asymptote
    if bool(large.any()):
        h_large, root_large = h[large], root[large]
        if centric:
            # French and Wilson's expansion with the h^-6 term cctbx adds.
            h2 = 1.0 / (h_large * h_large)
            h4 = h2 * h2
            h6 = h2 * h4
            F[large] = (
                torch.sqrt(h_large)
                * (
                    1.0
                    - (3.0 / 8.0) * h2
                    - (87.0 / 128.0) * h4
                    - (2889.0 / 1024.0) * h6
                )
                * root_large
            )
            sigma_F[large] = (
                torch.sqrt(
                    h_large
                    * ((1.0 / 4.0) * h2 + (15.0 / 32.0) * h4 + (273.0 / 128.0) * h6)
                )
                * root_large
            )
        else:
            F_large = torch.sqrt(h_large) * root_large
            F[large] = F_large
            sigma_F[large] = 0.5 * root_large * root_large / F_large
    return F, sigma_F


def french_wilson(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor | None = None,
    min_i_over_sigma: float = DEFAULT_MIN_I_OVER_SIGMA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    French-Wilson posterior amplitudes from intensities and their prior.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities of shape (...).
    sigma_I : torch.Tensor
        Standard deviations of intensities of shape (...).
    mean_intensity : torch.Tensor
        Expected intensity of each reflection under the Wilson prior, of shape
        (...), e.g. from :func:`fit_mean_intensity`.
    is_centric : torch.Tensor, optional
        Boolean mask of centric reflections of shape (...). If None, all are
        treated as acentric.
    min_i_over_sigma : float, optional
        Lower cut on ``I/sigma_I``, as for :func:`french_wilson_valid_mask`.
        Default -3.7.

    Returns
    -------
    F : torch.Tensor
        Posterior mean amplitudes of shape (...). Computed for rejected rows
        too, so the caller decides what to do with them; NaN where the prior
        mean is not positive or ``sigma_I`` is degenerate, which have no
        posterior.
    sigma_F : torch.Tensor
        Posterior standard deviations of shape (...), NaN where ``F`` is.
    valid_mask : torch.Tensor
        :func:`french_wilson_valid_mask`, shape (...).
    """
    h = french_wilson_h(I, sigma_I, mean_intensity, is_centric)
    centric = (
        torch.zeros_like(I, dtype=torch.bool)
        if is_centric is None
        else is_centric.to(torch.bool)
    )
    F = torch.full_like(I, float("nan"))
    sigma_F = torch.full_like(I, float("nan"))
    defined = torch.isfinite(h)
    for flag in (False, True):
        rows = defined & (centric == flag)
        if bool(rows.any()):
            F[rows], sigma_F[rows] = _posterior_amplitude(
                h[rows], sigma_I[rows], centric=flag
            )
    return F, sigma_F, _keep(h, I, sigma_I, min_i_over_sigma)


def intensities_from_amplitudes(
    F: torch.Tensor, sigma_F: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate intensities from amplitudes, for datasets that supply only F.

    Uses ``I = F^2`` and the delta-method ``sigma_I = 2 F sigma_F``.

    **This is not an inverse of French-Wilson.** In the acentric asymptotic
    branch the conversion satisfies ``F^2 = h sigma_I = I - sigma_I^2/S``, and
    the table-driven branch has no closed form at all, so the round trip is
    lossy in a specific and unavoidable direction: French-Wilson output ``F`` is
    a strictly positive posterior mean, so every trace of a negative intensity
    is gone.

    The consequence for outlier detection is that on an amplitude-only dataset
    the French-Wilson guard cannot detect an inexplicably negative intensity --
    only an absurd ``sigma_F``. That is a property of amplitudes as input, not a
    deficiency of this function; nothing can recover information the posterior
    mean discarded.

    Parameters
    ----------
    F : torch.Tensor
        Structure factor amplitudes.
    sigma_F : torch.Tensor
        Their standard deviations (same shape as F).

    Returns
    -------
    I : torch.Tensor
        ``F**2``.
    sigma_I : torch.Tensor
        ``2 * F * sigma_F``. Zero wherever ``F`` or ``sigma_F`` is non-positive
        or non-finite; callers must treat a non-positive ``sigma_I`` as
        "no usable measurement" rather than dividing by it.
    """
    usable = torch.isfinite(F) & torch.isfinite(sigma_F) & (F > 0) & (sigma_F > 0)
    I = torch.where(usable, F * F, torch.zeros_like(F))
    sigma_I = torch.where(usable, 2.0 * F * sigma_F, torch.zeros_like(F))
    return I, sigma_I


def _bspline(x: torch.Tensor, n: int) -> torch.Tensor:
    """Clamped B-spline basis on [-1, 1], shape (len(x), n).

    Degree ``min(3, n - 1)`` with uniformly spaced knots, so ``n <= 4`` is the
    Bernstein polynomial basis. The functions sum to one everywhere, so equal
    coefficients give a constant curve.
    """
    degree = min(3, n - 1)
    inner = torch.linspace(-1.0, 1.0, n - degree + 1, dtype=x.dtype, device=x.device)
    knots = torch.cat([inner[:1].repeat(degree), inner, inner[-1:].repeat(degree)])
    x = torch.clamp(x, -1.0, 1.0).unsqueeze(1)
    basis = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
    # The intervals are half-open; x = 1 belongs to the last non-empty one.
    basis[:, n - 1] = torch.where(x[:, 0] >= 1.0, 1.0, basis[:, n - 1])

    def ratio(num, den):
        return torch.where(den > 0, num / torch.where(den > 0, den, 1.0), 0.0)

    for k in range(1, degree + 1):
        m = basis.shape[1] - 1
        lo, lo_next = knots[:m], knots[1 : 1 + m]
        hi, hi_next = knots[k : k + m], knots[k + 1 : k + 1 + m]
        left = ratio(x - lo, hi - lo)
        right = ratio(hi_next - x, hi_next - lo_next)
        basis = left * basis[:, :-1] + right * basis[:, 1:]
    return basis


def _solve_normal(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve the symmetric positive definite system ``A x = b`` for a vector ``b``.

    Cholesky with two triangular solves, because MPS implements neither
    ``lu_solve`` nor ``cholesky_solve``. A ridge proportional to the largest
    diagonal element keeps it posed when a basis direction carries almost no
    weight; if the factorisation still fails, a general solve takes over.
    """
    A = A + 1e-10 * torch.diagonal(A).max() * torch.eye(
        A.shape[0], dtype=A.dtype, device=A.device
    )
    rhs = b.unsqueeze(-1)
    chol = torch.linalg.cholesky_ex(A)
    if int(chol.info) != 0:
        return torch.linalg.solve(A, rhs).squeeze(-1)
    y = torch.linalg.solve_triangular(chol.L, rhs, upper=False)
    return torch.linalg.solve_triangular(chol.L.mT, y, upper=True).squeeze(-1)


def _anisotropy_design(
    hkl: torch.Tensor,
    space_group: SpaceGroupLike,
    radial: torch.Tensor,
    s2: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Direction-only quadratic forms in the Miller indices the Laue group allows.

    An ellipsoidal fall-off ``exp(-2 pi^2 s^T U s)`` is ``exp(h^T M h)`` with
    the cell folded into ``M``, so it needs only ``hkl``. Averaging ``h_i h_j``
    over the symmetry copies of each reflection leaves the forms the Laue group
    allows. Each then loses, by least squares over ``rows``, whatever the
    radial curve or ``s^2`` itself can represent -- ``s^2 = h^T G* h`` is the
    isotropic form, which the radial curve does not reproduce exactly at low
    resolution. What is left is direction only, and nothing in the fit
    duplicates the radial columns. The result is orthonormal over ``rows``.

    How many directions are left is fixed by symmetry, not by the data: it is
    the dimension of the Laue-invariant quadratic forms, the trace of the
    averaging operator over the group's rotations, less the isotropic one -- 5
    for triclinic down to 0 for cubic. Reading it off the data instead would
    let rounding in a nearly isotropic direction pass for anisotropy.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices, shape (n, 3).
    space_group : SpaceGroupLike
        The data's space group.
    radial : torch.Tensor
        The radial basis evaluated at every row, shape (n, m).
    s2 : torch.Tensor
        ``1/d^2`` in Å⁻², shape (n,), zero where ``d`` is not finite.
    rows : torch.Tensor
        Boolean mask of shape (n,) of the rows the fit uses.

    Returns
    -------
    torch.Tensor
        Shape (n, r) with ``r`` between 0 and 5.
    """
    dtype = radial.dtype
    group = SpaceGroup(space_group, device=hkl.device)
    copies, _, _, _ = group.equivalent_hkl(
        hkl, include_friedel=False, device=hkl.device
    )

    # Rotations acting on h, recovered as the images of the unit vectors. The
    # averaging operator maps M to mean(R M R^T); in the coordinates
    # (M00, M11, M22, M01, M02, M12) its trace counts the invariant forms.
    unit = torch.eye(3, dtype=hkl.dtype, device=hkl.device)
    images, _, _, _ = group.equivalent_hkl(
        unit, include_friedel=False, device=hkl.device
    )
    R = images.to(dtype).cpu().reshape(-1, 3, 3)
    trace = 0.0
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        E = torch.zeros(3, 3, dtype=dtype)
        E[i, j] = E[j, i] = 1.0
        trace += float((R @ E @ R.mT).mean(0)[i, j])
    n_directions = round(trace) - 1
    if n_directions < 1:
        return torch.zeros(hkl.shape[0], 0, dtype=dtype, device=hkl.device)
    n_ops = copies.shape[0] // hkl.shape[0]
    h = copies.to(dtype).reshape(n_ops, hkl.shape[0], 3)
    forms = torch.stack(
        [
            h[..., 0] * h[..., 0],
            h[..., 1] * h[..., 1],
            h[..., 2] * h[..., 2],
            2.0 * h[..., 0] * h[..., 1],
            2.0 * h[..., 0] * h[..., 2],
            2.0 * h[..., 1] * h[..., 2],
        ],
        dim=-1,
    ).mean(0)
    # Unit scale before any sums: raw h_i h_j reach 1e4, and their squares
    # summed over 1e5 rows lose the digits that separate the anisotropic part
    # from the isotropic one in float32.
    forms = forms / forms[rows].pow(2).mean(0).sqrt().clamp(min=1e-30)

    # Two projections in turn rather than one onto [radial, s^2]: s^2 is nearly
    # a radial function, and the joint normal equations lose in float32 the
    # very difference that is being kept.
    on_rows = radial[rows]
    gram = on_rows.T @ on_rows

    def off_radial(columns):
        coefficients = torch.stack(
            [_solve_normal(gram, on_rows.T @ col[rows]) for col in columns.T], dim=1
        )
        return columns - radial @ coefficients

    residual = off_radial(forms)
    s2_off = off_radial((s2 / s2[rows].pow(2).mean().sqrt()).unsqueeze(1))[:, 0]
    w = s2_off[rows]
    residual = residual - s2_off.unsqueeze(1) * (
        (residual[rows].T @ w) / (w @ w).clamp(min=1e-30)
    )

    # A 6x6 eigenproblem, solved on the host: it is tiny, and MPS has no eigh.
    # The leading directions are the anisotropic ones; what follows them is
    # what the symmetry or the projection removed, down to rounding.
    fitted = residual[rows]
    values, vectors = torch.linalg.eigh((fitted.T @ fitted / fitted.shape[0]).cpu())
    values, vectors = values[-n_directions:], vectors[:, -n_directions:]
    basis = (vectors / values.clamp(min=1e-30).sqrt()).to(device=hkl.device)
    return residual @ basis


def _fisher_scoring(
    design: torch.Tensor,
    I: torch.Tensor,
    offset: torch.Tensor,
    kappa: torch.Tensor,
    log_noise: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    """Coefficients of ``log Sigma`` maximising :func:`fit_mean_intensity`'s
    quasi-likelihood, from the starting ``coeff``.

    The model for row ``i`` is mean ``exp(design_i c + offset_i)`` and variance
    ``kappa_i mean^2 + noise_i^2``. Fisher scoring rather than a generic
    optimiser: each step is scaled by the information in every direction, so
    weakly determined directions get full-sized steps instead of stalling.
    """
    noise = torch.exp(log_noise)
    variance = noise * noise
    eps = torch.finfo(I.dtype).eps
    root_kappa = kappa.sqrt()
    log_kappa = kappa.log()

    # The negated quasi-likelihood, integral of (I - t)/(k t^2 + v) from the
    # observation to the mean, constant dropped. atan2(noise, sqrt(k) mean)
    # rather than pi/2 - atan(sqrt(k) mean/noise) keeps the strong end, where
    # it is the familiar exponential term I/mean, free of cancellation, and it
    # stays bounded as the mean goes to zero. Also returned: the rounding error
    # of the sum, below which two values do not differ.
    def objective(c):
        eta = design @ c + offset
        terms = (I / (root_kappa * noise)) * torch.atan2(
            noise, root_kappa * torch.exp(eta)
        ) + (0.5 / kappa) * torch.logaddexp(2.0 * eta + log_kappa, 2.0 * log_noise)
        return float(terms.sum()), eta, eps * float(terms.abs().sum())

    loss, eta, rounding = objective(coeff)
    # A step that raises the objective by more than its rounding error is
    # halved; a float32 sum cannot resolve the last steps, and refusing them
    # would stop the fit short in exactly the weakly determined directions.
    # Convergence is judged on the curve where the mean is at least a hundredth
    # of the noise: below that every reflection is rejected whatever the mean
    # is, and a region of pure noise keeps sliding towards zero long after the
    # rest has settled.
    for _ in range(_MAX_ITER):
        mu = torch.exp(eta)
        total = kappa * mu * mu + variance
        weight = mu * mu / total
        score = mu * (I - mu) / total
        step = _solve_normal(
            design.T @ (design * weight.unsqueeze(1)), design.T @ score
        )
        for _ in range(_MAX_HALVINGS):
            trial_loss, trial_eta, trial_rounding = objective(coeff + step)
            if trial_loss <= loss + rounding:
                break
            step = step * 0.5
        else:
            break
        live = mu > 0.01 * noise
        moved = float((trial_eta - eta)[live].abs().max()) if bool(live.any()) else 0.0
        coeff, loss, eta, rounding = coeff + step, trial_loss, trial_eta, trial_rounding
        if moved < _CURVE_TOL:
            break
    return coeff


def fit_mean_intensity(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    d_spacings: torch.Tensor,
    fit_mask: torch.Tensor | None = None,
    n_coeff: int = DEFAULT_N_COEFF,
    *,
    hkl: torch.Tensor | None = None,
    space_group: SpaceGroupLike | None = None,
    is_centric: torch.Tensor | None = None,
    epsilon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Wilson mean intensity ``Sigma`` per reflection, as a smooth positive curve.

    ``log Sigma`` is a cubic B-spline in ``s^3 = 1/d^3`` with evenly spaced
    knots. Reflections are spread evenly over ``s^3``, so the knots are
    equal-count, as shells are; each basis function has local support, so
    neither end of the curve rests on a handful of reflections the way the end
    of a single high-degree polynomial does.

    It is fitted by quasi-likelihood with the first two moments of an acentric
    Wilson intensity measured with error, ``E[I] = Sigma`` and
    ``Var[I] = Sigma^2 + sbar^2``, where ``sbar(s)`` is the typical measurement
    error at that resolution, a smooth least-squares fit to ``log sigma_I``.
    The estimating equation,

        sum_h  x_h (I_h - Sigma_h) Sigma_h / (Sigma_h^2 + sbar_h^2) = 0,

    makes ``Sigma`` a resolution-local mean intensity, as a shell mean is, but:

    - ``Sigma > 0`` everywhere. Where the local mean is zero or below, as in a
      region of pure noise, the weight vanishes with ``Sigma`` and the fit
      settles towards zero instead of crossing it.
    - There are no shell edges, and one reflection's pull is spread over the
      support of its basis functions instead of landing on its own shell.
    - Reflections at a given resolution get the same weight. Per-reflection
      sigmas are not used for weighting because they correlate with the
      intensity -- counting statistics make strong reflections noisier, and
      merging can understate the error of rarely measured ones -- and
      weighting by them would bias the mean in whichever direction that
      correlation runs. The exception is a reflection more than ten times
      noisier than typical, which takes its own sigma in place of ``sbar``
      and so cannot drag the curve with an intensity that is mostly noise.

    Given ``hkl`` and ``space_group``, ``log Sigma`` also carries an
    ellipsoidal anisotropy, ``h^T M h`` with ``M`` restricted to what the Laue
    group allows, its isotropic part left to the radial curve (see
    :func:`_anisotropy_design`): at most five parameters, each informed by
    every reflection. Without them the curve is isotropic.

    Given ``epsilon``, a reflection's expected intensity is ``epsilon Sigma``:
    on a symmetry element the copies of each atom that map ``h`` onto itself
    scatter in phase, so the symmetry concentrates the same total intensity on
    fewer, stronger reflections. It enters the fit as a fixed offset on
    ``log Sigma``, and the result is returned multiplied by it.

    Parameters
    ----------
    I : torch.Tensor
        Intensities of shape (n,), any sign.
    sigma_I : torch.Tensor
        Their standard deviations, shape (n,). Rows whose ``I`` or ``sigma_I``
        is not finite, or whose ``sigma_I`` is not positive, do not inform the
        fit.
    d_spacings : torch.Tensor
        Resolution in Å, shape (n,).
    fit_mask : torch.Tensor, optional
        Boolean mask of shape (n,) selecting the rows that inform the fit, e.g.
        the acentric ones. Every row with a finite ``d`` receives a ``Sigma``
        regardless.
    n_coeff : int, optional
        B-spline coefficients. Reduced to one per 400 fitted reflections, so a
        small dataset gets a stiffer curve; ``1`` gives a single mean
        intensity.
    hkl : torch.Tensor, optional
        Miller indices of shape (n, 3). With ``space_group``, makes the prior
        anisotropic.
    space_group : str, int, or gemmi.SpaceGroup, optional
        The data's space group; see ``hkl``.
    is_centric : torch.Tensor, optional
        Boolean mask of shape (n,). Centric rows that inform the fit are given
        the centric variance, ``2 Sigma^2 + sbar^2``; their mean is the same
        ``Sigma``. None treats every row as acentric.
    epsilon : torch.Tensor, optional
        Multiplicity of shape (n,): the number of symmetry operations mapping
        ``h`` onto itself, Friedel mates not counted (``SpaceGroup.epsilon``
        with ``friedel=False``). A factor common to every row, such as a
        lattice centring, cancels in the result. None means 1 everywhere.

    Returns
    -------
    torch.Tensor
        The expected intensity ``epsilon Sigma`` of each reflection, shape
        (n,), positive; NaN where ``d`` is not finite, and
        everywhere if no row can inform the fit. Detached from autograd: a
        fitted constant, not a function of ``I`` that gradients flow through.
        The fit is Fisher scoring, which reads its objective back to the host
        on every step.
    """
    I = I.detach()
    sigma_I = sigma_I.detach()
    s3 = 1.0 / (d_spacings.detach() ** 3)
    expected = torch.full_like(I, float("nan"))

    placed = torch.isfinite(s3)
    usable = placed & torch.isfinite(I) & torch.isfinite(sigma_I) & (sigma_I > 0)
    if fit_mask is not None:
        usable = usable & fit_mask.to(torch.bool)
    n_fit = int(usable.sum())
    if n_fit == 0:
        return expected

    # The basis spans every row that will receive a Sigma, not only the fitted
    # ones, so nothing is evaluated outside the fitted range.
    lo, hi = s3[placed].min(), s3[placed].max()
    if hi > lo:
        x = 2.0 * (s3 - lo) / (hi - lo) - 1.0
        n_terms = max(1, min(int(n_coeff), n_fit // _ROWS_PER_COEFF))
    else:
        x = torch.zeros_like(s3)
        n_terms = 1
    radial = torch.zeros(len(I), n_terms, dtype=I.dtype, device=I.device)
    radial[placed] = _bspline(x[placed], n_terms)

    log_epsilon = torch.zeros_like(I)
    if epsilon is not None:
        log_epsilon = torch.log(epsilon.detach().to(I.dtype).clamp(min=1.0))
    kappa = torch.ones_like(I)
    if is_centric is not None:
        kappa = torch.where(is_centric.to(torch.bool), 2.0, kappa)

    log_sigma = torch.log(sigma_I[usable])
    fitted_radial = radial[usable]
    log_sbar = fitted_radial @ _solve_normal(
        fitted_radial.T @ fitted_radial, fitted_radial.T @ log_sigma
    )
    log_noise = torch.maximum(log_sbar, log_sigma - math.log(_NOISY_SIGMA_RATIO))

    design = radial
    if hkl is not None and space_group is not None:
        s2 = torch.where(placed, 1.0 / (d_spacings.detach() ** 2), 0.0).to(I.dtype)
        design = torch.cat(
            [radial, _anisotropy_design(hkl, space_group, radial, s2, usable)], dim=1
        )

    # Start from a flat curve at the mean intensity, or at the noise level when
    # the data average to nothing; only the scale of the start matters. Equal
    # B-spline coefficients are a constant curve.
    offset = log_epsilon[usable]
    level = math.log(
        max(float((I[usable] / offset.exp()).mean()), float(log_noise.exp().median()))
    )
    start = torch.zeros(design.shape[1], dtype=I.dtype, device=I.device)
    start[:n_terms] = level
    coeff = _fisher_scoring(
        design[usable], I[usable], offset, kappa[usable], log_noise, start
    )

    log_floor = float(log_sbar.median()) - _MAX_LOG_SIGMA_RATIO
    log_Sigma = torch.clamp(design[placed] @ coeff, min=log_floor)
    expected[placed] = torch.exp(log_Sigma + log_epsilon[placed])
    return expected


def french_wilson_auto(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    hkl: torch.Tensor,
    d_spacings: torch.Tensor,
    space_group: SpaceGroupLike = "P1",
    min_i_over_sigma: float = DEFAULT_MIN_I_OVER_SIGMA,
    *,
    n_coeff: int = DEFAULT_N_COEFF,
    exclude_from_fit: torch.Tensor | None = None,
    anisotropic: bool = True,
    epsilon: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert intensities to amplitudes, fitting the prior from the data.

    Every per-reflection input must be row-aligned: the prior and centric flag
    of row ``i`` are taken from ``hkl[i]`` and ``d_spacings[i]``.

    The prior is :func:`fit_mean_intensity`. Anisotropic, it is fitted to every
    row, centric ones with their own variance. Isotropic, it is fitted to the
    acentric rows only: a centric zone is a single plane of reciprocal space,
    and on anisotropic data its mean intensity is that of one direction, not
    of the shell. Systematic absences never inform the prior: their intensity
    is zero by symmetry, not a sample of the Wilson distribution.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities of shape (n_reflections,).
    sigma_I : torch.Tensor
        Standard deviations of intensities of shape (n_reflections,).
    hkl : torch.Tensor
        Miller indices of shape (n_reflections, 3).
    d_spacings : torch.Tensor
        Resolution (d-spacing) in Å for each reflection of shape
        (n_reflections,).
    space_group : str, int, or gemmi.SpaceGroup, optional
        Space group specification. Default is "P1".
    min_i_over_sigma : float, optional
        Lower cut on ``I/sigma_I``, as for :func:`french_wilson_valid_mask`.
        Default -3.7.
    n_coeff : int, optional
        B-spline coefficients in ``log Sigma``, as for
        :func:`fit_mean_intensity`.
    exclude_from_fit : torch.Tensor, optional
        Boolean mask of shape (n_reflections,) of rows that must not inform the
        prior -- the free (test) set, so that nothing fitted has seen it. They
        are still given a prior from the curve and converted like every other
        row. Ignored if it would leave no row to fit.
    anisotropic : bool, optional
        Fit an ellipsoidal anisotropy into the prior. Default True.
    epsilon : bool, optional
        Give each reflection the expected intensity ``epsilon Sigma``, with
        ``epsilon`` counted from ``space_group``, which must therefore be the
        crystal's true symmetry: a reflection on a symmetry element is that
        much stronger whatever group the data were merged in. Default True.
        Absences take the general value.

    Returns
    -------
    F : torch.Tensor
        Structure factor amplitudes of shape (n_reflections,).
    sigma_F : torch.Tensor
        Standard deviations of F of shape (n_reflections,).
    valid_mask : torch.Tensor
        Boolean mask, ``True`` = keep. ``False`` both for rows too negative
        for their own sigma and for rows with NaN ``I`` or ``sigma_I`` or a
        non-finite ``d``, whose ``F`` and ``sigma_F`` are NaN.

    Examples
    --------
    ::

        hkl = torch.tensor([[1, 2, 3], [2, 0, 0], [0, 3, 0], [1, 1, 1]])
        I = torch.tensor([100.0, 50.0, 30.0, 200.0])
        sigma_I = torch.tensor([10.0, 8.0, 7.0, 15.0])
        d_spacings = torch.tensor([2.5, 3.0, 2.8, 2.0])
        F, sigma_F, valid = french_wilson_auto(I, sigma_I, hkl, d_spacings, "P212121")
    """
    F = torch.full_like(I, float("nan"))
    sigma_F = torch.full_like(sigma_I, float("nan"))
    # NaN rows are never converted, so they are not kept either.
    valid_mask = torch.zeros_like(I, dtype=torch.bool)

    finite = ~(torch.isnan(I) | torch.isnan(sigma_I))
    if not finite.any():
        return F, sigma_F, valid_mask
    hkl_f = hkl[finite]

    group = SpaceGroup(space_group, device=hkl.device)
    is_centric = group.is_centric(hkl_f)
    absent = group.is_absent(hkl_f)

    fit_mask = ~absent
    if not anisotropic and bool((~is_centric).any()):
        fit_mask = fit_mask & ~is_centric
    if exclude_from_fit is not None:
        held_out = exclude_from_fit.to(device=I.device, dtype=torch.bool)[finite]
        fit_mask = fit_mask & ~held_out
    # A test set drawn on a tiny dataset can cover every row; with nothing left
    # to fit there would be no prior at all.
    if not bool(fit_mask.any()):
        fit_mask = torch.ones_like(fit_mask)

    multiplicity = None
    if epsilon:
        # Operations whose rotation is the identity are the lattice centrings;
        # their count is epsilon for a general reflection, and for an absence.
        eye = torch.eye(3, dtype=group.matrices.dtype, device=group.matrices.device)
        centrings = (group.matrices - eye).abs().amax(dim=(1, 2)) < 1e-6
        multiplicity = group.epsilon(hkl_f, friedel=False).to(I.dtype)
        multiplicity = torch.where(absent, float(centrings.sum()), multiplicity)

    mean_intensity = fit_mean_intensity(
        I[finite],
        sigma_I[finite],
        d_spacings[finite],
        fit_mask=fit_mask,
        n_coeff=n_coeff,
        hkl=hkl_f if anisotropic else None,
        space_group=space_group if anisotropic else None,
        is_centric=is_centric,
        epsilon=multiplicity,
    )

    F[finite], sigma_F[finite], valid_mask[finite] = french_wilson(
        I[finite],
        sigma_I[finite],
        mean_intensity,
        is_centric=is_centric,
        min_i_over_sigma=min_i_over_sigma,
    )
    return F, sigma_F, valid_mask
