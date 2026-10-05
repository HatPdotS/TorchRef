"""
PyTorch implementation of French-Wilson conversion from intensities to structure factors.

Reference: French, S. & Wilson, K. (1978). Acta Cryst. A34, 517-525
The lookup tables, asymptotic branches and rejection rule follow the Phenix
implementation in cctbx/french_wilson.py; the Wilson prior does not (see below).

Usage::

        from torchref.base.french_wilson import french_wilson_auto

        F, sigma_F, valid = french_wilson_auto(
            I, sigma_I, hkl, d_spacings, space_group='P212121'
        )

This is a plain function on purpose: the conversion runs once per dataset, and
a cached estimator holding per-row buffers goes stale the moment the rows are
reordered (as ``ReflectionData`` canonicalization does).

The conversion needs a Wilson prior, the mean intensity ``Sigma`` expected at
each reflection's resolution. :func:`fit_mean_intensity` supplies it as a smooth
resolution-local mean intensity, positive by construction;
:func:`french_wilson` turns ``I``, ``sigma_I`` and ``Sigma`` into
posterior amplitudes; :func:`french_wilson_auto` does both.
:func:`estimate_mean_intensity_by_resolution` is the plain binned mean, kept for
comparison -- it can come out zero or negative and is not a usable prior.
"""

import math
import warnings

import torch

from torchref.config import get_float_dtype
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
#: any reflection where ``Sigma`` still matters (see the loop).
_CURVE_TOL = 1e-4
#: Floor on ``Sigma`` below the typical measurement error, as a log. Far below
#: anything that changes a French-Wilson result -- every reflection is rejected
#: long before it -- and it keeps ``Sigma`` representable and positive where a
#: pure-noise region drives the fit towards zero.
_MAX_LOG_SIGMA_RATIO = 30.0

# Acentric lookup tables from French-Wilson supplement (1978)
AC_ZJ = torch.tensor(
    [
        0.226,
        0.230,
        0.235,
        0.240,
        0.246,
        0.251,
        0.257,
        0.263,
        0.270,
        0.276,
        0.283,
        0.290,
        0.298,
        0.306,
        0.314,
        0.323,
        0.332,
        0.341,
        0.351,
        0.362,
        0.373,
        0.385,
        0.397,
        0.410,
        0.424,
        0.439,
        0.454,
        0.470,
        0.487,
        0.505,
        0.525,
        0.545,
        0.567,
        0.590,
        0.615,
        0.641,
        0.668,
        0.698,
        0.729,
        0.762,
        0.798,
        0.835,
        0.875,
        0.917,
        0.962,
        1.009,
        1.059,
        1.112,
        1.167,
        1.226,
        1.287,
        1.352,
        1.419,
        1.490,
        1.563,
        1.639,
        1.717,
        1.798,
        1.882,
        1.967,
        2.055,
        2.145,
        2.236,
        2.329,
        2.422,
        2.518,
        2.614,
        2.710,
        2.808,
        2.906,
        3.004,
    ],
    dtype=get_float_dtype(),
)

AC_ZJ_SD = torch.tensor(
    [
        0.217,
        0.221,
        0.226,
        0.230,
        0.235,
        0.240,
        0.245,
        0.250,
        0.255,
        0.261,
        0.267,
        0.273,
        0.279,
        0.286,
        0.292,
        0.299,
        0.307,
        0.314,
        0.322,
        0.330,
        0.339,
        0.348,
        0.357,
        0.367,
        0.377,
        0.387,
        0.398,
        0.409,
        0.421,
        0.433,
        0.446,
        0.459,
        0.473,
        0.488,
        0.503,
        0.518,
        0.535,
        0.551,
        0.568,
        0.586,
        0.604,
        0.622,
        0.641,
        0.660,
        0.679,
        0.698,
        0.718,
        0.737,
        0.757,
        0.776,
        0.795,
        0.813,
        0.831,
        0.848,
        0.865,
        0.881,
        0.895,
        0.909,
        0.921,
        0.933,
        0.943,
        0.953,
        0.961,
        0.968,
        0.974,
        0.980,
        0.984,
        0.988,
        0.991,
        0.994,
        0.996,
    ],
    dtype=get_float_dtype(),
)

AC_ZF = torch.tensor(
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

AC_ZF_SD = torch.tensor(
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

# Centric lookup tables from French-Wilson supplement (1978)
C_ZJ = torch.tensor(
    [
        0.114,
        0.116,
        0.119,
        0.122,
        0.124,
        0.127,
        0.130,
        0.134,
        0.137,
        0.141,
        0.145,
        0.148,
        0.153,
        0.157,
        0.162,
        0.166,
        0.172,
        0.177,
        0.183,
        0.189,
        0.195,
        0.202,
        0.209,
        0.217,
        0.225,
        0.234,
        0.243,
        0.253,
        0.263,
        0.275,
        0.287,
        0.300,
        0.314,
        0.329,
        0.345,
        0.363,
        0.382,
        0.402,
        0.425,
        0.449,
        0.475,
        0.503,
        0.534,
        0.567,
        0.603,
        0.642,
        0.684,
        0.730,
        0.779,
        0.833,
        0.890,
        0.952,
        1.018,
        1.089,
        1.164,
        1.244,
        1.327,
        1.416,
        1.508,
        1.603,
        1.703,
        1.805,
        1.909,
        2.015,
        2.123,
        2.233,
        2.343,
        2.453,
        2.564,
        2.674,
        2.784,
        2.894,
        3.003,
        3.112,
        3.220,
        3.328,
        3.435,
        3.541,
        3.647,
        3.753,
        3.962,
    ],
    dtype=get_float_dtype(),
)

C_ZJ_SD = torch.tensor(
    [
        0.158,
        0.161,
        0.165,
        0.168,
        0.172,
        0.176,
        0.179,
        0.184,
        0.188,
        0.192,
        0.197,
        0.202,
        0.207,
        0.212,
        0.218,
        0.224,
        0.230,
        0.236,
        0.243,
        0.250,
        0.257,
        0.265,
        0.273,
        0.282,
        0.291,
        0.300,
        0.310,
        0.321,
        0.332,
        0.343,
        0.355,
        0.368,
        0.382,
        0.397,
        0.412,
        0.428,
        0.445,
        0.463,
        0.481,
        0.501,
        0.521,
        0.543,
        0.565,
        0.589,
        0.613,
        0.638,
        0.664,
        0.691,
        0.718,
        0.745,
        0.773,
        0.801,
        0.828,
        0.855,
        0.881,
        0.906,
        0.929,
        0.951,
        0.971,
        0.989,
        1.004,
        1.018,
        1.029,
        1.038,
        1.044,
        1.049,
        1.052,
        1.054,
        1.054,
        1.053,
        1.051,
        1.049,
        1.047,
        1.044,
        1.041,
        1.039,
        1.036,
        1.034,
        1.031,
        1.029,
        1.028,
    ],
    dtype=get_float_dtype(),
)

C_ZF = torch.tensor(
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

C_ZF_SD = torch.tensor(
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


def interpolate_table(
    h: torch.Tensor, table: torch.Tensor, h_min: float = -4.0
) -> torch.Tensor:
    """
    Interpolate values from French-Wilson lookup table.

    Parameters
    ----------
    h : torch.Tensor
        Normalized parameter (tensor of any shape).
    table : torch.Tensor
        Lookup table tensor (1D).
    h_min : float, optional
        Minimum h value. Default is -4.0.

    Returns
    -------
    torch.Tensor
        Interpolated values (same shape as h).
    """
    # Map h to table index: point = 10.0 * (h - h_min)
    # For h_min = -4.0, this gives point = 10.0 * (h + 4.0)
    point = 10.0 * (h - h_min)
    point = torch.clamp(point, 0.0, len(table) - 1.001)  # Clamp to valid range

    # Linear interpolation
    pt_1 = point.long()
    pt_2 = torch.clamp(pt_1 + 1, max=len(table) - 1)
    delta = point - pt_1.float()

    # Interpolate: (1-delta)*table[pt_1] + delta*table[pt_2]
    val_1 = table[pt_1]
    val_2 = table[pt_2]
    result = (1.0 - delta) * val_1 + delta * val_2

    return result


def french_wilson_h(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor = None,
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
    ``J^(-1/2) exp(-J/2S)`` yields the factor of two. So ``h`` answers "how
    probable is this observation under Wilson conditions", folding the shell mean
    and the measurement sigma into one number, and a cut on ``h`` is a
    tail-probability cut (``h >= -4`` corresponds to ``p ~ 3e-5``).

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities (any shape).
    sigma_I : torch.Tensor
        Standard deviations of intensities (same shape as I).
    mean_intensity : torch.Tensor
        Wilson mean intensity ``Sigma`` for each reflection (same shape as I).
        Must be positive: there is no Wilson prior with a mean at or below
        zero.
    is_centric : torch.Tensor or bool, optional
        Boolean mask of centric reflections, or a plain ``bool`` when the whole
        input is known to be one or the other (as it is for the pre-split
        callers below). If None, all are treated as acentric.

    Returns
    -------
    torch.Tensor
        ``h`` for each reflection (same shape as I). NaN wherever
        ``mean_intensity`` is not positive, so such rows fail every cut on
        ``h`` rather than passing one.
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


def french_wilson_valid_mask(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor = None,
    h_min: float = -4.0,
) -> torch.Tensor:
    """
    French-Wilson's own rejection criterion, as a keep-mask.

    ``True`` means the observation is explainable as a noisy measurement of a
    Wilson-distributed reflection and should be kept. This deliberately keeps
    negative intensities that noise accounts for -- only observations too
    negative to be explained by *any* Wilson-distributed true intensity, given
    their own sigma and their shell's mean, are rejected.

    Parameters
    ----------
    I, sigma_I, mean_intensity, is_centric
        As for :func:`french_wilson_h`.
    h_min : float, optional
        Rejection threshold on ``h``. Default -4.0.

        Raising this does **not** find more outliers -- it discards valid weak
        measurements. On a typical dataset ``h_min=-4`` rejects nothing,
        ``-2`` rejects ~0.3% and ``0`` rejects ~6%, and those are noise-
        explainable reflections, not bad ones.

    Returns
    -------
    torch.Tensor
        Boolean keep-mask (same shape as I).
    """
    h = french_wilson_h(I, sigma_I, mean_intensity, is_centric)
    # A non-finite h means sigma_I was degenerate or mean_intensity was not a
    # positive number; such a reflection has no posterior and must not be kept
    # on the strength of a NaN comparison (which is False anyway, but not by
    # intent).
    return torch.isfinite(h) & (I / sigma_I >= h_min + 0.3) & (h >= h_min)


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
    is gone. Measured on 4BX9, a reflection whose true ``h`` is -3.76 comes back
    with ``h = +0.01``.

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
    usable = (
        torch.isfinite(F) & torch.isfinite(sigma_F) & (F > 0) & (sigma_F > 0)
    )
    I = torch.where(usable, F * F, torch.zeros_like(F))
    sigma_I = torch.where(usable, 2.0 * F * sigma_F, torch.zeros_like(F))
    return I, sigma_I


def french_wilson_acentric(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    h_min: float = -4.0,
    i_sig_min: float = -3.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    French-Wilson conversion for acentric reflections.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities (any shape).
    sigma_I : torch.Tensor
        Standard deviations of intensities (same shape as I).
    mean_intensity : torch.Tensor
        Wilson mean intensity ``Sigma`` for each reflection (same shape as I).
    h_min : float, optional
        Minimum h value for rejection. Default is -4.0.
    i_sig_min : float, optional
        Minimum I/sigma_I for rejection. Default is -3.7 (h_min + 0.3).

    Returns
    -------
    F : torch.Tensor
        Structure factor amplitudes (same shape as I). NaN where
        ``mean_intensity`` is not positive.
    sigma_F : torch.Tensor
        Standard deviations of F (same shape as I), NaN where ``F`` is.
    valid_mask : torch.Tensor
        Boolean mask indicating valid (not rejected) reflections.
    """
    device = I.device
    dtype = I.dtype

    # Move lookup tables to same device and dtype
    ac_zj = AC_ZJ.to(device=device, dtype=dtype)
    ac_zj_sd = AC_ZJ_SD.to(device=device, dtype=dtype)
    ac_zf = AC_ZF.to(device=device, dtype=dtype)
    ac_zf_sd = AC_ZF_SD.to(device=device, dtype=dtype)

    # Compute normalized parameter h (shared with the rejection criterion, so
    # the guard and the conversion can never drift apart)
    h = french_wilson_h(I, sigma_I, mean_intensity, is_centric=False)

    # Clamp h to valid table range [-4.0, ...] to avoid extrapolation issues
    # Very weak reflections (h < h_min) get the boundary value from lookup table
    h_clamped = torch.clamp(h, min=h_min)

    # Initialize outputs
    F = torch.zeros_like(I)
    sigma_F = torch.zeros_like(I)

    # Case 1: Small h (h < 3.0) - use lookup tables
    small_h_mask = h_clamped < 3.0
    if small_h_mask.any():
        h_small = h_clamped[small_h_mask]
        sigma_I_small = sigma_I[small_h_mask]

        # Interpolate from tables
        zf = interpolate_table(h_small, ac_zf, h_min=h_min)
        zf_sd = interpolate_table(h_small, ac_zf_sd, h_min=h_min)

        F[small_h_mask] = zf * torch.sqrt(sigma_I_small)
        sigma_F[small_h_mask] = zf_sd * torch.sqrt(sigma_I_small)

    # Case 2: Large h (h >= 3.0) - use asymptotic formula
    large_h_mask = h_clamped >= 3.0
    if large_h_mask.any():
        h_large = h_clamped[large_h_mask]
        sigma_I_large = sigma_I[large_h_mask]

        J = h_large * sigma_I_large
        F_large = torch.sqrt(J)
        sigma_F_large = 0.5 * (sigma_I_large / F_large)

        F[large_h_mask] = F_large
        sigma_F[large_h_mask] = sigma_F_large

    # Where h is undefined (a prior mean that is not positive, or a degenerate
    # sigma) there is no posterior, so there is no amplitude either.
    undefined = ~torch.isfinite(h)
    F = F.masked_fill(undefined, float("nan"))
    sigma_F = sigma_F.masked_fill(undefined, float("nan"))

    # Rejection criterion, computed but not used to zero out values: the caller
    # decides what to do with it. See french_wilson_valid_mask.
    valid_mask = torch.isfinite(h) & (I / sigma_I >= i_sig_min) & (h >= h_min)

    return F, sigma_F, valid_mask


def french_wilson_centric(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    h_min: float = -4.0,
    i_sig_min: float = -3.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    French-Wilson conversion for centric reflections.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities (any shape).
    sigma_I : torch.Tensor
        Standard deviations of intensities (same shape as I).
    mean_intensity : torch.Tensor
        Wilson mean intensity ``Sigma`` for each reflection (same shape as I).
    h_min : float, optional
        Minimum h value for rejection. Default is -4.0.
    i_sig_min : float, optional
        Minimum I/sigma_I for rejection. Default is -3.7 (h_min + 0.3).

    Returns
    -------
    F : torch.Tensor
        Structure factor amplitudes (same shape as I). NaN where
        ``mean_intensity`` is not positive.
    sigma_F : torch.Tensor
        Standard deviations of F (same shape as I), NaN where ``F`` is.
    valid_mask : torch.Tensor
        Boolean mask indicating valid (not rejected) reflections.
    """
    device = I.device
    dtype = I.dtype

    # Move lookup tables to same device and dtype
    c_zj = C_ZJ.to(device=device, dtype=dtype)
    c_zj_sd = C_ZJ_SD.to(device=device, dtype=dtype)
    c_zf = C_ZF.to(device=device, dtype=dtype)
    c_zf_sd = C_ZF_SD.to(device=device, dtype=dtype)

    # Compute normalized parameter h (note factor of 2 for centric!)
    h = french_wilson_h(I, sigma_I, mean_intensity, is_centric=True)

    # Clamp h to valid table range [-4.0, ...] to avoid extrapolation issues
    # Very weak reflections (h < h_min) get the boundary value from lookup table
    h_clamped = torch.clamp(h, min=h_min)

    # Initialize outputs
    F = torch.zeros_like(I)
    sigma_F = torch.zeros_like(I)

    # Case 1: Small h (h < 4.0) - use lookup tables
    small_h_mask = h_clamped < 4.0
    if small_h_mask.any():
        h_small = h_clamped[small_h_mask]
        sigma_I_small = sigma_I[small_h_mask]

        # Interpolate from tables
        zf = interpolate_table(h_small, c_zf, h_min=h_min)
        zf_sd = interpolate_table(h_small, c_zf_sd, h_min=h_min)

        F[small_h_mask] = zf * torch.sqrt(sigma_I_small)
        sigma_F[small_h_mask] = zf_sd * torch.sqrt(sigma_I_small)

    # Case 2: Large h (h >= 4.0) - use extended asymptotic formula
    large_h_mask = h_clamped >= 4.0
    if large_h_mask.any():
        h_large = h_clamped[large_h_mask]
        sigma_I_large = sigma_I[large_h_mask]

        # Extended asymptotic expansion (Phenix extension)
        h_2 = 1.0 / (h_large * h_large)
        h_4 = h_2 * h_2
        h_6 = h_2 * h_4

        # Posterior mean of F
        post_F = torch.sqrt(h_large) * (
            1.0 - (3.0 / 8.0) * h_2 - (87.0 / 128.0) * h_4 - (2889.0 / 1024.0) * h_6
        )

        # Posterior standard deviation of F
        post_sig_F = torch.sqrt(
            h_large * ((1.0 / 4.0) * h_2 + (15.0 / 32.0) * h_4 + (273.0 / 128.0) * h_6)
        )

        F[large_h_mask] = post_F * torch.sqrt(sigma_I_large)
        sigma_F[large_h_mask] = post_sig_F * torch.sqrt(sigma_I_large)

    # Where h is undefined (a prior mean that is not positive, or a degenerate
    # sigma) there is no posterior, so there is no amplitude either.
    undefined = ~torch.isfinite(h)
    F = F.masked_fill(undefined, float("nan"))
    sigma_F = sigma_F.masked_fill(undefined, float("nan"))

    # Rejection criterion, computed but not used to zero out values: the caller
    # decides what to do with it. See french_wilson_valid_mask.
    valid_mask = torch.isfinite(h) & (I / sigma_I >= i_sig_min) & (h >= h_min)

    return F, sigma_F, valid_mask


def french_wilson(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    mean_intensity: torch.Tensor,
    is_centric: torch.Tensor = None,
    h_min: float = -4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    French-Wilson conversion from intensities to structure factors.

    Automatically handles both centric and acentric reflections.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities of shape (...).
    sigma_I : torch.Tensor
        Standard deviations of intensities of shape (...).
    mean_intensity : torch.Tensor
        Wilson mean intensity ``Sigma`` for each reflection, of shape (...),
        e.g. from :func:`fit_mean_intensity`. Rows where it is not positive have
        no posterior: ``F`` and ``sigma_F`` are NaN and ``valid_mask`` is False.
    is_centric : torch.Tensor, optional
        Boolean mask indicating centric reflections of shape (...).
        If None, assumes all reflections are acentric.
    h_min : float, optional
        Minimum h value for rejection. Default is -4.0.

    Returns
    -------
    F : torch.Tensor
        Structure factor amplitudes of shape (...).
    sigma_F : torch.Tensor
        Standard deviations of F of shape (...).
    valid_mask : torch.Tensor
        Boolean mask indicating valid (not rejected) reflections of shape (...).

    Examples
    --------
    ::

        I = torch.tensor([100.0, 5.0, -15.0, 200.0])
        sigma_I = torch.tensor([10.0, 10.0, 10.0, 15.0])
        mean_I = torch.tensor([80.0, 80.0, 80.0, 150.0])
        F, sigma_F, valid = french_wilson(I, sigma_I, mean_I)
        print(f"F = {F}")
    """
    i_sig_min = h_min + 0.3

    # Initialize outputs
    F = torch.zeros_like(I)
    sigma_F = torch.zeros_like(I)
    valid_mask = torch.zeros_like(I, dtype=torch.bool)

    if is_centric is None:
        # All acentric
        F, sigma_F, valid_mask = french_wilson_acentric(
            I, sigma_I, mean_intensity, h_min, i_sig_min
        )
    else:
        # Process acentric reflections
        acentric_mask = ~is_centric
        if acentric_mask.any():
            F_acen, sigma_F_acen, valid_acen = french_wilson_acentric(
                I[acentric_mask],
                sigma_I[acentric_mask],
                mean_intensity[acentric_mask],
                h_min,
                i_sig_min,
            )
            F[acentric_mask] = F_acen
            sigma_F[acentric_mask] = sigma_F_acen
            valid_mask[acentric_mask] = valid_acen

        # Process centric reflections
        if is_centric.any():
            F_cen, sigma_F_cen, valid_cen = french_wilson_centric(
                I[is_centric],
                sigma_I[is_centric],
                mean_intensity[is_centric],
                h_min,
                i_sig_min,
            )
            F[is_centric] = F_cen
            sigma_F[is_centric] = sigma_F_cen
            valid_mask[is_centric] = valid_cen

    return F, sigma_F, valid_mask


def estimate_mean_intensity_by_resolution(
    I: torch.Tensor, d_spacings: torch.Tensor, n_bins: int = 60, min_per_bin: int = 40
) -> torch.Tensor:
    """
    Estimate mean intensity for each reflection based on resolution binning.

    Uses linear interpolation between bin centers for smooth mean intensity
    estimates.

    This is an unweighted arithmetic mean of every row in a bin, and **not a
    usable French-Wilson prior**: a bin of pure noise averages to about zero
    and comes out negative as often as not, and one reflection with a huge
    sigma can outweigh the rest of its bin. :func:`fit_mean_intensity` is the
    prior :func:`french_wilson_auto` uses. It is also the wrong thing for
    outlier detection: a strong outlier raises the mean of its own bin and so
    raises its own ``Sigma``, hiding itself. Use
    :func:`~torchref.base.wilson_outliers.robust_mean_intensity` there.

    Parameters
    ----------
    I : torch.Tensor
        Measured intensities of shape (n_reflections,).
    d_spacings : torch.Tensor
        Resolution (d-spacing) for each reflection of shape (n_reflections,).
    n_bins : int, optional
        Number of resolution bins. Default is 60.
    min_per_bin : int, optional
        Minimum reflections per bin. Default is 40.

    Returns
    -------
    torch.Tensor
        Estimated mean intensity for each reflection of shape (n_reflections,).
        Can be zero or negative.
    """
    n_reflections = len(I)

    # Adjust number of bins to ensure minimum per bin
    reflections_per_bin = max(min_per_bin, n_reflections // n_bins)
    actual_n_bins = max(1, n_reflections // reflections_per_bin)

    # Sort by resolution (d-spacing, high to low)
    sort_idx = torch.argsort(d_spacings, descending=True)
    I_sorted = I[sort_idx]
    d_sorted = d_spacings[sort_idx]

    # Compute bin boundaries and mean intensities using vectorized operations
    # Create bin indices for each sorted reflection
    bin_indices = torch.arange(n_reflections, device=I.device) // reflections_per_bin
    bin_indices = torch.clamp(bin_indices, max=actual_n_bins - 1)

    # Use scatter_add to compute sum of intensities per bin
    bin_sums = torch.zeros(actual_n_bins, dtype=I.dtype, device=I.device)
    bin_counts = torch.zeros(actual_n_bins, dtype=bin_indices.dtype, device=I.device)
    bin_sums.scatter_add_(0, bin_indices, I_sorted)
    bin_counts.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))

    # Compute mean intensity per bin
    bin_means = bin_sums / bin_counts.to(I.dtype)

    # Compute bin centers using scatter for min/max d-spacings
    bin_d_max = torch.full(
        (actual_n_bins,), -float("inf"), dtype=I.dtype, device=I.device
    )
    bin_d_min = torch.full(
        (actual_n_bins,), float("inf"), dtype=I.dtype, device=I.device
    )
    bin_d_max.scatter_reduce_(
        0, bin_indices, d_sorted, reduce="amax", include_self=False
    )
    bin_d_min.scatter_reduce_(
        0, bin_indices, d_sorted, reduce="amin", include_self=False
    )
    bin_centers = (bin_d_max + bin_d_min) / 2.0

    # Now interpolate for each reflection based on ORIGINAL (unsorted) d_spacings
    # bin_centers are in descending order (high to low d-spacing)

    # For each d_spacing, find which two bins it falls between
    # torch.searchsorted expects ascending order, so we flip
    bin_centers_ascending = bin_centers.flip(0)

    # Find the insertion point for each d_spacing in ascending order
    # right=True means if d_spacing equals a bin center, use the bin to the right
    insert_idx = torch.searchsorted(bin_centers_ascending, d_spacings, right=True)

    # Convert back to descending order indexing
    # In descending order, the "left" bin is at position (n_bins - insert_idx)
    # and the "right" bin is at position (n_bins - insert_idx - 1)
    left_idx = actual_n_bins - insert_idx
    right_idx = left_idx - 1

    # Clamp to valid range
    left_idx = torch.clamp(left_idx, 0, actual_n_bins - 1)
    right_idx = torch.clamp(right_idx, 0, actual_n_bins - 1)

    # Handle edge cases first (beyond first or last bin)
    # If d >= first bin center, use first bin
    beyond_first = d_spacings >= bin_centers[0]
    # If d <= last bin center, use last bin
    beyond_last = d_spacings <= bin_centers[-1]

    # Get bin centers and means for interpolation
    d1 = bin_centers[left_idx]
    d2 = bin_centers[right_idx]
    m1 = bin_means[left_idx]
    m2 = bin_means[right_idx]

    # Linear interpolation weight
    d_diff = d1 - d2
    # Avoid division by zero
    safe_d_diff = torch.where(
        torch.abs(d_diff) > 1e-10, d_diff, torch.ones_like(d_diff)
    )
    weight = (d1 - d_spacings) / safe_d_diff
    weight = torch.clamp(weight, 0.0, 1.0)

    # Interpolate
    mean_I = (1 - weight) * m1 + weight * m2

    # Apply edge case handling
    mean_I = torch.where(beyond_first, bin_means[0], mean_I)
    mean_I = torch.where(beyond_last, bin_means[-1], mean_I)

    return mean_I


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
    Sigma = torch.full_like(I, float("nan"))

    placed = torch.isfinite(s3)
    usable = placed & torch.isfinite(I) & torch.isfinite(sigma_I) & (sigma_I > 0)
    if fit_mask is not None:
        usable = usable & fit_mask.to(torch.bool)
    n_fit = int(usable.sum())
    if n_fit == 0:
        return Sigma

    # The basis spans every row that will receive a Sigma, not only the fitted
    # ones, so nothing is evaluated outside the fitted range.
    lo, hi = s3[placed].min(), s3[placed].max()
    if hi > lo:
        x = 2.0 * (s3 - lo) / (hi - lo) - 1.0
        n_terms = max(1, min(int(n_coeff), n_fit // _ROWS_PER_COEFF))
    else:
        x = torch.zeros_like(s3)
        n_terms = 1
    radial = _bspline(x[usable], n_terms)
    I_fit = I[usable]
    log_epsilon = torch.zeros_like(I)
    if epsilon is not None:
        log_epsilon = torch.log(epsilon.detach().to(I.dtype).clamp(min=1.0))
    offset = log_epsilon[usable]
    kappa = torch.ones_like(I_fit)
    if is_centric is not None:
        kappa = torch.where(is_centric.to(torch.bool)[usable], 2.0, kappa)

    log_sigma = torch.log(sigma_I[usable])
    log_sbar = radial @ _solve_normal(radial.T @ radial, radial.T @ log_sigma)
    log_noise = torch.maximum(log_sbar, log_sigma - math.log(_NOISY_SIGMA_RATIO))
    noise = torch.exp(log_noise)
    variance = noise * noise

    anisotropy = None
    if hkl is not None and space_group is not None:
        radial_all = torch.zeros(len(I), n_terms, dtype=I.dtype, device=I.device)
        radial_all[placed] = _bspline(x[placed], n_terms)
        s2 = torch.where(placed, 1.0 / (d_spacings.detach() ** 2), 0.0).to(I.dtype)
        anisotropy = _anisotropy_design(hkl, space_group, radial_all, s2, usable)
    if anisotropy is not None and anisotropy.shape[1] > 0:
        design = torch.cat([radial, anisotropy[usable]], dim=1)
    else:
        anisotropy, design = None, radial

    # The negated quasi-likelihood, integral of (I - t)/(k t^2 + v) from the
    # observation to Sigma, with v the noise variance and k = 1 (acentric) or 2
    # (centric), Sigma-independent constant dropped. atan2(noise, sqrt(k) Sigma)
    # rather than pi/2 - atan(sqrt(k) Sigma/noise) keeps the strong end, where
    # it is the familiar exponential term I/Sigma, free of cancellation, and it
    # stays bounded as Sigma goes to zero. Also returned: the rounding error of
    # the sum, below which two values do not differ.
    eps = torch.finfo(I.dtype).eps
    root_kappa = kappa.sqrt()
    log_kappa = kappa.log()

    def objective(c):
        eta = design @ c + offset
        terms = (I_fit / (root_kappa * noise)) * torch.atan2(
            noise, root_kappa * torch.exp(eta)
        ) + (0.5 / kappa) * torch.logaddexp(2.0 * eta + log_kappa, 2.0 * log_noise)
        return float(terms.sum()), eta, eps * float(terms.abs().sum())

    # Start from a flat curve at the mean intensity, or at the noise level when
    # the data average to nothing; only the scale of the start matters.
    level = math.log(max(float((I_fit / offset.exp()).mean()), float(noise.median())))
    coeff = torch.zeros(design.shape[1], dtype=I.dtype, device=I.device)
    coeff[:n_terms] = level
    loss, eta, rounding = objective(coeff)

    # Fisher scoring rather than a generic optimiser: each step is scaled by the
    # information in every direction, so weakly determined directions get
    # full-sized steps instead of stalling. A step that raises the objective by
    # more than its rounding error is halved; a float32 sum cannot resolve the
    # last steps, and refusing them would stop the fit short in exactly those
    # directions. Convergence is judged on the curve, where Sigma is at least a
    # hundredth of the noise: below that every reflection is rejected whatever
    # Sigma is, and a region of pure noise keeps sliding towards zero long
    # after the rest has settled.
    for _ in range(_MAX_ITER):
        mu = torch.exp(eta)
        total = kappa * mu * mu + variance
        weight = mu * mu / total
        score = mu * (I_fit - mu) / total
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

    log_floor = float(log_sbar.median()) - _MAX_LOG_SIGMA_RATIO
    log_Sigma = _bspline(x[placed], n_terms) @ coeff[:n_terms]
    if anisotropy is not None:
        log_Sigma = log_Sigma + anisotropy[placed] @ coeff[n_terms:]
    log_Sigma = torch.clamp(log_Sigma, min=log_floor) + log_epsilon[placed]
    Sigma[placed] = torch.exp(log_Sigma)
    return Sigma


def french_wilson_auto(
    I: torch.Tensor,
    sigma_I: torch.Tensor,
    hkl: torch.Tensor,
    d_spacings: torch.Tensor,
    space_group: SpaceGroupLike = "P1",
    n_bins: int | None = None,
    min_per_bin: int | None = None,
    h_min: float = -4.0,
    *,
    n_coeff: int = DEFAULT_N_COEFF,
    exclude_from_fit: torch.Tensor | None = None,
    anisotropic: bool = True,
    epsilon: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert intensities to amplitudes, fitting the Wilson prior and centricity.

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
    n_bins, min_per_bin : int, optional
        Deprecated and ignored: the prior has no resolution bins. Size it with
        ``n_coeff``.
    h_min : float, optional
        Minimum h value for rejection. Default is -4.0.
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
        Boolean mask, ``True`` = keep. ``False`` both for rows French-Wilson
        rejects as too negative and for rows with NaN ``I`` or ``sigma_I`` or
        a non-finite ``d``, whose ``F`` and ``sigma_F`` are NaN.

    Examples
    --------
    ::

        hkl = torch.tensor([[1, 2, 3], [2, 0, 0], [0, 3, 0], [1, 1, 1]])
        I = torch.tensor([100.0, 50.0, 30.0, 200.0])
        sigma_I = torch.tensor([10.0, 8.0, 7.0, 15.0])
        d_spacings = torch.tensor([2.5, 3.0, 2.8, 2.0])
        F, sigma_F, valid = french_wilson_auto(I, sigma_I, hkl, d_spacings, "P212121")
    """
    if n_bins is not None or min_per_bin is not None:
        warnings.warn(
            "french_wilson_auto: n_bins and min_per_bin have no effect and will be "
            "removed; the Wilson prior is a smooth fit sized by n_coeff "
            "(see fit_mean_intensity).",
            DeprecationWarning,
            stacklevel=2,
        )
    F = torch.full_like(I, float("nan"))
    sigma_F = torch.full_like(sigma_I, float("nan"))
    # NaN rows are never converted, so they are not kept either.
    valid_mask = torch.zeros_like(I, dtype=torch.bool)

    finite = ~(torch.isnan(I) | torch.isnan(sigma_I))
    if not finite.any():
        return F, sigma_F, valid_mask

    group = SpaceGroup(space_group, device=hkl.device)
    is_centric = group.is_centric(hkl[finite])
    absent = group.is_absent(hkl[finite])
    acentric = ~is_centric
    if anisotropic or not bool(acentric.any()):
        fit_mask = ~absent
    else:
        fit_mask = acentric & ~absent
    if not bool(fit_mask.any()):
        fit_mask = torch.ones_like(acentric)
    multiplicity = None
    if epsilon:
        # The operations whose rotation is the identity are the lattice
        # centrings; their count is epsilon for a general reflection.
        eye = torch.eye(3, dtype=group.matrices.dtype, device=group.matrices.device)
        centring = (group.matrices - eye).abs().amax(dim=(1, 2)) < 1e-6
        multiplicity = group.epsilon(hkl[finite], friedel=False).to(I.dtype)
        multiplicity = torch.where(absent, float(centring.sum()), multiplicity)
    if exclude_from_fit is not None:
        held_out = exclude_from_fit.to(device=I.device, dtype=torch.bool)
        working = fit_mask & ~held_out[finite]
        # A test set drawn on a tiny dataset can cover every row, and with
        # nothing left to fit there would be no prior at all.
        if bool(working.any()):
            fit_mask = working
    mean_intensity = fit_mean_intensity(
        I[finite],
        sigma_I[finite],
        d_spacings[finite],
        fit_mask=fit_mask,
        n_coeff=n_coeff,
        hkl=hkl[finite] if anisotropic else None,
        space_group=space_group if anisotropic else None,
        is_centric=is_centric,
        epsilon=multiplicity,
    )

    F[finite], sigma_F[finite], valid_mask[finite] = french_wilson(
        I[finite],
        sigma_I[finite],
        mean_intensity,
        is_centric=is_centric,
        h_min=h_min,
    )
    return F, sigma_F, valid_mask
