"""
Statistical distributions for Maximum Likelihood molecular replacement.

This module provides numerically stable implementations of the probability
distributions used in crystallographic ML target functions:

- Rice distribution for acentric reflections
- Woolfson (folded normal) distribution for centric reflections
- Stable log-Bessel I_0 computation for large arguments

The key numerical challenge is computing log(I_0(x)) for large x (up to ~10000),
which requires an asymptotic expansion to avoid overflow.

References
----------
- Read, R.J. (2001). Pushing the boundaries of molecular replacement with
  maximum likelihood. Acta Cryst. D57, 1373-1382.
- McCoy et al. (2007). Phaser crystallographic software. J. Appl. Cryst. 40, 658-674.
"""

import math

import torch


def stable_log_bessel_i0(x: torch.Tensor) -> torch.Tensor:
    """
    Compute log(I_0(x)) with numerical stability for large x.

    The modified Bessel function I_0(x) grows exponentially, so direct
    computation overflows for x > ~700. This implementation uses:
    - For small x (< 50): log(I_0e(x)) + |x| using torch.special.i0e
    - For large x (>= 50): asymptotic expansion x - 0.5*log(2*pi*x)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of non-negative values.

    Returns
    -------
    torch.Tensor
        log(I_0(x)) for each element, same shape as input.

    Notes
    -----
    The asymptotic expansion is:
        log(I_0(x)) ~ x - 0.5*log(2*pi*x) + 1/(8x) - 1/(128x^2) + ...

    For x >= 50, the leading two terms give excellent accuracy.
    For very large x (> 10000), higher order terms can be added if needed.

    Examples
    --------
    >>> x = torch.tensor([0.1, 10.0, 100.0, 1000.0])
    >>> log_i0 = stable_log_bessel_i0(x)
    >>> torch.all(torch.isfinite(log_i0))
    True
    """
    # Ensure non-negative input
    x_abs = torch.abs(x)

    # Initialize output
    result = torch.zeros_like(x)

    # Small x regime: use i0e which is I_0(x) * exp(-|x|)
    # So log(I_0(x)) = log(I_0e(x)) + |x|
    small_mask = x_abs < 50.0
    if small_mask.any():
        x_small = x_abs[small_mask]
        i0e_val = torch.special.i0e(x_small)
        # Guard against i0e returning 0 for very small x
        i0e_val = torch.clamp(i0e_val, min=1e-300)
        result[small_mask] = torch.log(i0e_val) + x_small

    # Large x regime: asymptotic expansion
    # log(I_0(x)) ~ x - 0.5*log(2*pi*x) + 1/(8x) - 1/(128x^2) + ...
    large_mask = ~small_mask
    if large_mask.any():
        x_large = x_abs[large_mask]
        # Leading terms of asymptotic expansion
        log_i0_asymp = (
            x_large
            - 0.5 * torch.log(2.0 * math.pi * x_large)
            + 1.0 / (8.0 * x_large)
            - 1.0 / (128.0 * x_large * x_large)
        )
        result[large_mask] = log_i0_asymp

    return result


def rice_log_likelihood(
    F_obs: torch.Tensor,
    F_mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """
    Log-likelihood for acentric reflections (Rice distribution).

    The Rice distribution describes the distribution of |F_obs| given
    the expected |F_calc| and variance for acentric reflections:

        p(F_obs | F_mean, sigma^2) = (F_obs / sigma^2) *
            exp(-(F_obs^2 + F_mean^2) / (2*sigma^2)) * I_0(F_obs*F_mean / sigma^2)

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes |F_obs|.
    F_mean : torch.Tensor
        Expected structure factor amplitudes D * |F_calc|.
    variance : torch.Tensor
        Variance parameter sigma^2 = epsilon * (Sigma_N - D^2 * <|F_calc|^2>).

    Returns
    -------
    torch.Tensor
        Log-likelihood for each reflection.

    Notes
    -----
    The log-likelihood is:
        log p = log(F_obs) - log(variance) - (F_obs^2 + F_mean^2)/(2*variance)
                + log(I_0(F_obs * F_mean / variance))

    Numerical stability is ensured by using stable_log_bessel_i0.
    """
    # Numerical guards
    variance_safe = torch.clamp(variance, min=1e-8)
    F_obs_safe = torch.clamp(F_obs, min=1e-10)
    F_mean_safe = torch.clamp(F_mean, min=0.0)

    # Compute Bessel argument
    bessel_arg = F_obs_safe * F_mean_safe / variance_safe

    # Log-likelihood components
    log_likelihood = (
        torch.log(F_obs_safe)
        - torch.log(variance_safe)
        - (F_obs_safe**2 + F_mean_safe**2) / (2.0 * variance_safe)
        + stable_log_bessel_i0(bessel_arg)
    )

    return log_likelihood


def woolfson_log_likelihood(
    F_obs: torch.Tensor,
    F_mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """
    Log-likelihood for centric reflections (folded normal / Woolfson distribution).

    For centric reflections, the phase is restricted to 0 or pi, so the
    distribution becomes a folded normal (Woolfson distribution):

        p(F_obs | F_mean, sigma^2) = (2/sigma) * (2*pi)^(-0.5) *
            cosh(F_obs * F_mean / sigma^2) * exp(-(F_obs^2 + F_mean^2)/(2*sigma^2))

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes |F_obs|.
    F_mean : torch.Tensor
        Expected structure factor amplitudes D * |F_calc|.
    variance : torch.Tensor
        Variance parameter (note: 2x larger than acentric due to phase restriction).

    Returns
    -------
    torch.Tensor
        Log-likelihood for each centric reflection.

    Notes
    -----
    The log-likelihood is:
        log p = log(2) - 0.5*log(2*pi*variance) - (F_obs^2 + F_mean^2)/(2*variance)
                + log(cosh(F_obs * F_mean / variance))

    For numerical stability with large arguments:
        log(cosh(x)) ~ |x| - log(2) for |x| > ~20
    """
    # Numerical guards
    variance_safe = torch.clamp(variance, min=1e-8)
    F_obs_safe = torch.clamp(F_obs, min=1e-10)
    F_mean_safe = torch.clamp(F_mean, min=0.0)

    # For centric, variance is 2x larger (sigma^2 / 2 for each component)
    # The effective sigma for centric is sqrt(2 * variance)
    sigma = torch.sqrt(variance_safe)

    # Compute argument for cosh
    cosh_arg = F_obs_safe * F_mean_safe / variance_safe

    # Stable log(cosh(x)): use |x| - log(2) for large x
    log_cosh = torch.where(
        cosh_arg < 20.0,
        torch.log(torch.cosh(cosh_arg)),
        torch.abs(cosh_arg) - math.log(2.0),
    )

    # Log-likelihood components
    log_likelihood = (
        math.log(2.0)
        - 0.5 * torch.log(2.0 * math.pi * variance_safe)
        - (F_obs_safe**2 + F_mean_safe**2) / (2.0 * variance_safe)
        + log_cosh
    )

    return log_likelihood


def combined_log_likelihood(
    F_obs: torch.Tensor,
    F_mean: torch.Tensor,
    variance: torch.Tensor,
    centric_flags: torch.Tensor,
) -> torch.Tensor:
    """
    Combined log-likelihood dispatching to Rice/Woolfson based on centric flags.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes |F_obs|.
    F_mean : torch.Tensor
        Expected structure factor amplitudes D * |F_calc|.
    variance : torch.Tensor
        Variance parameters for each reflection.
    centric_flags : torch.Tensor
        Boolean mask, True for centric reflections.

    Returns
    -------
    torch.Tensor
        Log-likelihood for each reflection using the appropriate distribution.

    Examples
    --------
    >>> F_obs = torch.tensor([10.0, 20.0, 15.0])
    >>> F_mean = torch.tensor([9.5, 18.0, 14.0])
    >>> variance = torch.tensor([5.0, 8.0, 6.0])
    >>> centric = torch.tensor([False, True, False])
    >>> ll = combined_log_likelihood(F_obs, F_mean, variance, centric)
    """
    # Initialize with Rice likelihood (acentric default)
    log_likelihood = rice_log_likelihood(F_obs, F_mean, variance)

    # Replace centric reflections with Woolfson likelihood
    if centric_flags.any():
        centric_ll = woolfson_log_likelihood(
            F_obs[centric_flags],
            F_mean[centric_flags],
            variance[centric_flags],
        )
        log_likelihood[centric_flags] = centric_ll

    return log_likelihood


def acentric_pdf(
    F_obs: torch.Tensor,
    F_mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """
    Probability density for acentric reflections (Rice distribution).

    This is the PDF (not log) for cases where the actual probability is needed.
    For numerical reasons, prefer rice_log_likelihood when possible.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_mean : torch.Tensor
        Expected structure factor amplitudes.
    variance : torch.Tensor
        Variance parameter.

    Returns
    -------
    torch.Tensor
        Probability density for each reflection.
    """
    return torch.exp(rice_log_likelihood(F_obs, F_mean, variance))


def centric_pdf(
    F_obs: torch.Tensor,
    F_mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    """
    Probability density for centric reflections (Woolfson distribution).

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_mean : torch.Tensor
        Expected structure factor amplitudes.
    variance : torch.Tensor
        Variance parameter.

    Returns
    -------
    torch.Tensor
        Probability density for each reflection.
    """
    return torch.exp(woolfson_log_likelihood(F_obs, F_mean, variance))
