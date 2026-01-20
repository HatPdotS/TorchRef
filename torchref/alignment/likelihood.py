"""
Maximum Likelihood target functions for molecular replacement.

This module provides the core ML target functions for crystallographic
molecular replacement:

- D factor computation (Luzzati-style model error parameterization)
- Variance estimation for the likelihood function
- MLTF (Maximum Likelihood Translation Function)

The ML approach properly accounts for model errors and provides
superior signal-to-noise compared to traditional R-factor targets.

References
----------
- Read, R.J. (2001). Pushing the boundaries of molecular replacement with
  maximum likelihood. Acta Cryst. D57, 1373-1382.
- Storoni, L.C., McCoy, A.J. & Read, R.J. (2004). Likelihood-enhanced fast
  rotation functions. Acta Cryst. D60, 432-438.
"""

import math
from typing import Optional

import torch

from torchref.alignment.distributions import combined_log_likelihood


def compute_d_factors(
    resolution: torch.Tensor,
    rms_error: float = 1.0,
    f_p: float = 1.0,
    f_sol: float = 0.95,
    b_sol: float = 300.0,
) -> torch.Tensor:
    """
    Compute Luzzati D factors as a function of resolution.

    The D factor models the expected correlation between observed and
    calculated structure factors, accounting for:
    - Coordinate errors (rms_error in Angstroms)
    - Solvent contribution (f_sol, b_sol)
    - Overall completeness factor (f_p)

    The formula is:
        D(d) = f_p * exp(-2*pi^2*sigma_r^2/d^2) * (1 - f_sol*exp(-b_sol/4d^2))

    where d is the resolution (d-spacing) and sigma_r is the RMS coordinate error.

    Parameters
    ----------
    resolution : torch.Tensor
        Resolution (d-spacing) values in Angstroms.
    rms_error : float, optional
        RMS coordinate error in Angstroms. Default is 1.0 A.
        Typical values: 0.5-1.0 for good models, 1.5-2.0 for poor models.
    f_p : float, optional
        Fraction of partial structure (completeness). Default is 1.0.
    f_sol : float, optional
        Solvent contribution factor. Default is 0.95.
    b_sol : float, optional
        Solvent B-factor in A^2. Default is 300.0.

    Returns
    -------
    torch.Tensor
        D factors for each resolution, same shape as input.

    Notes
    -----
    At low resolution, D approaches f_p (good correlation).
    At high resolution, D decreases due to coordinate errors.
    The solvent term reduces D at low resolution due to solvent disorder.

    Examples
    --------
    ::

        d = torch.tensor([10.0, 5.0, 3.0, 2.0, 1.5])  # Angstroms
        D = compute_d_factors(d, rms_error=1.0)
        D[0] > D[-1]  # D decreases with resolution
    True
    """
    # Guard against zero resolution
    d = torch.clamp(resolution, min=1e-6)

    # Coordinate error term: exp(-2*pi^2*sigma_r^2/d^2)
    # This is the Debye-Waller-like decay due to coordinate uncertainty
    coord_term = torch.exp(-2.0 * (math.pi**2) * (rms_error**2) / (d**2))

    # Solvent term: (1 - f_sol * exp(-b_sol/(4*d^2)))
    # At low resolution, solvent scattering reduces correlation
    solvent_term = 1.0 - f_sol * torch.exp(-b_sol / (4.0 * d**2))

    # Combined D factor
    D = f_p * coord_term * solvent_term

    # Clamp to valid range [0, 1]
    D = torch.clamp(D, min=0.0, max=1.0)

    return D


def estimate_mltf_variance(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    D: torch.Tensor,
    epsilon: torch.Tensor,
    N_expected: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Estimate variance for the ML translation function.

    Uses the Read (2001) formulation where variance represents the
    uncertainty in the expected structure factor amplitude:

        sigma^2 = epsilon * Sigma_N * (1 - D^2)

    This formulation ensures variance is always positive when D < 1
    and properly accounts for model incompleteness.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factor amplitudes (used for Sigma_N estimate).
    D : torch.Tensor
        D factors for each reflection.
    epsilon : torch.Tensor
        Multiplicity (epsilon) factors for each reflection.
    N_expected : torch.Tensor, optional
        Expected total scattering power per reflection (Sigma_N).
        If None, estimated from F_obs^2 / epsilon.

    Returns
    -------
    torch.Tensor
        Variance values for each reflection.

    Notes
    -----
    The variance represents the expected squared deviation between
    observed and model amplitudes due to:
    - Missing atoms in the model (1 - D^2 term)
    - Coordinate errors (embedded in D)
    - Solvent scattering (embedded in D)

    References
    ----------
    Read, R.J. (2001). Pushing the boundaries of molecular replacement
    with maximum likelihood. Acta Cryst. D57, 1373-1382.
    """
    # Estimate Sigma_N (expected total scattering power) if not provided
    if N_expected is None:
        # Use Wilson-like estimate: |F_obs|^2 / epsilon
        # This represents the total expected scattering per reflection
        epsilon_safe = torch.clamp(epsilon, min=1.0)
        N_expected = (F_obs**2) / epsilon_safe

    # Variance: epsilon * Sigma_N * (1 - D^2)
    # This is always positive when D < 1
    # The (1 - D^2) term represents the fraction of scattering not
    # explained by the model
    D_safe = torch.clamp(D, min=0.0, max=0.9999)
    variance = epsilon * N_expected * (1.0 - D_safe**2)

    # Ensure variance is positive (numerical guard)
    variance = torch.clamp(variance, min=1e-8)

    return variance


def mltf(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    D: torch.Tensor,
    epsilon: torch.Tensor,
    centric_flags: torch.Tensor,
    N_expected: Optional[torch.Tensor] = None,
    return_per_reflection: bool = False,
) -> torch.Tensor:
    """
    Maximum Likelihood Translation Function (MLTF).

    Computes the log-likelihood gain (LLG) for the calculated structure
    factors given the observed data. The MLTF is the target function
    for molecular replacement refinement.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes |F_obs|.
    F_calc : torch.Tensor
        Calculated structure factor amplitudes |F_calc|.
    D : torch.Tensor
        D factors (coordinate error model) for each reflection.
    epsilon : torch.Tensor
        Multiplicity factors for each reflection.
    centric_flags : torch.Tensor
        Boolean mask, True for centric reflections.
    N_expected : torch.Tensor, optional
        Expected total scattering power. If None, estimated from data.
    return_per_reflection : bool, optional
        If True, return per-reflection LLG instead of sum.

    Returns
    -------
    torch.Tensor
        Log-likelihood gain. Higher values indicate better model fit.
        If return_per_reflection=False: scalar sum of LLG.
        If return_per_reflection=True: per-reflection LLG tensor.

    Notes
    -----
    The LLG is computed as the difference between the log-likelihood
    of the model and a random model:

        LLG = sum[ log(P(F_obs|model)) - log(P(F_obs|random)) ]

    For acentric reflections: Rice distribution
    For centric reflections: Woolfson (folded normal) distribution

    Examples
    --------
    ::

        F_obs = torch.tensor([100.0, 80.0, 60.0])
        F_calc = torch.tensor([95.0, 75.0, 55.0])
        D = torch.tensor([0.9, 0.85, 0.8])
        epsilon = torch.tensor([1.0, 1.0, 1.0])
        centric = torch.tensor([False, False, False])
        llg = mltf(F_obs, F_calc, D, epsilon, centric)
        llg > 0  # Good model should have positive LLG
    True
    """
    # Estimate variance
    variance = estimate_mltf_variance(F_obs, F_calc, D, epsilon, N_expected)

    # Expected amplitude: D * |F_calc|
    F_mean = D * torch.abs(F_calc)

    # Compute log-likelihood for model
    log_likelihood_model = combined_log_likelihood(
        F_obs, F_mean, variance, centric_flags
    )

    # Compute log-likelihood for random model (F_mean = 0)
    # This is the Wilson distribution expectation
    F_random = torch.zeros_like(F_mean)
    variance_random = estimate_mltf_variance(
        F_obs, F_random, D * 0, epsilon, N_expected
    )
    log_likelihood_random = combined_log_likelihood(
        F_obs, F_random, variance_random, centric_flags
    )

    # Log-likelihood gain
    llg = log_likelihood_model - log_likelihood_random

    if return_per_reflection:
        return llg
    else:
        return llg.sum()


def compute_llg(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    resolution: torch.Tensor,
    epsilon: torch.Tensor,
    centric_flags: torch.Tensor,
    rms_error: float = 1.0,
) -> torch.Tensor:
    """
    Convenience function to compute LLG from structure factors.

    Combines D factor computation and MLTF into a single call.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factor amplitudes.
    resolution : torch.Tensor
        Resolution (d-spacing) for each reflection in Angstroms.
    epsilon : torch.Tensor
        Multiplicity factors.
    centric_flags : torch.Tensor
        Boolean mask for centric reflections.
    rms_error : float, optional
        RMS coordinate error in Angstroms. Default is 1.0.

    Returns
    -------
    torch.Tensor
        Scalar log-likelihood gain.
    """
    D = compute_d_factors(resolution, rms_error=rms_error)
    return mltf(F_obs, F_calc, D, epsilon, centric_flags)


def estimate_rms_from_correlation(
    correlation: torch.Tensor,
    resolution: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate RMS coordinate error from observed correlation.

    Inverts the D factor formula to estimate the coordinate error
    that would produce the observed correlation at a given resolution.

    Parameters
    ----------
    correlation : torch.Tensor
        Observed correlation coefficient (can be per-resolution-bin).
    resolution : torch.Tensor
        Resolution (d-spacing) in Angstroms.

    Returns
    -------
    torch.Tensor
        Estimated RMS coordinate error in Angstroms.

    Notes
    -----
    Assumes correlation ~ D, so:
        sigma_r = d / (sqrt(2) * pi) * sqrt(-log(correlation))

    This is a rough estimate; actual error estimation requires more
    sophisticated analysis.
    """
    # Clamp correlation to valid range
    corr = torch.clamp(correlation, min=1e-6, max=0.9999)

    # Invert the main term of D factor: D ~ exp(-2*pi^2*sigma^2/d^2)
    # log(D) = -2*pi^2*sigma^2/d^2
    # sigma = d / (sqrt(2) * pi) * sqrt(-log(D))
    rms = resolution / (math.sqrt(2.0) * math.pi) * torch.sqrt(-torch.log(corr))

    return rms


class MLTargetFunction(torch.nn.Module):
    """
    Reusable ML target function with precomputed factors.

    This class precomputes resolution-dependent factors (D, epsilon, centric)
    for efficiency when the target function is called repeatedly during
    optimization.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    resolution : torch.Tensor
        Resolution for each reflection in Angstroms.
    epsilon : torch.Tensor
        Multiplicity factors.
    centric_flags : torch.Tensor
        Boolean mask for centric reflections.
    rms_error : float, optional
        Initial RMS coordinate error estimate. Default is 1.0.

    Attributes
    ----------
    D : torch.Tensor
        Precomputed D factors.
    N_expected : torch.Tensor
        Estimated expected scattering power.
    """

    def __init__(
        self,
        F_obs: torch.Tensor,
        resolution: torch.Tensor,
        epsilon: torch.Tensor,
        centric_flags: torch.Tensor,
        rms_error: float = 1.0,
    ):
        super().__init__()

        # Store observed data as buffers (not parameters)
        self.register_buffer("F_obs", F_obs)
        self.register_buffer("resolution", resolution)
        self.register_buffer("epsilon", epsilon)
        self.register_buffer("centric_flags", centric_flags)

        # Compute D factors
        D = compute_d_factors(resolution, rms_error=rms_error)
        self.register_buffer("D", D)

        # Estimate expected scattering power
        epsilon_safe = torch.clamp(epsilon, min=1.0)
        N_mean = (F_obs**2 / epsilon_safe).mean()
        N_expected = torch.full_like(F_obs, N_mean.item())
        self.register_buffer("N_expected", N_expected)

        self.rms_error = rms_error

    def update_rms_error(self, rms_error: float) -> None:
        """Update D factors with new RMS error estimate."""
        self.rms_error = rms_error
        D = compute_d_factors(self.resolution, rms_error=rms_error)
        self.D = D

    def forward(self, F_calc: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood gain for optimization.

        Parameters
        ----------
        F_calc : torch.Tensor
            Calculated structure factor amplitudes.

        Returns
        -------
        torch.Tensor
            Negative LLG (for minimization).
        """
        llg = mltf(
            self.F_obs,
            F_calc,
            self.D,
            self.epsilon,
            self.centric_flags,
            N_expected=self.N_expected,
        )
        # Return negative for minimization
        return -llg

    def llg(self, F_calc: torch.Tensor) -> torch.Tensor:
        """Compute LLG (not negated)."""
        return mltf(
            self.F_obs,
            F_calc,
            self.D,
            self.epsilon,
            self.centric_flags,
            N_expected=self.N_expected,
        )
