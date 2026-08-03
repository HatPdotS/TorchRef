"""
Loss functions for crystallographic refinement.

Functions for computing various loss/likelihood functions used in
structure refinement.
"""

import torch


def _nll_xray_per_refl(F_obs, F_calc, sigma_F_obs):
    """Shared per-reflection Gaussian NLL and mask for the ``nll_xray`` family.

    ``0.5 (F_obs - |F_calc|)^2 / sigma^2 + log sigma + 0.5 log 2pi``

    The sigma floor is data-dependent: each sigma is clamped to a minimum of
    ``median(sigma_F_obs) * 0.1`` rather than a fixed constant, so the effective
    minimum scales with the data.
    """
    # MaskedTensor inputs: use torch.where rather than boolean indexing, which
    # triggers nonzero() and forces a CPU-GPU sync.
    mask = None
    if hasattr(F_obs, "get_mask"):
        mask = F_obs.get_mask()
        F_obs = torch.where(mask, F_obs.get_data(), torch.zeros_like(F_obs.get_data()))
        F_calc = torch.where(mask, F_calc, torch.zeros_like(F_calc))
        sigma_raw = (
            sigma_F_obs.get_data() if hasattr(sigma_F_obs, "get_mask") else sigma_F_obs
        )
        sigma_F_obs = torch.where(mask, sigma_raw, torch.ones_like(sigma_raw))
    elif hasattr(sigma_F_obs, "get_mask"):
        mask = sigma_F_obs.get_mask()
        F_obs = torch.where(mask, F_obs, torch.zeros_like(F_obs))
        F_calc = torch.where(mask, F_calc, torch.zeros_like(F_calc))
        sigma_F_obs = torch.where(
            mask, sigma_F_obs.get_data(), torch.ones_like(sigma_F_obs.get_data())
        )

    diff = F_obs - torch.abs(F_calc)
    eps = torch.median(sigma_F_obs) * 1e-1
    sigma_save = torch.clamp(sigma_F_obs, min=eps)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi))
    nll = 0.5 * (diff**2) / (sigma_save**2) + torch.log(sigma_save) + 0.5 * log_2pi
    return nll, mask


def nll_xray(
    F_obs: torch.Tensor, F_calc: torch.Tensor, sigma_F_obs: torch.Tensor
) -> torch.Tensor:
    """
    Summed X-ray negative log-likelihood assuming a Gaussian on the amplitude.

    Parameters
    ----------
    F_obs : torch.Tensor or MaskedTensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    sigma_F_obs : torch.Tensor or MaskedTensor
        Standard deviations of observed amplitudes.

    Returns
    -------
    torch.Tensor
        Summed negative log-likelihood (scalar).

    Notes
    -----
    **This returns a SUM.** Every other target in TorchRef is on an absolute
    scale -- all five x-ray target maths (``xray_likelihoods``, ``xray_ml_full``,
    ``xray_ls``, ``xray_nll``), every geometry term
    (bond, angle, chiral, planarity, torsion, ramachandran, nonbonded) and the ADP
    terms all return ``.sum()``. This function previously returned a ``.mean()``,
    which made it the sole exception: any caller combining it with another term
    (e.g. an anisotropy ``sum(U**2)`` penalty) silently got a relative weight wrong
    by a factor of the reflection count. It now matches the convention, so absolute
    magnitudes reported by it are ~n times larger than before.

    ``nll_xray_sum`` is retained as an alias.

    The sigma floor is data-dependent: each sigma is clamped to a minimum of
    ``median(sigma_F_obs) * 0.1``, so the effective minimum scales with the data.
    """
    nll, mask = _nll_xray_per_refl(F_obs, F_calc, sigma_F_obs)
    if mask is not None:
        return (nll * mask).sum()
    return nll.sum()


def nll_xray_mean(
    F_obs: torch.Tensor, F_calc: torch.Tensor, sigma_F_obs: torch.Tensor
) -> torch.Tensor:
    """Per-reflection *mean* Gaussian NLL.

    Use only for reporting a number that should be comparable between datasets of
    different sizes. Never mix it with another loss term -- see the convention note
    in :func:`nll_xray`.
    """
    nll, mask = _nll_xray_per_refl(F_obs, F_calc, sigma_F_obs)
    if mask is not None:
        return (nll * mask).sum() / mask.sum()
    return nll.mean()


# Alias: `nll_xray` is itself a sum now, so these are the same function. Kept
# because it is re-exported from torchref.base, torchref.base.math_torch and
# torchref.base.metrics.
nll_xray_sum = nll_xray


def nll_xray_lognormal(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma_F_obs: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute X-ray negative log-likelihood assuming lognormal distribution.

    This is a more realistic model for structure factor amplitudes, which must
    be positive. For a lognormal distribution LogNormal(mu, sigma^2), the NLL is:
    NLL = 0.5*(log(x) - mu)^2/sigma^2 + log(x) + log(sigma) + 0.5*log(2*pi)

    Where mu and sigma are derived from F_obs and sigma_F_obs using:
    - sigma = sqrt(log(1 + (sigma_F/F)^2))
    - mu = log(F) - sigma^2/2

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    sigma_F_obs : torch.Tensor
        Standard deviations of observed amplitudes.
    eps : float, optional
        Small value to avoid numerical issues. Default is 1e-10.

    Returns
    -------
    torch.Tensor
        Mean negative log-likelihood.
    """
    # Compute amplitude of calculated structure factors
    F_calc_amp = torch.abs(F_calc)

    # Ensure positive values
    F_obs_safe = torch.clamp(F_obs, min=eps)
    F_calc_safe = torch.clamp(F_calc_amp, min=eps)
    sigma_F_safe = torch.clamp(sigma_F_obs, min=eps)

    # Convert Gaussian parameters to lognormal parameters
    # For lognormal: CV² = exp(σ²) - 1, where CV = sigma_F/F
    CV = sigma_F_safe / F_obs_safe
    CV_squared = CV**2
    sigma_ln = torch.sqrt(torch.log1p(CV_squared))  # σ of lognormal

    # μ = log(F) - σ²/2
    mu_ln = torch.log(F_obs_safe) - 0.5 * sigma_ln**2

    # Lognormal NLL: 0.5*(log(x) - μ)²/σ² + log(x) + log(σ) + 0.5*log(2π)
    log_F_calc = torch.log(F_calc_safe)
    diff = log_F_calc - mu_ln

    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=F_obs.device))
    nll = (
        0.5 * (diff**2) / (sigma_ln**2 + eps)
        + log_F_calc
        + torch.log(sigma_ln + eps)
        + 0.5 * log_2pi
    )

    # Mean over all reflections
    return nll.mean()


def log_loss(
    F_obs: torch.Tensor, F_calc: torch.Tensor, sigma_F_obs: torch.Tensor
) -> torch.Tensor:
    """
    Compute log-space loss between observed and calculated structure factors.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    sigma_F_obs : torch.Tensor
        Standard deviations of observed amplitudes (unused).

    Returns
    -------
    torch.Tensor
        Mean absolute difference in log space.
    """
    # Compute amplitude of calculated structure factors
    F_calc_amp = torch.abs(F_calc)

    # Compute residual
    diff = torch.log(F_obs) - torch.log(F_calc_amp)
    return torch.mean(torch.abs(diff))


def estimate_sigma_I(I):
    """
    Estimate standard deviation of intensities.

    Separates positive and negative values for robust estimation.

    Parameters
    ----------
    I : torch.Tensor
        Intensity values.

    Returns
    -------
    torch.Tensor
        Estimated standard deviations.
    """
    if torch.any(I < 0):
        neg_I_sig = torch.mean(I[I < 0] ** 2) ** 0.5
        sigma = I * 0.05 + neg_I_sig
    else:
        sigma = I * 0.05 + torch.mean(I) * 0.01
    return sigma


def estimate_sigma_F(F):
    """
    Estimate standard deviation of structure factor amplitudes.

    Parameters
    ----------
    F : torch.Tensor
        Structure factor amplitudes.

    Returns
    -------
    torch.Tensor
        Estimated standard deviations.
    """
    sigma = F * 0.05 + torch.mean(F) * 0.01
    return sigma


def gaussian_to_lognormal_sigma(
    F: torch.Tensor, sigma_F: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Approximate the sigma parameter of a lognormal distribution from Gaussian statistics.

    If we assume F comes from a lognormal distribution X ~ LogNormal(mu, sigma^2), then:
    - Mean: E[X] = F
    - Std: sqrt(Var[X]) = sigma_F

    For lognormal distribution:
    - E[X] = exp(mu + sigma^2/2)
    - Var(X) = exp(2*mu + sigma^2)(exp(sigma^2) - 1)

    We can derive:
    - CV^2 = Var[X]/E[X]^2 = exp(sigma^2) - 1
    - sigma = sqrt(log(1 + CV^2))

    where CV = sigma_F/F is the coefficient of variation.

    Parameters
    ----------
    F : torch.Tensor
        Structure factor amplitudes (mean of the distribution).
    sigma_F : torch.Tensor
        Standard deviations.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-10.

    Returns
    -------
    torch.Tensor
        Sigma parameter for lognormal distribution.
    """
    # Avoid division by zero
    F_safe = torch.clamp(F, min=eps)
    sigma_F_safe = torch.clamp(sigma_F, min=eps)

    # Compute coefficient of variation (CV)
    CV = sigma_F_safe / F_safe

    # Compute CV²
    CV_squared = CV**2

    # For lognormal: CV² = exp(σ²) - 1
    # Therefore: σ = √(log(1 + CV²))
    sigma_lognormal = torch.sqrt(torch.log1p(CV_squared))

    return sigma_lognormal


def gaussian_to_lognormal_mu(
    F: torch.Tensor, sigma_lognormal: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Calculate the mu parameter of a lognormal distribution given F and sigma.

    For lognormal distribution X ~ LogNormal(mu, sigma^2):
    - E[X] = exp(mu + sigma^2/2)

    Solving for mu:
    - mu = log(E[X]) - sigma^2/2

    Parameters
    ----------
    F : torch.Tensor
        Structure factor amplitudes (mean of the distribution).
    sigma_lognormal : torch.Tensor
        Sigma parameter from lognormal distribution.
    eps : float, optional
        Small value to avoid log of zero. Default is 1e-10.

    Returns
    -------
    torch.Tensor
        Mu parameter for lognormal distribution.
    """
    F_safe = torch.clamp(F, min=eps)
    mu_lognormal = torch.log(F_safe) - 0.5 * sigma_lognormal**2
    return mu_lognormal
