"""
Amplitude-space loss/likelihood functions for crystallographic refinement.

Note the reduction convention: the ``nll_xray`` family returns a **sum** (so it combines
with the geometry and ADP targets at the intended relative weight) while the lognormal and
log-space losses return a **mean**.
"""

import torch


def _nll_xray_per_refl(F_obs, F_calc, sigma_F_obs):
    """Per-reflection Gaussian NLL + mask shared by the ``nll_xray`` family.

    ``0.5 (F_obs - |F_calc|)^2 / sigma^2 + log sigma + 0.5 log 2pi``, each sigma clamped to
    a data-dependent floor of ``median(sigma_F_obs) * 0.1``.
    """
    # MaskedTensor inputs: torch.where rather than boolean indexing, which triggers
    # nonzero() and forces a CPU-GPU sync.
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

    Returns a **sum**, matching every other TorchRef target, so it combines with them at
    the intended relative weight; :func:`nll_xray_mean` is for reporting only.
    ``nll_xray_sum`` is an alias. Each sigma is clamped to ``median(sigma_F_obs) * 0.1``.

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


# Alias kept because it is re-exported from torchref.base,
# torchref.base.math_torch and torchref.base.metrics.
nll_xray_sum = nll_xray


def nll_xray_lognormal(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma_F_obs: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    X-ray negative log-likelihood assuming a lognormal on the amplitude.

    ``(mu, sigma)`` are moment-matched from ``(F_obs, sigma_F_obs)``:
    ``sigma = sqrt(log(1 + (sigma_F/F)^2))``, ``mu = log(F) - sigma^2/2``. Note the model
    is centred on the *observation*, so ``|F_calc|`` is the variate.

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    sigma_F_obs : torch.Tensor
        Standard deviations of observed amplitudes.
    eps : float, optional
        Floor guarding the logs and the division. Default 1e-10.

    Returns
    -------
    torch.Tensor
        **Mean** negative log-likelihood -- unlike :func:`nll_xray`, which sums.
    """
    F_calc_amp = torch.abs(F_calc)

    F_obs_safe = torch.clamp(F_obs, min=eps)
    F_calc_safe = torch.clamp(F_calc_amp, min=eps)
    sigma_F_safe = torch.clamp(sigma_F_obs, min=eps)

    # Gaussian -> lognormal parameters: CV² = exp(σ²) - 1 with CV = sigma_F/F,
    # μ = log(F) - σ²/2.
    CV = sigma_F_safe / F_obs_safe
    CV_squared = CV**2
    sigma_ln = torch.sqrt(torch.log1p(CV_squared))
    mu_ln = torch.log(F_obs_safe) - 0.5 * sigma_ln**2

    log_F_calc = torch.log(F_calc_safe)
    diff = log_F_calc - mu_ln

    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=F_obs.device))
    nll = (
        0.5 * (diff**2) / (sigma_ln**2 + eps)
        + log_F_calc
        + torch.log(sigma_ln + eps)
        + 0.5 * log_2pi
    )

    return nll.mean()


def log_loss(
    F_obs: torch.Tensor, F_calc: torch.Tensor, sigma_F_obs: torch.Tensor
) -> torch.Tensor:
    """
    Mean absolute difference of ``log F_obs`` and ``log |F_calc|``.

    ``sigma_F_obs`` is accepted for signature compatibility and **ignored**, so this is
    unweighted; non-positive ``F_obs`` yields non-finite values (there is no clamp).

    Parameters
    ----------
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    F_calc : torch.Tensor
        Calculated structure factors (complex).
    sigma_F_obs : torch.Tensor
        Unused.

    Returns
    -------
    torch.Tensor
        Mean absolute difference in log space.
    """
    F_calc_amp = torch.abs(F_calc)
    diff = torch.log(F_obs) - torch.log(F_calc_amp)
    return torch.mean(torch.abs(diff))


def estimate_sigma_I(I):
    """
    Heuristic standard deviation for intensities that carry no measured sigma.

    5% of the intensity plus a floor: the RMS of the negative intensities when any are
    present, otherwise 1% of the mean. Not a measurement -- a stand-in.

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
    """Heuristic sigma for amplitudes: 5% of ``F`` plus 1% of its mean. Not measured."""
    sigma = F * 0.05 + torch.mean(F) * 0.01
    return sigma


def gaussian_to_lognormal_sigma(
    F: torch.Tensor, sigma_F: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Sigma parameter of a lognormal moment-matched to Gaussian ``(F, sigma_F)``.

    ``CV^2 = Var/E^2 = exp(sigma^2) - 1``, hence ``sigma = sqrt(log(1 + CV^2))`` with
    ``CV = sigma_F/F``.

    Parameters
    ----------
    F : torch.Tensor
        Structure factor amplitudes (mean of the distribution).
    sigma_F : torch.Tensor
        Standard deviations.
    eps : float, optional
        Floor on both inputs, guarding the division. Default 1e-10.

    Returns
    -------
    torch.Tensor
        The lognormal ``sigma``.
    """
    F_safe = torch.clamp(F, min=eps)
    sigma_F_safe = torch.clamp(sigma_F, min=eps)

    CV = sigma_F_safe / F_safe
    CV_squared = CV**2

    # CV² = exp(σ²) - 1  =>  σ = √(log(1 + CV²))
    sigma_lognormal = torch.sqrt(torch.log1p(CV_squared))

    return sigma_lognormal


def gaussian_to_lognormal_mu(
    F: torch.Tensor, sigma_lognormal: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """
    Mu parameter of a lognormal with mean ``F``: ``mu = log(F) - sigma^2/2``.

    Takes the lognormal ``sigma`` (from :func:`gaussian_to_lognormal_sigma`), not the
    Gaussian ``sigma_F``.

    Parameters
    ----------
    F : torch.Tensor
        Structure factor amplitudes (mean of the distribution).
    sigma_lognormal : torch.Tensor
        Sigma parameter of the lognormal.
    eps : float, optional
        Floor on ``F``, guarding the log. Default 1e-10.

    Returns
    -------
    torch.Tensor
        The lognormal ``mu``.
    """
    F_safe = torch.clamp(F, min=eps)
    mu_lognormal = torch.log(F_safe) - 0.5 * sigma_lognormal**2
    return mu_lognormal
