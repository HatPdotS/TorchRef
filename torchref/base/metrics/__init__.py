"""
R-factors, amplitude-space likelihoods and per-bin scaling.

Two reduction conventions live side by side here: the ``nll_xray`` family sums, the
lognormal and log-space losses average. See :mod:`~torchref.base.metrics.loss`.
"""

from .rfactor import (
    rfactor,
    get_rfactors,
    rfactor_work_free,
    bin_wise_rfactors,
)

from .loss import (
    nll_xray,
    nll_xray_sum,
    nll_xray_mean,
    nll_xray_lognormal,
    log_loss,
    estimate_sigma_I,
    estimate_sigma_F,
    gaussian_to_lognormal_sigma,
    gaussian_to_lognormal_mu,
)

from .binwise_scale import binwise_scale

__all__ = [
    # R-factor
    "rfactor",
    "get_rfactors",
    "rfactor_work_free",
    "bin_wise_rfactors",
    "binwise_scale",
    # Loss functions
    "nll_xray",
    "nll_xray_sum",
    "nll_xray_mean",
    "nll_xray_lognormal",
    "log_loss",
    "estimate_sigma_I",
    "estimate_sigma_F",
    "gaussian_to_lognormal_sigma",
    "gaussian_to_lognormal_mu",
]
