"""
Metrics and loss functions for crystallographic refinement.

This submodule provides functions for:
- R-factor calculations
- Negative log-likelihood loss functions
- Statistical metrics
"""

from .rfactor import (
    get_rfactors,
    rfactor_work_free,
    bin_wise_rfactors,
)

from .loss import (
    nll_xray,
    nll_xray_sum,
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
    "get_rfactors",
    "rfactor_work_free",
    "bin_wise_rfactors",
    "binwise_scale",
    # Loss functions
    "nll_xray",
    "nll_xray_sum",
    "nll_xray_lognormal",
    "log_loss",
    "estimate_sigma_I",
    "estimate_sigma_F",
    "gaussian_to_lognormal_sigma",
    "gaussian_to_lognormal_mu",
]
