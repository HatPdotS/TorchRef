"""
R-factors, amplitude-space likelihoods and per-bin scaling.

Two reduction conventions live side by side here: the ``nll_xray`` family sums, the
lognormal loss averages. See :mod:`~torchref.base.metrics.loss`.
"""

from .rfactor import (
    rfactor,
    get_rfactors,
    rfactor_work_free,
)

from .loss import (
    nll_xray,
    nll_xray_mean,
    nll_xray_lognormal,
    estimate_sigma_F,
)

from .binwise_scale import binwise_scale

__all__ = [
    # R-factor
    "rfactor",
    "get_rfactors",
    "rfactor_work_free",
    "binwise_scale",
    # Loss functions
    "nll_xray",
    "nll_xray_mean",
    "nll_xray_lognormal",
    "estimate_sigma_F",
]
