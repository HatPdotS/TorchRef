"""Dispatcher for the ``nll`` target's Gaussian, at ``var = sigma_obs**2``.

This module holds **no math**. The Gaussian lives once, in
:func:`torchref.base.targets.xray_likelihoods.nll_math`, and the amplitude variance is built
once, by :func:`~torchref.base.targets.xray_likelihoods.amplitude_var_from_sigma_obs`. What
is left here is the Triton fast path and the choice between it and the eager primitive.

Was ``xray_gaussian.py``, which carried its own full copy of the Gaussian -- a copy that was
bit-for-bit the same arithmetic as the one ``nll_beta`` used, differing only in how the
variance was constructed. Both copies are gone.

The Triton kernel is kept and is **not** rewritten to take ``var``: it is a fused kernel for
this specific case (it computes the sigma clamp inside), rewriting it would mean editing CUDA
that the development node cannot execute, and an eager/Triton pair per loss is the
established pattern in this package -- ``bond``, ``angle`` and ``ls`` all have one, and
``tests/unit/test_gradient_correctness.py`` cosine-compares them.
"""

import torch

from ._dispatch import use_triton
from .xray_likelihoods import amplitude_var_from_sigma_obs, nll_math


def nll_sigma_obs_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Gaussian amplitude NLL weighted by the experimental sigma alone (``--xray-mode nll``).

    No model-error term, so it does not control overfitting. ``mask`` defaults to all
    reflections (``None``); compact (already-subset) inputs need none.

    Verified bit-identical -- loss *and* gradient -- to the eager Gaussian this replaced,
    over 3000 random sigmas in float64.
    """
    if mask is None:
        mask = torch.ones(F_obs.shape[0], dtype=torch.bool, device=F_obs.device)
    if use_triton(F_calc, F_obs, sigma):
        from .triton.xray_nll import nll_sigma_obs_math_triton

        return nll_sigma_obs_math_triton(F_obs, F_calc, sigma, mask)
    return nll_math(F_obs, F_calc, amplitude_var_from_sigma_obs(sigma), mask=mask)
