import torch
from typing import TYPE_CHECKING

from torchref.base.targets.xray_likelihoods import (
    amplitude_var_from_sigma_obs,
    nll_per_refl,
)
from torchref.base.targets.xray_nll import nll_sigma_obs_math

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class NLLXrayTarget(XrayTarget):
    """``--xray-mode nll``: Gaussian amplitude NLL weighted by the experimental sigma.

        NLL = 0.5*(F_obs - |F_calc|)²/σ² + log(σ) + 0.5*log(2π)

    No model-error term, so it does **not** control overfitting -- the only x-ray target
    here that does not. Was ``GaussianXrayTarget``; the taxonomy names the row, and
    "Gaussian" named the distribution, which ``nll_beta`` shares.

    **Not a** :class:`SigmaAXrayTarget`, and deliberately so. Beyond needing no estimate, it
    reads its amplitudes through :meth:`XrayTarget.get_data`, which goes via
    ``ReflectionData._corrected_or_raw`` and falls back to **raw** amplitudes when the scaler
    has not run; the sigma_A path calls ``get_corrected_data()``, which raises instead.
    Moving this target onto that path would turn a silent fallback into a hard failure on
    unscaled data -- a behaviour change, not a refactor. It would also lose the fused Triton
    kernel and the ``median(sigma)*0.1`` clamp, neither of which the beta-variance path has.

    Attributes
    ----------
    target_value : float
        Reference value carried for the generic ``Target`` machinery. Note the
        loss returned by ``forward`` is a summed (not per-reflection normalized)
        NLL, so this value is not a tight per-reflection target.
    """

    target_value: float = 1.0

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the sigma-weighted Gaussian NLL.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Summed NLL loss on this target's set.
        """
        F_obs, F_calc, sigma, _, _ = self.get_data(fcalc=fcalc)
        return nll_sigma_obs_math(F_obs, F_calc, sigma)

    def _per_refl(self, ctx) -> torch.Tensor:
        """The eager twin of :meth:`forward`'s fused kernel.

        ``forward`` keeps ``nll_sigma_obs_math``, which on CUDA float32 fuses the
        per-reflection Gaussian and the sum into one Triton kernel and never
        materialises this tensor. That fusion is why the two are written out
        separately instead of ``forward`` being ``_masked_sum(self._per_refl(...))``;
        ``tests/unit/refinement/test_xray_residuals.py`` pins them to each other.

        The variance builder must stay :func:`amplitude_var_from_sigma_obs` -- it
        carries the ``median(sigma)*0.1`` clamp that the Triton kernel also applies
        internally, and dropping it here would make the two disagree on weak data.
        """
        F_obs, F_calc, sigma, _, _ = ctx
        return nll_per_refl(F_obs, F_calc, amplitude_var_from_sigma_obs(sigma))
