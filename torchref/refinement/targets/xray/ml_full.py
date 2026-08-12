"""``--xray-mode ml_full``: the full-form MLF, marginalising the measurement error."""

from typing import Tuple

import torch

from torchref.base.targets.xray_likelihoods import rice_marginal_per_refl
from torchref.base.targets.xray_ml_full import parity_indices

from .sigma_a import AlphaCentredMixin, SigmaAXrayTarget


class MLFullXrayTarget(AlphaCentredMixin, SigmaAXrayTarget):
    """Rice (x) Gaussian, marginalised over the unknown error-free amplitude.

    The observation error enters as an amplitude-only Gaussian while ``beta`` keeps the phase
    component it physically has, instead of the two error kinds being folded into a single
    variance -- which is what the Green (1979) inflation shortcut
    (:func:`~torchref.base.targets.xray_likelihoods.inflate_with_sigma_obs`) does, and what
    Refmac uses. The most principled of the five, and the most expensive: a 32-node
    Gauss-Legendre quadrature per loss evaluation, ~4x the per-gradient cost of ``ml``.

    It is kept because it is the correct treatment of the two error kinds.

    The only row that overrides :meth:`_model_error`, and the only one carrying extra state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parity split, needed by this row alone. `nonzero()` forces a device sync, so it
        # must not be recomputed per forward. One entry per subset view rather than a
        # single slot: `forward` and `residuals` ask for different views of the same data
        # and would otherwise evict each other on every call.
        self._parity_cache = {}
        self._parity_dataid = None

    def _model_error(self, est):
        """``beta_model``: the model error ALONE.

        This likelihood integrates ``sigma_obs`` explicitly, so a variance that already
        contains the measurement error counts it twice. Every other row reads the total
        ``beta`` -- reading ``beta_model`` there would understate the variance exactly where
        ``sigma_obs`` matters (weak, high-resolution data), a measurable and directional
        error.
        """
        return est.beta_model

    def _parity(self, sub) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cached ``(acentric_idx, centric_idx)``.

        The gather/scatter split is far cheaper than a ``torch.where`` over the quadrature,
        but ``nonzero()`` forces a device sync, so it is cached per (data, subset).

        Keyed on ``sub.kind`` and not on the subset size alone: ``forward`` and
        ``residuals`` reach this with different views of the same data, and two
        views can hold the same number of reflections while being different
        reflections in a different order.
        """
        dataid = id(self._data)
        if self._parity_dataid != dataid:
            self._parity_cache = {}
            self._parity_dataid = dataid
        key = (sub.kind, int(sub.centric.shape[0]))
        idx = self._parity_cache.get(key)
        if idx is None:
            idx = parity_indices(sub.centric)
            self._parity_cache[key] = idx
        return idx

    def _per_refl(self, ctx):
        # `Sigma` is already `epsilon * beta_model` and `alpha` is already folded into
        # `F_calc` by `_mean`, so neither is passed down again -- passing either twice would
        # apply it twice.
        return rice_marginal_per_refl(
            ctx.F_obs,
            ctx.F_calc,
            ctx.Sigma,
            ctx.compact(ctx.sigma_obs_full),
            ctx.centric,
            idx=self._parity(ctx.sub),
        )
