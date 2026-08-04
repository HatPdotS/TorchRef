"""The shared sigma_A data path, and the three hooks a sigma_A likelihood fills.

Four selectable modes -- ``ml``, ``ml_noalpha``, ``nll_beta``, ``ml_full`` -- differ only in
their likelihood, their conditional mean and (for ``ml_full``) which variance field they read.
Everything else lives here: one estimator, one geometry cache, one estimator call, one
compaction, one ``maintenance``.

**No code in this module branches on which mode is running**, and none should: the three
differences are the :meth:`SigmaAXrayTarget._model_error`, :meth:`SigmaAXrayTarget._mean` and
:meth:`SigmaAXrayTarget._loss` overrides, so a new row is a new class rather than a new
``elif``. One class per mode is checked at import by
:class:`~torchref.refinement.targets.xray._specs.XrayTargetTable`.

``nll`` is deliberately *not* a subclass: it needs no estimate, and it reads amplitudes
through :meth:`XrayTarget.get_data`, which falls back to **raw** amplitudes when the scaler
has not run, where this path calls ``get_corrected_data()`` and raises. Moving it here would
turn that silent fallback into a hard failure and lose its fused Triton kernel and its
``median(sigma)*0.1`` clamp.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from torchref.base.reciprocal import get_scattering_vectors
from torchref.base.targets.xray_likelihoods import complex_var_from_beta
from torchref.refinement.model_error_estimation.sigma_a import (
    SigmaAEstimate,
    SigmaAEstimator,
    epsilon_from_hkl,
)

from .base import XrayTarget


@dataclass(frozen=True)
class SigmaALossInputs:
    """Everything a sigma_A likelihood is evaluated from, on one reflection subset.

    Frozen and passed whole rather than spread over keyword arguments, since the four
    likelihoods each read a different subset of these fields.

    Attributes
    ----------
    F_obs
        Compact corrected amplitudes, from the SAME full-size array the estimator was fed.
    F_calc
        Compact scaled amplitude, **already centred** by :meth:`SigmaAXrayTarget._mean`:
        ``|F_c|``, or ``alpha*|F_c|`` for the rows whose mean says so.
    Sigma
        Compact **complex** variance -- what a Rice denominator takes, ``epsilon`` times
        whatever :meth:`SigmaAXrayTarget._model_error` selected. ``nll_beta`` converts it to
        an amplitude variance itself; that conversion is the large-signal limit and is the
        one place the two variance conventions must not be confused.
    centric
        ``sub.centric``.
    sigma_obs_full
        FULL-SIZE corrected sigmas, uncompacted **on purpose** -- only ``ml_full`` reads them,
        so compacting here would be work the other three rows discard. Call :meth:`compact`.
    est
        The whole estimate, full-size, for a hook needing a field this context does not name
        (:meth:`SigmaAXrayTarget._mean` reads ``est.alpha``).
    sub
        The ``_ReflectionSubset`` view. ``ml_full`` needs it for its parity cache key.
    """

    F_obs: torch.Tensor
    F_calc: torch.Tensor
    Sigma: torch.Tensor
    centric: torch.Tensor
    sigma_obs_full: torch.Tensor
    est: SigmaAEstimate
    sub: object

    def compact(self, t: torch.Tensor) -> torch.Tensor:
        """Restrict a full-size array to this subset, in ``F_obs``'s dtype."""
        return self.sub.select(t).to(self.F_obs.dtype)


class SigmaAXrayTarget(XrayTarget):
    """Abstract base for the four sigma_A-family targets. Not itself selectable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Constructed ONCE, here -- never in forward(). The estimator caches internally
        # until `maintenance()` resets it, which `LossState.optimize` calls after the step
        # loop, so one estimate serves a whole optimizer-step block (every LBFGS inner
        # iteration and line-search evaluation included). Rebuilding it per forward is
        # correct and ruinous. The invalidation half is pinned by
        # tests/unit/refinement/test_ml_sigmaa.py::test_maintenance_resets_cache; that the
        # construction stays HERE is not asserted anywhere, so read the indentation.
        self._sigma_a = SigmaAEstimator()
        # Model-independent per-reflection geometry (multiplicity + d*^2), cached per
        # ReflectionData instance (mirrors the bins cache pattern).
        self._eps_cache: torch.Tensor = None
        self._dss_cache: torch.Tensor = None
        self._geom_dataid: int = None

    # --- estimator knobs, read from the one config every x-ray target carries ---
    @property
    def sigma_a_max(self) -> float:
        return self.sigma_a_config.sigma_a_max

    @property
    def shrink(self) -> bool:
        """ONE shrinkage setting for every target; never make it row-dependent, or a
        comparison between two rows measures the estimator as well as the likelihood.
        """
        return self.sigma_a_config.shrink

    # --- the three hooks -------------------------------------------------------
    def _model_error(self, est: SigmaAEstimate) -> torch.Tensor:
        """Full-size model-error variance. Default ``est.beta``, the TOTAL variance --
        concrete, not abstract, because it is the safe answer for anything added later.
        Override only in a likelihood that accounts for ``sigma_obs`` itself (``ml_full``):
        reading ``beta_model`` elsewhere understates the variance exactly where ``sigma_obs``
        matters, a measurable and directional error.
        """
        return est.beta

    def _mean(self, F_calc: torch.Tensor, est: SigmaAEstimate, sub) -> torch.Tensor:
        """Compact amplitude the likelihood centres on, default ``|F_calc|``. Takes loose
        arguments rather than a context because the context is built *from* its result.
        """
        return F_calc

    def _loss(self, ctx: SigmaALossInputs) -> torch.Tensor:
        """The likelihood. One per selectable row; no row is a branch."""
        raise NotImplementedError

    # --- shared helper the hooks may call; forward() never does ----------------
    @staticmethod
    def _alpha_centred(F_calc, est: SigmaAEstimate, sub) -> torch.Tensor:
        """``alpha * |F_calc|``; the ONLY place ``sub.select(alpha)`` appears.

        Casts ``alpha`` to ``F_calc.dtype`` rather than going through the generic
        ``ctx.compact``, which casts to ``F_obs.dtype``: the context is built *from*
        this result, so it does not exist yet here. ``alpha`` stays detached, so this
        reweights the residual without adding a parameter to the gradient.
        """
        return F_calc * sub.select(est.alpha).to(F_calc.dtype)

    # --- shared machinery -----------------------------------------------------
    def _geom(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(epsilon, d_star_sq)`` over the full data HKL. Both are model-independent
        (pure multiplicity / reciprocal geometry), so they are computed once per
        :class:`ReflectionData` and cached against its id.
        """
        dataid = id(self._data)
        if self._eps_cache is None or self._geom_dataid != dataid:
            sg = getattr(self._data, "spacegroup", None)
            eps = epsilon_from_hkl(self._data.hkl, sg)
            s = get_scattering_vectors(self._data.hkl, self._data.cell)
            # d*^2 = |s|^2 = 4 * (|s|/2)^2 (the scaler's _s_half_sq convention).
            dss = (torch.norm(s, dim=1) ** 2).to(eps.dtype)
            self._eps_cache = eps
            self._dss_cache = dss
            self._geom_dataid = dataid
        return self._eps_cache, self._dss_cache

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Read-MLF-family loss with free-set sigma_A variance ``epsilon * beta``.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, used instead of computing from the
            model.

        Returns
        -------
        torch.Tensor
            Summed loss on this target's set (work, free or validation).
        """
        sub = self._subset()

        # Full-size scaled |F_calc| (aligned to data.hkl). beta is estimated on the full
        # free set, so it needs the full-size arrays.
        if fcalc is not None:
            F_calc_full = self.get_F_calc_scaled(fcalc=fcalc)
        else:
            F_calc_full = self.get_F_calc_scaled(self._data.hkl_for_sf(), recalc=False)

        eps_full, dss_full = self._geom()
        eps_full = eps_full.to(F_calc_full.dtype)
        dss_full = dss_full.to(F_calc_full.dtype)
        # ONE data path. `sub.F` / `sub.sigF` go through `_corrected_or_raw()`, which
        # silently falls back to RAW amplitudes when the scaler has not run, while the
        # estimator below is fed `get_corrected_data()`, which raises instead. Mixing the two
        # can put raw amplitudes and a scaled-data variance in the same loss.
        F_obs_full, sigma_full = self._data.get_corrected_data()
        F_obs_full = F_obs_full.to(F_calc_full.dtype).reshape(-1)
        centric_full = self._data.centric
        # ALWAYS the free set, independent of this target's own `use_set`.
        est_mask = self._data.free.mask

        # Detached, cached until maintenance() resets it.
        #
        # `sigma_obs` is passed for EVERY row, not just the ones that read `beta_model`. It
        # defines what sigma_A means: with it, sigma_A is the correlation with the noise-free
        # true amplitudes (Read, since Sigma_N = B - S2); without it, with the noisy data.
        # Passing it selectively gives the rows different sigma_A definitions.
        est = self._sigma_a.get(
            F_obs_full, F_calc_full, centric_full, eps_full, dss_full, est_mask,
            sigma_obs=sigma_full.to(F_calc_full.dtype).reshape(-1),
            sigma_a_max=self.sigma_a_max,
            shrink=self.shrink,
        )

        F_obs = sub.select(F_obs_full)
        eps = est.epsilon
        eps_c = sub.select(eps).to(F_obs.dtype) if eps is not None else None
        return self._loss(
            SigmaALossInputs(
                F_obs=F_obs,
                F_calc=self._mean(sub.select(F_calc_full), est, sub),
                Sigma=complex_var_from_beta(
                    sub.select(self._model_error(est)).to(F_obs.dtype), eps_c
                ),
                centric=sub.centric,
                sigma_obs_full=sigma_full,
                est=est,
                sub=sub,
            )
        )

    def maintenance(self) -> None:
        """Invalidate the cached estimate so it refits from the updated model on the next
        forward. ``LossState`` calls this after each optimizer-step block -- see the note in
        :meth:`__init__` on why that cadence is load-bearing."""
        self._sigma_a.reset()


class AlphaCentredMixin:
    """Centre the likelihood on ``alpha*|F_calc|`` instead of ``|F_calc|``.

    A mixin rather than a copied override so ``sub.select(alpha)`` stays in exactly one
    place (:meth:`SigmaAXrayTarget._alpha_centred`).
    """

    def _mean(self, F_calc, est, sub):
        return self._alpha_centred(F_calc, est, sub)
