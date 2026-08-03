"""The shared sigma_A data path, and the three hooks a sigma_A likelihood fills.

Four selectable modes -- ``ml``, ``ml_noalpha``, ``nll_beta``, ``ml_full`` -- differ in their
likelihood, their conditional mean, and (for ``ml_full``) which variance field they read.
In nothing else. Everything else is here: one estimator, one geometry cache, one estimator
call, one compaction, one ``maintenance``.

**No code in this module branches on which mode is running.** It used to: one class served
all four rows and ``forward`` read ``self.spec`` in four places -- the distribution, the
variance field, whether ``alpha`` centres the mean, and the alpha fold itself. Those are now
:meth:`SigmaAXrayTarget._model_error`, :meth:`SigmaAXrayTarget._mean` and
:meth:`SigmaAXrayTarget._loss` overrides, so a new row is a new class and *cannot* be a new
``elif``. The one-class-per-mode invariant is checked at import time by
:class:`~torchref.refinement.targets.xray._specs.XrayTargetTable`.

Deliberately **not** a class-attribute table (``_LOSS_FN = staticmethod(...)``): that is the
retired ``_beta_loss_fn`` hook, removed because ``ml_full`` had to override ``forward``
wholesale and so bypassed it. It would work now that ``ml_full`` overrides only ``_loss``,
but it buys four lines and costs the reader an indirection.

Two other shapes were considered and rejected, recorded so they are not re-litigated:

* **Mixins** (``Variance`` x ``Likelihood``). The axes are not orthogonal -- reading
  ``beta_model`` in a likelihood that does not itself account for ``sigma_obs`` understates
  the variance exactly where ``sigma_obs`` matters (weak, high-resolution data), a measurable
  and directional error. Mixins make the invalid combinations constructible and MRO order
  load-bearing.
* **Composition**, injecting a variance provider and a likelihood. That is ``self.spec``
  relocated into constructor arguments; the factory would need a conditional to pick the
  pair, so the mode switch survives verbatim -- the opposite of the point.

``nll`` is **not** a subclass of this, and that is not an oversight. Beyond needing no
estimate, it reads its amplitudes through :meth:`XrayTarget.get_data`, which goes via
``ReflectionData._corrected_or_raw`` and falls back to **raw** amplitudes when the scaler has
not run, where this path calls ``get_corrected_data()``, which raises. Moving it here would
turn a silent fallback into a hard failure on unscaled data -- a behaviour change, not a
refactor -- and would also lose its fused Triton kernel and its ``median(sigma)*0.1`` clamp.
See :class:`~torchref.refinement.targets.xray.nll.NLLXrayTarget`.
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

    Frozen and passed whole rather than spread over keyword arguments: the four likelihoods
    read different subsets of these fields, so a kwargs signature would have to be their
    union in all four ``_loss`` definitions (or ``**_``, which stops catching typos), and
    adding a field later would edit four signatures. Same argument
    :class:`SigmaAEstimate` records for replacing its own tuples.

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
        FULL-SIZE corrected sigmas, uncompacted **on purpose**. Only ``ml_full`` consumes
        them, and before the split every row paid for an ``index_select`` it discarded.
        Carrying it uncompacted costs nothing -- the array is already in hand because the
        estimator needs it -- and removes that dead work with no conditional and no laziness
        machinery. Call :meth:`compact`.
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
        # Constructed ONCE, here -- never in forward(). `SigmaAEstimator.get` wraps its whole
        # body in `if self._cache is None`, so the first forward after a reset estimates and
        # every later forward in the same block is a cache hit, including each LBFGS inner
        # iteration and each strong-Wolfe line-search evaluation. `LossState.optimize` calls
        # `maintenance()` AFTER the step loop, not between steps, so one estimate serves a
        # whole optimizer-step block. Building it per forward would give a fresh empty cache
        # every call: correct, and ruinous. Pinned by
        # tests/unit/refinement/test_xray_target_parity.py.
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
        """ONE shrinkage setting for every target.

        It used to be spec-dependent (on where the spec consumed ``alpha``, off otherwise),
        which meant ``ml`` and ``ml_full`` were fitted with *differently configured*
        estimators, so every ``ml`` vs ``ml_full`` comparison confounded the target with the
        estimator. The measurement that justified the split was taken under the old
        ``alpha == 1`` solve, which no longer exists.
        """
        return self.sigma_a_config.shrink

    # --- the three hooks -------------------------------------------------------
    def _model_error(self, est: SigmaAEstimate) -> torch.Tensor:
        """Full-size model-error variance. Default ``est.beta``, the TOTAL variance.

        Correct for any likelihood that does not account for ``sigma_obs`` itself. Only
        ``ml_full`` overrides this, because only ``ml_full`` accounts for the measurement
        error explicitly -- reading ``beta_model`` anywhere else understates the variance
        exactly where ``sigma_obs`` matters, a measurable and directional error. Concrete
        rather than abstract *because* that is the safe answer for anything added later.
        """
        return est.beta

    def _mean(self, F_calc: torch.Tensor, est: SigmaAEstimate, sub) -> torch.Tensor:
        """Compact amplitude the likelihood centres on. Default ``|F_calc|``.

        Takes the three locals it needs rather than a context, because the context is built
        *from* its result.
        """
        return F_calc

    def _loss(self, ctx: SigmaALossInputs) -> torch.Tensor:
        """The likelihood. One per selectable row; no row is a branch."""
        raise NotImplementedError

    # --- shared helper the hooks may call; forward() never does ----------------
    @staticmethod
    def _alpha_centred(F_calc, est: SigmaAEstimate, sub) -> torch.Tensor:
        """``alpha * |F_calc|``. The ONLY place ``sub.select(alpha)`` appears.

        Cast to ``F_calc.dtype``, **not** ``F_obs.dtype``: ``F_calc`` is complex and
        ``F_obs`` is real, so the generic ``ctx.compact`` would change the cast. ``alpha`` is
        detached, so this reweights the residual without adding a parameter to the gradient.
        """
        return F_calc * sub.select(est.alpha).to(F_calc.dtype)

    # --- shared machinery -----------------------------------------------------
    def _geom(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(epsilon, d_star_sq)`` over the full data HKL, cached per data id.

        Both are model-independent (pure multiplicity / reciprocal geometry), so they are
        computed once per :class:`ReflectionData` and reused.
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
        # ALWAYS the free set, independent of this target's own `use_set`. Refmac/Servalcat
        # uses all reflections; measured indistinguishable here (and that arm was one of the
        # five the plumbing bug had disabled).
        est_mask = self._data.free.mask

        # Detached, cached until maintenance() resets it.
        #
        # `sigma_obs` is passed ALWAYS, for every row. It defines what sigma_A means: with
        # it, sigma_A is the correlation with the noise-free true amplitudes (Read's
        # definition, since Sigma_N = B - S2); without it, the correlation with the noisy
        # data. Passing it only for the rows that read `beta_model` -- which this used to do
        # -- gave the targets different sigma_A definitions and so confounded every
        # `ml vs ml_full` comparison with an estimator difference.
        est = self._sigma_a.get(
            F_obs_full, F_calc_full, centric_full, eps_full, dss_full, est_mask,
            sigma_obs=sigma_full.to(F_calc_full.dtype).reshape(-1),
            sigma_a_max=self.sigma_a_max,
            shrink=self.shrink,
        )

        F_obs = sub.select(F_obs_full)
        eps = est.epsilon
        # The `eps is not None` guard is behaviour, not defensiveness: `est.epsilon` is never
        # None in practice, but the pre-split code carried it and this is a refactor.
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

    One line, but a mixin rather than a copied override so ``sub.select(alpha)`` appears
    exactly once in the codebase (in :meth:`SigmaAXrayTarget._alpha_centred`) and the two
    rows that want it declare the intent rather than restate the mechanism.
    """

    def _mean(self, F_calc, est, sub):
        return self._alpha_centred(F_calc, est, sub)
