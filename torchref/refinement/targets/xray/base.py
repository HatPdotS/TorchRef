"""Shared base for the X-ray targets.

:class:`XrayTarget` owns the work/free/validation subset selection, the compact
``get_data`` view every subclass' ``forward`` consumes, the model-independent
per-reflection geometry, and the one R-factor implementation
(:meth:`XrayTarget.get_rfactor`). Subclasses supply only the likelihood -- and, if
they own their own scale, :meth:`_scaled_F_calc_full`.

The likelihood is supplied **per reflection**, as :meth:`_per_refl`, so that both
the summed loss (:meth:`forward`) and the unsummed one (:meth:`residuals`) come
from one expression and cannot encode different objectives. ``residuals`` is what
lets anything outside the target ask which reflections the model fails to explain,
in the target's own currency.
"""

from typing import TYPE_CHECKING, Dict, Tuple

import torch

from torchref.base.metrics.rfactor import rfactor_work_free
from torchref.base.reciprocal import get_scattering_vectors
from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from ..base import DataTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.model.model_ft import ModelFT
    from torchref.scaling.scaler_base import Scaler


class XrayTarget(DataTarget):
    """Base class for X-ray targets, with or without a model.

    With a model, F_calc is recomputed each forward pass; without one, a
    pre-computed ``fcalc`` is passed to ``forward()``/``get_data()``. The
    R-factor and statistics paths honour the same dual mode.

    Parameters
    ----------
    data : ReflectionData, optional
        Reference to the ReflectionData object. Required for ``forward()``.
    model : Model or ModelFT, optional
        Model used for F_calc. If None, ``fcalc`` must be passed to ``forward()``.
    scaler : Scaler, optional
        Reference to the Scaler object.
    use_work_set : bool, optional
        Legacy bool, default True; ignored when ``use_set`` is given.
    use_set : str, optional
        Canonical 3-way subset selector, ``"work"``/``"free"``/``"val"``. Takes
        precedence over ``use_work_set``; if None, derived from it.
    verbose : int, optional
        Verbosity level. Default is 0.
    sigma_a : SigmaAConfig, optional
        Model-error estimator configuration. Carried by every x-ray target so the
        factory has one construction call for all taxonomy rows, but read only by
        the sigma_A family -- ``nll``, ``ls`` and ``ls_wunit_k1`` ignore it.
    """

    name: str = "xray"  # Will be overridden based on work/test set

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        use_work_set: bool = True,
        verbose: int = 0,
        use_set: str = None,
        device=None,
        sigma_a=None,
    ):
        """Initialize X-ray target; see the class docstring for parameters."""
        super().__init__(
            data=data, model=model, scaler=scaler, verbose=verbose, device=device
        )
        # ``use_set`` is canonical and the legacy ``use_work_set`` bool maps onto
        # it. Keep BOTH consistent below: ``get_data`` reads ``use_set`` while some
        # subclass ``forward`` paths read ``use_work_set``, and if the two disagree
        # the loss and the reported statistics silently use different subsets.
        if sigma_a is None:
            from torchref.refinement.model_error_estimation.sigma_a import SigmaAConfig

            sigma_a = SigmaAConfig()
        self.sigma_a_config = sigma_a
        if use_set is None:
            use_set = "work" if use_work_set else "free"
        self.use_set = use_set
        self.use_work_set = use_set == "work"
        self.name = {
            "work": "xray_work",
            "free": "xray_test",
            "val": "xray_validation",
        }.get(use_set, "xray_work")
        # Model-independent per-reflection geometry (multiplicity + d*^2), cached per
        # ReflectionData instance (mirrors the bins cache pattern).
        self._eps_cache: torch.Tensor = None
        self._dss_cache: torch.Tensor = None
        self._geom_dataid: int = None

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

    def reset_get_data_cache(self):
        """Deprecated no-op, kept for compatibility: the subset indices and scaled
        ``(F, F_sigma)`` now live on the :class:`ReflectionData` and self-invalidate
        by fingerprint, so there is nothing here to reset.
        """
        pass

    def _subset(self):
        """The ``_ReflectionSubset`` view for this target's ``use_set``. Single
        source of truth for the selection -- both :meth:`get_data` and the subclass
        ``forward`` paths go through here, so loss and stats cannot diverge.
        """
        if self.use_set == "free":
            return self._data.free
        elif self.use_set == "val":
            return self._data.validation
        return self._data.work

    def get_data(
        self, fcalc: torch.Tensor = None, sub=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """
        Get compact F_obs, F_calc, sigma, centric and the subset view for
        this target's set (work, free or validation).

        Goes through the :class:`ReflectionData` subset accessor, which applies the
        validity masks and caches the remapped indices. The returned amplitude
        tensors are **compact** -- already restricted to the subset -- so a
        full-size, model-computed array must be passed through ``sub.select(t)``
        before it can be combined with them.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from the model.
        sub : _ReflectionSubset, optional
            Which reflections to return. Defaults to this target's own
            ``use_set``; :meth:`residuals` passes ``data.all`` to get every
            reflection, masks included.

        Returns
        -------
        tuple
            ``(F_obs, F_calc, sigma, centric, sub)`` — the first four compact;
            ``sub`` is the ``_ReflectionSubset`` view (``.indices``/``.select``/``.n``).
        """
        if sub is None:
            sub = self._subset()

        F_obs = sub.F

        # Sigma: scaled (corrected) experimental uncertainty.
        sigma = sub.sigF

        centric = sub.centric

        # F_calc depends on the live model state — always computed fresh, then
        # restricted to the same subset.
        if fcalc is not None:
            F_calc_full = self.get_F_calc_scaled(fcalc=fcalc)
        else:
            F_calc_full = self.get_F_calc_scaled(self._data.hkl_for_sf(), recalc=False)
        F_calc = sub.select(F_calc_full)

        return F_obs, F_calc, sigma, centric, sub

    # --- the per-reflection seam ---------------------------------------------
    #
    # ``_loss_inputs`` gathers what the likelihood reads on ONE subset view, and
    # ``_per_refl`` evaluates the likelihood on it without reducing. ``forward``
    # and ``residuals`` differ only in which view they gather and whether they
    # sum, so the two can never drift apart into different objectives.

    def _loss_inputs(self, fcalc: torch.Tensor = None, sub=None):
        """Everything this row's :meth:`_per_refl` reads, restricted to ``sub``.

        The default is :meth:`get_data`'s 5-tuple. The sigma_A family overrides it
        with a :class:`~.sigma_a.SigmaALossInputs`, which additionally carries the
        model-error estimate; the two shapes never mix because each class pairs its
        own ``_loss_inputs`` with its own ``_per_refl``.
        """
        return self.get_data(fcalc=fcalc, sub=sub)

    def _per_refl(self, ctx) -> torch.Tensor:
        """The likelihood, per reflection and **unreduced**, on ``ctx``'s subset.

        One per selectable row; no row is a branch. The summed form is
        ``_masked_sum`` of this -- which is how :meth:`forward` is written wherever
        the sum is not fused into a kernel.
        """
        raise NotImplementedError

    def residuals(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Per-reflection loss over EVERY reflection, aligned to ``data.hkl``.

        The unsummed :meth:`forward`: same mean, same variance, same likelihood.
        Three deliberate differences, all of them so the result can be used to
        *judge* the data rather than to fit it:

        * **Full size, not this target's subset.** Length matches ``data.hkl``, so
          entry ``i`` is reflection ``i`` and the array is directly comparable
          against masks, resolution or the work/free split.
        * **Masks are not applied.** A reflection the masks exclude still gets a
          value, so the array can be used to ask *why* it was excluded rather than
          only reflecting the answer back; see :attr:`ReflectionData.all`.
        * **Non-finite values survive.** ``forward`` substitutes ``1e6`` so one NaN
          cannot poison a gradient; here a NaN is a finding, not a nuisance.

        Not wrapped in ``no_grad`` -- it stays differentiable like any other target
        expression, so a caller doing diagnostics should wrap it themselves.

        Returns
        -------
        torch.Tensor
            Shape ``(len(data.hkl),)``.
        """
        return self._per_refl(self._loss_inputs(fcalc=fcalc, sub=self._data.all))

    def _scaled_F_calc_full(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Full-size ``|F_calc|`` under THIS target's objective scaling.

        Defaults to the scaler's; targets owning their own scale override it so the
        reported R-factor uses the very scale the loss saw.
        """
        if fcalc is not None:
            return self.get_F_calc_scaled(fcalc=fcalc)
        return self.get_F_calc_scaled(self._data.hkl_for_sf(), recalc=False)

    def get_rfactor(self, fcalc: torch.Tensor = None):
        """Compute ``(R_work, R_free)`` for this target.

        Single source of truth for the X-ray R-factor: it uses exactly the
        scaled ``|F_calc|`` this target's loss sees (the scaler's scaling by
        default; the detached per-bin closed-form scale for ``binwise_optimal``).
        ``R_work`` is computed on the work subset and ``R_free`` on the free
        subset — the same subsets the loss uses, so any validation reflections
        are excluded from both. All X-ray targets share this implementation;
        only the *scale* (:meth:`_scaled_F_calc_full`) varies by target.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, used instead of
            computing from the model (e.g. rigid-body / model-less targets).

        Returns
        -------
        tuple
            ``(R_work, R_free)`` as Python floats.
        """
        with torch.no_grad():
            F_calc_full = self._scaled_F_calc_full(fcalc=fcalc)
            return rfactor_work_free(self._data, F_calc_full)

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """
        Get statistics for this X-ray target.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors.

        Returns
        -------
        dict
            Statistics dict with StatEntry values containing verbosity levels.
        """
        F_obs, F_calc, sigma, _, sub = self.get_data(fcalc=fcalc)
        F_calc_amp = torch.abs(F_calc)
        diff = F_obs - F_calc_amp

        loss = self.forward(fcalc=fcalc)

        rwork, rfree = self.get_rfactor()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(sub.n, VERBOSITY_DEBUG),
            "rwork": stat(rwork, VERBOSITY_STANDARD),
            "rfree": stat(rfree, VERBOSITY_STANDARD),
        }
