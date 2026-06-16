from typing import TYPE_CHECKING, Dict, Tuple

import torch

from torchref.base.metrics.rfactor import rfactor_work_free
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
    """
    Base class for X-ray targets.

    Provides common functionality for accessing F_obs, F_calc, etc.
    Supports two modes of operation:

    1. With Model: Computes F_calc from model on each forward pass
    2. Without Model: Uses pre-computed F_calc passed to forward()/get_data()

    Parameters
    ----------
    data : ReflectionData, optional
        Reference to the ReflectionData object. Required for forward().
    model : Model or ModelFT, optional
        Reference to Model object for F_calc computation.
        If None, fcalc must be provided to forward().
    scaler : Scaler, optional
        Reference to the Scaler object.
    use_work_set : bool, optional
        If True, compute loss on work set; if False, on test set. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Attributes
    ----------
    use_work_set : bool
        Whether to use work set or test set.
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
    ):
        """
        Initialize X-ray target.

        Parameters
        ----------
        data : ReflectionData, optional
            Reference to the ReflectionData object. Required for forward().
        model : Model or ModelFT, optional
            Reference to Model object for F_calc computation.
            If None, fcalc must be provided to forward().
        scaler : Scaler, optional
            Reference to the Scaler object.
        use_work_set : bool, optional
            If True, compute loss on work set; if False, on test set. Default is True.

        verbose : int, optional
            Verbosity level. Default is 0.
        """
        super().__init__(data=data, model=model, scaler=scaler, verbose=verbose)
        self.use_work_set = use_work_set
        # Set name based on work/test set
        self.name = "xray_work" if use_work_set else "xray_test"

    def reset_get_data_cache(self):
        """Deprecated no-op.

        The work/free subset indices and the scaled ``(F, F_sigma)`` are now
        cached on the :class:`ReflectionData` and self-invalidate via
        fingerprints, so there is nothing to reset here. Retained for
        backward compatibility.
        """
        pass

    def get_data(
        self, fcalc: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """
        Get compact F_obs, F_calc, sigma, centric, and the subset view for the
        appropriate set (work, free, or validation).

        Uses the :class:`ReflectionData` ``work``/``free``/``validation``
        accessor, which applies the validity masks and the work/test/validation
        selection and caches the remapped integer indices. The returned
        amplitude tensors are **compact** (already restricted to the subset);
        use ``sub.select(t)`` to align a full-size, model-computed array to the
        same subset.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from the model.

        Returns
        -------
        tuple
            ``(F_obs, F_calc, sigma, centric, sub)`` — the first four compact;
            ``sub`` is the ``_ReflectionSubset`` view (``.indices``/``.select``/``.n``).
        """
        if self.use_set == "free":
            sub = self._data.free
        elif self.use_set == "val":
            sub = self._data.validation
        else:
            sub = self._data.work

        F_obs = sub.F

        # Sigma: scaled experimental, or per-shell effective from the scaler.
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

    def _scaled_F_calc_full(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Full-size ``|F_calc|`` under THIS target's objective scaling.

        Default is the scaler's scaling (:meth:`get_F_calc_scaled`). Targets that
        own their scale (e.g. :class:`LeastSquaresXrayTarget` in
        ``binwise_optimal`` mode) override this so the reported R-factor uses the
        very scale the loss sees.
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
