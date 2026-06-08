from typing import TYPE_CHECKING, Dict, Tuple

import torch

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
        sigma_mode: str = "raw",
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
            Ignored when ``use_set`` is provided.
        use_set : {'work', 'free', 'val'}, optional
            Three-class flag selector (added for ensemble refinement). When
            provided, takes precedence over ``use_work_set``. Use ``'val'``
            to target the validation set, distinct from the R-free test set.
        sigma_mode : str, optional
            Which sigma to use in the likelihood. Options:

            - ``'raw'`` (default): use the raw experimental sigmas from the
              data file. Empirically gives the best Rfree across the
              mid-resolution regime (1.5-3.0 A) when paired with appropriate
              group weights.
            - ``'effective'``: use per-shell effective sigmas estimated from
              scaling residuals (capped SIGMAA-style correction). Opt-in for
              high-resolution refinement (< 1.5 A) or datasets with known
              sigma miscalibration. Note: ``Scaler.estimate_sigma_eff`` is
              *always* called so the estimates are available regardless of
              which mode the target uses.

        verbose : int, optional
            Verbosity level. Default is 0.
        """
        super().__init__(data=data, model=model, scaler=scaler, verbose=verbose)
        if use_set is not None:
            if use_set not in ("work", "free", "val"):
                raise ValueError(
                    f"use_set must be 'work', 'free', or 'val'; got {use_set!r}"
                )
            self.use_set = use_set
            # Keep use_work_set in sync for any legacy code that reads it.
            self.use_work_set = (use_set == "work")
        else:
            self.use_set = "work" if use_work_set else "free"
            self.use_work_set = use_work_set
        if sigma_mode not in ("effective", "raw"):
            raise ValueError(
                f"sigma_mode must be 'effective' or 'raw', got {sigma_mode!r}"
            )
        self.sigma_mode = sigma_mode
        # Set name based on which set we target.
        if self.use_set == "work":
            self.name = "xray_work"
        elif self.use_set == "free":
            self.name = "xray_test"
        else:
            self.name = "xray_val"

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
        if self.sigma_mode == "effective" and self._scaler is not None:
            sigma_eff = getattr(self._scaler, "sigma_eff", None)
            if sigma_eff is not None and sigma_eff.shape[0] == len(self._data.hkl):
                sigma = sub.select(sigma_eff)

        centric = sub.centric

        # F_calc depends on the live model state — always computed fresh, then
        # restricted to the same subset.
        if fcalc is not None:
            F_calc_full = self.get_F_calc_scaled(fcalc=fcalc)
        else:
            F_calc_full = self.get_F_calc_scaled(self._data.hkl_for_sf(), recalc=False)
        F_calc = sub.select(F_calc_full)

        return F_obs, F_calc, sigma, centric, sub

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
