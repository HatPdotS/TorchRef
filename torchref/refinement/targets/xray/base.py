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
        self.use_work_set = use_work_set
        if sigma_mode not in ("effective", "raw"):
            raise ValueError(
                f"sigma_mode must be 'effective' or 'raw', got {sigma_mode!r}"
            )
        self.sigma_mode = sigma_mode
        # Set name based on work/test set
        self.name = "xray_work" if use_work_set else "xray_test"

        # Cache for the bookkeeping pieces of get_data: F_obs_sel, sigma_sel,
        # mask, centric_sel. These depend only on the data (rfree, validity,
        # centric) and the ReflectionData scaling parameters (log_scale,
        # U_aniso). None of those change during xyz/adp refinement, so we
        # recompute only when their fingerprint changes (see
        # ``_data_fingerprint``).
        self._cached_get_data = None
        self._cached_get_data_fp = None
        self._cached_sigma_mode = None

    def _data_fingerprint(self):
        """Fingerprint everything in ``self._data`` that get_data's
        cached pieces depend on. Used to detect when the bookkeeping
        cache (F_obs_sel / sigma_sel / mask / centric_sel) must be
        rebuilt.

        We probe the (data_ptr, _version) of any param/buffer the data
        scale or anisotropy correction depends on. During typical
        xyz/adp refinement these never change, so the cache stays warm
        across every closure call.
        """
        d = self._data
        entries = []
        for attr in ("log_scale", "U_aniso"):
            t = getattr(d, attr, None)
            if isinstance(t, torch.Tensor):
                entries.append((attr, t.data_ptr(), t._version))
        # If sigma_mode flips, the cache also needs to be rebuilt.
        # Tracked separately in self._cached_sigma_mode.
        return tuple(entries)

    def _build_get_data_cache(self):
        """Recompute the bookkeeping tensors that don't depend on fcalc.

        Returns ``(F_obs_sel, sigma_sel, mask, centric_sel)``.
        """
        _hkl, F_obs, sigma_F_obs, rfree_mask = self._data()

        # Sigma selection: use per-shell effective sigma from scaler if requested
        if self.sigma_mode == "effective" and self._scaler is not None:
            sigma_eff = getattr(self._scaler, "sigma_eff", None)
            if sigma_eff is not None and sigma_eff.shape == sigma_F_obs.shape:
                sigma_F_obs = sigma_eff

        centric_all = self._data.centric

        if hasattr(F_obs, "get_mask"):
            validity_mask = F_obs.get_mask()
            F_obs_data = F_obs.get_data()
            sigma_data = (
                sigma_F_obs.get_data()
                if hasattr(sigma_F_obs, "get_mask") else sigma_F_obs
            )
            rfree_bool = rfree_mask.bool()
            mask = (
                validity_mask & rfree_bool
                if self.use_work_set
                else validity_mask & ~rfree_bool
            )
        else:
            F_obs_data = F_obs
            sigma_data = sigma_F_obs
            rfree_bool = rfree_mask.bool()
            mask = rfree_bool if self.use_work_set else ~rfree_bool

        F_obs_sel = torch.where(mask, F_obs_data, torch.zeros_like(F_obs_data))
        sigma_sel = torch.where(mask, sigma_data, torch.ones_like(sigma_data))
        centric_sel = (centric_all & mask) if centric_all is not None else None
        return F_obs_sel, sigma_sel, mask, centric_sel

    def reset_get_data_cache(self):
        """Drop the cached bookkeeping tensors.

        Call this if you mutate ``self._data.log_scale`` /
        ``self._data.U_aniso`` in-place outside of the normal fingerprint-
        tracked flow, or if you want to free the memory.
        """
        self._cached_get_data = None
        self._cached_get_data_fp = None
        self._cached_sigma_mode = None

    def get_data(
        self, fcalc: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get F_obs, F_calc, sigma, and centric flags for the appropriate set.

        Bookkeeping tensors (F_obs_sel, sigma_sel, mask, centric_sel) are
        cached and reused as long as the upstream scaling parameters
        (``log_scale``, ``U_aniso``) of the ReflectionData haven't been
        mutated. Only ``F_calc_sel`` is recomputed from the live fcalc on
        each call.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model. Useful when model is not set.

        Returns
        -------
        tuple
            ``(F_obs_sel, F_calc_sel, sigma_sel, centric_sel, mask)``.
        """
        fp = self._data_fingerprint()
        if (
            self._cached_get_data is None
            or self._cached_get_data_fp != fp
            or self._cached_sigma_mode != self.sigma_mode
        ):
            self._cached_get_data = self._build_get_data_cache()
            self._cached_get_data_fp = fp
            self._cached_sigma_mode = self.sigma_mode
        F_obs_sel, sigma_sel, mask, centric_sel = self._cached_get_data

        # F_calc must always be computed fresh — depends on the live model
        # state (xyz / adp / occ).
        if fcalc is not None:
            F_calc = self.get_F_calc_scaled(fcalc=fcalc)
        else:
            F_calc = self.get_F_calc_scaled(self._data.hkl_for_sf(), recalc=False)

        F_calc_sel = torch.where(mask, F_calc, torch.zeros_like(F_calc))
        return F_obs_sel, F_calc_sel, sigma_sel, centric_sel, mask

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
        F_obs, F_calc, sigma, _, mask = self.get_data(fcalc=fcalc)
        F_calc_amp = torch.abs(F_calc)
        diff = F_obs - F_calc_amp

        loss = self.forward(fcalc=fcalc)

        rwork, rfree = self.get_rfactor()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(mask.sum().item(), VERBOSITY_DEBUG),
            "rwork": stat(rwork, VERBOSITY_STANDARD),
            "rfree": stat(rfree, VERBOSITY_STANDARD),
        }
