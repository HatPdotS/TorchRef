"""Difference targets for time-resolved crystallography.

Four ways to compare a light/dark pair: :class:`DifferenceXrayTarget` (Gaussian
NLL on amplitude differences), :class:`PhaseInformedDifferenceTarget` and
:class:`TaylorCorrectedDifferenceTarget` (complex differences built from model
phases), and :class:`RiceDifferenceTarget` (Rice likelihood on the non-negative
complex-difference magnitude). All grafted phases are detached -- gradients must
reach only ΔF_calc.
"""

import torch
from torch import nn
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

from .base import Target
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.io.datasets import DatasetCollection
    from torchref.model.model_ft import ModelFT
    from torchref.model import MixedModel
    from torchref.scaling.scaler_base import Scaler


class DifferenceXrayTarget(Target):
    """Time-resolved difference target: Gaussian NLL on amplitude differences.

    With ΔF_calc = |F_light_calc| - |F_dark_calc|, ΔF_obs = F_light_obs -
    F_dark_obs and σ_diff = sqrt(σ_light² + σ_dark²), the loss is
    0.5·(ΔF_obs - ΔF_calc)²/σ_diff² + log σ_diff + 0.5·log 2π, summed.

    Either pass a ``dataset_collection`` of pre-aligned 'dark'/'light' datasets
    (preferred; lets scalers be shared with other targets) or individual
    ``data_light``/``data_dark``, which are HKL-matched automatically. The two
    modes return arrays of different length -- see :meth:`get_delta_F_obs`.

    Parameters
    ----------
    dataset_collection : DatasetCollection, optional
        Collection containing 'dark' and 'light' datasets (pre-aligned HKL).
        If provided, ``data_light``/``data_dark`` are ignored.
    data_light, data_dark : ReflectionData, optional
        Per-state reflection data; used only without a collection.
    model_light : ModelFT or MixedModel
        Model for the light (excited) state.
    model_dark : ModelFT
        Model for the dark (ground) state.
    scaler_light, scaler_dark : Scaler, optional
        Per-state F_calc scalers. May be shared with other targets.
    use_work_set : bool, optional
        Loss on the work set if True (default), else on the test set.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "difference_xray"

    def __init__(
        self,
        dataset_collection: "DatasetCollection" = None,
        data_light: "ReflectionData" = None,
        data_dark: "ReflectionData" = None,
        model_light: "ModelFT" = None,
        model_dark: "ModelFT" = None,
        scaler_light: "Scaler" = None,
        scaler_dark: "Scaler" = None,
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        """Initialize DifferenceXrayTarget."""
        super().__init__(verbose=verbose)

        self._dataset_collection = dataset_collection

        if dataset_collection is not None:
            if "dark" not in dataset_collection:
                raise ValueError(
                    "DatasetCollection must contain a 'dark' dataset"
                )
            if "light" not in dataset_collection:
                raise ValueError(
                    "DatasetCollection must contain a 'light' dataset"
                )
            self._data_dark = dataset_collection["dark"]
            self._data_light = dataset_collection["light"]
            self._use_collection = True
        else:
            self._data_light = data_light
            self._data_dark = data_dark
            self._use_collection = False

        self.add_module("_model_light", model_light)
        self.add_module("_model_dark", model_dark)
        self.add_module("_scaler_light", scaler_light)
        self.add_module("_scaler_dark", scaler_dark)
        self.use_work_set = use_work_set

        # Cache for matched reflection indices (only used in non-collection mode)
        self._matched_indices_light = None
        self._matched_indices_dark = None
        self._common_hkl = None

        # Match reflections if using separate datasets
        if not self._use_collection and data_light is not None and data_dark is not None:
            self._match_reflections()

    @property
    def dataset_collection(self):
        """DatasetCollection if using collection mode."""
        return self._dataset_collection

    @property
    def data_light(self) -> "ReflectionData":
        """Light state reflection data."""
        return self._data_light

    @property
    def data_dark(self) -> "ReflectionData":
        """Dark state reflection data."""
        return self._data_dark

    @property
    def model_light(self) -> "ModelFT":
        """Light state model."""
        return self._model_light

    @property
    def model_dark(self) -> "ModelFT":
        """Dark state model."""
        return self._model_dark

    @property
    def scaler_light(self) -> "Scaler":
        """Light state scaler."""
        return self._scaler_light

    @property
    def scaler_dark(self) -> "Scaler":
        """Dark state scaler."""
        return self._scaler_dark

    @property
    def hkl(self) -> torch.Tensor:
        """Common HKL: the collection's if aligned, else the matched subset."""
        if self._use_collection:
            return self._dataset_collection.hkl
        else:
            if self._common_hkl is None:
                self._match_reflections()
            return self._common_hkl

    def _hkl_to_hash(self, hkl: torch.Tensor) -> torch.Tensor:
        """Hash (n, 3) Miller indices to ``h*p1 + k*p2 + l``. Not injective --
        large or negative indices can alias, so the sorted merge below relies on
        the actual indices not colliding.
        """
        p1 = 1000003
        p2 = 1000033

        h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
        return h * p1 + k * p2 + l

    def _match_reflections(self):
        """Cache the light/dark matched-index pair via an O(N log N) hash merge.

        No-op in collection mode, where the datasets are already aligned.
        """
        if self._use_collection:
            return

        hkl_light = self._data_light.hkl
        hkl_dark = self._data_dark.hkl

        hash_light = self._hkl_to_hash(hkl_light)
        hash_dark = self._hkl_to_hash(hkl_dark)

        sorted_light, sort_idx_light = torch.sort(hash_light)
        sorted_dark, sort_idx_dark = torch.sort(hash_dark)

        matched_light = []
        matched_dark = []

        i, j = 0, 0
        n_light, n_dark = len(sorted_light), len(sorted_dark)

        while i < n_light and j < n_dark:
            if sorted_light[i] < sorted_dark[j]:
                i += 1
            elif sorted_light[i] > sorted_dark[j]:
                j += 1
            else:
                # Map back to pre-sort positions.
                matched_light.append(sort_idx_light[i].item())
                matched_dark.append(sort_idx_dark[j].item())
                i += 1
                j += 1

        device = hkl_light.device
        self._matched_indices_light = torch.tensor(
            matched_light, dtype=torch.long, device=device  # dtype-ok: matched atom indices used for indexing; PyTorch requires int64
        )
        self._matched_indices_dark = torch.tensor(
            matched_dark, dtype=torch.long, device=device  # dtype-ok: matched atom indices used for indexing; PyTorch requires int64
        )

        # Store common HKL (using light indices, they should be identical)
        self._common_hkl = hkl_light[self._matched_indices_light]

        if self.verbose > 0:
            print(
                f"DifferenceXrayTarget: matched {len(matched_light)} reflections "
                f"({len(hkl_light)} light, {len(hkl_dark)} dark)"
            )

    def get_delta_F_obs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get observed difference structure factors with error propagation.

        Returns
        -------
        delta_F_obs : torch.Tensor
            ΔF_obs = F_light_obs - F_dark_obs
        sigma_diff : torch.Tensor
            σ_diff = sqrt(σ_light² + σ_dark²)
        mask : torch.Tensor
            Boolean mask for work/test set selection and valid data. Note the
            two code paths differ in length: in collection mode
            (``_get_delta_F_obs_collection``) the returned arrays are
            full-length and aligned across datasets, whereas in matched mode
            (``_get_delta_F_obs_matched``) they are subset to the matched
            reflections only. Callers must account for this shape difference.
        """
        if self._use_collection:
            return self._get_delta_F_obs_collection()
        else:
            return self._get_delta_F_obs_matched()

    def _get_delta_F_obs_collection(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get delta F_obs when using DatasetCollection (aligned HKL)."""
        F_obs_light, sigma_light = self._data_light.get_corrected_data()
        rfree_light = self._data_light.rfree_flags
        validity_light = self._data_light.masks().to(torch.bool)
        F_obs_dark, sigma_dark = self._data_dark.get_corrected_data()
        rfree_dark = self._data_dark.rfree_flags
        validity_dark = self._data_dark.masks().to(torch.bool)

        delta_F_obs = F_obs_light - F_obs_dark
        sigma_diff = torch.sqrt(sigma_light**2 + sigma_dark**2)

        # A reflection counts only if valid in BOTH datasets.
        valid_both = validity_light & validity_dark

        # rfree flags may be int32 0/1; ``&`` on ints is bitwise, not logical.
        rfree_light_bool = rfree_light.bool()
        rfree_dark_bool = rfree_dark.bool()
        if self.use_work_set:
            set_mask = rfree_light_bool & rfree_dark_bool
        else:
            set_mask = ~rfree_light_bool & ~rfree_dark_bool

        mask = valid_both & set_mask

        return delta_F_obs, sigma_diff, mask

    def _get_delta_F_obs_matched(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get delta F_obs when using matched indices (non-collection mode)."""
        if self._matched_indices_light is None:
            self._match_reflections()

        F_obs_light, sigma_light = self._data_light.get_corrected_data()
        rfree_light = self._data_light.rfree_flags
        validity_light = self._data_light.masks().to(torch.bool)
        F_obs_dark, sigma_dark = self._data_dark.get_corrected_data()
        rfree_dark = self._data_dark.rfree_flags
        validity_dark = self._data_dark.masks().to(torch.bool)

        F_light = F_obs_light[self._matched_indices_light]
        F_dark = F_obs_dark[self._matched_indices_dark]
        sig_light = sigma_light[self._matched_indices_light]
        sig_dark = sigma_dark[self._matched_indices_dark]
        valid_light = validity_light[self._matched_indices_light]
        valid_dark = validity_dark[self._matched_indices_dark]

        delta_F_obs = F_light - F_dark
        sigma_diff = torch.sqrt(sig_light**2 + sig_dark**2)

        # A reflection counts only if valid in BOTH datasets.
        valid_both = valid_light & valid_dark

        rfree_light_matched = rfree_light[self._matched_indices_light]
        rfree_dark_matched = rfree_dark[self._matched_indices_dark]

        # rfree flags may be int32 0/1; ``&`` on ints is bitwise, not logical.
        rfree_light_bool = rfree_light_matched.bool()
        rfree_dark_bool = rfree_dark_matched.bool()
        if self.use_work_set:
            set_mask = rfree_light_bool & rfree_dark_bool
        else:
            set_mask = ~rfree_light_bool & ~rfree_dark_bool

        mask = valid_both & set_mask

        return delta_F_obs, sigma_diff, mask

    def get_delta_F_calc(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = False,
    ) -> torch.Tensor:
        """
        Compute calculated difference structure factors.

        ΔF_calc = |F_light_calc| - |F_dark_calc|

        Parameters
        ----------
        fcalc_light : torch.Tensor, optional
            Pre-computed light state structure factors.
        fcalc_dark : torch.Tensor, optional
            Pre-computed dark state structure factors.
        recalc : bool, optional
            Force recalculation if True. Default is False.

        Returns
        -------
        torch.Tensor
            ΔF_calc for all reflections (full size, use mask from get_delta_F_obs).
        """
        hkl = self.hkl

        if fcalc_light is None:
            if self._model_light is None:
                raise RuntimeError(
                    "Cannot compute F_calc_light: no model_light set."
                )
            fcalc_light = self._model_light(hkl, recalc=recalc)

        if self._scaler_light is not None:
            fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            if self._model_dark is None:
                raise RuntimeError("Cannot compute F_calc_dark: no model_dark set.")
            fcalc_dark = self._model_dark(hkl, recalc=recalc)

        if self._scaler_dark is not None:
            fcalc_dark = self._scaler_dark(fcalc_dark)

        F_light_amp = torch.abs(fcalc_light)
        F_dark_amp = torch.abs(fcalc_dark)
        delta_F_calc = F_light_amp - F_dark_amp

        return delta_F_calc

    def forward(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = False,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss for difference structure factors.

        NLL = 0.5 * (ΔF_obs - ΔF_calc)² / σ_diff² + log(σ_diff) + 0.5*log(2π)

        Parameters
        ----------
        fcalc_light : torch.Tensor, optional
            Pre-computed light state structure factors.
        fcalc_dark : torch.Tensor, optional
            Pre-computed dark state structure factors.
        recalc : bool, optional
            Force recalculation if True. Default is False.

        Returns
        -------
        torch.Tensor
            Summed Gaussian NLL over included reflections (scalar).
        """
        delta_F_obs, sigma_diff, mask = self.get_delta_F_obs()

        delta_F_calc = self.get_delta_F_calc(
            fcalc_light=fcalc_light, fcalc_dark=fcalc_dark, recalc=recalc
        )

        # Mask via torch.where, not boolean indexing: no nonzero() device sync.
        delta_F_obs = torch.where(mask, delta_F_obs, torch.zeros_like(delta_F_obs))
        delta_F_calc = torch.where(mask, delta_F_calc, torch.zeros_like(delta_F_calc))
        sigma_diff = torch.where(mask, sigma_diff, torch.ones_like(sigma_diff))

        diff = delta_F_obs - delta_F_calc

        # Floor sigma at 10% of its median so a zero sigma cannot blow up.
        eps = torch.median(sigma_diff) * 1e-1
        sigma_safe = torch.clamp(sigma_diff, min=eps)

        log_2pi = torch.log(
            torch.tensor(2.0 * torch.pi, device=sigma_diff.device, dtype=sigma_diff.dtype)
        )
        nll = (
            0.5 * (diff**2) / (sigma_safe**2)
            + torch.log(sigma_safe)
            + 0.5 * log_2pi
        )

        return (nll * mask).sum()

    def stats(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
    ) -> Dict[str, StatEntry]:
        """
        Get statistics for difference refinement.

        Parameters
        ----------
        fcalc_light : torch.Tensor, optional
            Pre-computed light state structure factors.
        fcalc_dark : torch.Tensor, optional
            Pre-computed dark state structure factors.

        Returns
        -------
        dict
            Statistics dict with correlation, R_diff, etc.
        """
        # Get observed and calculated differences
        delta_F_obs, sigma_diff, mask = self.get_delta_F_obs()
        delta_F_calc = self.get_delta_F_calc(
            fcalc_light=fcalc_light, fcalc_dark=fcalc_dark
        )

        # Apply mask
        delta_F_obs = delta_F_obs[mask]
        delta_F_calc = delta_F_calc[mask]
        sigma_diff = sigma_diff[mask]

        # Compute loss
        loss = self.forward(fcalc_light=fcalc_light, fcalc_dark=fcalc_dark)

        # Compute correlation coefficient
        obs_mean = delta_F_obs.mean()
        calc_mean = delta_F_calc.mean()
        obs_centered = delta_F_obs - obs_mean
        calc_centered = delta_F_calc - calc_mean

        covariance = (obs_centered * calc_centered).mean()
        obs_std = torch.sqrt((obs_centered**2).mean())
        calc_std = torch.sqrt((calc_centered**2).mean())

        correlation = covariance / (obs_std * calc_std + 1e-8)

        # Compute R_diff = Σ|ΔF_obs - ΔF_calc| / Σ|ΔF_obs|
        diff = delta_F_obs - delta_F_calc
        r_diff = torch.abs(diff).sum() / (torch.abs(delta_F_obs).sum() + 1e-8)

        # RMS difference
        rms_diff = torch.sqrt((diff**2).mean())

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(delta_F_obs), VERBOSITY_DEBUG),
            "correlation": stat(correlation.item(), VERBOSITY_STANDARD),
            "r_diff": stat(r_diff.item(), VERBOSITY_STANDARD),
            "rms_diff": stat(rms_diff.item(), VERBOSITY_DETAILED),
            "mean_sigma_diff": stat(sigma_diff.mean().item(), VERBOSITY_DEBUG),
        }


class PhaseInformedDifferenceTarget(Target):
    """Difference target comparing complex differences via grafted model phases.

    Detached model phases φ turn the observed amplitude difference into a
    complex one, which is then compared with the calculated complex difference::

        ΔF_calc      = F_light_calc - F_dark_calc
        ΔF_obs_cplx  = (|F_light_obs| - |F_dark_obs|) * exp(i·φ)
        Loss         = |ΔF_obs_cplx - ΔF_calc|² / σ_diff²

    The light state is ``F_light_calc`` throughout even when the light model is
    a mixed/excited-state model (called "mixed" elsewhere).

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection containing 'dark' and 'light' datasets.
    model_light : ModelFT or MixedModel
        Model for the light/excited state.
    model_dark : ModelFT
        Model for the dark/ground state.
    scaler_light, scaler_dark : Scaler, optional
        Per-state F_calc scalers.
    phase_source : {"dark", "difference", "mixed"}, optional
        Where φ comes from: the dark model, the calculated difference ΔF_calc,
        or the mixed/light model. Default is "difference".
    use_work_set : bool, optional
        Loss on the work set if True (default), else on the test set.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "phase_informed_difference"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_light: "ModelFT" = None,
        model_dark: "ModelFT" = None,
        scaler_light: "Scaler" = None,
        scaler_dark: "Scaler" = None,
        phase_source: Literal["dark", "difference", "mixed"] = "difference",
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        if "dark" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'dark' dataset")
        if "light" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'light' dataset")

        if phase_source not in ("dark", "difference", "mixed"):
            raise ValueError(f"phase_source must be 'dark', 'difference', or 'mixed', got {phase_source}")

        self._dataset_collection = dataset_collection
        self._data_dark = dataset_collection["dark"]
        self._data_light = dataset_collection["light"]

        self.add_module("_model_light", model_light)
        self.add_module("_model_dark", model_dark)
        self.add_module("_scaler_light", scaler_light)
        self.add_module("_scaler_dark", scaler_dark)

        self.phase_source = phase_source
        self.use_work_set = use_work_set

        # Precompute sigma_diff
        self._setup_data()

    def _setup_data(self):
        """Setup observed data and masks."""
        F_light, sigma_light = self._data_light.get_corrected_data()
        rfree_light = self._data_light.rfree_flags
        F_dark, sigma_dark = self._data_dark.get_corrected_data()
        rfree_dark = self._data_dark.rfree_flags

        self.register_buffer("_F_obs_light", F_light)
        self.register_buffer("_F_obs_dark", F_dark)
        self.register_buffer("_sigma_light", sigma_light)
        self.register_buffer("_sigma_dark", sigma_dark)
        self.register_buffer("_sigma_diff", torch.sqrt(sigma_light**2 + sigma_dark**2))

        # Work/test set mask
        if self.use_work_set:
            mask = rfree_light.bool() & rfree_dark.bool()
        else:
            mask = ~rfree_light.bool() & ~rfree_dark.bool()
        self.register_buffer("_mask", mask)

    @property
    def hkl(self) -> torch.Tensor:
        """Common HKL indices."""
        return self._dataset_collection.hkl

    def _get_phases(
        self,
        F_dark_calc: torch.Tensor,
        F_mixed_calc: torch.Tensor
    ) -> torch.Tensor:
        """Phase for the observed difference, per ``phase_source``; detached, so
        gradients reach only ΔF_calc -- letting them into the reconstructed
        ΔF_obs adds spurious terms that stall refinement part-way.
        """
        if self.phase_source == "dark":
            return torch.angle(F_dark_calc).detach()
        elif self.phase_source == "mixed":
            return torch.angle(F_mixed_calc).detach()
        elif self.phase_source == "difference":
            delta_F_calc = F_mixed_calc - F_dark_calc
            return torch.angle(delta_F_calc).detach()
        else:
            raise ValueError(f"Unknown phase_source: {self.phase_source}")

    def forward(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = True,
    ) -> torch.Tensor:
        """
        Compute phase-informed difference loss.

        Parameters
        ----------
        fcalc_light : torch.Tensor, optional
            Pre-computed light state structure factors.
        fcalc_dark : torch.Tensor, optional
            Pre-computed dark state structure factors.
        recalc : bool, optional
            Force recalculation if True. Default is True.

        Returns
        -------
        torch.Tensor
            Summed weighted squared error over included reflections (scalar).
        """
        hkl = self.hkl

        if fcalc_light is None:
            if self._model_light is None:
                raise RuntimeError("No model_light set")
            fcalc_light = self._model_light(hkl, recalc=recalc)

        if self._scaler_light is not None:
            fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            if self._model_dark is None:
                raise RuntimeError("No model_dark set")
            fcalc_dark = self._model_dark(hkl, recalc=recalc)

        if self._scaler_dark is not None:
            fcalc_dark = self._scaler_dark(fcalc_dark)

        phi = self._get_phases(fcalc_dark, fcalc_light)

        delta_F_obs = self._F_obs_light - self._F_obs_dark
        delta_F_obs_complex = delta_F_obs * torch.exp(1j * phi)
        delta_F_calc = fcalc_light - fcalc_dark

        delta_F_obs_complex = delta_F_obs_complex[self._mask]
        delta_F_calc = delta_F_calc[self._mask]
        sigma_diff = self._sigma_diff[self._mask]

        diff = delta_F_obs_complex - delta_F_calc

        # χ²-style NLL: summed, unnormalised.
        loss = (torch.abs(diff)**2 / sigma_diff**2).sum()

        return loss

    def stats(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
    ) -> Dict[str, StatEntry]:
        """
        Get statistics for the difference refinement.

        Returns
        -------
        dict
            Dictionary with loss, correlation, R_diff, etc.
        """
        hkl = self.hkl

        # Compute F_calc
        if fcalc_light is None:
            fcalc_light = self._model_light(hkl, recalc=True)
            if self._scaler_light is not None:
                fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            fcalc_dark = self._model_dark(hkl, recalc=True)
            if self._scaler_dark is not None:
                fcalc_dark = self._scaler_dark(fcalc_dark)

        with torch.no_grad():
            loss = self.forward(fcalc_light, fcalc_dark, recalc=False)

            # Amplitude difference correlation
            delta_F_obs = (self._F_obs_light - self._F_obs_dark)[self._mask]
            delta_F_calc_amp = (torch.abs(fcalc_light) - torch.abs(fcalc_dark))[self._mask]

            obs_centered = delta_F_obs - delta_F_obs.mean()
            calc_centered = delta_F_calc_amp - delta_F_calc_amp.mean()

            correlation = (
                (obs_centered * calc_centered).sum() /
                (torch.sqrt((obs_centered**2).sum() * (calc_centered**2).sum()) + 1e-8)
            ).item()

            # R_diff
            r_diff = (
                torch.abs(delta_F_obs - delta_F_calc_amp).sum() /
                (torch.abs(delta_F_obs).sum() + 1e-8)
            ).item()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(self._mask.sum().item(), VERBOSITY_DETAILED),
            "correlation": stat(correlation, VERBOSITY_STANDARD),
            "r_diff": stat(r_diff, VERBOSITY_STANDARD),
            "phase_source": stat(self.phase_source, VERBOSITY_DETAILED),
        }

    def __repr__(self) -> str:
        return f"PhaseInformedDifferenceTarget(phase_source={self.phase_source})"


class TaylorCorrectedDifferenceTarget(Target):
    """Difference target using the exact complex-exponential phase-shift identity.

    Builds the observed complex difference from the model phase rotation with no
    small-angle approximation -- the "Taylor" name is historical, the expression
    is an exact algebraic identity::

        ΔF_obs = exp(i·φ_dark)·[F_obs_dark·(exp(i·dφ) - 1) + dF_obs·exp(i·dφ)]

    with dφ = φ_light_calc - φ_dark_calc the model phase rotation and
    dF_obs = F_obs_light - F_obs_dark the observed amplitude difference. The loss
    is |ΔF_obs - ΔF_calc|² / σ_diff², summed.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection containing 'dark' and 'light' datasets.
    model_light : ModelFT or MixedModel
        Model for the light/excited state.
    model_dark : ModelFT
        Model for the dark/ground state.
    scaler_light, scaler_dark : Scaler, optional
        Per-state F_calc scalers.
    use_work_set : bool, optional
        Loss on the work set if True (default), else on the test set.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "taylor_corrected_difference"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_light: "ModelFT" = None,
        model_dark: "ModelFT" = None,
        scaler_light: "Scaler" = None,
        scaler_dark: "Scaler" = None,
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        if "dark" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'dark' dataset")
        if "light" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'light' dataset")

        self._dataset_collection = dataset_collection
        self._data_dark = dataset_collection["dark"]
        self._data_light = dataset_collection["light"]

        self.add_module("_model_light", model_light)
        self.add_module("_model_dark", model_dark)
        self.add_module("_scaler_light", scaler_light)
        self.add_module("_scaler_dark", scaler_dark)

        self.use_work_set = use_work_set

        # Precompute sigma_diff
        self._setup_data()

    def _setup_data(self):
        """Setup observed data and masks."""
        F_light, sigma_light = self._data_light.get_corrected_data()
        rfree_light = self._data_light.rfree_flags
        valid_light = self._data_light.masks().to(torch.bool)
        F_dark, sigma_dark = self._data_dark.get_corrected_data()
        rfree_dark = self._data_dark.rfree_flags
        valid_dark = self._data_dark.masks().to(torch.bool)

        # Build validity mask first (needed for cleanup below)
        valid_mask = torch.ones_like(rfree_light, dtype=torch.bool)
        if valid_light is not None:
            valid_mask = valid_mask & valid_light
        if valid_dark is not None:
            valid_mask = valid_mask & valid_dark

        # Clean invalid values to avoid NaN propagation in torch.where path
        sigma_diff = torch.sqrt(sigma_light**2 + sigma_dark**2)
        F_light = torch.where(valid_mask, F_light, torch.zeros_like(F_light))
        F_dark = torch.where(valid_mask, F_dark, torch.zeros_like(F_dark))
        sigma_diff = torch.where(valid_mask, sigma_diff, torch.ones_like(sigma_diff))

        self.register_buffer("_F_obs_light", F_light)
        self.register_buffer("_F_obs_dark", F_dark)
        self.register_buffer("_sigma_light", sigma_light)
        self.register_buffer("_sigma_dark", sigma_dark)
        self.register_buffer("_sigma_diff", sigma_diff)
        work_mask = rfree_light.bool() & rfree_dark.bool() & valid_mask
        free_mask = ~rfree_light.bool() & ~rfree_dark.bool() & valid_mask

        if self.use_work_set:
            mask = work_mask
        else:
            mask = free_mask
        self.register_buffer("_mask", mask)
        self.register_buffer("_work_mask", work_mask)
        self.register_buffer("_free_mask", free_mask)

    @property
    def hkl(self) -> torch.Tensor:
        """Common HKL indices."""
        return self._dataset_collection.hkl

    def forward(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = True,
    ) -> torch.Tensor:
        """
        Compute the Taylor-corrected difference loss (identity in the class doc).

        Parameters
        ----------
        fcalc_light : torch.Tensor, optional
            Pre-computed light state structure factors.
        fcalc_dark : torch.Tensor, optional
            Pre-computed dark state structure factors.
        recalc : bool, optional
            Force recalculation if True. Default is True.

        Returns
        -------
        torch.Tensor
            Summed weighted squared error over included reflections (scalar).
        """
        hkl = self.hkl

        if fcalc_light is None:
            if self._model_light is None:
                raise RuntimeError("No model_light set")
            fcalc_light = self._model_light(hkl, recalc=recalc)

        if self._scaler_light is not None:
            fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            if self._model_dark is None:
                raise RuntimeError("No model_dark set")
            fcalc_dark = self._model_dark(hkl, recalc=recalc)

        if self._scaler_dark is not None:
            fcalc_dark = self._scaler_dark(fcalc_dark)

        # Both phases are detached so gradients reach only ΔF_calc, never the
        # reconstructed ΔF_obs -- spurious terms there stall refinement.
        phi_dark = torch.angle(fcalc_dark).detach()
        phi_light = torch.angle(fcalc_light).detach()
        dphi = torch.exp(1j * (phi_light - phi_dark))

        dF_obs = self._F_obs_light - self._F_obs_dark

        # exp(iφ)·[(F + dF)·exp(i·dφ) - F], regrouped, with observed F and dF.
        delta_F_obs_complex = torch.exp(1j * phi_dark) * (
            self._F_obs_dark * (dphi - 1) + dF_obs * dphi
        )

        delta_F_calc = fcalc_light - fcalc_dark

        # Mask via torch.where, not boolean indexing: no nonzero() device sync.
        zero_c = torch.zeros_like(delta_F_obs_complex)
        delta_F_obs_complex = torch.where(self._mask, delta_F_obs_complex, zero_c)
        delta_F_calc = torch.where(self._mask, delta_F_calc, zero_c)

        # Complex difference loss (invalid: diff=0, sigma=1 → loss=0)
        diff = delta_F_obs_complex - delta_F_calc
        loss = torch.abs(diff)**2 / self._sigma_diff**2

        return (loss * self._mask).sum()

    def compute_free_metrics(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Compute loss and correlation on the FREE (test) set.

        This is the key metric for detecting overfitting in the α-δF degeneracy.
        The correct solution should have better free set metrics.

        Returns
        -------
        dict
            Dictionary with 'free_loss' and 'free_correlation'.
        """
        hkl = self.hkl

        # Compute F_calc if not provided
        if fcalc_light is None:
            fcalc_light = self._model_light(hkl, recalc=True)
            if self._scaler_light is not None:
                fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            fcalc_dark = self._model_dark(hkl, recalc=True)
            if self._scaler_dark is not None:
                fcalc_dark = self._scaler_dark(fcalc_dark)

        with torch.no_grad():
            # Compute phases (detached)
            phi_dark = torch.angle(fcalc_dark).detach()
            phi_light = torch.angle(fcalc_light).detach()
            dphi = torch.exp(1j * (phi_light - phi_dark))

            # Observed amplitude difference
            dF_obs = self._F_obs_light - self._F_obs_dark

            # Taylor-corrected observed complex difference
            delta_F_obs_complex = torch.exp(1j * phi_dark) * (
                self._F_obs_dark * (dphi - 1) + dF_obs * dphi
            )

            # Calculated complex difference
            delta_F_calc = fcalc_light - fcalc_dark

            # Apply FREE mask
            delta_F_obs_free = delta_F_obs_complex[self._free_mask]
            delta_F_calc_free = delta_F_calc[self._free_mask]
            sigma_diff_free = self._sigma_diff[self._free_mask]

            # Free loss
            diff_free = delta_F_obs_free - delta_F_calc_free
            free_loss = (torch.abs(diff_free)**2 / sigma_diff_free**2).mean().item()

            # Free correlation (amplitude difference)
            delta_F_obs_amp = (self._F_obs_light - self._F_obs_dark)[self._free_mask]
            delta_F_calc_amp = (torch.abs(fcalc_light) - torch.abs(fcalc_dark))[self._free_mask]

            obs_centered = delta_F_obs_amp - delta_F_obs_amp.mean()
            calc_centered = delta_F_calc_amp - delta_F_calc_amp.mean()

            free_correlation = (
                (obs_centered * calc_centered).sum() /
                (torch.sqrt((obs_centered**2).sum() * (calc_centered**2).sum()) + 1e-8)
            ).item()

        return {
            'free_loss': free_loss,
            'free_correlation': free_correlation,
            'n_free': self._free_mask.sum().item(),
        }

    def stats(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
    ) -> Dict[str, StatEntry]:
        """
        Get statistics for the difference refinement.

        Returns
        -------
        dict
            Dictionary with loss, correlation, R_diff, etc.
        """
        hkl = self.hkl

        # Compute F_calc
        if fcalc_light is None:
            fcalc_light = self._model_light(hkl, recalc=True)
            if self._scaler_light is not None:
                fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            fcalc_dark = self._model_dark(hkl, recalc=True)
            if self._scaler_dark is not None:
                fcalc_dark = self._scaler_dark(fcalc_dark)

        with torch.no_grad():
            loss = self.forward(fcalc_light, fcalc_dark, recalc=False)

            # Amplitude difference correlation
            delta_F_obs = (self._F_obs_light - self._F_obs_dark)[self._mask]
            delta_F_calc_amp = (torch.abs(fcalc_light) - torch.abs(fcalc_dark))[self._mask]

            obs_centered = delta_F_obs - delta_F_obs.mean()
            calc_centered = delta_F_calc_amp - delta_F_calc_amp.mean()

            correlation = (
                (obs_centered * calc_centered).sum() /
                (torch.sqrt((obs_centered**2).sum() * (calc_centered**2).sum()) + 1e-8)
            ).item()

            # R_diff
            r_diff = (
                torch.abs(delta_F_obs - delta_F_calc_amp).sum() /
                (torch.abs(delta_F_obs).sum() + 1e-8)
            ).item()

            # Phase difference statistics
            phi_dark = torch.angle(fcalc_dark)[self._mask]
            phi_light = torch.angle(fcalc_light)[self._mask]
            dphi = phi_light - phi_dark
            # Wrap to [-pi, pi]
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
            mean_abs_dphi = torch.abs(dphi).mean().item()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(self._mask.sum().item(), VERBOSITY_DETAILED),
            "correlation": stat(correlation, VERBOSITY_STANDARD),
            "r_diff": stat(r_diff, VERBOSITY_STANDARD),
            "mean_abs_dphi_deg": stat(mean_abs_dphi * 180 / 3.14159, VERBOSITY_DETAILED),
        }

    def __repr__(self) -> str:
        return "TaylorCorrectedDifferenceTarget()"


class RiceDifferenceTarget(Target):
    """Rice-likelihood difference target for time-resolved crystallography.

    Grafts detached model phases onto the observed amplitudes, differences in
    complex space, then takes magnitudes -- non-negative by construction, so a
    Rice likelihood (magnitude of a complex signal plus Gaussian noise) applies
    where an amplitude difference could not::

        A_obs = |F_obs_light·exp(i·φ_light) - F_obs_dark·exp(i·φ_dark)|
        ν     = |F_calc_light - F_calc_dark|
        NLL   = -log(A_obs/σ²) + (A_obs² + ν²)/(2σ²) - log I₀(A_obs·ν/σ²)

    ``forward`` evaluates the Bessel term as ``log(i0e(x)) + x`` for numerical
    stability; the math is equivalent.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection containing 'dark' and 'light' datasets.
    model_light : ModelFT or MixedModel
        Model for the light/excited state.
    model_dark : ModelFT
        Model for the dark/ground state.
    scaler_light, scaler_dark : Scaler, optional
        Per-state F_calc scalers.
    use_work_set : bool, optional
        Loss on the work set if True (default), else on the test set.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "rice_difference"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_light: "ModelFT" = None,
        model_dark: "ModelFT" = None,
        scaler_light: "Scaler" = None,
        scaler_dark: "Scaler" = None,
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        if "dark" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'dark' dataset")
        if "light" not in dataset_collection:
            raise ValueError("DatasetCollection must contain a 'light' dataset")

        self._dataset_collection = dataset_collection
        self._data_dark = dataset_collection["dark"]
        self._data_light = dataset_collection["light"]

        self.add_module("_model_light", model_light)
        self.add_module("_model_dark", model_dark)
        self.add_module("_scaler_light", scaler_light)
        self.add_module("_scaler_dark", scaler_dark)

        self.use_work_set = use_work_set

        self._setup_data()

    def _setup_data(self):
        """Setup observed data and masks."""
        F_light, sigma_light = self._data_light.get_corrected_data()
        rfree_light = self._data_light.rfree_flags
        valid_light = self._data_light.masks().to(torch.bool)
        F_dark, sigma_dark = self._data_dark.get_corrected_data()
        rfree_dark = self._data_dark.rfree_flags
        valid_dark = self._data_dark.masks().to(torch.bool)

        # Build validity mask
        valid_mask = torch.ones_like(rfree_light, dtype=torch.bool)
        if valid_light is not None:
            valid_mask = valid_mask & valid_light
        if valid_dark is not None:
            valid_mask = valid_mask & valid_dark

        # Clean invalid values to avoid NaN propagation in torch.where path
        sigma_diff = torch.sqrt(sigma_light**2 + sigma_dark**2)
        F_light = torch.where(valid_mask, F_light, torch.zeros_like(F_light))
        F_dark = torch.where(valid_mask, F_dark, torch.zeros_like(F_dark))
        sigma_diff = torch.where(valid_mask, sigma_diff, torch.ones_like(sigma_diff))

        self.register_buffer("_F_obs_light", F_light)
        self.register_buffer("_F_obs_dark", F_dark)
        self.register_buffer("_sigma_diff", sigma_diff)

        work_mask = rfree_light.bool() & rfree_dark.bool() & valid_mask
        free_mask = ~rfree_light.bool() & ~rfree_dark.bool() & valid_mask

        if self.use_work_set:
            mask = work_mask
        else:
            mask = free_mask
        self.register_buffer("_mask", mask)
        self.register_buffer("_work_mask", work_mask)
        self.register_buffer("_free_mask", free_mask)

    @property
    def hkl(self) -> torch.Tensor:
        """Common HKL indices."""
        return self._dataset_collection.hkl

    def forward(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = True,
    ) -> torch.Tensor:
        """
        Compute Rice distribution NLL loss for difference structure factors.

        Parameters
        ----------
        fcalc_light : torch.Tensor, optional
            Pre-computed light state structure factors.
        fcalc_dark : torch.Tensor, optional
            Pre-computed dark state structure factors.
        recalc : bool, optional
            Force recalculation if True. Default is True.

        Returns
        -------
        torch.Tensor
            Summed Rice NLL over included reflections (scalar).
        """
        hkl = self.hkl

        if fcalc_light is None:
            if self._model_light is None:
                raise RuntimeError("No model_light set")
            fcalc_light = self._model_light(hkl, recalc=recalc)

        if self._scaler_light is not None:
            fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            if self._model_dark is None:
                raise RuntimeError("No model_dark set")
            fcalc_dark = self._model_dark(hkl, recalc=recalc)

        if self._scaler_dark is not None:
            fcalc_dark = self._scaler_dark(fcalc_dark)

        # Phases detached: gradients must reach only ΔF_calc.
        phi_light = torch.angle(fcalc_light).detach()
        phi_dark = torch.angle(fcalc_dark).detach()

        F_obs_light_complex = self._F_obs_light * torch.exp(1j * phi_light)
        F_obs_dark_complex = self._F_obs_dark * torch.exp(1j * phi_dark)

        delta_F_obs_complex = F_obs_light_complex - F_obs_dark_complex
        delta_F_calc = fcalc_light - fcalc_dark

        A_obs = torch.abs(delta_F_obs_complex)
        nu = torch.abs(delta_F_calc)

        # Rice NLL via i0e: log(I₀(x)) = log(i0e(x)) + x.
        sigma_sq = self._sigma_diff**2
        sigma_sq_safe = torch.clamp(sigma_sq, min=1e-8)
        A_safe = torch.clamp(A_obs, min=1e-12)

        term1 = -torch.log(A_safe / sigma_sq_safe)
        term2 = (A_obs**2 + nu**2) / (2 * sigma_sq_safe)

        arg_bessel = A_obs * nu / sigma_sq_safe
        arg_bessel = torch.clamp(arg_bessel, max=1e6)
        term3 = -(torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel)

        nll = term1 + term2 + term3

        # Replace NaN/Inf with large finite value to maintain gradient signal
        nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))

        # Apply mask using torch.where (no nonzero sync)
        nll = torch.where(self._mask, nll, torch.zeros_like(nll))

        return (nll * self._mask).sum()

    def compute_free_metrics(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Compute loss and correlation on the FREE (test) set.

        Returns
        -------
        dict
            Dictionary with 'free_loss' and 'free_correlation'.
        """
        hkl = self.hkl

        if fcalc_light is None:
            fcalc_light = self._model_light(hkl, recalc=True)
            if self._scaler_light is not None:
                fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            fcalc_dark = self._model_dark(hkl, recalc=True)
            if self._scaler_dark is not None:
                fcalc_dark = self._scaler_dark(fcalc_dark)

        with torch.no_grad():
            # Amplitude difference correlation on free set
            delta_F_obs_amp = (self._F_obs_light - self._F_obs_dark)[self._free_mask]
            delta_F_calc_amp = (
                torch.abs(fcalc_light) - torch.abs(fcalc_dark)
            )[self._free_mask]

            obs_centered = delta_F_obs_amp - delta_F_obs_amp.mean()
            calc_centered = delta_F_calc_amp - delta_F_calc_amp.mean()

            free_correlation = (
                (obs_centered * calc_centered).sum()
                / (
                    torch.sqrt(
                        (obs_centered**2).sum() * (calc_centered**2).sum()
                    )
                    + 1e-8
                )
            ).item()

            # Free loss via Rice NLL
            phi_light = torch.angle(fcalc_light).detach()
            phi_dark = torch.angle(fcalc_dark).detach()

            F_obs_light_c = self._F_obs_light * torch.exp(1j * phi_light)
            F_obs_dark_c = self._F_obs_dark * torch.exp(1j * phi_dark)

            delta_obs_c = (F_obs_light_c - F_obs_dark_c)[self._free_mask]
            delta_calc = (fcalc_light - fcalc_dark)[self._free_mask]
            sigma_sq = self._sigma_diff[self._free_mask] ** 2
            sigma_sq_safe = torch.clamp(sigma_sq, min=1e-8)

            A_obs = torch.abs(delta_obs_c)
            nu = torch.abs(delta_calc)
            A_safe = torch.clamp(A_obs, min=1e-12)

            arg_bessel = A_obs * nu / sigma_sq_safe
            arg_bessel = torch.clamp(arg_bessel, max=1e6)

            nll = (
                -torch.log(A_safe / sigma_sq_safe)
                + (A_obs**2 + nu**2) / (2 * sigma_sq_safe)
                - (torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel)
            )
            nll = torch.where(torch.isfinite(nll), nll, torch.full_like(nll, 1e6))
            free_loss = nll.mean().item()

        return {
            "free_loss": free_loss,
            "free_correlation": free_correlation,
            "n_free": self._free_mask.sum().item(),
        }

    def stats(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
    ) -> Dict[str, StatEntry]:
        """
        Get statistics for the Rice difference refinement.

        Returns
        -------
        dict
            Dictionary with loss, correlation, R_diff, etc.
        """
        hkl = self.hkl

        if fcalc_light is None:
            fcalc_light = self._model_light(hkl, recalc=True)
            if self._scaler_light is not None:
                fcalc_light = self._scaler_light(fcalc_light)

        if fcalc_dark is None:
            fcalc_dark = self._model_dark(hkl, recalc=True)
            if self._scaler_dark is not None:
                fcalc_dark = self._scaler_dark(fcalc_dark)

        with torch.no_grad():
            loss = self.forward(fcalc_light, fcalc_dark, recalc=False)

            # Amplitude difference correlation
            delta_F_obs = (self._F_obs_light - self._F_obs_dark)[self._mask]
            delta_F_calc_amp = (
                torch.abs(fcalc_light) - torch.abs(fcalc_dark)
            )[self._mask]

            obs_centered = delta_F_obs - delta_F_obs.mean()
            calc_centered = delta_F_calc_amp - delta_F_calc_amp.mean()

            correlation = (
                (obs_centered * calc_centered).sum()
                / (
                    torch.sqrt(
                        (obs_centered**2).sum() * (calc_centered**2).sum()
                    )
                    + 1e-8
                )
            ).item()

            # R_diff
            r_diff = (
                torch.abs(delta_F_obs - delta_F_calc_amp).sum()
                / (torch.abs(delta_F_obs).sum() + 1e-8)
            ).item()

            # Phase difference statistics
            phi_dark = torch.angle(fcalc_dark)[self._mask]
            phi_light = torch.angle(fcalc_light)[self._mask]
            dphi = phi_light - phi_dark
            dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
            mean_abs_dphi = torch.abs(dphi).mean().item()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(self._mask.sum().item(), VERBOSITY_DETAILED),
            "correlation": stat(correlation, VERBOSITY_STANDARD),
            "r_diff": stat(r_diff, VERBOSITY_STANDARD),
            "mean_abs_dphi_deg": stat(
                mean_abs_dphi * 180 / 3.14159, VERBOSITY_DETAILED
            ),
        }

    def __repr__(self) -> str:
        return "RiceDifferenceTarget()"
