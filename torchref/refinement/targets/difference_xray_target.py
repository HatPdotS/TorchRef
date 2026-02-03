# =============================================================================
# Difference Target for Time-Resolved Crystallography
# =============================================================================

from .targets import Target
import torch
from typing import Tuple, Dict
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)


class DifferenceXrayTarget(Target):
    """
    Target for time-resolved crystallography comparing light/dark states.

    Computes difference structure factors and compares against observed differences:

    - ΔF_calc = |F_light_calc| - |F_dark_calc|
    - ΔF_obs = F_light_obs - F_dark_obs

    Uses Gaussian NLL with proper error propagation:

    - σ_diff = sqrt(σ_light² + σ_dark²)
    - NLL = 0.5 * (ΔF_obs - ΔF_calc)² / σ_diff² + log(σ_diff) + 0.5*log(2π)

    Supports two initialization modes:

    1. **DatasetCollection mode** (recommended): Pass a DatasetCollection with
       pre-aligned datasets. This is more efficient and ensures consistency
       with other targets using the same data.

    2. **Separate datasets mode**: Pass individual ReflectionData objects.
       HKL matching is performed automatically.

    Parameters
    ----------
    dataset_collection : DatasetCollection, optional
        Collection containing 'dark' and 'light' datasets (pre-aligned HKL).
        If provided, data_light and data_dark are ignored.
    data_light : ReflectionData, optional
        Reflection data for the light (excited) state.
    data_dark : ReflectionData, optional
        Reflection data for the dark (ground) state.
    model_light : ModelFT or MixedModel
        Model for the light state structure factor calculation.
    model_dark : ModelFT
        Model for the dark state structure factor calculation.
    scaler_light : ScalerBase, optional
        Scaler for the light state F_calc. Can be shared with other targets.
    scaler_dark : ScalerBase, optional
        Scaler for the dark state F_calc. Can be shared with other targets.
    use_work_set : bool, optional
        If True, compute loss on work set. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    Using DatasetCollection (recommended for sharing scalers)::

        # Create collection with aligned HKL
        collection = DatasetCollection()
        collection.add_dataset('dark', data_dark, set_as_reference=True)
        collection.add_dataset('light', data_light)

        # Create shared scalers
        scaler_dark = IsotropicScaler(data=collection['dark'], model=model_dark)
        scaler_light = IsotropicScaler(data=collection['light'], model=model_mixed)

        # Create targets that share scalers
        xray_dark = GaussianXrayTarget(
            data=collection['dark'], model=model_dark, scaler=scaler_dark
        )
        xray_light = GaussianXrayTarget(
            data=collection['light'], model=model_mixed, scaler=scaler_light
        )
        diff_target = DifferenceXrayTarget(
            dataset_collection=collection,
            model_light=model_mixed,
            model_dark=model_dark,
            scaler_light=scaler_light,
            scaler_dark=scaler_dark,
        )

        # Combined loss
        loss = xray_dark() + xray_light() + diff_target()

    Using separate datasets::

        diff_target = DifferenceXrayTarget(
            data_light=data_light,
            data_dark=data_dark,
            model_light=model_light,
            model_dark=model_dark,
        )
        loss = diff_target()

    With mixed model for partial occupancy::

        mixed_light = MixedModel([model_dark, model_light], [0.7, 0.3])
        diff_target = DifferenceXrayTarget(
            dataset_collection=collection,
            model_light=mixed_light,
            model_dark=model_dark,
            scaler_light=scaler_light,
            scaler_dark=scaler_dark,
        )
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

        # Store collection reference
        self._dataset_collection = dataset_collection

        # Handle DatasetCollection mode
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
        """
        Common HKL indices for both datasets.

        Returns the aligned HKL from DatasetCollection if available,
        otherwise the matched HKL computed from separate datasets.
        """
        if self._use_collection:
            return self._dataset_collection.hkl
        else:
            if self._common_hkl is None:
                self._match_reflections()
            return self._common_hkl

    def _hkl_to_hash(self, hkl: torch.Tensor) -> torch.Tensor:
        """
        Convert HKL indices to unique hash values for efficient matching.

        Uses a simple polynomial hash: hash = h * p1 + k * p2 + l
        where p1 and p2 are large primes.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).

        Returns
        -------
        torch.Tensor
            Hash values with shape (n_reflections,).
        """
        # Use large primes for hashing
        p1 = 1000003
        p2 = 1000033

        h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
        return h * p1 + k * p2 + l

    def _match_reflections(self):
        """
        Find common HKL indices between light and dark datasets.

        Uses hash-based matching for O(N log N) efficiency.
        Stores matched indices for both datasets.

        This method is only used when datasets are not pre-aligned
        via DatasetCollection.
        """
        if self._use_collection:
            # Datasets are already aligned - no matching needed
            return

        hkl_light, _, _, _ = self._data_light()
        hkl_dark, _, _, _ = self._data_dark()

        # Compute hashes
        hash_light = self._hkl_to_hash(hkl_light)
        hash_dark = self._hkl_to_hash(hkl_dark)

        # Sort hashes and get indices
        sorted_light, sort_idx_light = torch.sort(hash_light)
        sorted_dark, sort_idx_dark = torch.sort(hash_dark)

        # Find intersection using sorted merge
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
                # Match found - map back to original indices
                matched_light.append(sort_idx_light[i].item())
                matched_dark.append(sort_idx_dark[j].item())
                i += 1
                j += 1

        # Store matched indices as tensors
        device = hkl_light.device
        self._matched_indices_light = torch.tensor(
            matched_light, dtype=torch.long, device=device
        )
        self._matched_indices_dark = torch.tensor(
            matched_dark, dtype=torch.long, device=device
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
            Boolean mask for work/test set selection and valid data.
        """
        if self._use_collection:
            return self._get_delta_F_obs_collection()
        else:
            return self._get_delta_F_obs_matched()

    def _get_delta_F_obs_collection(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get delta F_obs when using DatasetCollection (aligned HKL)."""
        # Get observed data - datasets are already aligned
        _, F_obs_light, sigma_light, rfree_light = self._data_light()
        _, F_obs_dark, sigma_dark, rfree_dark = self._data_dark()

        # Handle MaskedTensor inputs and get validity masks
        if hasattr(F_obs_light, "get_mask"):
            validity_light = F_obs_light.get_mask()
            F_obs_light = F_obs_light.get_data()
            sigma_light = sigma_light.get_data()
        else:
            validity_light = torch.ones(len(F_obs_light), dtype=torch.bool,
                                        device=F_obs_light.device)

        if hasattr(F_obs_dark, "get_mask"):
            validity_dark = F_obs_dark.get_mask()
            F_obs_dark = F_obs_dark.get_data()
            sigma_dark = sigma_dark.get_data()
        else:
            validity_dark = torch.ones(len(F_obs_dark), dtype=torch.bool,
                                       device=F_obs_dark.device)

        # Compute difference and propagated error
        delta_F_obs = F_obs_light - F_obs_dark
        sigma_diff = torch.sqrt(sigma_light**2 + sigma_dark**2)

        # Combined mask: valid in both datasets AND in work/test set
        # Reflections must be valid (not masked) in BOTH datasets
        valid_both = validity_light & validity_dark

        # Work/test set selection
        if self.use_work_set:
            set_mask = rfree_light & rfree_dark  # Work set in both
        else:
            set_mask = ~rfree_light & ~rfree_dark  # Test set in both

        mask = valid_both & set_mask

        return delta_F_obs, sigma_diff, mask

    def _get_delta_F_obs_matched(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get delta F_obs when using matched indices (non-collection mode)."""
        if self._matched_indices_light is None:
            self._match_reflections()

        # Get observed data
        _, F_obs_light, sigma_light, rfree_light = self._data_light()
        _, F_obs_dark, sigma_dark, rfree_dark = self._data_dark()

        # Handle MaskedTensor inputs and get validity masks
        if hasattr(F_obs_light, "get_mask"):
            validity_light = F_obs_light.get_mask()
            F_obs_light = F_obs_light.get_data()
            sigma_light = sigma_light.get_data()
        else:
            validity_light = torch.ones(len(F_obs_light), dtype=torch.bool,
                                        device=F_obs_light.device)

        if hasattr(F_obs_dark, "get_mask"):
            validity_dark = F_obs_dark.get_mask()
            F_obs_dark = F_obs_dark.get_data()
            sigma_dark = sigma_dark.get_data()
        else:
            validity_dark = torch.ones(len(F_obs_dark), dtype=torch.bool,
                                       device=F_obs_dark.device)

        # Extract matched reflections
        F_light = F_obs_light[self._matched_indices_light]
        F_dark = F_obs_dark[self._matched_indices_dark]
        sig_light = sigma_light[self._matched_indices_light]
        sig_dark = sigma_dark[self._matched_indices_dark]
        valid_light = validity_light[self._matched_indices_light]
        valid_dark = validity_dark[self._matched_indices_dark]

        # Compute difference and propagated error
        delta_F_obs = F_light - F_dark
        sigma_diff = torch.sqrt(sig_light**2 + sig_dark**2)

        # Combined validity: must be valid in BOTH datasets
        valid_both = valid_light & valid_dark

        # Work/test set mask (use intersection of both masks)
        rfree_light_matched = rfree_light[self._matched_indices_light]
        rfree_dark_matched = rfree_dark[self._matched_indices_dark]

        # Only include reflections that are valid AND in work/test set for BOTH
        if self.use_work_set:
            set_mask = rfree_light_matched & rfree_dark_matched
        else:
            set_mask = ~rfree_light_matched & ~rfree_dark_matched

        mask = valid_both & set_mask

        return delta_F_obs, sigma_diff, mask

    def get_delta_F_calc(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = True,
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
            Force recalculation if True. Default is True.

        Returns
        -------
        torch.Tensor
            ΔF_calc for all reflections (full size, use mask from get_delta_F_obs).
        """
        # Get HKL to use
        hkl = self.hkl

        # Compute F_calc for light state
        if fcalc_light is None:
            if self._model_light is None:
                raise RuntimeError(
                    "Cannot compute F_calc_light: no model_light set."
                )
            fcalc_light = self._model_light(hkl, recalc=recalc)

        # Apply scaler if available
        if self._scaler_light is not None:
            fcalc_light = self._scaler_light(fcalc_light)

        # Compute F_calc for dark state
        if fcalc_dark is None:
            if self._model_dark is None:
                raise RuntimeError("Cannot compute F_calc_dark: no model_dark set.")
            fcalc_dark = self._model_dark(hkl, recalc=recalc)

        # Apply scaler if available
        if self._scaler_dark is not None:
            fcalc_dark = self._scaler_dark(fcalc_dark)

        # Compute amplitude difference
        F_light_amp = torch.abs(fcalc_light)
        F_dark_amp = torch.abs(fcalc_dark)
        delta_F_calc = F_light_amp - F_dark_amp

        return delta_F_calc

    def forward(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = True,
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
            Force recalculation if True. Default is True.

        Returns
        -------
        torch.Tensor
            Mean NLL loss value.
        """
        # Get observed differences
        delta_F_obs, sigma_diff, mask = self.get_delta_F_obs()

        # Get calculated differences
        delta_F_calc = self.get_delta_F_calc(
            fcalc_light=fcalc_light, fcalc_dark=fcalc_dark, recalc=recalc
        )

        # Apply mask
        delta_F_obs = delta_F_obs[mask]
        delta_F_calc = delta_F_calc[mask]
        sigma_diff = sigma_diff[mask]

        # Compute Gaussian NLL
        diff = delta_F_obs - delta_F_calc

        # Avoid division by zero
        eps = torch.median(sigma_diff).item() * 1e-1
        sigma_safe = torch.clamp(sigma_diff, min=eps)

        log_2pi = torch.log(
            torch.tensor(2.0 * torch.pi, device=sigma_diff.device, dtype=sigma_diff.dtype)
        )
        nll = (
            0.5 * (diff**2) / (sigma_safe**2)
            + torch.log(sigma_safe)
            + 0.5 * log_2pi
        )

        return nll.mean()

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
