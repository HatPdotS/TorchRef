"""
Taylor-Corrected Difference Target for Time-Resolved Crystallography.

This target uses an exact Taylor expansion to properly account for the phase
shift between dark and light states, eliminating the false minimum that occurs
with naive phase-informed difference targets.

The key mathematical insight is that the true difference has its own phase
that differs from the dark phase by up to ~96°. The naive approach of simply
applying dark phases to observed amplitude differences creates a competing
force that stops refinement early (~70% of the way to the true light structure).

Mathematical Derivation
-----------------------
For the light state: F_light = (F + dF) * exp(i*(φ + dφ))
For the dark state:  F_dark = F * exp(i*φ)

Exact expansion of F_light - F_dark:
    ΔF = F_light - F_dark
       = (F + dF) * exp(i*φ) * exp(i*dφ) - F * exp(i*φ)
       = exp(i*φ) * [(F + dF) * exp(i*dφ) - F]
       = exp(i*φ) * [F * (exp(i*dφ) - 1) + dF * exp(i*dφ)]

This gives the corrected observed difference:
    ΔF_obs_corrected = exp(i*φ_dark) * [F_obs_dark * (exp(i*dφ) - 1) + dF_obs * exp(i*dφ)]

Where:
    - dF_obs = F_obs_light - F_obs_dark (observed amplitude difference)
    - φ_dark = angle(F_dark_calc) (dark phase from model)
    - exp(i*dφ) = exp(i*(φ_light_calc - φ_dark_calc)) (complex phase rotation from model)
"""

import torch
from torch import nn
from typing import TYPE_CHECKING, Dict

from .targets import Target
from torchref.utils.stats import (
    VERBOSITY_STANDARD,
    VERBOSITY_DETAILED,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.io.datasets import DatasetCollection
    from torchref.model import ModelFT, MixedModel
    from torchref.scaling import Scaler


class TaylorCorrectedDifferenceTarget(Target):
    """
    Taylor-corrected difference target for time-resolved crystallography.

    Uses an exact Taylor expansion to properly account for the phase shift
    between dark and light states when constructing observed complex differences:

        ΔF_obs = exp(i*φ_dark) * [F_obs_dark * (exp(i*dφ) - 1) + dF_obs * exp(i*dφ)]

    Where:
        - dφ = φ_light_calc - φ_dark_calc (phase rotation from model)
        - dF_obs = F_obs_light - F_obs_dark (observed amplitude difference)

    This formulation:
        1. Uses the exact complex exponential (no small-angle approximation)
        2. Properly accounts for both the amplitude difference and phase rotation
        3. Eliminates the false minimum that causes refinement to stop at ~70%

    The loss is computed as:
        Loss = |ΔF_obs_corrected - ΔF_calc|² / σ_diff²

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection containing 'dark' and 'light' datasets.
    model_light : ModelFT or MixedModel
        Model for the light/excited state.
    model_dark : ModelFT
        Model for the dark/ground state.
    scaler_light : Scaler, optional
        Scaler for light state F_calc.
    scaler_dark : Scaler, optional
        Scaler for dark state F_calc.
    use_work_set : bool, optional
        If True, compute loss on work set only. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    Basic usage::

        target = TaylorCorrectedDifferenceTarget(
            dataset_collection=collection,
            model_light=mixed_model,
            model_dark=model_dark,
        )

    With scalers::

        target = TaylorCorrectedDifferenceTarget(
            dataset_collection=collection,
            model_light=mixed_model,
            model_dark=model_dark,
            scaler_light=scaler_light,
            scaler_dark=scaler_dark,
        )
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
        _, F_light, sigma_light, rfree_light = self._data_light()
        _, F_dark, sigma_dark, rfree_dark = self._data_dark()

        # Handle MaskedTensor — extract data AND validity masks
        valid_light = valid_dark = None
        if hasattr(F_light, "get_mask"):
            valid_light = F_light.get_mask()
            F_light = F_light.get_data()
            sigma_light = sigma_light.get_data()
        if hasattr(F_dark, "get_mask"):
            valid_dark = F_dark.get_mask()
            F_dark = F_dark.get_data()
            sigma_dark = sigma_dark.get_data()

        self.register_buffer("_F_obs_light", F_light)
        self.register_buffer("_F_obs_dark", F_dark)
        self.register_buffer("_sigma_light", sigma_light)
        self.register_buffer("_sigma_dark", sigma_dark)
        self.register_buffer("_sigma_diff", torch.sqrt(sigma_light**2 + sigma_dark**2))

        # Work/test set masks incorporating data validity
        # Note: rfree flags indicate work set (True=work, False=free)
        valid_mask = torch.ones_like(rfree_light, dtype=torch.bool)
        if valid_light is not None:
            valid_mask = valid_mask & valid_light
        if valid_dark is not None:
            valid_mask = valid_mask & valid_dark
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
        Compute Taylor-corrected difference loss.

        The observed complex difference is constructed using the exact Taylor expansion:
            ΔF_obs = exp(i*φ_dark) * [F_obs_dark * (exp(i*dφ) - 1) + dF_obs * exp(i*dφ)]

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
            Mean weighted squared error.
        """
        hkl = self.hkl

        # Get F_calc for light/mixed
        if fcalc_light is None:
            if self._model_light is None:
                raise RuntimeError("No model_light set")
            fcalc_light = self._model_light(hkl, recalc=recalc)

        if self._scaler_light is not None:
            fcalc_light = self._scaler_light(fcalc_light)

        # Get F_calc for dark
        if fcalc_dark is None:
            if self._model_dark is None:
                raise RuntimeError("No model_dark set")
            fcalc_dark = self._model_dark(hkl, recalc=recalc)

        if self._scaler_dark is not None:
            fcalc_dark = self._scaler_dark(fcalc_dark)

        # Dark phase (dark model is typically frozen, but detach anyway for safety)
        phi_dark = torch.angle(fcalc_dark).detach()

        # Phase difference as complex exponential exp(i*dφ)
        # This is exact, no small-angle approximation needed
        # IMPORTANT: Detach phi_light so gradients only flow through ΔF_calc,
        # not through the reconstructed ΔF_obs_complex. Otherwise we get
        # spurious gradients that can cause refinement to stop at ~50%.
        phi_light = torch.angle(fcalc_light).detach()
        dphi = torch.exp(1j * (phi_light - phi_dark))  # complex unit vector (no gradients)

        # Observed amplitude difference
        dF_obs = self._F_obs_light - self._F_obs_dark

        # Exact Taylor expansion of F_light - F_dark:
        # ΔF = (F + dF) * exp(i*φ) * exp(i*dφ) - F * exp(i*φ)
        #    = exp(i*φ) * [(F + dF) * exp(i*dφ) - F]
        #    = exp(i*φ) * [F * (exp(i*dφ) - 1) + dF * exp(i*dφ)]
        #
        # Substituting observed values:
        delta_F_obs_complex = torch.exp(1j * phi_dark) * (
            self._F_obs_dark * (dphi - 1) + dF_obs * dphi
        )

        # Calculated complex difference
        delta_F_calc = fcalc_light - fcalc_dark

        # Apply mask
        delta_F_obs_complex = delta_F_obs_complex[self._mask]
        delta_F_calc = delta_F_calc[self._mask]
        sigma_diff = self._sigma_diff[self._mask]

        # Complex difference loss
        diff = delta_F_obs_complex - delta_F_calc
        loss = (torch.abs(diff)**2 / sigma_diff**2).mean()

        return loss

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
