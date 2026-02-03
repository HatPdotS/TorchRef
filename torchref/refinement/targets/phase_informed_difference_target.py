"""
Phase-Informed Difference Target for Time-Resolved Crystallography.

This target uses model phases to create a complex difference loss,
providing cleaner gradients compared to amplitude-only difference targets.

The key insight is that in difference Fourier methods, the signal (peaks)
is localized while noise is distributed uniformly. This allows even weak
difference signals to be detected and used for refinement.

By using current model phases, the target iteratively improves - as the
model gets better, the phases improve, leading to better gradients.
"""

import torch
from torch import nn
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

from .targets import Target
from torchref.utils.stats import (
    VERBOSITY_STANDARD,
    VERBOSITY_DETAILED,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.io.datasets import ReflectionData, DatasetCollection
    from torchref.model import ModelFT, MixedModel
    from torchref.scaling import Scaler


class PhaseInformedDifferenceTarget(Target):
    """
    Phase-informed difference target for time-resolved crystallography.

    Uses model phases to create complex observed differences, then compares
    with calculated complex differences:

        ΔF_calc = F_mixed_calc - F_dark_calc  (complex)
        ΔF_obs_complex = ΔF_obs * exp(i * φ)  (using model phases)
        Loss = |ΔF_obs_complex - ΔF_calc|² / σ_diff²

    The phase source can be configured:
    - "dark": Use dark model phases (stable reference)
    - "difference": Use phase of calculated difference ΔF_calc (self-consistent)
    - "mixed": Use mixed/light model phases

    Using current model phases is standard practice in difference Fourier
    methods. The iterative nature of refinement self-corrects any phase bias,
    and the localized nature of difference peaks allows detection of weak signals.

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
    phase_source : str, optional
        Source for phases: "dark", "difference", or "mixed". Default is "difference".
    use_work_set : bool, optional
        If True, compute loss on work set only. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    Using difference phases (recommended)::

        target = PhaseInformedDifferenceTarget(
            dataset_collection=collection,
            model_light=mixed_model,
            model_dark=model_dark,
            phase_source="difference",
        )

    Using dark phases::

        target = PhaseInformedDifferenceTarget(
            dataset_collection=collection,
            model_light=mixed_model,
            model_dark=model_dark,
            phase_source="dark",
        )
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
        _, F_light, sigma_light, rfree_light = self._data_light()
        _, F_dark, sigma_dark, rfree_dark = self._data_dark()

        # Handle MaskedTensor
        if hasattr(F_light, "get_data"):
            F_light = F_light.get_data()
            sigma_light = sigma_light.get_data()
        if hasattr(F_dark, "get_data"):
            F_dark = F_dark.get_data()
            sigma_dark = sigma_dark.get_data()

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
        """
        Get phases based on phase_source setting.

        IMPORTANT: Phases are detached from the computation graph so that
        gradients only flow through ΔF_calc, not through the reconstructed
        ΔF_obs_complex. Otherwise we get spurious gradients that can cause
        refinement to stop at ~50%.

        Parameters
        ----------
        F_dark_calc : complex tensor
            Dark model structure factors
        F_mixed_calc : complex tensor
            Mixed/light model structure factors

        Returns
        -------
        torch.Tensor
            Phases to use for observed difference (detached from gradient graph)
        """
        if self.phase_source == "dark":
            return torch.angle(F_dark_calc).detach()
        elif self.phase_source == "mixed":
            return torch.angle(F_mixed_calc).detach()
        elif self.phase_source == "difference":
            # Use phase of calculated difference
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

        # Get phases based on phase_source setting
        phi = self._get_phases(fcalc_dark, fcalc_light)

        # Observed amplitude difference
        delta_F_obs = self._F_obs_light - self._F_obs_dark

        # Make observed difference complex using model phases
        delta_F_obs_complex = delta_F_obs * torch.exp(1j * phi)

        # Calculated complex difference
        delta_F_calc = fcalc_light - fcalc_dark

        # Apply mask
        delta_F_obs_complex = delta_F_obs_complex[self._mask]
        delta_F_calc = delta_F_calc[self._mask]
        sigma_diff = self._sigma_diff[self._mask]

        # Complex difference
        diff = delta_F_obs_complex - delta_F_calc

        # Weighted MSE
        loss = (torch.abs(diff)**2 / sigma_diff**2).mean()

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
