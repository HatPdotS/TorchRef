"""
SampledMLPhaseTarget - Phase-aware ML target using reparameterized sampling.

This module provides a maximum-likelihood refinement target that accounts for
phase uncertainty derived from amplitude errors. Uses the reparameterization
trick to enable differentiable Monte Carlo estimation of expected structure
factor errors in complex space.

Key insight: Standard ML refinement treats calculated phases as exact. This
approach samples phases from a distribution whose width depends on amplitude
discrepancy, allowing gradients to naturally account for phase uncertainty.
"""

import numpy as np
import torch
from typing import TYPE_CHECKING, Dict, Tuple

from torchref.refinement.targets.base import Target
from torchref.refinement.targets.xray import XrayTarget
from torchref.utils.stats import (
    VERBOSITY_STANDARD,
    VERBOSITY_DETAILED,
    VERBOSITY_DEBUG,
    StatEntry,
    stat,
)

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model import Model, ModelFT
    from torchref.scaling import Scaler


class SampledMLPhaseTarget(XrayTarget):
    """
    Phase-aware ML target using reparameterized sampling.

    Computes E[|F_obs*exp(i*phi) - F_calc|^2] where phi ~ N(phi_ref, sigma_phi^2)
    with sigma_phi derived from amplitude errors and discrepancy.

    Uses French-Wilson posteriors for amplitude estimation and supports
    both Monte Carlo sampling and analytical evaluation.

    Parameters
    ----------
    data : ReflectionData
        Reference to the ReflectionData object.
    model : Model or ModelFT, optional
        Reference to Model object for F_calc computation.
    scaler : Scaler, optional
        Reference to the Scaler object.
    phi_ref : torch.Tensor, optional
        Reference phases (e.g., from dark state). If None, uses phi_calc.
    n_samples : int, optional
        Number of MC samples. Default is 32.
    sigma_model_log : float, optional
        Model error in log(I) space (~R_work). Default is 0.15.
    use_analytical : bool, optional
        Use closed-form instead of MC sampling. Default is False.
    use_antithetic : bool, optional
        Use antithetic sampling for variance reduction. Default is True.
    use_work_set : bool, optional
        If True, compute loss on work set. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Attributes
    ----------
    name : str
        Target name for LossState registration.

    Examples
    --------
    Basic usage with model::

        target = SampledMLPhaseTarget(
            data=reflection_data,
            model=model,
            scaler=scaler,
            n_samples=32,
        )
        loss = target()  # Computes F_calc internally

    With pre-computed F_calc::

        target = SampledMLPhaseTarget(data=reflection_data)
        loss = target(fcalc=F_calc_precomputed)

    With reference phases from dark state::

        target = SampledMLPhaseTarget(
            data=light_data,
            model=light_model,
            phi_ref=torch.angle(F_dark_calc),
        )
    """

    name: str = "xray_sampled_ml"

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        phi_ref: torch.Tensor = None,
        n_samples: int = 32,
        sigma_model_log: float = 0.15,
        use_analytical: bool = False,
        use_antithetic: bool = True,
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            verbose=verbose,
        )

        # Update name based on work/test set
        self.name = "xray_sampled_ml_work" if use_work_set else "xray_sampled_ml_test"

        # Register tunable parameters as buffers for state_dict access
        self.register_buffer("_n_samples", torch.tensor(n_samples, dtype=torch.int64))
        self.register_buffer("_sigma_model_log", torch.tensor(sigma_model_log))
        self.register_buffer("_use_analytical", torch.tensor(use_analytical))
        self.register_buffer("_use_antithetic", torch.tensor(use_antithetic))

        # Reference phases (optional - if None, uses phi_calc)
        if phi_ref is not None:
            self.register_buffer("_phi_ref", phi_ref)
        else:
            self._phi_ref = None

        # Cache for diagnostics (populated on forward)
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    # =========================================================================
    # Property accessors for buffer parameters
    # =========================================================================

    @property
    def n_samples(self) -> int:
        """Get number of MC samples."""
        return self._n_samples.item()

    @n_samples.setter
    def n_samples(self, value: int):
        """Set number of MC samples."""
        self._n_samples.fill_(value)

    @property
    def sigma_model_log(self) -> float:
        """Get model error in log(I) space."""
        return self._sigma_model_log.item()

    @sigma_model_log.setter
    def sigma_model_log(self, value: float):
        """Set model error in log(I) space."""
        self._sigma_model_log.fill_(value)

    @property
    def use_analytical(self) -> bool:
        """Get whether to use analytical form."""
        return self._use_analytical.item()

    @use_analytical.setter
    def use_analytical(self, value: bool):
        """Set whether to use analytical form."""
        self._use_analytical.fill_(value)

    @property
    def use_antithetic(self) -> bool:
        """Get whether to use antithetic sampling."""
        return self._use_antithetic.item()

    @use_antithetic.setter
    def use_antithetic(self, value: bool):
        """Set whether to use antithetic sampling."""
        self._use_antithetic.fill_(value)

    # =========================================================================
    # Core computation methods
    # =========================================================================

    def french_wilson_moments(
        self,
        I_obs: torch.Tensor,
        sigma_I: torch.Tensor,
        Sigma_wilson: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance of |F_true| given I_obs.

        Properly handles negative and weak intensities using numerical
        integration over a grid.

        Parameters
        ----------
        I_obs : torch.Tensor
            Observed intensities.
        sigma_I : torch.Tensor
            Intensity uncertainties.
        Sigma_wilson : torch.Tensor, optional
            Wilson expected intensities.

        Returns
        -------
        F_mean : torch.Tensor
            Posterior mean of |F|.
        F_var : torch.Tensor
            Posterior variance of |F|.
        """
        device = I_obs.device
        dtype = I_obs.dtype

        if Sigma_wilson is None:
            Sigma_wilson = torch.clamp(I_obs, min=sigma_I)

        # Integration grid (30 points is sufficient for smooth posteriors)
        n_points = 30
        F_max = (
            torch.sqrt(torch.clamp(I_obs + 4 * sigma_I, min=sigma_I))
            + torch.sqrt(Sigma_wilson)
        )

        t = torch.linspace(0.01, 1.0, n_points, device=device, dtype=dtype)
        F_grid = t.unsqueeze(0) * F_max.unsqueeze(1)  # (n_refl, n_points)
        dF = F_grid[:, 1:2] - F_grid[:, :1]

        # Log posterior: measurement likelihood + Wilson prior
        I_obs_exp = I_obs.unsqueeze(1)
        sigma_I_exp = sigma_I.unsqueeze(1)
        Sigma_exp = Sigma_wilson.unsqueeze(1)

        log_post = (
            -(I_obs_exp - F_grid**2) ** 2 / (2 * sigma_I_exp**2)  # Gaussian in I
            + torch.log(F_grid + 1e-10)  # Jacobian
            - F_grid**2 / Sigma_exp  # Wilson prior
        )
        log_post = log_post - log_post.max(dim=1, keepdim=True)[0]

        post = torch.exp(log_post)
        post = post / (post.sum(dim=1, keepdim=True) * dF + 1e-10)

        # Moments
        F_mean = (post * F_grid * dF).sum(dim=1)
        F_sq_mean = (post * F_grid**2 * dF).sum(dim=1)
        F_var = torch.clamp(F_sq_mean - F_mean**2, min=1e-6)

        return F_mean, F_var

    def compute_sigma_phi(
        self,
        F_obs: torch.Tensor,
        sigma_F_obs: torch.Tensor,
        F_calc_amp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phase uncertainty from amplitude uncertainties and discrepancy.

        The phase uncertainty has three components:
        1. Measurement uncertainty: sigma_F_obs / |F_obs|
        2. Model uncertainty: sigma_model_log (multiplicative)
        3. Excess from amplitude discrepancy beyond expected

        Parameters
        ----------
        F_obs : torch.Tensor
            Observed amplitudes (or French-Wilson means).
        sigma_F_obs : torch.Tensor
            Amplitude uncertainties.
        F_calc_amp : torch.Tensor
            Calculated amplitudes |F_calc|.

        Returns
        -------
        sigma_phi : torch.Tensor
            Phase uncertainty in radians.
        """
        # Model error (lognormal -> multiplicative in |F|)
        sigma_F_model = self.sigma_model_log * F_obs

        # Total amplitude uncertainty
        sigma_F_total = torch.sqrt(sigma_F_obs**2 + sigma_F_model**2)

        # Base phase uncertainty from amplitude uncertainties
        sigma_phi_meas = sigma_F_obs / (F_obs + 1e-6)
        sigma_phi_model = sigma_F_model / (F_calc_amp + 1e-6)

        # Excess from amplitude discrepancy
        amplitude_discrepancy = torch.abs(F_obs - F_calc_amp)
        expected_discrepancy = sigma_F_total
        excess = torch.clamp(amplitude_discrepancy - expected_discrepancy, min=0)

        sigma_phi_excess = excess / (torch.minimum(F_obs, F_calc_amp) + 1e-6)

        # Combined
        sigma_phi = torch.sqrt(
            sigma_phi_meas**2 + sigma_phi_model**2 + sigma_phi_excess**2
        )
        sigma_phi = torch.clamp(sigma_phi, min=0.01, max=2.0)

        return sigma_phi

    def forward(
        self,
        fcalc: torch.Tensor = None,
        recalc: bool = True,
    ) -> torch.Tensor:
        """
        Compute phase-aware ML loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed complex structure factors. If provided, uses these
            instead of computing from model.
        recalc : bool, optional
            Force recalculation if True. Default is True.

        Returns
        -------
        torch.Tensor
            Mean weighted loss value.
        """
        # Get data using XrayTarget.get_data() - handles work/test selection
        F_obs, F_calc, sigma_F_obs, centric_flags = self.get_data(fcalc=fcalc)

        device = F_obs.device
        n_refl = F_obs.shape[0]

        F_calc_amp = torch.abs(F_calc)
        phi_calc = torch.angle(F_calc)

        # Use reference phases if provided, otherwise use calculated phases
        if self._phi_ref is not None:
            # Need to select same reflections as F_obs
            # This assumes phi_ref was computed on full dataset
            rfree_mask = self._data.rfree_flags
            # Note: rfree_mask may be int32 (0/1), must convert to bool for proper masking
            rfree_bool = rfree_mask.bool()
            if self.use_work_set:
                phi_ref = self._phi_ref[rfree_bool]
            else:
                phi_ref = self._phi_ref[~rfree_bool]
        else:
            phi_ref = phi_calc

        # Get intensity data for French-Wilson
        # Convert F_obs to I_obs (approximation for French-Wilson input)
        I_obs = F_obs**2
        sigma_I = 2 * F_obs * sigma_F_obs  # Error propagation

        # French-Wilson posterior moments
        F_mean, F_var = self.french_wilson_moments(I_obs, sigma_I)
        sigma_F_meas = torch.sqrt(F_var)

        # Phase uncertainty
        sigma_phi = self.compute_sigma_phi(F_mean, sigma_F_meas, F_calc_amp)

        # Effective figure of merit
        m_eff = torch.exp(-sigma_phi**2 / 2)

        if self.use_analytical:
            # Analytical form (faster, no sampling noise)
            phase_shift = phi_ref - phi_calc
            expected_sq_error = (
                F_mean**2
                + F_calc_amp**2
                - 2 * F_mean * F_calc_amp * torch.cos(phase_shift) * m_eff
            )
        else:
            # Monte Carlo with reparameterization
            n_samples = self.n_samples

            if self.use_antithetic:
                eps = torch.randn(n_refl, n_samples // 2, device=device)
                eps = torch.cat([eps, -eps], dim=1)
            else:
                eps = torch.randn(n_refl, n_samples, device=device)

            # Sample phases: phi = phi_ref + sigma_phi * eps
            phi_samples = phi_ref.unsqueeze(1) + sigma_phi.unsqueeze(1) * eps

            # Construct complex F_obs samples
            F_obs_real = F_mean.unsqueeze(1) * torch.cos(phi_samples)
            F_obs_imag = F_mean.unsqueeze(1) * torch.sin(phi_samples)

            # Complex error: F_obs_sample - F_calc
            error_real = F_obs_real - F_calc.real.unsqueeze(1)
            error_imag = F_obs_imag - F_calc.imag.unsqueeze(1)

            # |error|^2 for each sample
            sq_error_samples = error_real**2 + error_imag**2

            # Monte Carlo estimate
            expected_sq_error = sq_error_samples.mean(dim=1)

        # Model error for weighting
        sigma_F_model = self.sigma_model_log * F_mean
        sigma_F_total = torch.sqrt(sigma_F_meas**2 + sigma_F_model**2)

        # Weighted loss
        weights = 1.0 / (sigma_F_total**2 + 1e-6)
        loss = (weights * expected_sq_error).mean()

        # Store diagnostics
        self._last_diagnostics = {
            "F_mean": F_mean.detach(),
            "sigma_F_total": sigma_F_total.detach(),
            "sigma_phi": sigma_phi.detach(),
            "m_eff": m_eff.detach(),
            "expected_sq_error": expected_sq_error.detach(),
            "weights": weights.detach(),
        }

        return loss

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """
        Get statistics for this target.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors.

        Returns
        -------
        dict
            Statistics dict with StatEntry values containing verbosity levels.
        """
        # Compute loss to populate diagnostics
        with torch.no_grad():
            loss = self.forward(fcalc=fcalc)

        diag = self._last_diagnostics

        # R-factor computation
        F_obs, F_calc, _, _ = self.get_data(fcalc=fcalc)
        F_calc_amp = torch.abs(F_calc)

        r_factor = (
            torch.abs(F_obs - F_calc_amp).sum() / (F_obs.sum() + 1e-8)
        ).item()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(F_obs), VERBOSITY_DEBUG),
            "r_factor": stat(r_factor, VERBOSITY_STANDARD),
            "mean_m_eff": stat(diag["m_eff"].mean().item(), VERBOSITY_STANDARD),
            "mean_sigma_phi": stat(diag["sigma_phi"].mean().item(), VERBOSITY_DETAILED),
            "min_m_eff": stat(diag["m_eff"].min().item(), VERBOSITY_DETAILED),
            "max_sigma_phi": stat(diag["sigma_phi"].max().item(), VERBOSITY_DETAILED),
            "n_samples": stat(self.n_samples, VERBOSITY_DEBUG),
            "sigma_model_log": stat(self.sigma_model_log, VERBOSITY_DEBUG),
        }

    def __repr__(self) -> str:
        mode = "analytical" if self.use_analytical else f"MC({self.n_samples})"
        return f"SampledMLPhaseTarget(mode={mode}, sigma_model={self.sigma_model_log:.3f})"


class SampledMLDifferenceTarget(Target):
    """
    Phase-aware difference target for two-dataset refinement.

    Uses dark state phases as reference, with phase uncertainty
    informed by amplitude changes between states. Jointly refines
    against both dark and light datasets.

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
    n_samples : int, optional
        Number of MC samples. Default is 32.
    sigma_model_log : float, optional
        Model error in log(I) space. Default is 0.15.
    use_work_set : bool, optional
        If True, compute loss on work set. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    Basic usage::

        target = SampledMLDifferenceTarget(
            dataset_collection=collection,
            model_light=mixed_model,
            model_dark=model_dark,
            n_samples=32,
        )
        loss = target()
    """

    name: str = "sampled_ml_difference"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_light: "ModelFT" = None,
        model_dark: "ModelFT" = None,
        scaler_light: "Scaler" = None,
        scaler_dark: "Scaler" = None,
        n_samples: int = 32,
        sigma_model_log: float = 0.15,
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        # Validation
        if "dark" not in dataset_collection:
            raise ValueError("DatasetCollection must contain 'dark' dataset")
        if "light" not in dataset_collection:
            raise ValueError("DatasetCollection must contain 'light' dataset")

        self._dataset_collection = dataset_collection
        self._data_dark = dataset_collection["dark"]
        self._data_light = dataset_collection["light"]

        # Register models as submodules
        self.add_module("_model_light", model_light)
        self.add_module("_model_dark", model_dark)
        self.add_module("_scaler_light", scaler_light)
        self.add_module("_scaler_dark", scaler_dark)

        # Tunable parameters as buffers
        self.register_buffer("_n_samples", torch.tensor(n_samples, dtype=torch.int64))
        self.register_buffer("_sigma_model_log", torch.tensor(sigma_model_log))

        self.use_work_set = use_work_set

        # Cache for diagnostics
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

        # Setup data buffers
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

    @property
    def n_samples(self) -> int:
        """Get number of MC samples."""
        return self._n_samples.item()

    @n_samples.setter
    def n_samples(self, value: int):
        """Set number of MC samples."""
        self._n_samples.fill_(value)

    @property
    def sigma_model_log(self) -> float:
        """Get model error in log(I) space."""
        return self._sigma_model_log.item()

    @sigma_model_log.setter
    def sigma_model_log(self, value: float):
        """Set model error in log(I) space."""
        self._sigma_model_log.fill_(value)

    def _compute_sigma_phi(
        self,
        F_obs: torch.Tensor,
        sigma_F_obs: torch.Tensor,
        F_calc_amp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phase uncertainty from amplitude uncertainties and discrepancy.

        Parameters
        ----------
        F_obs : torch.Tensor
            Observed amplitudes.
        sigma_F_obs : torch.Tensor
            Amplitude uncertainties.
        F_calc_amp : torch.Tensor
            Calculated amplitudes |F_calc|.

        Returns
        -------
        sigma_phi : torch.Tensor
            Phase uncertainty in radians.
        """
        sigma_F_model = self.sigma_model_log * F_obs
        sigma_F_total = torch.sqrt(sigma_F_obs**2 + sigma_F_model**2)

        sigma_phi_meas = sigma_F_obs / (F_obs + 1e-6)
        sigma_phi_model = sigma_F_model / (F_calc_amp + 1e-6)

        amplitude_discrepancy = torch.abs(F_obs - F_calc_amp)
        excess = torch.clamp(amplitude_discrepancy - sigma_F_total, min=0)
        sigma_phi_excess = excess / (torch.minimum(F_obs, F_calc_amp) + 1e-6)

        sigma_phi = torch.sqrt(
            sigma_phi_meas**2 + sigma_phi_model**2 + sigma_phi_excess**2
        )
        return torch.clamp(sigma_phi, min=0.01, max=2.0)

    def forward(
        self,
        fcalc_light: torch.Tensor = None,
        fcalc_dark: torch.Tensor = None,
        recalc: bool = True,
    ) -> torch.Tensor:
        """
        Compute phase-aware difference loss.

        Jointly refines against dark and light datasets using dark phases
        as reference. Phase uncertainty increases for reflections with
        large amplitude changes between states.

        Parameters
        ----------
        fcalc_light : torch.Tensor, optional
            Pre-computed light state structure factors.
        fcalc_dark : torch.Tensor, optional
            Pre-computed dark state structure factors.
        recalc : bool, optional
            Passed to the model when structure factors are computed
            internally (i.e. when the corresponding ``fcalc_*`` argument is
            None); ignored when pre-computed structure factors are supplied.
            Default is True.

        Returns
        -------
        torch.Tensor
            Combined loss for both datasets.
        """
        hkl = self.hkl
        device = hkl.device

        # Get F_calc for light
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

        # Reference phases from dark state (detached to prevent spurious gradients)
        phi_dark = torch.angle(fcalc_dark).detach()

        # Apply mask
        mask = self._mask
        F_dark_obs = self._F_obs_dark[mask]
        F_light_obs = self._F_obs_light[mask]
        sigma_dark = self._sigma_dark[mask]
        sigma_light = self._sigma_light[mask]
        phi_dark = phi_dark[mask]
        F_calc_dark = fcalc_dark[mask]
        F_calc_light = fcalc_light[mask]

        n_refl = F_dark_obs.shape[0]

        # Compute phase uncertainties for dark dataset
        sigma_F_model_dark = self.sigma_model_log * F_dark_obs
        sigma_F_dark_total = torch.sqrt(sigma_dark**2 + sigma_F_model_dark**2)

        sigma_phi_dark = self._compute_sigma_phi(
            F_dark_obs, sigma_dark, torch.abs(F_calc_dark)
        )

        # Additional uncertainty for light from amplitude change
        delta_F_obs = torch.abs(F_light_obs - F_dark_obs)
        delta_F_calc = torch.abs(torch.abs(F_calc_light) - torch.abs(F_calc_dark))
        delta_discrepancy = torch.abs(delta_F_obs - delta_F_calc)
        sigma_phi_change = delta_discrepancy / (F_light_obs + 1e-6)

        sigma_phi_light_base = self._compute_sigma_phi(
            F_light_obs, sigma_light, torch.abs(F_calc_light)
        )
        sigma_phi_light = torch.sqrt(sigma_phi_light_base**2 + sigma_phi_change**2)
        sigma_phi_light = torch.clamp(sigma_phi_light, min=0.01, max=2.0)

        # Reparameterized sampling with antithetic variates
        n_samples = self.n_samples
        eps = torch.randn(n_refl, n_samples // 2, device=device)
        eps = torch.cat([eps, -eps], dim=1)

        # Dark samples
        phi_samples_dark = phi_dark.unsqueeze(1) + sigma_phi_dark.unsqueeze(1) * eps
        F_dark_samples = F_dark_obs.unsqueeze(1) * torch.exp(1j * phi_samples_dark)

        # Light samples (using dark phases as reference)
        phi_samples_light = phi_dark.unsqueeze(1) + sigma_phi_light.unsqueeze(1) * eps
        F_light_samples = F_light_obs.unsqueeze(1) * torch.exp(1j * phi_samples_light)

        # Losses
        sq_error_dark = torch.abs(F_dark_samples - F_calc_dark.unsqueeze(1)) ** 2
        sq_error_light = torch.abs(F_light_samples - F_calc_light.unsqueeze(1)) ** 2

        expected_error_dark = sq_error_dark.mean(dim=1)
        expected_error_light = sq_error_light.mean(dim=1)

        # Weights
        sigma_F_model_light = self.sigma_model_log * F_light_obs
        sigma_light_total = torch.sqrt(sigma_light**2 + sigma_F_model_light**2)

        weights_dark = 1.0 / (sigma_F_dark_total**2 + 1e-6)
        weights_light = 1.0 / (sigma_light_total**2 + 1e-6)

        loss = (weights_dark * expected_error_dark).mean() + (
            weights_light * expected_error_light
        ).mean()

        # Store diagnostics
        self._last_diagnostics = {
            "sigma_phi_dark": sigma_phi_dark.detach(),
            "sigma_phi_light": sigma_phi_light.detach(),
            "m_dark": torch.exp(-sigma_phi_dark**2 / 2).detach(),
            "m_light": torch.exp(-sigma_phi_light**2 / 2).detach(),
            "delta_F_obs": delta_F_obs.detach(),
            "delta_discrepancy": delta_discrepancy.detach(),
        }

        return loss

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
            Statistics dict with StatEntry values.
        """
        with torch.no_grad():
            loss = self.forward(fcalc_light=fcalc_light, fcalc_dark=fcalc_dark)

        diag = self._last_diagnostics

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(self._mask.sum().item(), VERBOSITY_DEBUG),
            "mean_m_dark": stat(diag["m_dark"].mean().item(), VERBOSITY_STANDARD),
            "mean_m_light": stat(diag["m_light"].mean().item(), VERBOSITY_STANDARD),
            "mean_sigma_phi_dark": stat(
                diag["sigma_phi_dark"].mean().item(), VERBOSITY_DETAILED
            ),
            "mean_sigma_phi_light": stat(
                diag["sigma_phi_light"].mean().item(), VERBOSITY_DETAILED
            ),
            "mean_delta_F_obs": stat(
                diag["delta_F_obs"].mean().item(), VERBOSITY_DETAILED
            ),
            "mean_delta_discrepancy": stat(
                diag["delta_discrepancy"].mean().item(), VERBOSITY_DETAILED
            ),
        }

    def __repr__(self) -> str:
        return (
            f"SampledMLDifferenceTarget(n_samples={self.n_samples}, "
            f"sigma_model={self.sigma_model_log:.3f})"
        )


# =============================================================================
# Factory functions
# =============================================================================


def create_sampled_ml_target(
    data: "ReflectionData" = None,
    model: "Model" = None,
    scaler: "Scaler" = None,
    phi_ref: torch.Tensor = None,
    n_samples: int = 32,
    sigma_model_log: float = 0.15,
    use_analytical: bool = False,
    use_work_set: bool = True,
    verbose: int = 0,
) -> SampledMLPhaseTarget:
    """
    Factory function to create SampledMLPhaseTarget.

    See SampledMLPhaseTarget for parameter documentation.

    Returns
    -------
    SampledMLPhaseTarget
        Configured target instance.
    """
    return SampledMLPhaseTarget(
        data=data,
        model=model,
        scaler=scaler,
        phi_ref=phi_ref,
        n_samples=n_samples,
        sigma_model_log=sigma_model_log,
        use_analytical=use_analytical,
        use_work_set=use_work_set,
        verbose=verbose,
    )


def create_sampled_ml_difference_target(
    dataset_collection: "DatasetCollection",
    model_light: "ModelFT" = None,
    model_dark: "ModelFT" = None,
    scaler_light: "Scaler" = None,
    scaler_dark: "Scaler" = None,
    n_samples: int = 32,
    sigma_model_log: float = 0.15,
    use_work_set: bool = True,
    verbose: int = 0,
) -> SampledMLDifferenceTarget:
    """
    Factory function to create SampledMLDifferenceTarget.

    See SampledMLDifferenceTarget for parameter documentation.

    Returns
    -------
    SampledMLDifferenceTarget
        Configured target instance.
    """
    return SampledMLDifferenceTarget(
        dataset_collection=dataset_collection,
        model_light=model_light,
        model_dark=model_dark,
        scaler_light=scaler_light,
        scaler_dark=scaler_dark,
        n_samples=n_samples,
        sigma_model_log=sigma_model_log,
        use_work_set=use_work_set,
        verbose=verbose,
    )


__all__ = [
    "SampledMLPhaseTarget",
    "SampledMLDifferenceTarget",
    "create_sampled_ml_target",
    "create_sampled_ml_difference_target",
]
