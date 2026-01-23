"""
Target Functions for Crystallographic Refinement

This module provides target (loss) functions for crystallographic refinement.
Each target is instantiated once with a reference to the refinement object,
then evaluated on each iteration by calling the target.

Target Types:
- X-ray targets: Least Squares, Maximum Likelihood, Gaussian NLL
- Geometry restraint targets: Bonds, Angles, Torsions
- ADP restraint targets: Similarity (SIMU), Rigid Bond (DELU)

LossState Integration:
- Targets can optionally receive a LossState and add their loss to it
"""

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.special import i0

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)
if TYPE_CHECKING:
    from torchref.model.model import Model
    from torchref.model.model_ft import ModelFT
    from torchref.refinement.loss_state import LossState
    from torchref.io.reflection_data import ReflectionData
    from torchref.scaling.scaler_base import Scaler


# =============================================================================
# Base Target Class
# =============================================================================


class Target(nn.Module):
    """
    Abstract base class for all target functions.

    All tunable parameters should be registered as buffers using register_buffer()
    so they can be accessed/modified via state_dict notation.

    Supports empty initialization for state_dict loading::

        target = Target()  # Creates empty shell
        target.load_state_dict(torch.load('target.pt'))

    LossState Integration:
        Targets can work with LossState for the new pipeline::

            state = target.add_to_state(state)  # Adds loss to state

    Parameters
    ----------
    verbose : int, optional
        Verbosity level. Default is 0.
    target_value : float, optional
        Target value for this loss. Default is 0.0.
    sigma : float, optional
        Sigma parameter for weighting. Default is 0.5.

    Attributes
    ----------
    name : str
        Unique name for this target (used as loss key in LossState).
    verbose : int
        Verbosity level.
    _target_value : torch.Tensor (buffer)
        Target value for this loss. Used for computing weights based on deviation.
    _sigma : torch.Tensor (buffer)
        Sigma parameter for weighting. Subclasses should override defaults.
    """

    # Class attribute: unique name for this target type
    # Subclasses should override this
    name: str = "base_target"

    def __init__(
        self,
        verbose: int = 0,
        target_value: float = 0.0,
        sigma: float = 0.5,
    ):
        """
        Initialize target.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Default is 0.
        target_value : float, optional
            Target value for this loss. Default is 0.0.
        sigma : float, optional
            Sigma parameter for weighting. Default is 0.5.
        """
        super().__init__()
        self.verbose = verbose
        # Register target_value and sigma as buffers for state_dict access
        self.register_buffer("_target_value", torch.tensor(target_value))
        self.register_buffer("_sigma", torch.tensor(sigma))

    @property
    def target_value(self) -> float:
        """Get target value as float."""
        return self._target_value.item()

    @target_value.setter
    def target_value(self, value: float):
        """Set target value."""
        self._target_value.fill_(value)

    @property
    def sigma(self) -> float:
        """Get sigma as float."""
        return self._sigma.item()

    @sigma.setter
    def sigma(self, value: float):
        """Set sigma."""
        self._sigma.fill_(value)

    def get_deviation_weight(self, current_value: float = None) -> float:
        """
        Compute weight based on deviation from target value.

        The weight increases with distance from the target value,
        allowing targets that are far from their ideal to contribute
        more to the optimization.

        Parameters
        ----------
        current_value : float, optional
            Current loss value. If None, forward() is called to compute it.

        Returns
        -------
        float
            Weight factor based on |current_value - target_value|.
        """
        if current_value is None:
            with torch.no_grad():
                current_value = self.forward().item()
        return abs(current_value - self.target_value)

    def forward(self) -> torch.Tensor:
        """Compute and return the loss. Override in subclasses."""
        raise NotImplementedError

    def add_to_state(self, state: "LossState") -> "LossState":
        """
        Compute loss and add it to the LossState.

        This method enables the new LossState pipeline pattern where targets
        receive a state object, compute their loss, add it to the state,
        and return the state for chaining.

        Parameters
        ----------
        state : LossState
            Current loss state with computed data.

        Returns
        -------
        LossState
            State with this target's loss added.
        """
        loss = self.forward()
        state.add_loss(self.name, loss)
        return state


# =============================================================================
# Model-Only Target Base Class
# =============================================================================


class ModelTarget(Target):
    """
    Base class for targets that only need a Model reference.

    This class provides a simpler interface for geometry and ADP targets
    that don't need access to reflection data or refinement machinery.
    Targets inherit from this class when they only need the atomic model.

    The model is registered as a proper submodule, allowing PyTorch to
    handle device movement and state_dict operations automatically.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    target_value : float, optional
        Target value for this loss. Default is 0.0.
    sigma : float, optional
        Sigma parameter for weighting. Default is 0.5.

    Attributes
    ----------
    name : str
        Unique name for this target (used as loss key in LossState).
    _model : Model
        Reference to the model object (registered as submodule).
    verbose : int
        Verbosity level.
    """

    name: str = "model_target"

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        target_value: float = 0.0,
        sigma: float = 0.5,
    ):
        """
        Initialize model target.

        Parameters
        ----------
        model : Model, optional
            Reference to the Model object (optional for empty init).
        verbose : int, optional
            Verbosity level. Default is 0.
        target_value : float, optional
            Target value for this loss. Default is 0.0.
        sigma : float, optional
            Sigma parameter for weighting. Default is 0.5.
        """
        super().__init__(verbose=verbose, target_value=target_value, sigma=sigma)
        # Register model as a proper submodule (not in state_dict but handles device)
        # Use add_module to allow None values
        self.add_module("_model", model)

    @property
    def model(self) -> "Model":
        """Access the model object."""
        return self._model

    @property
    def restraints(self):
        """Access model's restraints (built lazily on first access)."""
        if self._model is None:
            return None
        return self._model.restraints


# =============================================================================
# Data Target Base Class (for X-ray targets)
# =============================================================================


class DataTarget(Target):
    """
    Base class for targets that need ReflectionData and optionally Model/Scaler.

    This class provides a flexible interface for X-ray targets that can work
    in two modes:

    1. With Model: Computes F_calc from the model on each forward pass
    2. Without Model: Uses pre-computed F_calc passed directly

    This decoupling allows targets to be used for:
    - Standard refinement (with model)
    - Analysis/scoring of pre-computed structure factors (without model)
    - Testing and validation workflows

    All objects (model, data, scaler) are registered as proper submodules,
    allowing PyTorch to handle device movement and state_dict operations.

    Parameters
    ----------
    data : ReflectionData, optional
        Reference to the ReflectionData object. Required for forward().
    model : Model or ModelFT, optional
        Reference to a Model object for F_calc computation.
        If None, F_calc must be provided to forward().
    scaler : Scaler, optional
        Reference to the Scaler object for scaling F_calc.
    verbose : int, optional
        Verbosity level. Default is 0.
    target_value : float, optional
        Target value for this loss. Default is 0.0.
    sigma : float, optional
        Sigma parameter for weighting. Default is 0.5.

    Attributes
    ----------
    name : str
        Unique name for this target (used as loss key in LossState).
    _model : Model
        Reference to the model object (registered as submodule).
    _data : ReflectionData
        Reference to the reflection data object (registered as submodule).
    _scaler : Scaler
        Reference to the scaler object (registered as submodule).
    verbose : int
        Verbosity level.
    """

    name: str = "data_target"

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        verbose: int = 0,
        target_value: float = 0.0,
        sigma: float = 0.5,
    ):
        """
        Initialize data target.

        Parameters
        ----------
        data : ReflectionData, optional
            Reference to the ReflectionData object. Required for forward().
        model : Model or ModelFT, optional
            Reference to Model object for F_calc computation.
            If None, F_calc must be provided when calling forward().
        scaler : Scaler, optional
            Reference to the Scaler object.
        verbose : int, optional
            Verbosity level. Default is 0.
        target_value : float, optional
            Target value for this loss. Default is 0.0.
        sigma : float, optional
            Sigma parameter for weighting. Default is 0.5.
        """
        super().__init__(verbose=verbose, target_value=target_value, sigma=sigma)
        # Register as proper submodules (allows None values)
        self.add_module("_model", model)
        self._data = data
        self.add_module("_scaler", scaler)

    @property
    def model(self) -> "Model":
        """Access the model object."""
        return self._model

    @property
    def data(self) -> "ReflectionData":
        """Access the reflection data object."""
        return self._data

    @property
    def scaler(self) -> "Scaler":
        """Access the scaler object."""
        return self._scaler

    @property
    def has_model(self) -> bool:
        """Check if a model is available for F_calc computation."""
        return self._model is not None

    def get_fcalc(self, hkl=None, recalc=False):
        """
        Compute structure factors from model.

        Parameters
        ----------
        hkl : torch.Tensor, optional
            Miller indices. If None, uses data's hkl.
        recalc : bool, optional
            Force recalculation. Default is False.

        Returns
        -------
        torch.Tensor
            Complex structure factors.

        Raises
        ------
        RuntimeError
            If no model is set.
        """
        if self._model is None:
            raise RuntimeError(
                "Cannot compute F_calc: no model set. "
                "Either provide a model or pass fcalc directly."
            )
        if hkl is None:
            hkl, _, _, _ = self._data()
        return self._model(hkl, recalc=recalc)

    def get_fcalc_scaled(self, hkl=None, recalc=False, fcalc=None):
        """
        Compute or scale structure factors.

        Parameters
        ----------
        hkl : torch.Tensor, optional
            Miller indices. If None, uses data's hkl.
        recalc : bool, optional
            Force recalculation. Default is False.
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, skips model computation.

        Returns
        -------
        torch.Tensor
            Scaled complex structure factors.
        """
        if fcalc is None:
            fcalc = self.get_fcalc(hkl, recalc=recalc)
        if self._scaler is not None:
            return self._scaler(fcalc)
        return fcalc

    def get_F_calc_scaled(self, hkl=None, recalc=False, fcalc=None):
        """
        Compute scaled structure factor amplitudes.

        Parameters
        ----------
        hkl : torch.Tensor, optional
            Miller indices. If None, uses data's hkl.
        recalc : bool, optional
            Force recalculation. Default is False.
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, skips model computation.

        Returns
        -------
        torch.Tensor
            Scaled structure factor amplitudes |F_calc|.
        """
        return torch.abs(self.get_fcalc_scaled(hkl, recalc=recalc, fcalc=fcalc))

    def get_rfactor(self):
        """
        Compute R-factors using scaler.

        Returns
        -------
        tuple
            (R_work, R_free) values.

        Raises
        ------
        RuntimeError
            If no scaler is set.
        """
        if self._scaler is None:
            raise RuntimeError("Cannot compute R-factor: no scaler set.")
        return self._scaler.rfactor()


# =============================================================================
# X-ray Target Functions
# =============================================================================


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

    def get_data(
        self, fcalc: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get F_obs, F_calc, sigma, and centric flags for the appropriate set.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model. Useful when model is not set.

        Returns
        -------
        tuple
            Tuple of (F_obs, F_calc, sigma_F_obs, centric_flags).

        Raises
        ------
        RuntimeError
            If neither model nor fcalc is available.
        """
        hkl, F_obs, sigma_F_obs, rfree_mask = self._data()

        # Get F_calc: either from argument or compute from model
        if fcalc is not None:
            F_calc = self.get_F_calc_scaled(fcalc=fcalc)
        else:
            F_calc = self.get_F_calc_scaled(hkl, recalc=True)

        # Get centric flags (full size, unfiltered)
        centric_all = self._data.centric

        # Handle MaskedTensor inputs: extract valid data first
        if hasattr(F_obs, "get_mask"):
            # MaskedTensor case: F_obs and sigma have validity mask built-in
            validity_mask = F_obs.get_mask()  # True = valid reflection
            F_obs_data = F_obs.get_data()
            sigma_data = (
                sigma_F_obs.get_data()
                if hasattr(sigma_F_obs, "get_mask")
                else sigma_F_obs
            )

            # Combine validity mask with work/free selection
            if self.use_work_set:
                combined_mask = validity_mask & rfree_mask
            else:
                combined_mask = validity_mask & ~rfree_mask

            F_obs_sel = F_obs_data[combined_mask]
            F_calc_sel = F_calc[combined_mask]
            sigma_sel = sigma_data[combined_mask]
            centric_sel = (
                centric_all[combined_mask] if centric_all is not None else None
            )
        else:
            # Traditional indexed tensor case
            if self.use_work_set:
                mask = rfree_mask
            else:
                mask = ~rfree_mask

            F_obs_sel = F_obs[mask]
            F_calc_sel = F_calc[mask]
            sigma_sel = sigma_F_obs[mask]
            centric_sel = centric_all[mask] if centric_all is not None else None

        return F_obs_sel, F_calc_sel, sigma_sel, centric_sel

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
        F_obs, F_calc, sigma, _ = self.get_data(fcalc=fcalc)
        F_calc_amp = torch.abs(F_calc)
        diff = F_obs - F_calc_amp

        loss = self.forward(fcalc=fcalc)

        rwork, rfree = self.get_rfactor()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(F_obs), VERBOSITY_DEBUG),
            "rwork": stat(rwork, VERBOSITY_STANDARD),
            "rfree": stat(rfree, VERBOSITY_STANDARD),
        }


class GaussianXrayTarget(XrayTarget):
    """
    Simple Gaussian NLL target for X-ray data.

    NLL = 0.5*(F_obs - |F_calc|)²/σ² + log(σ) + 0.5*log(2π)
    """

    target_value: float = 1.0  # Ideal normalized NLL

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Mean NLL loss value.
        """
        F_obs, F_calc, sigma, _ = self.get_data(fcalc=fcalc)

        F_calc_amp = torch.abs(F_calc)
        diff = F_obs - F_calc_amp

        # Avoid division by zero
        eps = torch.median(sigma).item() * 1e-1
        sigma_safe = torch.clamp(sigma, min=eps)

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=sigma.device, dtype=sigma.dtype)
        )
        nll = 0.5 * (diff**2) / (sigma_safe**2) + torch.log(sigma_safe) + 0.5 * log_2pi

        return nll.mean()


class LeastSquaresXrayTarget(XrayTarget):
    """
    Least Squares target function.
    L_LS = Σ w_i * (|F_obs| - k * |F_calc|)²
    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        weighting: str = "sigma",
        use_work_set: bool = True,
        verbose: int = 0,
    ):
        super().__init__(data=data, model=model, scaler=scaler, use_work_set=use_work_set, verbose=verbose)
        self.weighting = weighting

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute least squares loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Mean weighted least squares loss.
        """
        F_obs, F_calc, sigma, _ = self.get_data(fcalc=fcalc)

        F_calc_amp = torch.abs(F_calc)
        diff = F_obs - F_calc_amp

        if self.weighting == "sigma":
            eps = torch.median(sigma).item() * 1e-1
            sigma_safe = torch.clamp(sigma, min=eps)
            weights = 1.0 / (sigma_safe**2)
        elif self.weighting == "unit":
            weights = torch.ones_like(F_obs)
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting}")

        loss = 0.5 * torch.sum(weights * (diff**2))
        return loss / F_obs.numel()


class MaximumLikelihoodXrayTarget(XrayTarget):
    """
    Maximum Likelihood target function with proper centric/acentric handling.
    """

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Compute maximum likelihood loss.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead
            of computing from model.

        Returns
        -------
        torch.Tensor
            Mean ML loss value.
        """
        F_obs, F_calc, sigma, centric_flags = self.get_data(fcalc=fcalc)

        # Default parameters if not available
        alpha = torch.ones_like(F_obs)
        beta = sigma**2
        epsilon = torch.ones_like(F_obs)

        if centric_flags is None:
            centric_flags = torch.zeros_like(F_obs, dtype=torch.bool)

        F_calc_amp = torch.abs(F_calc)

        # Precompute common terms
        eb = epsilon * beta
        eb = torch.clamp(eb, min=1e-6)

        # Acentric term
        term1 = -torch.log(2 * F_obs / eb + 1e-12)
        term2 = (F_obs**2) / eb
        term3 = (alpha * F_calc_amp) ** 2 / eb

        arg_bessel = 2 * alpha * F_obs * F_calc_amp / eb
        term4 = -(
            torch.log(torch.special.i0e(arg_bessel) + 1e-12) + torch.abs(arg_bessel)
        )

        loss_acentric = term1 + term2 + term3 + term4

        # Centric term
        term1_c = -0.5 * torch.log(2 / (np.pi * eb) + 1e-12)
        term2_c = (F_obs**2) / (2 * eb)
        term3_c = (alpha * F_calc_amp) ** 2 / (2 * eb)
        term4_c = -(alpha * F_obs * F_calc_amp) / eb

        arg_exp = -2 * alpha * F_obs * F_calc_amp / eb
        term5_c = -torch.log((1 + torch.exp(arg_exp)) / 2 + 1e-12)

        loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

        # Combine based on centric flags
        loss = torch.where(centric_flags, loss_centric, loss_acentric)

        return loss.mean()


# =============================================================================
# Geometry Restraint Targets
# =============================================================================


class GeometryTarget(ModelTarget):
    """
    Base class for geometry restraint targets.

    Geometry targets access the model's restraints property (built lazily)
    to compute losses for bonds, angles, torsions, planes, etc.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    target_value : float, optional
        Target value for this loss. Default is -1.0.
    sigma : float, optional
        Sigma parameter for weighting. Default is 0.5.
    """

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        target_value: float = -1.0,
        sigma: float = 0.5,
    ):
        super().__init__(model, verbose, target_value=target_value, sigma=sigma)

    def stats(self) -> Dict[str, StatEntry]:
        """
        Get statistics for this restraint type.

        Returns dict with StatEntry values. Filter with filter_stats() at display time.

        Returns
        -------
        dict
            Statistics dict with StatEntry values containing verbosity levels.
        """
        raise NotImplementedError("Subclasses should implement stats()")


class BondTarget(GeometryTarget):
    """
    Bond length restraint target (Gaussian NLL).

    NLL = 0.5 * ((d - d₀) / σ)² + log(σ) + 0.5 * log(2π)
    """

    name: str = "geometry/bond"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=1.0)

    def forward(self) -> torch.Tensor:
        deviations, sigmas = self.restraints.bond_deviations()

        if len(deviations) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=sigmas.device, dtype=sigmas.dtype)
        )
        nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi

        return nll.mean()

    def stats(self) -> Dict[str, StatEntry]:
        """Get bond restraint statistics."""
        deviations, sigmas = self.restraints.bond_deviations()
        if len(deviations) == 0:
            return {}

        z_scores = deviations / sigmas
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(deviations), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((deviations**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigmas.mean().item(), VERBOSITY_DEBUG),
        }


class AngleTarget(GeometryTarget):
    """
    Angle restraint target (Gaussian NLL).

    NLL = 0.5 * ((θ - θ₀) / σ)² + log(σ) + 0.5 * log(2π)
    """

    name: str = "geometry/angle"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=0.5)

    def forward(self) -> torch.Tensor:
        deviations, sigmas = self.restraints.angle_deviations()

        if len(deviations) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=sigmas.device, dtype=sigmas.dtype)
        )
        nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi

        return nll.mean()

    def stats(self) -> Dict[str, StatEntry]:
        """Get angle restraint statistics."""
        deviations_rad, sigmas_rad = self.restraints.angle_deviations()
        if len(deviations_rad) == 0:
            return {}

        # Convert to degrees for reporting
        deviations_deg = deviations_rad * (180.0 / np.pi)
        sigmas_deg = sigmas_rad * (180.0 / np.pi)
        z_scores = deviations_rad / sigmas_rad
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(deviations_rad), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((deviations_deg**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigmas_deg.mean().item(), VERBOSITY_DEBUG),
        }


class TorsionTarget(GeometryTarget):
    """
    Torsion angle restraint target (von Mises NLL).

    NLL = -κ*cos(φ - φ₀) + log(I₀(κ)) + log(2π)
    where κ = 1/σ²
    """

    name: str = "geometry/torsion"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=1.0, sigma=0.3)

    def forward(self) -> torch.Tensor:
        deviations_rad, sigmas_deg = self.restraints.torsion_deviations_with_sigmas()

        if len(deviations_rad) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)

        sigmas_rad = sigmas_deg * (np.pi / 180.0)
        kappa = torch.clamp(1.0 / (sigmas_rad**2), min=1e-3, max=1e4)

        # Compute log(I_0(kappa)) using stable approximation
        log_i0_kappa = torch.zeros_like(kappa)
        small_kappa_mask = kappa < 50.0
        large_kappa_mask = ~small_kappa_mask

        if small_kappa_mask.any():
            log_i0_kappa[small_kappa_mask] = torch.log(i0(kappa[small_kappa_mask]))

        if large_kappa_mask.any():
            kappa_large = kappa[large_kappa_mask]
            log_i0_kappa[large_kappa_mask] = kappa_large - 0.5 * torch.log(
                2.0 * np.pi * kappa_large
            )

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=sigmas_deg.device, dtype=sigmas_deg.dtype)
        )

        # NLL = -log P(theta)
        log_prob = kappa * torch.cos(deviations_rad) - log_i0_kappa - log_2pi

        return (-log_prob).mean()

    def stats(self) -> Dict[str, StatEntry]:
        """Get torsion angle statistics."""
        deviations_rad, sigmas_deg = self.restraints.torsion_deviations_with_sigmas()

        if len(deviations_rad) == 0:
            return {}

        deviations_deg = deviations_rad * (180.0 / np.pi)
        sigmas_rad = sigmas_deg * (np.pi / 180.0)
        z_scores = deviations_rad / sigmas_rad
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(deviations_rad), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((deviations_deg**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigmas_deg.mean().item(), VERBOSITY_DEBUG),
        }


class PlanarityTarget(GeometryTarget):
    """
    Planarity restraint target (Gaussian NLL).

    For each planar group (e.g., aromatic rings, peptide planes), computes the
    distance of each atom from the best-fit plane determined by SVD.

    The best-fit plane is found by:
    1. Computing the centroid of the atoms
    2. Centering the coordinates
    3. Finding the eigenvector with smallest eigenvalue via SVD
    4. This eigenvector is the plane normal

    NLL = 0.5 * (d_i / σ_i)² + log(σ_i) + 0.5 * log(2π)

    where d_i is the distance of atom i from the best-fit plane.

    Reference: cctbx/geometry_restraints/planarity.h
    """

    name: str = "geometry/planarity"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=0.2)

    def forward(self) -> torch.Tensor:
        xyz = self.model.xyz()
        device = xyz.device

        all_nlls = []

        if "plane" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))

        for key, plane_data in self.restraints.restraints["plane"].items():
            indices = plane_data.get("indices")
            sigmas = plane_data.get("sigmas")

            if indices is None or len(indices) == 0:
                continue

            # indices shape: (n_planes, n_atoms_per_plane)
            # sigmas shape: (n_planes, n_atoms_per_plane)

            # Gather all positions at once: (n_planes, n_atoms, 3)
            positions = xyz[indices]

            # Compute centroids: (n_planes, 1, 3)
            centroids = positions.mean(dim=1, keepdim=True)
            centered = positions - centroids  # (n_planes, n_atoms, 3)

            # BATCHED SVD: process all planes in single GPU kernel
            U, S, Vh = torch.linalg.svd(centered)  # Vh: (n_planes, 3, 3)
            normals = Vh[:, -1, :]  # (n_planes, 3) - smallest singular vector

            # Batched deviation calculation
            # deviations[p,a] = |centered[p,a] · normal[p]|
            deviations = torch.abs(torch.einsum("paj,pj->pa", centered, normals))

            # NLL calculation (all vectorized)
            nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
            all_nlls.append(nll.flatten())

        if all_nlls:
            return torch.cat(all_nlls).mean()
        return torch.tensor(0.0, device=device)

    def stats(self) -> Dict[str, any]:
        """Get planarity restraint statistics."""
        xyz = self.model.xyz()
        device = xyz.device

        if "plane" not in self.restraints.restraints:
            return {}

        all_deviations = []
        all_sigmas = []

        for key, plane_data in self.restraints.restraints["plane"].items():
            indices = plane_data.get("indices")
            sigmas = plane_data.get("sigmas")

            if indices is None or len(indices) == 0:
                continue

            # Gather all positions at once: (n_planes, n_atoms, 3)
            positions = xyz[indices]

            # Compute centroids: (n_planes, 1, 3)
            centroids = positions.mean(dim=1, keepdim=True)
            centered = positions - centroids  # (n_planes, n_atoms, 3)

            # BATCHED SVD
            U, S, Vh = torch.linalg.svd(centered)  # Vh: (n_planes, 3, 3)
            normals = Vh[:, -1, :]  # (n_planes, 3)

            # Batched deviation calculation
            deviations = torch.abs(torch.einsum("paj,pj->pa", centered, normals))

            all_deviations.append(deviations.flatten())
            all_sigmas.append(sigmas.flatten())

        if not all_deviations:
            return {"n": 0, "rms_delta": 0.0, "rms_z": 0.0, "mean_sigma": 0.0}

        all_deviations = torch.cat(all_deviations)
        all_sigmas = torch.cat(all_sigmas)
        z_scores = all_deviations / all_sigmas
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(all_deviations), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((all_deviations**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(all_sigmas.mean().item(), VERBOSITY_DEBUG),
        }


class ChiralTarget(GeometryTarget):
    """
    Chiral volume restraint target.

    Restrains the signed volume of tetrahedral chiral centers to maintain
    correct stereochemistry (R vs S configuration, L vs D amino acids).

    The chiral volume is computed as:
        V = v1 · (v2 × v3)

    where vi = position of neighbor i - position of center.

    For standard protein Cα atoms with ordering (N, C, CB):
    - L-amino acids: positive volume (~+2.5 Å³)
    - D-amino acids: negative volume (~-2.5 Å³)

    The loss function penalizes deviations from the ideal signed volume:
        NLL = 0.5 * ((V - V_ideal) / σ)² + log(σ) + 0.5 * log(2π)

    For achiral centers (volume_sign='both'), we restrain the absolute volume.
    """

    name: str = "geometry/chiral"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=-2.0, sigma=0.2)

    def forward(self) -> torch.Tensor:
        xyz = self.model.xyz()
        device = xyz.device

        if "chiral" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        chiral_data = self.restraints.restraints["chiral"]
        indices = chiral_data.get("indices")

        if indices is None or len(indices) == 0:
            return torch.tensor(0.0, device=device)

        ideal_volumes = chiral_data["ideal_volumes"]
        sigmas = chiral_data["sigmas"]

        # Get atom positions: indices is (N, 4) with [center, atom1, atom2, atom3]
        pos_center = xyz[indices[:, 0]]  # (N, 3)
        pos1 = xyz[indices[:, 1]]  # (N, 3)
        pos2 = xyz[indices[:, 2]]  # (N, 3)
        pos3 = xyz[indices[:, 3]]  # (N, 3)

        # Compute vectors from center to neighbors
        v1 = pos1 - pos_center  # (N, 3)
        v2 = pos2 - pos_center  # (N, 3)
        v3 = pos3 - pos_center  # (N, 3)

        # Compute chiral volume: V = v1 · (v2 × v3)
        cross_v2_v3 = torch.cross(v2, v3, dim=-1)  # (N, 3)
        volumes = torch.sum(v1 * cross_v2_v3, dim=-1)  # (N,)

        # Handle achiral centers (ideal_volume = 0) differently
        # For achiral: restrain |V| to typical value (2.5 Å³)
        achiral_mask = ideal_volumes == 0

        if achiral_mask.any():
            # For achiral centers, use absolute volume
            effective_ideal = torch.where(
                achiral_mask,
                torch.full_like(ideal_volumes, 2.5),  # Default magnitude
                ideal_volumes,
            )
            effective_volumes = torch.where(achiral_mask, torch.abs(volumes), volumes)
            deviations = effective_volumes - effective_ideal
        else:
            # All chiral - simple case
            deviations = volumes - ideal_volumes

        # Gaussian NLL
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))
        nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi

        return nll.mean()

    def get_violations(self, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Get information about chiral volume violations.

        Parameters
        ----------
        threshold : float, optional
            Report deviations larger than this (Å³). Default is 0.5.

        Returns
        -------
        dict
            Dictionary with 'indices', 'volumes', 'ideal_volumes', 'deviations'.
        """
        xyz = self.model.xyz()
        device = xyz.device

        if "chiral" not in self.restraints.restraints:
            return {
                "indices": torch.tensor([], dtype=torch.long, device=device).reshape(
                    0, 4
                ),
                "volumes": torch.tensor([], device=device),
                "ideal_volumes": torch.tensor([], device=device),
                "deviations": torch.tensor([], device=device),
            }

        chiral_data = self.restraints.restraints["chiral"]
        indices = chiral_data["indices"]
        ideal_volumes = chiral_data["ideal_volumes"]

        # Compute current volumes
        pos_center = xyz[indices[:, 0]]
        pos1 = xyz[indices[:, 1]]
        pos2 = xyz[indices[:, 2]]
        pos3 = xyz[indices[:, 3]]

        v1 = pos1 - pos_center
        v2 = pos2 - pos_center
        v3 = pos3 - pos_center

        cross_v2_v3 = torch.cross(v2, v3, dim=-1)
        volumes = torch.sum(v1 * cross_v2_v3, dim=-1)

        # Compute deviations (handle achiral)
        achiral_mask = ideal_volumes == 0
        if achiral_mask.any():
            effective_ideal = torch.where(
                achiral_mask, torch.full_like(ideal_volumes, 2.5), ideal_volumes
            )
            effective_volumes = torch.where(achiral_mask, torch.abs(volumes), volumes)
            deviations = torch.abs(effective_volumes - effective_ideal)
        else:
            deviations = torch.abs(volumes - ideal_volumes)

        # Filter by threshold
        mask = deviations > threshold

        return {
            "indices": indices[mask],
            "volumes": volumes[mask],
            "ideal_volumes": ideal_volumes[mask],
            "deviations": deviations[mask],
        }

    def stats(self) -> Dict[str, any]:
        """Get chiral volume statistics."""
        xyz = self.model.xyz()
        device = xyz.device

        if "chiral" not in self.restraints.restraints:
            return {}

        chiral_data = self.restraints.restraints["chiral"]
        indices = chiral_data["indices"]
        ideal_volumes = chiral_data["ideal_volumes"]
        sigmas = chiral_data["sigmas"]

        if len(indices) == 0:
            return {}

        # Compute current volumes
        pos_center = xyz[indices[:, 0]]
        pos1 = xyz[indices[:, 1]]
        pos2 = xyz[indices[:, 2]]
        pos3 = xyz[indices[:, 3]]

        v1 = pos1 - pos_center
        v2 = pos2 - pos_center
        v3 = pos3 - pos_center

        cross_v2_v3 = torch.cross(v2, v3, dim=-1)
        volumes = torch.sum(v1 * cross_v2_v3, dim=-1)

        # Handle achiral
        achiral_mask = ideal_volumes == 0
        if achiral_mask.any():
            effective_ideal = torch.where(
                achiral_mask, torch.full_like(ideal_volumes, 2.5), ideal_volumes
            )
            effective_volumes = torch.where(achiral_mask, torch.abs(volumes), volumes)
            deviations = effective_volumes - effective_ideal
        else:
            deviations = volumes - ideal_volumes

        z_scores = deviations / sigmas
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(indices), VERBOSITY_DEBUG),
            "rms_delta": stat(
                torch.sqrt((deviations**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
            "mean_sigma": stat(sigmas.mean().item(), VERBOSITY_DEBUG),
        }


class NonBondedTarget(GeometryTarget):
    """
    Non-bonded (van der Waals) restraint target using PROLSQ-style repulsion.

    Prevents atoms from clashing by penalizing distances shorter than the sum
    of van der Waals radii. Uses the PROLSQ/CNS repulsion function:

        E_vdw = c_rep * max(0, d_vdw - d)^r_exp

    Default parameters (from Phenix/PROLSQ):

    - c_rep = 16.0 (repulsion coefficient)
    - r_exp = 4.0 (repulsion exponent - makes it very steep near contact)

    This gives: E = 16 * max(0, d_vdw - d)^4

    The quartic (^4) function provides:

    - Zero energy when d >= d_vdw (no overlap)
    - Rapidly increasing energy as atoms approach
    - Smooth gradients for optimization

    Alternative modes:

    - 'prolsq': E = c_rep * max(0, d_vdw - d)^r_exp (default)
    - 'gaussian': Gaussian NLL for violations
    - 'soft': Soft repulsion with linear core

    Reference: cctbx/geometry_restraints/nonbonded.h, PROLSQ documentation

    Parameters
    ----------
    model : Model, optional
        Reference to Model object.
    mode : str, optional
        Repulsion function type ('prolsq', 'gaussian', 'soft'). Default is 'prolsq'.
    c_rep : float, optional
        Repulsion coefficient. Default is 16.0.
    r_exp : float, optional
        Repulsion exponent. Default is 4.0.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "geometry/nonbonded"

    def __init__(
        self,
        model: "Model" = None,
        mode: str = "prolsq",
        c_rep: float = 16.0,
        r_exp: float = 4.0,
        verbose: int = 0,
    ):
        """
        Initialize non-bonded target.

        Parameters
        ----------
        model : Model, optional
            Reference to Model object.
        mode : str, optional
            Repulsion function type ('prolsq', 'gaussian', 'soft'). Default is 'prolsq'.
        c_rep : float, optional
            Repulsion coefficient. Default is 16.0.
        r_exp : float, optional
            Repulsion exponent. Default is 4.0.
        verbose : int, optional
            Verbosity level. Default is 0.
        """
        super().__init__(model, verbose, target_value=0.5, sigma=1.2)
        self.mode = mode
        # Register c_rep and r_exp as buffers for state_dict access
        self.register_buffer("_c_rep", torch.tensor(c_rep))
        self.register_buffer("_r_exp", torch.tensor(r_exp))

    @property
    def c_rep(self) -> float:
        """Get repulsion coefficient."""
        return self._c_rep.item()

    @c_rep.setter
    def c_rep(self, value: float):
        """Set repulsion coefficient."""
        self._c_rep.fill_(value)

    @property
    def r_exp(self) -> float:
        """Get repulsion exponent."""
        return self._r_exp.item()

    @r_exp.setter
    def r_exp(self, value: float):
        """Set repulsion exponent."""
        self._r_exp.fill_(value)

    def forward(self) -> torch.Tensor:
        xyz = self.model.xyz()
        device = xyz.device

        if "vdw" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data.get("indices")

        if indices is None or len(indices) == 0:
            return torch.tensor(0.0, device=device)

        min_distances = vdw_data["min_distances"]  # Sum of VDW radii
        sigmas = vdw_data["sigmas"]

        # Get current positions
        pos1 = xyz[indices[:, 0]]
        pos2 = xyz[indices[:, 1]]

        # Compute actual distances
        actual_distances = torch.norm(pos2 - pos1, dim=-1)

        # Violations: where actual distance is less than VDW sum
        violations = torch.clamp(min_distances - actual_distances, min=0.0)

        if self.mode == "prolsq":
            # PROLSQ repulsion: E = c_rep * violation^r_exp
            # This is the standard crystallographic repulsion function
            energy = self.c_rep * (violations**self.r_exp)
            return energy.mean()

        elif self.mode == "gaussian":
            # Gaussian NLL for violations
            log_2pi = torch.log(
                torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype)
            )
            nll = 0.5 * (violations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
            return nll.mean()

        elif self.mode == "soft":
            # Soft repulsion with linear core to prevent gradient explosion
            # Quadratic near boundary, linear for severe clashes
            threshold = 0.5  # Å - switch to linear below this

            # Quadratic region
            quadratic_mask = violations <= threshold
            quadratic_energy = self.c_rep * (violations**2)

            # Linear region (for severe clashes)
            linear_mask = ~quadratic_mask
            linear_energy = self.c_rep * (2 * threshold * violations - threshold**2)

            energy = torch.where(quadratic_mask, quadratic_energy, linear_energy)
            return energy.mean()

        else:
            raise ValueError(f"Unknown non-bonded mode: {self.mode}")

    def get_violations(self, threshold: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Get information about VDW violations.

        Parameters
        ----------
        threshold : float, optional
            Only report violations greater than this (Å). Default is 0.0.

        Returns
        -------
        dict
            Dictionary with 'indices', 'violations', 'distances', 'min_distances'.
        """
        xyz = self.model.xyz()
        device = xyz.device

        if "vdw" not in self.restraints.restraints:
            return {
                "indices": torch.tensor([], dtype=torch.long, device=device).reshape(
                    0, 2
                ),
                "violations": torch.tensor([], device=device),
                "distances": torch.tensor([], device=device),
                "min_distances": torch.tensor([], device=device),
            }

        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data["indices"]
        min_distances = vdw_data["min_distances"]

        pos1 = xyz[indices[:, 0]]
        pos2 = xyz[indices[:, 1]]
        actual_distances = torch.norm(pos2 - pos1, dim=-1)
        violations = torch.clamp(min_distances - actual_distances, min=0.0)

        # Filter by threshold
        mask = violations > threshold

        return {
            "indices": indices[mask],
            "violations": violations[mask],
            "distances": actual_distances[mask],
            "min_distances": min_distances[mask],
        }

    def stats(self) -> Dict[str, any]:
        """Get non-bonded restraint statistics."""
        xyz = self.model.xyz()
        device = xyz.device

        if "vdw" not in self.restraints.restraints:
            return {}

        vdw_data = self.restraints.restraints["vdw"]
        indices = vdw_data.get("indices")

        if indices is None or len(indices) == 0:
            return {}

        min_distances = vdw_data["min_distances"]
        sigmas = vdw_data["sigmas"]

        pos1 = xyz[indices[:, 0]]
        pos2 = xyz[indices[:, 1]]
        actual_distances = torch.norm(pos2 - pos1, dim=-1)

        # Violations: where actual distance < VDW sum
        violations = torch.clamp(min_distances - actual_distances, min=0.0)
        n_violations = (violations > 0).sum().item()

        # RMS of violations only (for those that clash)
        if n_violations > 0:
            violation_mask = violations > 0
            rms_violation = torch.sqrt((violations[violation_mask] ** 2).mean()).item()
            max_violation = violations.max().item()
        else:
            rms_violation = 0.0
            max_violation = 0.0

        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n": stat(len(indices), VERBOSITY_DEBUG),
            "n_violations": stat(n_violations, VERBOSITY_DETAILED),
            "rms_violation": stat(rms_violation, VERBOSITY_DETAILED),
            "max_violation": stat(max_violation, VERBOSITY_DEBUG),
            "mean_sigma": stat(sigmas.mean().item(), VERBOSITY_DEBUG),
        }


class ADPTarget(ModelTarget):
    """
    Base class for ADP restraint targets.

    ADP targets access the model's ADP values and restraints for similarity,
    rigid bond, and other ADP-related restraints.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    target_value : float, optional
        Target value for this loss. Default is -1.0.
    sigma : float, optional
        Sigma parameter for weighting. Default is 1.0.
    """

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        target_value: float = -1.0,
        sigma: float = 1.0,
    ):
        super().__init__(model, verbose, target_value=target_value, sigma=sigma)


class ADPSimilarityTarget(ADPTarget):
    """
    ADP Similarity restraint (SIMU in Phenix/SHELX).

    Restrains B-factors of bonded atoms to be similar.
    NLL = 0.5 * ((B_i - B_j) / σ)² + log(σ) + 0.5 * log(2π)

    Tunable parameters (as buffers):
    - _simu_sigma: float, sigma for B-factor differences (default 2.0 Å²)
    """

    name: str = "adp/simu"

    def __init__(
        self, model: "Model" = None, simu_sigma: float = 2.0, verbose: int = 0
    ):
        super().__init__(model, verbose, target_value=4.0, sigma=1.2)
        # Register simu-specific sigma as buffer (separate from base sigma)
        self.register_buffer("_simu_sigma", torch.tensor(simu_sigma))

    @property
    def simu_sigma(self) -> float:
        """Get SIMU sigma value."""
        return self._simu_sigma.item()

    @simu_sigma.setter
    def simu_sigma(self, value: float):
        """Set SIMU sigma value."""
        self._simu_sigma.fill_(value)

    def forward(self) -> torch.Tensor:
        b_diffs = self.restraints.adp_b_differences()

        if len(b_diffs) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)

        log_2pi = torch.log(
            torch.tensor(2.0 * np.pi, device=b_diffs.device, dtype=b_diffs.dtype)
        )
        nll = (
            0.5 * (b_diffs / self.simu_sigma) ** 2
            + np.log(self.simu_sigma)
            + 0.5 * log_2pi
        )

        return nll.mean()

    def stats(self) -> Dict[str, any]:
        """Get SIMU restraint statistics."""
        b_diffs = self.restraints.adp_b_differences()

        if len(b_diffs) == 0:
            return {}

        b_diffs_abs = b_diffs.abs()
        z_scores = b_diffs_abs / self.simu_sigma
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "count": stat(len(b_diffs), VERBOSITY_DEBUG),
            "rms_delta_b": stat(
                torch.sqrt((b_diffs**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "mean_delta_b": stat(b_diffs_abs.mean().item(), VERBOSITY_DETAILED),
            "max_delta_b": stat(b_diffs_abs.max().item(), VERBOSITY_DETAILED),
            "mean_z": stat(z_scores.mean().item(), VERBOSITY_DEBUG),
            "rms_z": stat(torch.sqrt((z_scores**2).mean()).item(), VERBOSITY_DETAILED),
        }


class RigidBondTarget(ADPTarget):
    """
    Rigid Bond restraint (DELU in SHELX, Hirshfeld test).

    Based on Hirshfeld's rigid bond test (Acta Cryst. A32, 239, 1976).

    For a truly rigid bond, the mean-square displacement amplitudes (MSDA)
    of the two bonded atoms along the bond direction should be equal.
    This is because in a rigid bond, the atoms move together.

    For anisotropic ADPs (U tensors)::

        z_12 = l_12^T U_1 l_12 / |l_12|²  (MSDA of atom 1 along bond)
        z_21 = l_21^T U_2 l_21 / |l_21|²  (MSDA of atom 2 along bond)
        Δz = z_12 - z_21 should be ~0

    For isotropic B-factors, the difference in B_iso is used as a proxy::

        ΔB = B_1 - B_2

    This differs from SIMU (ADPSimilarityTarget) which restrains the full
    ADP tensors to be similar. Rigid bond only restrains the component
    along the bond direction.

    Energy: E = w * Δz²
    NLL: NLL = 0.5 * (Δz / σ)² + log(σ) + 0.5 * log(2π)

    References
    ----------
    - Hirshfeld, F.L. (1976). Acta Cryst. A32, 239.
    - cctbx/adp_restraints/rigid_bond.h

    Parameters
    ----------
    model : Model
        Reference to Model object.
    sigma : float, optional
        Target standard deviation for Δz. Default is 0.004 Å².
        Hirshfeld found typical values of 0.001 Å² for good structures.
    use_aniso : bool, optional
        If True and model has anisotropic ADPs, use proper tensor calculation.
        Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "adp/delu"

    def __init__(
        self,
        model: "Model" = None,
        sigma: float = 0.004,
        use_aniso: bool = True,
        verbose: int = 0,
    ):
        super().__init__(model, verbose)
        self.sigma = sigma
        self.use_aniso = use_aniso

    def forward(self) -> torch.Tensor:
        """
        Compute rigid bond restraint.

        For isotropic refinement, uses B-factor differences along bonds.
        For anisotropic refinement, computes proper MSDA differences.
        """
        device = self.model.xyz().device

        # Check if model has anisotropic ADPs
        has_aniso = hasattr(self.model, "u_aniso") and self.model.u_aniso is not None

        if has_aniso and self.use_aniso:
            return self._compute_aniso_rigid_bond()
        else:
            return self._compute_iso_rigid_bond()

    def _compute_iso_rigid_bond(self) -> torch.Tensor:
        """
        Compute rigid bond restraint for isotropic B-factors.

        For isotropic ADPs, the MSDA along any direction is B/(8π²).
        So the difference in MSDA is proportional to ΔB.

        We use ΔB directly and scale sigma accordingly.
        """
        adp = self.model.adp()
        xyz = self.model.xyz()
        device = xyz.device

        delta_z_list = []

        if "bond" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        for origin, restraint_group in self.restraints.restraints["bond"].items():
            if origin == "all":
                continue
            indices = restraint_group.get("indices")
            if indices is not None and len(indices) > 0:
                adp1 = adp[indices[:, 0]]
                adp2 = adp[indices[:, 1]]

                # For isotropic ADPs: U_iso = B / (8π²)
                # MSDA along bond = U_iso (same in all directions)
                # Δz = (B1 - B2) / (8π²)
                delta_z = (adp1 - adp2) / (8.0 * np.pi**2)
                delta_z_list.append(delta_z)

        if not delta_z_list:
            return torch.tensor(0.0, device=device)

        delta_z = torch.cat(delta_z_list, dim=0)

        # Gaussian NLL
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))
        nll = 0.5 * (delta_z / self.sigma) ** 2 + np.log(self.sigma) + 0.5 * log_2pi

        return nll.mean()

    def _compute_aniso_rigid_bond(self) -> torch.Tensor:
        """
        Compute rigid bond restraint for anisotropic ADPs.

        For each bond:
            l = (r2 - r1) / |r2 - r1|  (unit vector along bond)
            z_12 = l^T U_1 l  (MSDA of atom 1 along bond direction)
            z_21 = l^T U_2 l  (MSDA of atom 2 along bond direction)
            Δz = z_12 - z_21

        The U tensor is symmetric 3x3, stored as 6 unique values:
            U = [[U11, U12, U13],
                 [U12, U22, U23],
                 [U13, U23, U33]]
        """
        xyz = self.model.xyz()
        device = xyz.device

        # Get anisotropic U tensors (N, 6) -> (N, 3, 3)
        u_aniso = self.model.u_aniso  # Shape: (N, 6) for U11,U22,U33,U12,U13,U23

        # Convert to full symmetric matrices
        n_atoms = u_aniso.shape[0]
        U = torch.zeros(n_atoms, 3, 3, device=device, dtype=xyz.dtype)
        U[:, 0, 0] = u_aniso[:, 0]  # U11
        U[:, 1, 1] = u_aniso[:, 1]  # U22
        U[:, 2, 2] = u_aniso[:, 2]  # U33
        U[:, 0, 1] = u_aniso[:, 3]  # U12
        U[:, 1, 0] = u_aniso[:, 3]  # U12 (symmetric)
        U[:, 0, 2] = u_aniso[:, 4]  # U13
        U[:, 2, 0] = u_aniso[:, 4]  # U13 (symmetric)
        U[:, 1, 2] = u_aniso[:, 5]  # U23
        U[:, 2, 1] = u_aniso[:, 5]  # U23 (symmetric)

        delta_z_list = []

        if "bond" not in self.restraints.restraints:
            return torch.tensor(0.0, device=device)

        for origin, restraint_group in self.restraints.restraints["bond"].items():
            if origin == "all":
                continue
            indices = restraint_group.get("indices")
            if indices is not None and len(indices) > 0:
                idx1 = indices[:, 0]
                idx2 = indices[:, 1]

                # Get positions and compute bond vectors
                r1 = xyz[idx1]  # (n_bonds, 3)
                r2 = xyz[idx2]  # (n_bonds, 3)
                bond_vec = r2 - r1  # (n_bonds, 3)
                bond_length = torch.norm(bond_vec, dim=-1, keepdim=True)
                l = bond_vec / bond_length  # Unit vector along bond

                # Get U tensors for bonded atoms
                U1 = U[idx1]  # (n_bonds, 3, 3)
                U2 = U[idx2]  # (n_bonds, 3, 3)

                # Compute MSDA along bond direction: z = l^T U l
                # For batch: z = sum_ij l_i U_ij l_j
                # Using einsum: z = einsum('bi,bij,bj->b', l, U, l)
                z_12 = torch.einsum("bi,bij,bj->b", l, U1, l)
                z_21 = torch.einsum("bi,bij,bj->b", l, U2, l)

                delta_z = z_12 - z_21
                delta_z_list.append(delta_z)

        if not delta_z_list:
            return torch.tensor(0.0, device=device)

        delta_z = torch.cat(delta_z_list, dim=0)

        # Gaussian NLL
        log_2pi = torch.log(torch.tensor(2.0 * np.pi, device=device, dtype=xyz.dtype))
        nll = 0.5 * (delta_z / self.sigma) ** 2 + np.log(self.sigma) + 0.5 * log_2pi

        return nll.mean()

    def get_delta_z_stats(self) -> Dict[str, float]:
        """
        Get statistics of Δz values for analysis.

        Returns
        -------
        dict
            Dictionary with mean, std, max, min of |Δz| values and Z-scores.
        """
        adp = self.model.adp()
        xyz = self.model.xyz()
        device = xyz.device

        delta_z_list = []

        if "bond" not in self.restraints.restraints:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "rms": 0.0,
                "mean_z": 0.0,
                "rms_z": 0.0,
            }

        for origin, restraint_group in self.restraints.restraints["bond"].items():
            if origin == "all":
                continue
            indices = restraint_group.get("indices")
            if indices is not None and len(indices) > 0:
                adp1 = adp[indices[:, 0]]
                adp2 = adp[indices[:, 1]]
                delta_z = (adp1 - adp2) / (8.0 * np.pi**2)
                delta_z_list.append(delta_z)

        if not delta_z_list:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "rms": 0.0,
                "mean_z": 0.0,
                "rms_z": 0.0,
            }

        delta_z_all = torch.cat(delta_z_list, dim=0)
        delta_z_abs = delta_z_all.abs()

        # Z-scores (deviation / sigma)
        z_scores = delta_z_abs / self.sigma

        return {
            "count": len(delta_z_all),
            "mean": delta_z_abs.mean().item(),
            "std": delta_z_all.std().item(),
            "max": delta_z_abs.max().item(),
            "min": delta_z_abs.min().item(),
            "rms": torch.sqrt((delta_z_all**2).mean()).item(),
            "mean_z": z_scores.mean().item(),
            "rms_z": torch.sqrt((z_scores**2).mean()).item(),
        }

    def stats(self) -> Dict[str, any]:
        """
        Get rigid bond restraint statistics.

        Returns statistics including Δz values along bonds.
        """
        delta_z_stats = self.get_delta_z_stats()
        if delta_z_stats.get("count", 0) == 0:
            return {}

        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "count": stat(delta_z_stats["count"], VERBOSITY_DEBUG),
            "rms": stat(delta_z_stats["rms"], VERBOSITY_DETAILED),
            "mean": stat(delta_z_stats["mean"], VERBOSITY_DETAILED),
            "max": stat(delta_z_stats["max"], VERBOSITY_DETAILED),
            "rms_z": stat(delta_z_stats["rms_z"], VERBOSITY_DETAILED),
            "std": stat(delta_z_stats["std"], VERBOSITY_DEBUG),
            "min": stat(delta_z_stats["min"], VERBOSITY_DEBUG),
            "mean_z": stat(delta_z_stats["mean_z"], VERBOSITY_DEBUG),
        }


class ADPEntropyTarget(ADPTarget):
    """
    ADP Entropy regularization target.

    Uses the model's existing adp_kl_divergence_loss or similar.
    """

    name: str = "adp/KL"

    def __init__(self, model: "Model" = None, verbose: int = 0):
        super().__init__(model, verbose, target_value=0.5, sigma=0.5)

    def forward(self) -> torch.Tensor:
        return self.model.adp_kl_divergence_loss()

    def stats(self) -> Dict[str, any]:
        """Get KL divergence statistics."""
        adp = self.model.adp().detach()
        log_adp = torch.log(adp.clamp(min=1e-3))
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n_atoms": stat(len(adp), VERBOSITY_DEBUG),
            "mean_adp": stat(adp.mean().item(), VERBOSITY_DETAILED),
            "std_adp": stat(adp.std().item(), VERBOSITY_DETAILED),
            "min_adp": stat(adp.min().item(), VERBOSITY_DETAILED),
            "max_adp": stat(adp.max().item(), VERBOSITY_DETAILED),
            "mean_log_adp": stat(log_adp.mean().item(), VERBOSITY_DEBUG),
            "std_log_adp": stat(log_adp.std().item(), VERBOSITY_DEBUG),
        }


class ADPLocalityTarget(ADPTarget):
    """
    Proximity-based ADP restraint using K nearest neighbors.

    ADPs of nearby atoms should be similar because:

    1. Core residues (buried) tend to have low ADP
    2. Surface/loop residues tend to have high ADP
    3. Disorder propagates through space

    This restraint finds the K nearest neighbors for each atom and computes
    a weighted MSE on log(B) differences. The weight decays exponentially
    with distance, based on a correlation length parameter::

        w_ij = exp(-d_ij / xi)

    Where xi (correlation length) controls how quickly the restraint weakens
    with distance. Typical values: 4-8 Å for proteins.

    The loss is weighted MSE in log-space::

        loss = scale * mean_ij [w_ij * (log(B_i) - log(B_j))^2]

    This is simpler than NLL - no fake probabilistic sigma. The exponential
    decay has a clear physical interpretation: atoms within correlation
    length xi should have similar B-factors.

    Tunable parameters (as buffers):
    - _k_neighbors: int, number of nearest neighbors
    - _correlation_length: float, distance scale for weight decay (Å)
    - _scale: float, scaling factor for loss magnitude

    Parameters
    ----------
    model : Model
        Reference to Model object.
    k_neighbors : int, optional
        Number of nearest neighbors to consider. Default is 50.
    correlation_length : float, optional
        Distance scale for weight decay in Angstrom. Default is 5.0.
    scale : float, optional
        Scaling factor for loss magnitude. Default is 5.0.
    exclude_bonded : bool, optional
        Exclude directly bonded atoms. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "adp/locality"

    def __init__(
        self,
        model: "Model" = None,
        k_neighbors: int = 50,
        correlation_length: float = 5.0,  # xi in Angstrom
        scale: float = 5.0,  # Scale factor for loss magnitude (reduced from 10.0)
        exclude_bonded: bool = True,
        verbose: int = 0,
    ):
        super().__init__(model, verbose, target_value=0.3, sigma=0.2)
        # Register tunable parameters as buffers
        self.register_buffer(
            "_k_neighbors", torch.tensor(k_neighbors, dtype=torch.int64)
        )
        self.register_buffer("_correlation_length", torch.tensor(correlation_length))
        self.register_buffer("_scale", torch.tensor(scale))
        self.exclude_bonded = exclude_bonded

        # Cache for neighbor indices and distances
        self._neighbor_indices = None  # (N, k_neighbors)
        self._neighbor_distances = None  # (N, k_neighbors)
        self._last_xyz_hash = None

    @property
    def k_neighbors(self) -> int:
        """Get k_neighbors value."""
        return self._k_neighbors.item()

    @k_neighbors.setter
    def k_neighbors(self, value: int):
        """Set k_neighbors value."""
        self._k_neighbors.fill_(value)

    @property
    def correlation_length(self) -> float:
        """Get correlation_length value."""
        return self._correlation_length.item()

    @correlation_length.setter
    def correlation_length(self, value: float):
        """Set correlation_length value."""
        self._correlation_length.fill_(value)

    @property
    def scale(self) -> float:
        """Get scale value."""
        return self._scale.item()

    @scale.setter
    def scale(self, value: float):
        """Set scale value."""
        self._scale.fill_(value)

    def _build_neighbor_list(self) -> None:
        """
        Build list of K nearest neighbors for each atom.

        Stores:
            _neighbor_indices: (N, k_neighbors) indices of neighbors
            _neighbor_distances: (N, k_neighbors) distances to neighbors
        """
        xyz = self.model.xyz()
        device = xyz.device
        n_atoms = xyz.shape[0]

        # Compute all pairwise distances (O(N^2) but simple and fast for proteins)
        k = min(self.k_neighbors, n_atoms - 1)

        # Compute distance matrix
        diff = xyz.unsqueeze(0) - xyz.unsqueeze(1)  # (N, N, 3)
        dist_matrix = torch.sqrt((diff**2).sum(dim=-1)).detach()  # (N, N)

        # Create mask for diagonal (self-distances) and bonded pairs
        # Use non-inplace operations to avoid gradient issues
        mask = torch.eye(n_atoms, device=device, dtype=torch.bool)

        # Apply mask using torch.where (non-inplace)
        dist_matrix = torch.where(
            mask, torch.tensor(float("inf"), device=device), dist_matrix
        )

        # Get k nearest neighbors
        distances, indices = torch.topk(dist_matrix, k, dim=1, largest=False)

        self._neighbor_indices = indices  # (N, k)
        self._neighbor_distances = distances  # (N, k)

        if self.verbose > 1:
            mean_dist = distances.mean().item()
            min_dist = distances.min().item()
            max_dist = distances[:, -1].mean().item()  # Farthest of k neighbors
            print(
                f"    Built K-NN list: k={k}, mean dist={mean_dist:.2f}A, "
                f"min={min_dist:.2f}A, max (kth)={max_dist:.2f}A"
            )

    def forward(self, recompute_neighbors: bool = False) -> torch.Tensor:
        """
        Compute weighted MSE on log(B) differences with exponential decay.

        loss = scale * mean_ij [w_ij * (log(B_i) - log(B_j))^2]
        where w_ij = exp(-d_ij / correlation_length)
        """
        # Check if cached tensors are on wrong device and need rebuilding
        model_device = self.model.xyz().device
        cache_stale = (
            self._neighbor_indices is not None
            and self._neighbor_indices.device != model_device
        )
        if recompute_neighbors or self._neighbor_indices is None or cache_stale:
            self._build_neighbor_list()

        adp = self.model.adp()
        device = adp.device
        n_atoms = len(adp)

        if n_atoms == 0 or self._neighbor_indices is None:
            return torch.tensor(0.0, device=device)

        log_adp = torch.log(adp.clamp(min=1e-3))

        indices = self._neighbor_indices  # (N, k)
        distances = self._neighbor_distances  # (N, k)

        # Gather neighbor log(ADP) values
        neighbor_log_adp = log_adp[indices]  # (N, k)

        # Compute pairwise differences: diff_ij = log(ADP_i) - log(ADP_j)
        diff = log_adp.unsqueeze(1) - neighbor_log_adp  # (N, k)

        weights = 1 / distances + 1e-6  # Avoid div by zero

        weights = weights / (weights.mean() + 1e-8)  # Normalize weights per atom

        # Weighted MSE
        weighted_sq_diff = weights * (diff / 0.5) ** 2  # (N, k)

        # Normalize by sum of weights to get weighted average
        loss = weighted_sq_diff.mean()

        return loss

    def stats(self) -> Dict[str, any]:
        """Get locality restraint statistics."""
        self._build_neighbor_list()

        if self._neighbor_indices is None:
            return {}

        adp = self.model.adp().detach()
        log_adp = torch.log(adp.clamp(min=1e-3))

        indices = self._neighbor_indices
        distances = self._neighbor_distances

        neighbor_log_adp = log_adp[indices]
        diff = log_adp.unsqueeze(1) - neighbor_log_adp

        # Exponential weights
        weights = torch.exp(-distances / self.correlation_length)

        # Weighted RMS
        weighted_sq_diff = weights * (diff**2)
        weighted_rms = torch.sqrt(weighted_sq_diff.sum() / weights.sum()).item()
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n_atoms": stat(len(adp), VERBOSITY_DEBUG),
            "weighted_rms_log": stat(weighted_rms, VERBOSITY_DETAILED),
            "rms_deviation_log": stat(
                torch.sqrt((diff**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "max_deviation_log": stat(diff.abs().max().item(), VERBOSITY_DETAILED),
            "k_neighbors": stat(self.k_neighbors, VERBOSITY_DEBUG),
            "correlation_length": stat(self.correlation_length, VERBOSITY_DEBUG),
            "scale": stat(self.scale, VERBOSITY_DEBUG),
            "avg_neighbor_dist": stat(distances.mean().item(), VERBOSITY_DEBUG),
            "max_neighbor_dist": stat(distances.max().item(), VERBOSITY_DEBUG),
            "avg_weight": stat(weights.mean().item(), VERBOSITY_DEBUG),
        }


def create_xray_target(
    data: "ReflectionData" = None,
    model: "Model" = None,
    scaler: "Scaler" = None,
    mode: str = "gaussian",
    use_work_set: bool = True,
    verbose: int = 0,
) -> XrayTarget:
    """
    Factory function to create X-ray target.

    Parameters
    ----------
    data : ReflectionData
        Reference to ReflectionData object. Required for forward().
    model : Model or ModelFT, optional
        Reference to Model object for F_calc computation.
        If None, fcalc must be provided when calling forward().
    scaler : Scaler, optional
        Reference to Scaler object.
    mode : str, optional
        Target mode: 'gaussian', 'ls', or 'ml'. Default is 'gaussian'.
    use_work_set : bool, optional
        Use work set (True) or test set (False). Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    XrayTarget
        Appropriate XrayTarget instance.
    """
    if mode == "gaussian":
        return GaussianXrayTarget(
            data=data, model=model, scaler=scaler, use_work_set=use_work_set, verbose=verbose
        )
    elif mode == "ls":
        return LeastSquaresXrayTarget(
            data=data, model=model, scaler=scaler, use_work_set=use_work_set, verbose=verbose
        )
    elif mode == "ml":
        return MaximumLikelihoodXrayTarget(
            data=data, model=model, scaler=scaler, use_work_set=use_work_set, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown X-ray target mode: {mode}")


# =============================================================================
# Utility Functions for NLL Computation
# =============================================================================


def gaussian_nll(deviations: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Compute Gaussian negative log-likelihood.

    NLL = 0.5 * ((x - μ) / σ)² + log(σ) + 0.5 * log(2π)

    Parameters
    ----------
    deviations : torch.Tensor
        Deviations from target values (x - μ).
    sigmas : torch.Tensor
        Standard deviations.

    Returns
    -------
    torch.Tensor
        Tensor of NLL values (same shape as input).
    """
    log_2pi = torch.log(
        torch.tensor(2.0 * np.pi, device=sigmas.device, dtype=sigmas.dtype)
    )
    nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
    return nll


def von_mises_nll(
    deviations_rad: torch.Tensor, sigmas_deg: torch.Tensor
) -> torch.Tensor:
    """
    Compute von Mises negative log-likelihood for angular data.

    NLL = -κ*cos(θ) + log(I₀(κ)) + log(2π)
    where κ = 1/σ²

    Parameters
    ----------
    deviations_rad : torch.Tensor
        Angular deviations in radians.
    sigmas_deg : torch.Tensor
        Standard deviations in degrees.

    Returns
    -------
    torch.Tensor
        Tensor of NLL values (same shape as input).
    """
    sigmas_rad = sigmas_deg * (np.pi / 180.0)
    kappa = torch.clamp(1.0 / (sigmas_rad**2), min=1e-3, max=1e4)

    log_i0_kappa = torch.zeros_like(kappa)
    small_kappa_mask = kappa < 50.0
    large_kappa_mask = ~small_kappa_mask

    if small_kappa_mask.any():
        log_i0_kappa[small_kappa_mask] = torch.log(i0(kappa[small_kappa_mask]))

    if large_kappa_mask.any():
        kappa_large = kappa[large_kappa_mask]
        log_i0_kappa[large_kappa_mask] = kappa_large - 0.5 * torch.log(
            2.0 * np.pi * kappa_large
        )

    log_2pi = torch.log(
        torch.tensor(2.0 * np.pi, device=sigmas_deg.device, dtype=sigmas_deg.dtype)
    )
    log_prob = kappa * torch.cos(deviations_rad) - log_i0_kappa - log_2pi

    return -log_prob


def adp_similarity_nll(adp_diffs: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """
    Compute ADP similarity NLL (SIMU restraint).

    Parameters
    ----------
    adp_diffs : torch.Tensor
        ADP differences between bonded atoms.
    sigma : float, optional
        Target standard deviation. Default is 2.0 Å².

    Returns
    -------
    torch.Tensor
        Tensor of NLL values (same shape as input).
    """
    log_2pi = torch.log(
        torch.tensor(2.0 * np.pi, device=adp_diffs.device, dtype=adp_diffs.dtype)
    )
    nll = 0.5 * (adp_diffs / sigma) ** 2 + np.log(sigma) + 0.5 * log_2pi
    return nll
