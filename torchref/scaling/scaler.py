"""
A class for scaling and post corrections of scattering factors.

Currently implements:
- Overall scale per resolution bin
- B-factor per resolution bin
- Anisotropy correction
- Solvent model correction

This module provides the full-featured `Scaler` class that maintains a reference
to a Model object. For a model-independent scaler, see `ScalerBase`.
"""

from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from torchref.io import ReflectionData
from torchref.base.math_torch import (
    bin_wise_rfactors,
    get_rfactors,
    get_scattering_vectors,
    nll_xray,
    nll_xray_lognormal,
)
from torchref.scaling.scaler_base import ScalerBase
from torchref.scaling.solvent_new import SolventModel
from torchref.utils.utils import ModuleReference

if TYPE_CHECKING:
    from torchref.model import Model


class Scaler(ScalerBase):
    """
    Full-featured scaler with Model integration.

    Extends ScalerBase by maintaining a reference to a Model object
    and providing convenience methods that automatically compute F_calc
    when not provided.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        scaler = Scaler()  # Creates empty shell
        scaler.load_state_dict(torch.load('scaler.pt'))

    2. Full initialization with model and data::

        scaler = Scaler(model, reflection_data, nbins=20)
        scaler.initialize()

    Parameters
    ----------
    model : Model, optional
        Model object for structure factor calculation.
    data : ReflectionData, optional
        ReflectionData object with observed data.
    nbins : int, default 20
        Number of resolution bins.
    verbose : int, default 1
        Verbosity level.
    device : torch.device, default torch.device('cpu')
        Computation device.

    Attributes
    ----------
    device : torch.device
        Current computation device.
    nbins : int
        Number of resolution bins.
    frozen : bool
        Whether the scaler parameters are frozen.
    """

    def __init__(
        self,
        model: Optional["Model"] = None,
        data: Optional[ReflectionData] = None,
        nbins: int = 20,
        verbose: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize Scaler.

        If model and data are provided, fully initializes the scaler.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Parameters
        ----------
        model : Model, optional
            Model object for structure factor calculation.
        data : ReflectionData, optional
            ReflectionData object with observed data.
        nbins : int, default 20
            Number of resolution bins.
        verbose : int, default 1
            Verbosity level.
        device : torch.device, default torch.device('cpu')
            Computation device.
        """
        # Initialize base class with data
        super(Scaler, self).__init__(
            data=data,
            nbins=nbins,
            verbose=verbose,
            device=device,
        )

        # Model reference (only in Scaler, not in ScalerBase)
        if model is None:
            self._model = None
        else:
            self._model = ModuleReference(model)

    def set_model_and_data(self, model: "Model", data: ReflectionData):
        """
        Set model and data references after empty initialization.

        This is useful when loading from state_dict and then needing
        to reconnect to model/data objects.

        Parameters
        ----------
        model : Model
            Model object for structure factor calculation.
        data : ReflectionData
            ReflectionData object with observed data.
        """
        self._model = ModuleReference(model)
        # Use parent class method for data
        self.set_data(data)

    def initialize(self, fcalc: torch.Tensor = None):
        """
        Initialize scaling parameters.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        self.calc_initial_scale(fcalc)
        self.setup_solvent()
        self.setup_anisotropy_correction()

    def _compute_fcalc(self) -> torch.Tensor:
        """
        Compute F_calc from internal model.

        Returns
        -------
        torch.Tensor
            Calculated structure factors.

        Raises
        ------
        RuntimeError
            If no model is set.
        """
        if self._model is None:
            raise RuntimeError("No model set and no fcalc provided")
        return self._model(self.hkl)

    def calc_initial_scale(self, fcalc: torch.Tensor = None):
        """
        Calculate initial scale factors.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.

        Returns
        -------
        torch.nn.Parameter
            The log scale parameter for each resolution bin.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().calc_initial_scale(fcalc)

    def fit_anisotropy(self, fcalc: torch.Tensor = None, nsteps: int = 100):
        """
        Fit anisotropic correction.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.
        nsteps : int, default 100
            Number of optimization steps.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().fit_anisotropy(fcalc, nsteps=nsteps)

    def setup_solvent(self):
        """
        Setup solvent model using internal model.

        Creates a SolventModel using the internal model reference.
        """
        if self._model is None:
            raise RuntimeError("Model required for solvent setup")
        self.solvent = SolventModel(
            self._model,
            device=self.device,
            radius=1.1,
            k_solvent=0.35,
            b_solvent=46.0,
            verbose=self.verbose,
        )
        self.solvent.update_solvent()

    def fit_all_scales(self, fcalc: torch.Tensor = None):
        """
        Fit all scale parameters.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().fit_all_scales(fcalc)

    def screen_solvent_params(
        self,
        fcalc: torch.Tensor = None,
        steps: int = 15,
        use_low_res_weighting: bool = True,
        low_res_cutoff: float = 5.0,
        fit_on_low_res_only: bool = True,
        low_res_limit: float = 3.5,
    ):
        """
        Screen solvent parameters using grid search.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.
        steps : int, default 15
            Number of grid points for each parameter.
        use_low_res_weighting : bool, default True
            If True, weight low-resolution reflections more heavily.
        low_res_cutoff : float, default 5.0
            Resolution cutoff for weighting in Angstroms.
        fit_on_low_res_only : bool, default True
            If True, fit using only low-resolution reflections.
        low_res_limit : float, default 3.5
            Resolution limit for low-res only fitting in Angstroms.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().screen_solvent_params(
            fcalc,
            steps=steps,
            use_low_res_weighting=use_low_res_weighting,
            low_res_cutoff=low_res_cutoff,
            fit_on_low_res_only=fit_on_low_res_only,
            low_res_limit=low_res_limit,
        )

    def refine_lbfgs(
        self,
        fcalc: torch.Tensor = None,
        nsteps: int = 3,
        lr: float = 1.0,
        max_iter: int = 200,
        history_size: int = 10,
        verbose: bool = True,
    ):
        """
        Refine scale parameters using LBFGS optimizer.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.
        nsteps : int, default 3
            Number of LBFGS steps.
        lr : float, default 1.0
            Learning rate (typically 1.0 for LBFGS).
        max_iter : int, default 200
            Maximum iterations per line search.
        history_size : int, default 10
            Number of previous gradients to store for Hessian approximation.
        verbose : bool, default True
            Print progress information.

        Returns
        -------
        dict
            Dictionary with refinement metrics.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().refine_lbfgs(
            fcalc,
            nsteps=nsteps,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            verbose=verbose,
        )

    def rfactor(self, fcalc: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate R-factors.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.

        Returns
        -------
        tuple
            R-work and R-free values.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().rfactor(fcalc)

    def bin_wise_rfactor(self, fcalc: torch.Tensor = None):
        """
        Calculate bin-wise R-factors.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.

        Returns
        -------
        mean_res_per_bin : torch.Tensor
            Mean resolution per bin.
        rwork_per_bin : torch.Tensor
            R-work per bin.
        rfree_per_bin : torch.Tensor
            R-free per bin.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().bin_wise_rfactor(fcalc)

    def get_binwise_mean_intensity(self, fcalc: torch.Tensor = None):
        """
        Get bin-wise mean intensities.

        If fcalc is not provided, computes it from the internal model.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Calculated structure factors. If None, computed from model.

        Returns
        -------
        tuple
            Mean observed intensity, mean calculated intensity, and mean resolution per bin.
        """
        if fcalc is None:
            fcalc = self._compute_fcalc()
        return super().get_binwise_mean_intensity(fcalc)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the Scaler.

        This includes:

        - All registered buffers and parameters (via parent class)
        - Scaler-specific metadata (nbins, frozen state, etc.)
        - Solvent model state (if initialized)

        Note: Model and data references are NOT saved (managed separately).

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate.
        prefix : str, default ''
            Prefix for parameter names.
        keep_vars : bool, default False
            Whether to keep variables in computational graph.

        Returns
        -------
        dict
            Complete state dictionary.
        """
        # Use parent class implementation
        return super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Load the Scaler state from a dictionary.

        Note: This assumes model and data are already set via __init__ or assignment.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing scaler state.
        strict : bool, default True
            Whether to strictly enforce that keys match.
        """
        # Extract and load solvent model state if it exists
        solvent_state = state_dict.get("solvent", None)

        # If solvent state exists but module doesn't, instantiate it
        if solvent_state is not None and not hasattr(self, "solvent"):
            # We need to instantiate SolventModel.
            # It requires: model, radius, k_solvent, b_solvent, etc.
            if hasattr(self, "_model") and self._model is not None:
                self.solvent = SolventModel(
                    model=self._model.module, device=self.device, verbose=self.verbose
                )

        # Use parent class implementation (handles removing 'solvent' from state_dict)
        return super().load_state_dict(state_dict, strict=strict)
