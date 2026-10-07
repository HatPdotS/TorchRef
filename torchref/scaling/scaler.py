"""Scaling and post-corrections of calculated structure factors.

The full-featured :class:`Scaler`, which holds a reference to a ``Model`` and
computes ``F_calc`` itself; see :class:`~torchref.scaling.ScalerBase` for the
model-independent version. ``initialize()`` enables the isotropic overall scale,
the anisotropy correction and the solvent model.
"""

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from torchref.io import ReflectionData
from torchref.base.reciprocal import get_scattering_vectors
from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET, ScalerBase
from torchref.scaling.solvent import SolventModel
from torchref.utils.device_resolution import resolve_device
from torchref.utils.utils import ModuleReference

if TYPE_CHECKING:
    from torchref.model import Model


class Scaler(ScalerBase):
    """
    Full-featured scaler with Model integration.

    Extends :class:`~torchref.scaling.scaler_base.ScalerBase` with a reference to a
    ``Model``, so every method that needs ``F_calc`` computes it when not given one.
    Construct with both (``Scaler(model, data, nbins=20)``) and call ``initialize()``
    before use or before :meth:`load_state_dict`; ``Scaler()`` is a shell for a later
    :meth:`set_model_and_data` and ``initialize()``.

    Parameters
    ----------
    model : Model, optional
        Model object for structure factor calculation.
    data : ReflectionData, optional
        ReflectionData object with observed data.
    nbins : int, default 20
        Number of resolution bins used to seed the scale.
    n_iso_coeff : int, default 6
        Number of Chebyshev terms in the isotropic scale.
    verbose : int, default 1
        Verbosity level.
    device : torch.device, default: configured device.current
        Computation device.

    Attributes
    ----------
    device : torch.device
        Current computation device.
    nbins : int
        Number of resolution bins.
    c_iso : torch.nn.Parameter
        Chebyshev coefficients of the isotropic log scale (created during
        ``initialize()``).
    U : torch.nn.Parameter
        Anisotropic scaling parameters (created during ``initialize()``).
    solvent : SolventModel
        Bulk-solvent model (created during ``initialize()``).
    cell, bins, s
        Cell parameters, per-reflection bin indices, and scattering
        vectors set up from the data.
    """

    def __init__(
        self,
        model: Optional["Model"] = None,
        data: Optional[ReflectionData] = None,
        nbins: int = 20,
        n_iso_coeff: int = 6,
        verbose: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Scaler.

        With model and data, binds them and builds the reflection buffers; the scale
        parameters appear only with ``initialize()``. Without them, creates a shell for
        a later :meth:`set_model_and_data`.

        Parameters
        ----------
        model : Model, optional
            Model object for structure factor calculation.
        data : ReflectionData, optional
            ReflectionData object with observed data.
        nbins : int, default 20
            Number of resolution bins used to seed the scale.
        n_iso_coeff : int, default 6
            Number of Chebyshev terms in the isotropic scale.
        verbose : int, default 1
            Verbosity level.
        device : torch.device, optional
            Computation device.  If ``None``, derived from ``model``
            then ``data`` (model wins on mismatch); otherwise forces
            both onto the explicit device.  See
            :func:`torchref.utils.resolve_device`.
        """
        # Pin model+data onto a single device before super().__init__
        # registers buffers from ``data.hkl`` / ``data.cell``.
        resolved_device = resolve_device(model, data, device=device)

        super(Scaler, self).__init__(
            data=data,
            nbins=nbins,
            n_iso_coeff=n_iso_coeff,
            verbose=verbose,
            device=resolved_device,
        )

        # Wrap in ModuleReference to avoid registering the model as a
        # submodule (which would leak its state into the scaler's state_dict).
        self._model_ref = ModuleReference(model) if model is not None else None

    @property
    def model(self):
        """Access the model object (not a registered submodule)."""
        if self._model_ref is None:
            return None
        return self._model_ref.module

    @model.setter
    def model(self, value):
        """Set the model reference, bypassing nn.Module submodule registration."""
        ref = ModuleReference(value) if value is not None else None
        object.__setattr__(self, "_model_ref", ref)

    def set_model_and_data(self, model: "Model", data: ReflectionData):
        """
        Set model and data references after empty initialization.

        Follow with ``initialize()``, which creates the parameters that
        :meth:`load_state_dict` loads into.

        Parameters
        ----------
        model : Model
            Model object for structure factor calculation.
        data : ReflectionData
            ReflectionData object with observed data.

        Notes
        -----
        Receiver wins: the scaler already owns buffers by this point, so
        ``model`` and ``data`` are moved onto *its* device.
        """
        resolve_device(self, model, data)
        # Set _model_ref directly: `self.model = model` would be intercepted by
        # nn.Module.__setattr__ and registered as a submodule.
        self._model_ref = ModuleReference(model) if model is not None else None
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
            fcalc = self.compute_fcalc()
        self.calc_initial_scale(fcalc)
        self.setup_solvent()
        self.setup_anisotropy_correction()
        return self

    def compute_fcalc(self) -> torch.Tensor:
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
        if self.model is None:
            raise RuntimeError("No model set and no fcalc provided")
        # Canonical-ASU convention, with the anomalous (Bijvoet) amplitude
        # difference preserved; bulk solvent below is on the same index.
        return self._data.structure_factors(self.model)

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
            The Chebyshev coefficient parameter ``c_iso``.
        """
        if fcalc is None:
            fcalc = self.compute_fcalc()
        return super().calc_initial_scale(fcalc)

    def setup_solvent(self):
        """
        Setup solvent model using internal model.

        Creates a SolventModel using the internal model reference.
        """
        if self.model is None:
            raise RuntimeError("Model required for solvent setup")
        self.solvent = SolventModel(
            self.model,
            device=self.device,
            radius=1.1,
            k_solvent=0.35,
            verbose=self.verbose,
        )
        self.solvent.update_solvent()
        self._f_sol_raw = None  # Invalidate cached raw solvent SFs

    def refine_lbfgs(
        self,
        fcalc: torch.Tensor = None,
        nsteps: int = 3,
        lr: float = 1.0,
        max_iter: int = 200,
        history_size: int = 10,
        verbose: bool = True,
        scale_target: str = DEFAULT_SCALE_TARGET,
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
        scale_target : {'nll', 'ml_noalpha'}, default 'nll'
            Scale-fit objective, an :data:`XRAY_TARGETS` row; see
            :meth:`torchref.scaling.scaler_base.ScalerBase.refine_lbfgs` for why no
            ``alpha``-centred row is selectable.

        Returns
        -------
        dict
            Dictionary with refinement metrics.
        """
        if fcalc is None:
            fcalc = self.compute_fcalc()
        return super().refine_lbfgs(
            fcalc,
            nsteps=nsteps,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            verbose=verbose,
            scale_target=scale_target,
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Complete state of the Scaler; a pure pass-through to
        :meth:`ScalerBase.state_dict`, which does the serialization
        (buffers/parameters, nbins/verbose metadata, solvent state).
        Model and data references are NOT saved -- they are managed separately.
        """
        return super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Load the Scaler state from a dictionary.

        The scaler must be built with model and data and ``initialize()``-d first, so the
        parameters and the solvent model exist to load into.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing scaler state.
        strict : bool, default True
            Whether to strictly enforce that keys match.
        """
        solvent_state = state_dict.get("solvent", None)

        # A saved solvent state needs a SolventModel to load into.
        if solvent_state is not None and not hasattr(self, "solvent"):
            if hasattr(self, "model") and self.model is not None:
                self.solvent = SolventModel(
                    model=self.model, device=self.device, verbose=self.verbose
                )

        # Parent removes 'solvent' from state_dict before the strict key check.
        return super().load_state_dict(state_dict, strict=strict)
