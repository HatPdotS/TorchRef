"""Scaling and post-corrections of calculated structure factors.

The full-featured :class:`Scaler`, which holds a reference to a ``Model`` and
computes ``F_calc`` itself; see :class:`~torchref.scaling.scaler_base.ScalerBase` for
the model-independent version. ``initialize()`` enables the isotropic overall scale,
the anisotropy correction and the solvent model.
"""

from typing import Optional, TYPE_CHECKING

import torch

from torchref.io import ReflectionData
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
    device : torch.device, optional
        Computation device. If ``None``, derived from ``model`` then ``data`` (model
        wins on mismatch); an explicit device moves both. See
        :func:`torchref.utils.device_resolution.resolve_device`.

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
        """See the class docstring."""
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

        self.model = model

    def __setattr__(self, name, value):
        # nn.Module.__setattr__ registers any Module value before a property setter
        # could run, which would put the model's parameters into the scaler's
        # optimizer and state_dict; the scaler only borrows the model.
        if name == "model":
            ref = ModuleReference(value) if value is not None else None
            object.__setattr__(self, "_model_ref", ref)
        else:
            super().__setattr__(name, value)

    @property
    def model(self):
        """The bound ``Model``, held unregistered: absent from ``parameters()`` and
        ``state_dict()``."""
        if self._model_ref is None:
            return None
        return self._model_ref.module

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
        self.model = model
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
        Refine scale parameters with L-BFGS, computing ``fcalc`` from the model.

        See :meth:`torchref.scaling.scaler_base.ScalerBase.refine_lbfgs` for the other
        parameters, the return value and the errors raised.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Complex calculated structure factors, shape (n_reflections,). If ``None``,
            computed from the model.
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
