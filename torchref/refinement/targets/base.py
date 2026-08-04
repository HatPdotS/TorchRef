"""Base classes for the crystallographic refinement target (loss) functions.

A target is constructed once against the objects it scores, then called each
iteration -- directly, or via :meth:`Target.add_to_state` so its loss lands in a
:class:`~torchref.refinement.loss_state.LossState`. :class:`ModelTarget` adds a
``Model`` reference (geometry, ADP restraints); :class:`DataTarget` adds
``ReflectionData`` and an optional ``Scaler`` (X-ray targets). Also home to the
shared NLL primitives :func:`gaussian_nll`, :func:`von_mises_nll` and
:func:`adp_similarity_nll`.
"""

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.special import i0

from torchref.config import get_float_dtype, normalize_device
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)
if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model import Model
    from torchref.model.model_ft import ModelFT
    from torchref.refinement.loss_state import LossState
    from torchref.scaling.scaler_base import Scaler


# =============================================================================
# Base Target Class
# =============================================================================


class Target(DeviceMixin, nn.Module):
    """Abstract base class for all target functions.

    Register tunables as buffers (see :meth:`_register_scalar`) so they are
    reachable by state_dict path. ``Target()`` with no arguments is a valid empty
    shell for ``load_state_dict``.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level. Default is 0.

    Attributes
    ----------
    name : str
        Unique name for this target; the loss key in ``LossState``. Subclasses
        must override it.
    """

    name: str = "base_target"

    def __init__(
        self,
        verbose: int = 0,
        device=None,
        **kwargs,
    ):
        """
        Initialize target.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level. Default is 0.
        device : torch.device, optional
            Where this target allocates. Defaults to the configured default;
            :meth:`_adopt_device` refines it from whatever model / data /
            scaler the subclass is given.
        """
        super().__init__()
        self.verbose = verbose
        # Do not remove or defer these two assignments. Subclasses allocate
        # tunables in their own ``__init__``, so ``self.device`` /
        # ``self.dtype_float`` must already answer by then; and
        # ``DeviceMixin._refresh_device_trackers`` early-returns unless one of
        # these names is already in ``obj.__dict__``, so merely defining them is
        # what switches the mixin on for every target.
        self.device = normalize_device(device)
        self.dtype_float = get_float_dtype()

    # ---- device / dtype resolution --------------------------------------

    def _adopt_device(self, *sources, device=None):
        """Reconcile this target with the device-bearing objects it wraps.

        Call *after* attaching the sources and *before* allocating any buffer.
        ``None`` sources drop out, so empty-init keeps the seeded default and
        this stays re-runnable. ``self`` is deliberately not a source: an empty
        shell's default device must not outvote a model already on the
        accelerator. Returns (and stores on ``self.device``) the resolved device.
        """
        present = [s for s in sources if s is not None]
        if device is None and not present:
            return self.device
        self.device = resolve_device(*present, device=device)
        for src in present:
            src_dtype = getattr(src, "dtype_float", None)
            if isinstance(src_dtype, torch.dtype):
                self.dtype_float = src_dtype
                break
        return self.device

    def _register_scalar(self, name: str, value, dtype=None) -> None:
        """Register a 0-dim tunable on this target's device and float dtype.

        Placement is load-bearing: these reach Triton kernels as tensors, where
        a CPU-resident buffer is a host pointer, not a promotable scalar.
        """
        self.register_buffer(
            name,
            torch.tensor(
                value,
                device=self.device,
                dtype=self.dtype_float if dtype is None else dtype,
            ),
        )

    def forward(self) -> torch.Tensor:
        """Compute and return the loss. Override in subclasses."""
        raise NotImplementedError

    def add_to_state(self, state: "LossState") -> "LossState":
        """
        Compute this target's loss and add it to ``state`` under :attr:`name`.

        Parameters
        ----------
        state : LossState
            Current loss state with computed data.

        Returns
        -------
        LossState
            The same state, returned for chaining.
        """
        loss = self.forward()
        state.add_loss(self.name, loss)
        return state

    def maintenance(self) -> None:
        """Between-step housekeeping hook (no-op by default).

        ``LossState`` calls this on every registered target after each
        successful outer optimizer step; override it to rebuild internal state
        that parameter drift has staled (VDW pair lists, solvent masks).

        Overrides must be idempotent -- a second call on an unchanged model must
        not mutate the target -- and must put a cheap staleness check ahead of
        the rebuild, since the happy path is paid every outer step. They must
        also not raise on routine drift; if a rebuild genuinely fails, let the
        exception propagate rather than swallowing it -- that is a real bug.
        """
        pass


# =============================================================================
# Model-Only Target Base Class
# =============================================================================


class ModelTarget(Target):
    """Base class for targets that need only a Model reference.

    For geometry and ADP restraints, which never touch reflection data. The
    model is a real submodule, so device moves and state_dict follow it.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object. Omit for an empty shell.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "model_target"

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        device=None,
        **kwargs,
    ):
        """
        Initialize model target.

        Parameters
        ----------
        model : Model, optional
            Reference to the Model object (optional for empty init).
        verbose : int, optional
            Verbosity level. Default is 0.
        device : torch.device, optional
            Explicit device. When omitted, follows ``model``.
        """
        super().__init__(verbose=verbose, device=device)
        # add_module, not an attribute assignment: it accepts None.
        self.add_module("_model", model)
        # Follow the model rather than the global default, so subclass buffers
        # allocated below this line land beside the coordinates they act on.
        self._adopt_device(model, device=device)

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
    """Base class for X-ray targets: ReflectionData plus optional Model/Scaler.

    Works with a model (F_calc recomputed each forward pass) or without one
    (pre-computed F_calc passed in), which is what lets the same target both
    refine and score. Model and scaler are real submodules, so device moves and
    state_dict follow them.

    Parameters
    ----------
    data : ReflectionData, optional
        Reference to the ReflectionData object. Required for ``forward()``.
    model : Model or ModelFT, optional
        Model used for F_calc. If None, F_calc must be passed to ``forward()``.
    scaler : Scaler, optional
        Scaler applied to F_calc.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "data_target"

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        verbose: int = 0,
        device=None,
        **kwargs,
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
        device : torch.device, optional
            Explicit device. When omitted, model / data / scaler are
            reconciled onto one device, the model winning on disagreement.
        """
        super().__init__(verbose=verbose, device=device)
        # ``_data`` is a plain attribute, not a submodule: ReflectionData is a
        # dataclass, so add_module would reject it. DeviceMixin still walks it.
        self.add_module("_model", model)
        self._data = data
        self.add_module("_scaler", scaler)
        # The forward path mixes tensors from all three, so pin them together
        # here. Order is precedence: ``resolve_device`` is first-wins, and the
        # model is what the rest of the pipeline follows.
        self._adopt_device(model, data, scaler, device=device)

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
            # Signed HKL so Bijvoet mates get distinct |F_calc| (see
            # ReflectionData.hkl_for_sf).
            hkl = self._data.hkl_for_sf()
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

    For numerical stability, ``log(I₀(κ))`` is evaluated directly via ``i0``
    only for κ < 50; for κ ≥ 50 it uses the large-argument asymptotic
    ``log I₀(κ) ≈ κ − 0.5*log(2πκ)`` rather than a literal Bessel call.

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


def detach_phases(fcalc: torch.Tensor) -> torch.Tensor:
    """
    Extract phases from complex structure factors with gradient detachment.

    Not exported in ``targets/__init__.__all__``; treat its public-vs-private
    status as unresolved.

    Parameters
    ----------
    fcalc : torch.Tensor
        Complex structure factors.

    Returns
    -------
    torch.Tensor
        Detached phase angles in radians.
    """
    return torch.angle(fcalc).detach()
