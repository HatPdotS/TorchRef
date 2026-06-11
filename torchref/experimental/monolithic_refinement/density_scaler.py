"""Density-derived bulk solvent integrated into the Scaler (EXPERIMENTAL).

This wires the differentiable :class:`DensitySolventModel` into the production
:class:`~torchref.scaling.scaler.Scaler` by subclassing it, so a refinement can
select it through ``Refinement._scaler_class`` with no edits to core scaling code.

Two pieces:

``DensityDerivedSolvent``
    A thin ``nn.Module`` exposing the exact interface ``ScalerBase.forward``
    expects from its ``solvent`` attribute (``get_rec_solvent`` / ``log_k_solvent``
    / ``b_solvent`` / ``optimize_phase``), backed by a ``DensitySolventModel``.
    Unlike the vdW-mask ``SolventModel`` (whose mask FFT is detached / static),
    ``get_rec_solvent`` here is **live** -- it stays in the autograd graph, so
    ``F_sol`` tracks atom moves and gradients flow ``rho -> xyz/adp``. The raw
    solvent SF carries only the density *shape* (``rho_s`` frozen at 1); the
    scaler's ``k_sol`` is the refinable contrast and ``b_solvent`` the residual
    damping -- identical scaling machinery to the mask.

``DensitySolventScaler``
    ``Scaler`` subclass that installs ``DensityDerivedSolvent`` in
    ``setup_solvent`` and recomputes the solvent contribution live on every
    ``forward`` (the base caches ``_f_sol_raw`` across LBFGS line-search trial
    steps, which would both freeze the density to its first-call atoms and raise
    "backward through the graph a second time"). Same constructor signature as
    ``Scaler`` plus solvent-shape keyword defaults, so it is a drop-in.

Why ``rho0`` is frozen by default: the scaler caches nothing now, so ``rho0``
*could* be refined -- but a soft/Babinet-shaped solvent is rejected by the ML
target (k_sol -> 0), and the only accepted regime is the sharp mask. Freezing
``rho0`` keeps the refinement stable and adds no solvent-shape DOF beyond the
mask's own ``k_sol`` + ``b_solvent``.
"""

import torch
import torch.nn as nn

from torchref.config import get_default_device, get_float_dtype
from torchref.scaling.scaler import Scaler
from torchref.experimental.monolithic_refinement.density_solvent import (
    DensitySolventModel,
)


class DensityDerivedSolvent(nn.Module):
    """SolventModel-compatible, differentiable density-derived bulk solvent.

    Parameters
    ----------
    model : ModelFT
        Atomic model providing atoms / cell / spacegroup.
    rho0 : float, default 0.016
        Density-saturation level (sharp mask limit; the regime the ML target
        accepts). Frozen unless ``refine_rho0``.
    solvent_res : float, default 4.0
        Coarse solvent-grid resolution (A).
    occupancy : {"exp", "sigmoid"}, default "exp"
        Occupancy mapping passed to :class:`DensitySolventModel`.
    refine_rho0 : bool, default False
        If True, leave ``rho0`` refinable (co-refined by the host optimizer).
    k_solvent, b_solvent : float
        Initial contrast / residual B (init at the mask defaults so the scaler
        starts in the same basin and avoids the B=0 local-min trap).
    """

    def __init__(
        self,
        model,
        rho0=0.016,
        solvent_res=4.0,
        occupancy="exp",
        refine_rho0=False,
        k_solvent=0.35,
        b_solvent=46.0,
        device=None,
        dtype=None,
        verbose=0,
    ):
        super().__init__()
        device = device or get_default_device()
        dtype = dtype or get_float_dtype()
        self.optimize_phase = False
        # rho_s frozen at 1: the scaler's k_sol is the refinable contrast, so the
        # raw solvent SF returned here is pure density shape (ifft(M)).
        self.density = DensitySolventModel(
            model,
            rho_s=1.0,
            rho0=rho0,
            occupancy=occupancy,
            solvent_res=solvent_res,
            verbose=verbose,
            device=device,
            float_type=dtype,
        )
        self.density.log_rho_s.requires_grad_(False)
        if not refine_rho0:
            self.density.log_rho0.requires_grad_(False)
        self.log_k_solvent = nn.Parameter(
            torch.log(torch.tensor(k_solvent, dtype=dtype, device=device))
        )
        self.b_solvent = nn.Parameter(
            torch.tensor(b_solvent, dtype=dtype, device=device)
        )

    def get_rec_solvent(self, hkl):
        """Raw (contrast-free) solvent SF, LIVE in the autograd graph.

        Not detached: ``F_sol`` follows the moving atoms so gradients reach
        ``xyz``/``adp``. The scaler applies ``k_sol * exp(-B_sol s^2)`` on top.
        """
        return self.density(hkl.to(torch.long))

    def update_solvent(self):
        """No-op: the density mask is rebuilt live on every scaler forward."""
        return None


class DensitySolventScaler(Scaler):
    """:class:`Scaler` using a density-derived bulk solvent instead of the vdW mask.

    Drop-in for ``Scaler`` (same positional/keyword signature; the solvent-shape
    kwargs are keyword-only with defaults), so it can be selected via
    ``Refinement._scaler_class``.
    """

    def __init__(
        self,
        model=None,
        data=None,
        nbins: int = 20,
        verbose: int = 1,
        device=None,
        *,
        rho0: float = 0.016,
        solvent_res: float = 4.0,
        occupancy: str = "exp",
        refine_rho0: bool = False,
    ):
        # Plain dict assignment before super().__init__ is safe (mirrors
        # MonolithicRefinement setting _sigma_m_calib_bins pre-init).
        self._density_solvent_kwargs = dict(
            rho0=rho0,
            solvent_res=solvent_res,
            occupancy=occupancy,
            refine_rho0=refine_rho0,
        )
        super().__init__(
            model=model, data=data, nbins=nbins, verbose=verbose, device=device
        )

    def setup_solvent(self):
        if self.model is None:
            raise RuntimeError("Model required for solvent setup")
        self.solvent = DensityDerivedSolvent(
            self.model,
            device=self.device,
            verbose=self.verbose,
            **self._density_solvent_kwargs,
        )
        self._f_sol_raw = None  # Invalidate; rebuilt live each forward.

    def forward(self, fcalc, *args, **kwargs):
        # Bust the raw-solvent cache so the density mask is rebuilt live (in
        # graph) every call: it tracks atom moves and avoids the stale-graph
        # "backward through the graph a second time" the base cache would cause
        # under LBFGS line search.
        if isinstance(getattr(self, "solvent", None), DensityDerivedSolvent):
            self._f_sol_raw = None
        return super().forward(fcalc, *args, **kwargs)
