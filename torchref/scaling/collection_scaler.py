"""
Joint scaler for multi-dataset / multi-model kinetic refinement.

``CollectionScaler`` extends ``ScalerBase`` to operate on paired
``DatasetCollection`` + ``ModelCollection`` instances.  A single set of
scale parameters (log_scale, U, k_sol, B_sol, phase) is shared across
**all** data–model pairs, preventing artificial scale differences that
corrupt the difference signal in time-resolved refinement.

Per-component solvent models are created for each base model in the
``ModelCollection``.  For a mixed model at any timepoint the solvent
contribution is the linear combination of individual component solvent
structure factors, weighted by the same population fractions as the
structural models.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn as nn

from torchref.base.metrics import get_rfactors, nll_xray
from torchref.base.reciprocal import get_scattering_vectors
from torchref.base.targets.xray_ml_sigmaa import (
    SigmaAEstimator,
    epsilon_from_hkl,
    ml_xray_loss_beta_math,
)
from torchref.config import get_default_device, get_float_dtype
from torchref.scaling.scaler_base import ScalerBase
from torchref.scaling.solvent import SolventModel
from torchref.utils.utils import ModuleReference

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection


class CollectionScaler(ScalerBase):
    """
    Joint scaler for DatasetCollection + ModelCollection.

    Shares scale parameters (log_scale, U, bin_wise_bfactor, k_sol,
    B_sol, phase) across **all** data–model pairs.  Manages per-component
    solvent models so that the bulk-solvent contribution for a mixed model
    is the fraction-weighted sum of individual component solvent SFs.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Collection of reflection datasets keyed by timepoint name.
    model_collection : ModelCollection
        Collection of mixed models keyed by timepoint name.
    nbins : int
        Number of resolution bins.
    verbose : int
        Verbosity level.
    device : torch.device
        Computation device.

    Examples
    --------
    ::

        scaler = CollectionScaler(datasets, models, device=device)
        scaler.initialize()
        scaler.refine_lbfgs_joint()

        # In a target: scale a mixed-model F_calc with matching solvent
        f_scaled = scaler.forward_mixed(f_calc, model.fractions)
    """

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        nbins: int = 20,
        verbose: int = 1,
        device: torch.device = None,
    ):
        if device is None:
            device = get_default_device()
        # Bind to the dark/reference dataset for bins and scattering vectors
        dark_data = dataset_collection[model_collection.dark_key]
        super().__init__(
            data=dark_data,
            nbins=nbins,
            verbose=verbose,
            device=device,
        )

        self._dataset_collection = dataset_collection
        self._model_collection = model_collection

        # Per-component solvent models (one per base model)
        self._component_solvent_models: nn.ModuleList = nn.ModuleList()

        # Cached raw solvent SFs per component index
        self._f_sol_raw_components: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> "CollectionScaler":
        """
        One-shot initialization: joint initial scale, component solvents,
        anisotropy correction.

        Returns
        -------
        CollectionScaler
            Self, for method chaining.
        """
        self._calc_initial_scale_joint()
        self._setup_component_solvent_models()
        self.setup_anisotropy_correction()
        return self

    def _calc_initial_scale_joint(self):
        """
        Compute initial bin-wise log-scale using ALL data–model pairs.

        Averages log(F_obs / |F_calc|) per resolution bin across every
        matched timepoint in the collections.
        """
        dc = self._dataset_collection
        mc = self._model_collection

        # Use the configured float dtype so the scatter_add_ against
        # fobs-derived log_ratios does not raise under a float64 config.
        scales = torch.zeros(self.nbins, device=self.device, dtype=get_float_dtype())
        counts = torch.zeros(self.nbins, device=self.device, dtype=get_float_dtype())

        all_keys = [mc.dark_key] + mc.timepoint_names
        n_pairs = 0

        for name in all_keys:
            if name not in dc:
                continue

            data = dc[name]
            model = mc[name]

            hkl, fobs, sigma, rfree = data(mask=False)
            with torch.no_grad():
                fcalc = model(hkl)
            fcalc_amp = torch.abs(fcalc).clamp(min=1e-3).to(fobs.dtype)
            fobs_clamped = fobs.clamp(min=1e-3)

            # Mask: work set, positive intensities
            if hasattr(data, "I") and data.I is not None:
                pos_mask = data.I > 0
            else:
                pos_mask = torch.ones_like(fobs, dtype=torch.bool)
            mask = (data.masks() & rfree & pos_mask).to(torch.bool)

            bins = self.bins[mask].to(torch.int64)
            log_ratios = (
                torch.log(fobs_clamped[mask]) - torch.log(fcalc_amp[mask])
            ).to(self.device)
            bins = bins.to(self.device)

            scales.scatter_add_(0, bins, log_ratios)
            ones = torch.ones_like(log_ratios)
            counts.scatter_add_(0, bins, ones)
            n_pairs += 1

        log_scale = scales / (counts + 1e-6)
        self.log_scale = nn.Parameter(log_scale.detach())

        if self.verbose > 0:
            print(
                f"Joint initial scale from {n_pairs} data-model pairs "
                f"({self.nbins} bins)."
            )

    # ------------------------------------------------------------------
    # Component solvent models
    # ------------------------------------------------------------------

    def _setup_component_solvent_models(self):
        """
        Create a SolventModel for each base model in the ModelCollection.

        The **first** component's SolventModel is also stored as
        ``self.solvent`` — it owns the shared k_sol / B_sol / phase
        parameters that are optimised during scaling.  The remaining
        component models are used only for their raw solvent structure
        factors (mask FFT); their own k/B/phase parameters are frozen.
        """
        mc = self._model_collection
        self._component_solvent_models = nn.ModuleList()
        self._f_sol_raw_components = {}

        for i, base_model in enumerate(mc.base_models):
            sol = SolventModel(
                base_model,
                device=self.device,
                radius=1.1,
                k_solvent=0.35,
                b_solvent=46.0,
                verbose=max(0, self.verbose - 1),
            )
            sol.update_solvent()
            self._component_solvent_models.append(sol)

            if i == 0:
                # Primary solvent model — owns the shared learnable params
                self.solvent = sol
            else:
                # Freeze non-primary solvent params (only mask FFT used)
                for p in sol.parameters():
                    p.requires_grad = False

        # Invalidate any old cache
        self._f_sol_raw = None
        self._f_sol_raw_components = {}

        if self.verbose > 0:
            print(
                f"  Created {len(self._component_solvent_models)} component "
                f"solvent models."
            )

    def _get_component_f_sol_raw(self, idx: int) -> torch.Tensor:
        """
        Get (and cache) raw solvent SFs for component *idx*.

        Parameters
        ----------
        idx : int
            Index into ``_component_solvent_models``.

        Returns
        -------
        torch.Tensor
            Raw (un-damped) complex solvent structure factors.
        """
        if idx not in self._f_sol_raw_components:
            sol = self._component_solvent_models[idx]
            self._f_sol_raw_components[idx] = sol.get_rec_solvent(self.hkl)
        return self._f_sol_raw_components[idx]

    def get_mixed_solvent_raw(self, fractions: torch.Tensor) -> torch.Tensor:
        """
        Compute fraction-weighted raw solvent SFs.

        ``f_sol_mixed = sum_i(w_i * f_sol_raw_i)``

        Parameters
        ----------
        fractions : torch.Tensor
            Population fractions, shape ``(n_base_models,)``.

        Returns
        -------
        torch.Tensor
            Mixed raw solvent structure factors (complex, un-damped).
        """
        f_sol_mixed = None
        for i in range(len(self._component_solvent_models)):
            f_raw = self._get_component_f_sol_raw(i)
            contribution = fractions[i] * f_raw
            f_sol_mixed = (
                contribution if f_sol_mixed is None else f_sol_mixed + contribution
            )
        return f_sol_mixed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_mixed(
        self,
        fcalc: torch.Tensor,
        fractions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scale *fcalc* using the shared parameters **and** a fraction-
        weighted solvent contribution.

        This sets ``_f_sol_raw`` to the mixed solvent and then delegates
        to ``ScalerBase.forward()``, which applies k_sol / B_sol /
        phase damping and the overall + anisotropic scale.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        fractions : torch.Tensor
            Population fractions for the mixed model, shape
            ``(n_base_models,)``.

        Returns
        -------
        torch.Tensor
            Scaled structure factors.
        """
        f_sol_raw_mixed = self.get_mixed_solvent_raw(fractions)
        return super().forward(fcalc, f_sol_override=f_sol_raw_mixed)

    # ------------------------------------------------------------------
    # Joint LBFGS refinement
    # ------------------------------------------------------------------

    def refine_lbfgs_joint(
        self,
        nsteps: int = 3,
        lr: float = 1.0,
        max_iter: int = 200,
        history_size: int = 10,
        scale_smoothness: float = 1000.0,
        verbose: bool = True,
    ) -> dict:
        """
        Refine scale parameters using LBFGS against **all** datasets.

        The closure sums the NLL across every matched dataset–model pair,
        so a single set of scale parameters is fitted jointly.

        Parameters
        ----------
        nsteps : int
            Number of LBFGS outer steps.
        lr : float
            Learning rate (typically 1.0 for LBFGS).
        max_iter : int
            Maximum line-search iterations per step.
        history_size : int
            LBFGS history size.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Refinement metrics (steps, rwork, rfree of dark dataset).
        """
        from torchref.refinement.loss_state import LossState

        dc = self._dataset_collection
        mc = self._model_collection
        all_keys = [mc.dark_key] + mc.timepoint_names

        # Pre-compute all fcalc (detached) plus the per-dataset σ_A model-error
        # variance (beta/epsilon). beta is estimated ONCE on each dataset's free
        # set from the currently-scaled |F_calc| and held constant (detached)
        # during the scale optimisation — exactly as the single-dataset
        # ``ScalerBase.refine_lbfgs`` does. The σ_A (Read MLF) likelihood is what
        # keeps the per-bin log_scale from collapsing toward zero in weak shells;
        # the plain σ-weighted Gaussian ``nll_xray`` used previously drove the
        # scale down (model under-scaled, R blow-up).
        fcalc_cache = {}
        fractions_cache = {}
        data_cache = {}
        beta_cache = {}
        eps_cache = {}
        work_cache = {}
        centric_cache = {}
        for name in all_keys:
            if name not in dc:
                continue
            data = dc[name]
            model = mc[name]
            hkl = data.hkl
            fobs, sigma = data.get_corrected_data()
            rfree = data.rfree_flags
            with torch.no_grad():
                fc = model(hkl).detach()
                fracs = model.fractions.detach()
                f_sol_raw = self.get_mixed_solvent_raw(fracs)
                scaled0 = super(CollectionScaler, self).forward(
                    fc, f_sol_override=f_sol_raw
                )
                fc_amp0 = torch.abs(scaled0).reshape(-1)
                fobs0 = fobs.to(fc_amp0.dtype).reshape(-1)
                eps0 = epsilon_from_hkl(
                    hkl, getattr(data, "spacegroup", None)
                ).to(fc_amp0.dtype)
                s = get_scattering_vectors(hkl, data.cell)
                dss0 = (torch.norm(s, dim=1) ** 2).to(fc_amp0.dtype)
                beta0, eps0 = SigmaAEstimator().get(
                    fobs0, fc_amp0, data.centric, eps0, dss0, data.free.mask
                )
            fcalc_cache[name] = fc
            fractions_cache[name] = fracs
            data_cache[name] = (fobs, sigma, rfree)
            beta_cache[name] = beta0
            eps_cache[name] = eps0
            work_cache[name] = data.work
            centric_cache[name] = data.centric

        # Wrap the joint σ_A ML loss + U-penalty as a LossState target. fcalc is
        # detached, so the only leaves in the autograd graph are the scaler's own
        # parameters — LossState's probe picks them up at registration time, and
        # validate_loss inside state.step handles NaN/Inf rejection so no
        # per-target try/except is needed.
        scaler_self = self

        class _CollectionScalerJointTarget(nn.Module):
            name = "scaler/joint"

            def forward(self):
                total = torch.tensor(0.0, device=scaler_self.device)
                n = 0
                for nm in all_keys:
                    if nm not in fcalc_cache:
                        continue
                    fc = fcalc_cache[nm]
                    fracs = fractions_cache[nm]
                    f_sol_raw = scaler_self.get_mixed_solvent_raw(fracs)
                    scaled = super(CollectionScaler, scaler_self).forward(
                        fc, f_sol_override=f_sol_raw
                    )
                    # σ_A (Read MLF) scale-fit on the WORK set, with detached
                    # free-set beta/epsilon — same likelihood the body
                    # refinement uses.
                    amp = torch.abs(scaled).reshape(-1)
                    work = work_cache[nm]
                    F_obs = work.F.to(amp.dtype)
                    Fc = work.select(amp)
                    beta_w = work.select(beta_cache[nm]).to(F_obs.dtype)
                    eps_w = (
                        work.select(eps_cache[nm]).to(F_obs.dtype)
                        if eps_cache[nm] is not None
                        else None
                    )
                    centric_w = work.select(centric_cache[nm])
                    loss = ml_xray_loss_beta_math(
                        F_obs, Fc, beta_w, centric_w, epsilon=eps_w
                    )
                    if torch.isfinite(loss):
                        total = total + loss
                        n += 1
                if n > 0:
                    total = total / n
                u_penalty = torch.sum(scaler_self.U**2)
                # Smoothness (Tikhonov) penalty on the per-bin log_scale: the
                # lowest-resolution shells are solvent-dominated with large
                # model error, so the σ_A likelihood is nearly flat there and a
                # free per-bin scale runs away to -inf (model amplitudes -> 0,
                # R blow-up). Penalising squared first differences ties each bin
                # to its neighbours so under-constrained shells follow the
                # well-determined ones instead of collapsing.
                ls = scaler_self.log_scale
                smooth_penalty = scale_smoothness * torch.sum(
                    (ls[1:] - ls[:-1]) ** 2
                )
                return total + u_penalty + smooth_penalty

        state = LossState(device=self.device)
        state.register_target("scaler/joint", _CollectionScalerJointTarget())

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

        metrics = {
            "target": "scales_joint",
            "steps": [],
            "rwork": [],
            "rfree": [],
        }

        if verbose and self.verbose > 0:
            print("Refining scales jointly with LBFGS...")

        state.run(
            optimizer,
            nsteps=nsteps,
            log=False,
            context="collection_scaler.refine_lbfgs_joint",
        )

        # Evaluate metrics once on the dark dataset after refinement.
        with torch.no_grad():
            dark_name = mc.dark_key
            if dark_name in fcalc_cache:
                fc = fcalc_cache[dark_name]
                fracs = fractions_cache[dark_name]
                fobs, sigma, rfree = data_cache[dark_name]
                f_sol_raw = self.get_mixed_solvent_raw(fracs)
                scaled = super(CollectionScaler, self).forward(
                    fc, f_sol_override=f_sol_raw
                )
                rwork, rfree_val = get_rfactors(
                    torch.abs(fobs), torch.abs(scaled), rfree
                )
                metrics["steps"].append(nsteps)
                metrics["rwork"].append(rwork)
                metrics["rfree"].append(rfree_val)

        if verbose and self.verbose > 0:
            if metrics["rwork"]:
                print(
                    f"Joint scale refinement complete. "
                    f"rwork: {metrics['rwork'][-1]:.4f}, "
                    f"rfree: {metrics['rfree'][-1]:.4f}"
                )

        return metrics

    # ------------------------------------------------------------------
    # Solvent parameter screening
    # ------------------------------------------------------------------

    def screen_solvent_params_joint(self, steps: int = 15):
        """
        Grid-search k_sol / B_sol using NLL summed across all datasets.

        Parameters
        ----------
        steps : int
            Grid points per parameter.
        """
        if not self._component_solvent_models:
            raise RuntimeError("No component solvent models. Call initialize() first.")

        dc = self._dataset_collection
        mc = self._model_collection
        all_keys = [mc.dark_key] + mc.timepoint_names

        # Pre-compute fcalc (detached)
        pairs = []
        for name in all_keys:
            if name not in dc:
                continue
            data = dc[name]
            model = mc[name]
            hkl = data.hkl
            fobs, sigma = data.get_corrected_data()
            rfree = data.rfree_flags
            with torch.no_grad():
                fc = model(hkl)
            pairs.append((fc.detach(), model.fractions.detach(), fobs, sigma, rfree))

        sol = self.solvent
        best_log_k = sol.log_k_solvent.clone()
        best_b = sol.b_solvent.clone()
        best_loss = float("inf")

        ksol_start = torch.log(torch.tensor(0.1, device=self.device))
        ksol_end = torch.log(torch.tensor(0.6, device=self.device))

        for log_k in torch.linspace(
            ksol_start, ksol_end, steps=steps, device=self.device
        ):
            for b in torch.linspace(30.0, 100.0, steps=steps, device=self.device):
                sol.log_k_solvent.data = log_k.to(dtype=sol.log_k_solvent.dtype)
                sol.b_solvent.data = b.to(dtype=sol.b_solvent.dtype)

                total = 0.0
                for fc, fracs, fobs, sigma, rfree in pairs:
                    f_sol_raw = self.get_mixed_solvent_raw(fracs)
                    scaled = super(CollectionScaler, self).forward(
                        fc, f_sol_override=f_sol_raw
                    )
                    diff = fobs[rfree] - torch.abs(scaled[rfree])
                    sigma_safe = sigma[rfree].clamp(min=1e-3)
                    total += (0.5 * (diff**2) / sigma_safe**2).mean().item()

                if total < best_loss:
                    best_loss = total
                    best_log_k = log_k.clone()
                    best_b = b.clone()

        sol.log_k_solvent.data = best_log_k.to(dtype=sol.log_k_solvent.dtype)
        sol.b_solvent.data = best_b.to(dtype=sol.b_solvent.dtype)

        if self.verbose > 0:
            k_val = torch.exp(best_log_k).item()
            print(
                f"Joint solvent screening: k_sol={k_val:.4f}, "
                f"B_sol={best_b.item():.1f}, NLL={best_loss:.4f}"
            )

    # ------------------------------------------------------------------
    # Solvent mask updates
    # ------------------------------------------------------------------

    def update_all_solvent(self):
        """
        Recompute solvent masks for all component models.

        Call this after structure refinement changes base-model coordinates.
        """
        self._f_sol_raw_components = {}
        self._f_sol_raw = None

        for sol in self._component_solvent_models:
            sol.update_solvent()

        if self.verbose > 0:
            print("  Updated all component solvent masks.")

    def invalidate_solvent_cache(self):
        """Clear cached raw solvent SFs (forces recomputation on next call)."""
        self._f_sol_raw_components = {}
        self._f_sol_raw = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def component_solvent_models(self) -> nn.ModuleList:
        """Per-component SolventModel instances (read-only)."""
        return self._component_solvent_models

    def __repr__(self):
        n_comp = len(self._component_solvent_models)
        n_ds = self._dataset_collection.n_datasets if self._dataset_collection else 0
        return (
            f"CollectionScaler({n_comp} component solvents, "
            f"{n_ds} datasets, {self.nbins} bins)"
        )
