"""Joint scaler for multi-dataset / multi-model kinetic refinement.

``CollectionScaler`` extends ``ScalerBase`` to paired ``DatasetCollection`` +
``ModelCollection`` instances, sharing one set of scale parameters (c_iso,
U and the solvent falloff) across **all** data-model pairs so no artificial scale
difference corrupts the time-resolved difference signal. One solvent model is
built per base model; a mixed model's solvent contribution is their linear
combination at the same population fractions as the structural models.
"""

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn

from torchref.base.metrics.rfactor import rfactor_work_free
from torchref.config import get_float_dtype
from torchref.scaling.scaler_base import (
    DEFAULT_SCALE_TARGET,
    SCALE_TARGETS,
    ScalerBase,
)
from torchref.scaling.solvent import SS_HALF_BOUNDS, SolventModel
from torchref.utils.utils import ModuleReference

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection


class _DatasetScalerView(nn.Module):
    """One dataset's view of a shared :class:`CollectionScaler`.

    Exists so the scale fit can hand a plain scaler to a taxonomy row. A row calls
    ``self._scaler(fcalc)`` and knows nothing else about scaling, but the collection's
    bulk solvent is a fraction-weighted mixture that depends on *which* dataset is being
    scaled -- information :meth:`ScalerBase.forward` has no way to carry. This forwards to
    :meth:`CollectionScaler.forward_mixed` with the fractions bound.

    The parent is held through :class:`~torchref.utils.utils.ModuleReference`, so its
    parameters are **not** re-registered here: the optimiser is still built from the one
    scaler, and a row holding this view contributes no leaves of its own.
    """

    def __init__(self, parent: "CollectionScaler", fractions: torch.Tensor):
        super().__init__()
        self._parent = ModuleReference(parent)
        self.register_buffer("_fractions", fractions.detach().clone())

    def __getattr__(self, name):
        """Anything this view does not own belongs to the parent.

        ``nn.Module.__getattr__`` resolves parameters, buffers and submodules first; a
        row reading some other scaler attribute (bin edges, resolution limits) should see
        the parent's, not an AttributeError.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            parent = self.__dict__.get("_parent")
            if parent is None:
                raise
            return getattr(parent, name)

    def forward(self, fcalc: torch.Tensor) -> torch.Tensor:
        return self._parent.forward_mixed(fcalc, self._fractions)


class CollectionScaler(ScalerBase):
    """
    Joint scaler for DatasetCollection + ModelCollection.

    Shares scale parameters (c_iso, U and the solvent falloff) across **all**
    data-model pairs, and manages per-component solvent models so a mixed
    model's bulk solvent is the fraction-weighted sum of the component solvent
    SFs. The bin-wise B-factor correction is *not* set up by ``initialize()``
    and so is not a shared refined parameter by default.

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
        # Bind to the dark/reference dataset for bins and scattering vectors
        dark_data = dataset_collection[model_collection.dark_key]
        # Forward ``device`` through rather than resolving the global default
        # here: a resolved default arrives at ``ScalerBase`` as an *explicit*
        # device, which then drags ``dark_data`` onto it. Passing ``None``
        # lets ScalerBase derive from the data, which is the point.
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

            hkl = data.hkl
            fobs = data.get_corrected_data()[0]
            with torch.no_grad():
                fcalc = model(hkl)
            fcalc_amp = torch.abs(fcalc).clamp(min=1e-3).to(fobs.dtype)
            fobs_clamped = fobs.clamp(min=1e-3)

            # Mask: work subset (validity + work, validation carved out), and
            # positive intensities. ``data.work.mask`` is the standard subset
            # boolean mask, replacing the ad-hoc ``masks() & rfree``.
            work_mask = data.work.mask
            if hasattr(data, "I") and data.I is not None:
                pos_mask = data.I > 0
            else:
                pos_mask = torch.ones_like(fobs, dtype=torch.bool)
            mask = (work_mask & pos_mask).to(torch.bool)

            bins = self.bins[mask].to(torch.int64)  # dtype-ok: bin indices for scatter/index_select; PyTorch requires int64
            log_ratios = (
                torch.log(fobs_clamped[mask]) - torch.log(fcalc_amp[mask])
            ).to(self.device)
            bins = bins.to(self.device)

            scales.scatter_add_(0, bins, log_ratios)
            ones = torch.ones_like(log_ratios)
            counts.scatter_add_(0, bins, ones)
            n_pairs += 1

        per_bin = scales / (counts + 1e-6)
        with torch.no_grad():
            target = per_bin.detach()[self.bins.to(torch.int64)]  # dtype-ok: bin indices for advanced indexing; PyTorch requires int64
            design = self._iso_design.to(target.dtype)
            coeff = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
        self.c_iso = nn.Parameter(coeff.detach())

        if self.verbose > 0:
            print(
                f"Joint initial scale from {n_pairs} data-model pairs "
                f"({self.nbins} bins)."
            )

    # ------------------------------------------------------------------
    # Component solvent models
    # ------------------------------------------------------------------

    def _setup_component_solvent_models(self):
        """Create a SolventModel per base model; the first also becomes
        ``self.solvent`` and owns the shared solvent parameters.
        The rest contribute only their mask FFT and are frozen.
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
        """Raw (un-damped) complex solvent SFs for component *idx*, cached."""
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

    def compute_component_solvent_raw(self) -> torch.Tensor:
        """Raw (un-damped) complex solvent SFs for every component, stacked.

        The solvent counterpart of
        :meth:`~torchref.model.model_collection.ModelCollection.compute_component_fcalcs`.
        Each component's mask FFT is cached, so repeated calls are cheap.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``(n_components, n_reflections)``.
        """
        return torch.stack(
            [
                self._get_component_f_sol_raw(i)
                for i in range(len(self._component_solvent_models))
            ],
            dim=0,
        )

    def forward_batched(
        self,
        fcalc_batch: torch.Tensor,
        fractions_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Scale a batch of mixtures, each with its own fraction-weighted solvent.

        The batched form of :meth:`forward_mixed`: one shared set of scale parameters
        applied to ``T`` mixtures at once, with the bulk solvent mixed per row by the
        same weights. Since ``ScalerBase.forward`` is affine in ``fcalc`` and the mixed
        solvent is linear in the weights, passing a *derivative* of the fractions in
        place of the fractions returns the corresponding derivative of the scaled
        structure factors.

        Parameters
        ----------
        fcalc_batch : torch.Tensor
            Complex structure factors of shape ``(T, n_reflections)``.
        fractions_matrix : torch.Tensor
            Weights of shape ``(T, n_components)``, one row per member of the batch.

        Returns
        -------
        torch.Tensor
            Scaled complex structure factors of shape ``(T, n_reflections)``.
        """
        component_sol_raw = self.compute_component_solvent_raw()
        # Keep real fraction multiplication and component accumulation identical
        # to forward_mixed; complex GEMM changes rounding near solvent cancellation.
        f_sol_batch = sum(
            fractions_matrix[:, i, None] * component_sol_raw[i]
            for i in range(component_sol_raw.shape[0])
        )
        return super().forward(fcalc_batch, f_sol_override=f_sol_batch)

    # ------------------------------------------------------------------
    # Joint LBFGS refinement
    # ------------------------------------------------------------------

    def refine_lbfgs_joint(
        self,
        nsteps: int = 3,
        lr: float = 1.0,
        max_iter: int = 200,
        history_size: int = 10,
        verbose: bool = True,
        scale_target: str = DEFAULT_SCALE_TARGET,
    ) -> dict:
        """
        Refine the shared scale parameters against **all** datasets jointly.

        One set of scale parameters serves every matched dataset-model pair, so the
        closure sums a per-dataset objective. Each dataset's term is built from a row of
        :data:`~torchref.refinement.targets.xray._specs.XRAY_TARGETS`, exactly as
        :meth:`ScalerBase.refine_lbfgs` builds its single-dataset one -- so both scale
        fits evaluate the same likelihood code, and neither carries a private copy of it.

        The row sees this dataset's own **mixed** bulk solvent, via a
        :class:`_DatasetScalerView` that shares the parent's parameters and applies
        :meth:`forward_mixed`. That is why the scaler cannot simply be handed to the
        target: the solvent depends on which dataset's fractions are in play, and the
        plain :meth:`ScalerBase.forward` has no way to know.

        Amplitudes throughout, whatever observable the *refinement* target fits.
        Unit-weight least squares on intensities would put leverage where the data is
        strongest: the residual goes as ``2 F dF``, so the squared residual carries an
        extra factor of ``F**2`` and a global scale plus B plus anisotropy would be
        determined almost entirely by the strongest low-resolution reflections, leaving
        high resolution unconstrained.

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
        scale_target : str, optional
            Objective, one of :data:`~torchref.scaling.scaler_base.SCALE_TARGETS`.
            Defaults to :data:`~torchref.scaling.scaler_base.DEFAULT_SCALE_TARGET`
            (unit-weight ``ls``) -- the same default and the same reason as the
            single-dataset fit.

        Returns
        -------
        dict
            Refinement metrics (steps, rwork, rfree of dark dataset).

        Raises
        ------
        ValueError
            If ``scale_target`` is not in
            :data:`~torchref.scaling.scaler_base.SCALE_TARGETS`.
        """
        if scale_target not in SCALE_TARGETS:
            raise ValueError(
                f"scale_target must be one of {SCALE_TARGETS}, got {scale_target!r}"
            )
        # Local import, deliberately: `torchref.refinement` imports
        # `torchref.scaling` at module scope, so hoisting these to module level
        # closes an import cycle. Do not "tidy" them up.
        from torchref.refinement.loss_state import LossState
        from torchref.refinement.targets.xray import create_xray_target

        dc = self._dataset_collection
        mc = self._model_collection
        all_keys = [mc.dark_key] + mc.timepoint_names

        # Per dataset: a detached F_calc, and a table row that scales it through this
        # dataset's solvent view. `model=None`, so the row never recomputes structure
        # factors and the only leaves in the graph are this scaler's own parameters.
        # Rows that need a model-error estimate build and own one themselves, as under
        # the body refinement -- there is nothing to precompute here.
        fcalc_cache = {}
        fractions_cache = {}
        terms = []
        for name in all_keys:
            if name not in dc:
                continue
            data = dc[name]
            model = mc[name]
            with torch.no_grad():
                fcalc_cache[name] = model(data.hkl).detach()
                fractions_cache[name] = model.fractions.detach()
            view = _DatasetScalerView(self, fractions_cache[name])
            terms.append(
                (
                    create_xray_target(
                        data=data,
                        model=None,
                        scaler=view,
                        mode=scale_target,
                        use_set="work",
                        verbose=0,
                        device=self.device,
                    ),
                    fcalc_cache[name],
                )
            )

        # Use one fixed normalizer: LBFGS uses absolute tolerances, and a large
        # float32 loss can round away the decrease sought by the line search.
        with torch.no_grad():
            ssq = sum(
                float(dc[n].work.F.detach().pow(2).sum())
                for n in fcalc_cache
            )
        _norm = 1.0 / max(ssq, 1e-30)
        scaler_self = self

        class _CollectionScalerJointTarget(nn.Module):
            """The table rows, closed over their detached ``fcalc``."""

            name = "scaler/joint"

            def forward(self):
                total = torch.zeros((), device=scaler_self.device)
                for target, fc in terms:
                    loss = target(fcalc=fc)
                    # Skip a dataset whose term went non-finite rather than poisoning
                    # the whole joint gradient with it.
                    if torch.isfinite(loss):
                        total = total + loss
                return total * _norm

            def maintenance(self):
                """Forward the hook so sigma_A rows drop their ``beta`` cache after a
                step block, as they do under the body refinement."""
                for target, _ in terms:
                    maint = getattr(target, "maintenance", None)
                    if maint is not None:
                        maint()

        class _CollectionScalerUPenalty(nn.Module):
            """``sum(U**2)`` on the anisotropic scale tensor.

            Its normaliser is pinned to **amplitudes** rather than following the
            objective. Sharing ``_norm`` would make the penalty's weight relative to the
            likelihood depend on which objective was selected, which is a silent change
            of regularisation strength dressed up as a change of objective.
            """

            name = "scaler/u_penalty"

            def forward(self):
                return torch.sum(scaler_self.U**2) * _norm

        state = LossState(device=self.device)
        state.register_target("scaler/joint", _CollectionScalerJointTarget())
        state.register_target("scaler/u_penalty", _CollectionScalerUPenalty())

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

        # Evaluate metrics once on the dark dataset after refinement, through the
        # shared ``rfactor_work_free`` source of truth (validity-masked work/free
        # subsets, validation excluded from both) — the same partition the
        # refinement targets report.
        with torch.no_grad():
            dark_name = mc.dark_key
            if dark_name in fcalc_cache:
                fc = fcalc_cache[dark_name]
                fracs = fractions_cache[dark_name]
                f_sol_raw = self.get_mixed_solvent_raw(fracs)
                scaled = super(CollectionScaler, self).forward(
                    fc, f_sol_override=f_sol_raw
                )
                rwork, rfree_val = rfactor_work_free(
                    dc[dark_name], torch.abs(scaled)
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
            work_mask = data.work.mask
            with torch.no_grad():
                fc = model(hkl)
            pairs.append(
                (fc.detach(), model.fractions.detach(), fobs, sigma, work_mask)
            )

        sol = self.solvent
        best_log_k = sol.log_k_solvent.clone()
        best_log_ss = sol.log_ss_half.clone()
        best_loss = float("inf")

        ksol_start = torch.log(torch.tensor(0.1, device=self.device))
        ksol_end = torch.log(torch.tensor(0.6, device=self.device))
        ss_lo, ss_hi = SS_HALF_BOUNDS

        for log_k in torch.linspace(
            ksol_start, ksol_end, steps=steps, device=self.device
        ):
            for log_ss in torch.linspace(
                float(torch.log(torch.tensor(ss_lo))),
                float(torch.log(torch.tensor(ss_hi))),
                steps=steps,
                device=self.device,
            ):
                sol.log_k_solvent.data = log_k.to(dtype=sol.log_k_solvent.dtype)
                sol.log_ss_half.data = log_ss.to(dtype=sol.log_ss_half.dtype)

                total = 0.0
                for fc, fracs, fobs, sigma, work_mask in pairs:
                    f_sol_raw = self.get_mixed_solvent_raw(fracs)
                    scaled = super(CollectionScaler, self).forward(
                        fc, f_sol_override=f_sol_raw
                    )
                    diff = fobs[work_mask] - torch.abs(scaled[work_mask])
                    sigma_safe = sigma[work_mask].clamp(min=1e-3)
                    total += (0.5 * (diff**2) / sigma_safe**2).mean().item()

                if total < best_loss:
                    best_loss = total
                    best_log_k = log_k.clone()
                    best_log_ss = log_ss.clone()

        sol.log_k_solvent.data = best_log_k.to(dtype=sol.log_k_solvent.dtype)
        sol.log_ss_half.data = best_log_ss.to(dtype=sol.log_ss_half.dtype)

        if self.verbose > 0:
            k_val = torch.exp(best_log_k).item()
            d_half = 1.0 / (2.0 * torch.exp(best_log_ss).sqrt().item())
            print(
                f"Joint solvent screening: k_sol={k_val:.4f}, "
                f"d_half={d_half:.2f} A, NLL={best_loss:.4f}"
            )

    # ------------------------------------------------------------------
    # Solvent mask updates
    # ------------------------------------------------------------------

    def update_solvent(self):
        """
        Recompute solvent masks for all component models and drop the cached ``F_sol``.

        Collection override of :meth:`~torchref.scaling.scaler_base.ScalerBase.update_solvent`
        -- same contract, but every component has its own mask and cache entry.

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
