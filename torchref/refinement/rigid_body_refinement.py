"""
Multi-resolution rigid-body refinement strategy.

Wraps an existing :class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement`,
swaps the model's ``xyz`` container for a
:class:`~torchref.model.rigid_xyz.RigidXYZTensor` via
:meth:`~torchref.model.model.Model.use_rigid_xyz`, and runs an LBFGS step
at each cutoff in a coarse → fine resolution schedule.

At coarse resolutions (d > 6 Å) the xray target is Phenix-style
``ls_wunit_k1`` — least squares with unit weights and a per-bin optimal
scale recomputed at every gradient call (no external scaler). At high
resolutions (d ≤ 6 Å) the target switches to ``ml`` with the
normal Scaler module.
"""

from typing import List, Optional

import torch

from torchref.scaling.scaler import Scaler


class RigidBodyRefinementStep:
    """
    Multi-resolution rigid-body refinement.

    Parameters
    ----------
    refinement : LBFGSRefinement
        Refinement object whose ``model`` will be swapped for a rigid model
        for the duration of the run.
    cutoffs : list of float, optional
        High-resolution cutoffs (Å) to step through, coarse → fine. If
        ``None``, a default schedule is generated from the native data
        resolution via :meth:`default_cutoffs`.
    iterations_per_step : int, optional
        ``max_iter`` for the per-cutoff LBFGS step. Default 30.
    commit : bool, optional
        If ``True`` (default), the final rotated/translated coordinates are
        baked back into a plain ``ModelFT`` so subsequent regular refinement
        sees a normal per-atom xyz. If ``False``, the rigid model is left
        installed on the refinement.
    """

    DEFAULT_LBFGS_KWARGS = dict(
        lr=1.0,
        history_size=5,  # scitbx LBFGS default m=5 (vs PyTorch's typical 100)
        line_search_fn="strong_wolfe",
    )

    # Phenix's mmtbx.refinement.rigid_body target_auto_switch_resolution.
    TARGET_SWITCH_RES = 6.0

    def __init__(
        self,
        refinement,
        cutoffs: Optional[List[float]] = None,
        iterations_per_step: int = 30,
        commit: bool = True,
    ):
        self.refinement = refinement
        self.cutoffs = cutoffs
        self.iterations_per_step = int(iterations_per_step)
        self.commit = bool(commit)

    # -----------------------------------------------------------------------
    # Schedule
    # -----------------------------------------------------------------------
    @staticmethod
    def default_cutoffs(native_dmin: float) -> List[float]:
        """Geometric schedule from a coarse start down to ``native_dmin``."""
        native = float(native_dmin)
        if native <= 0:
            raise ValueError(f"native_dmin must be > 0, got {native}")
        if native >= 6.0:
            cuts = [native * 1.5, native * 1.2, native]
        else:
            coarse = max(6.0, native * 2.0)
            cuts = [coarse, (coarse * native) ** 0.5, native]
        # Enforce strictly decreasing and >= native.
        out: List[float] = []
        for c in cuts:
            c = max(float(c), native)
            if not out or c < out[-1] - 1e-6:
                out.append(c)
        if out[-1] > native + 1e-6:
            out.append(native)
        return out

    @classmethod
    def _xray_mode_for_cutoff(cls, d_min: float) -> str:
        """Phenix-style target auto-switch."""
        return "ls_wunit_k1" if d_min > cls.TARGET_SWITCH_RES else "ml"

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    def run(self):
        ref = self.refinement
        original_data = ref.reflection_data

        native_dmin = float(original_data.get_max_res())
        cutoffs = (
            self.cutoffs
            if self.cutoffs is not None
            else self.default_cutoffs(native_dmin)
        )

        # Swap the model's xyz container in place for a RigidXYZTensor.
        ref.model.use_rigid_xyz()

        history = []
        try:
            for d_min in cutoffs:
                xray_mode = self._xray_mode_for_cutoff(d_min)
                self._rebind_for_data(
                    original_data.cut_res(highres=float(d_min)),
                    xray_mode=xray_mode,
                )
                step_state = self._run_one_cutoff(d_min)
                history.append((float(d_min), step_state))
        finally:
            # Restore full-resolution data view.
            self._rebind_for_data(original_data)

        if self.commit:
            # Bake rigid coords into a fresh MixedTensor and re-bind
            # scaler/targets/loss-state against the per-atom xyz.
            ref.model.restore_xyz_from_rigid(commit=True)
            self._rebind_for_data(original_data)

        return history

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _rebind_for_data(self, data, model=None, xray_mode=None):
        """Point scaler + targets + loss-state at the given ReflectionData.

        For ``ls_wunit_k1`` cutoffs the Scaler is built with ``nbins=1``: the
        bulk-solvent term (mask-based, Phenix-style) is added to F_calc, and
        the closed-form per-bin scale ``c[bins]`` in the LS target owns the
        overall scaling. For ``ml`` and other modes a fresh full
        Scaler is built with ``ref.nbins`` bins.
        """
        ref = self.refinement
        if model is None:
            model = ref.model
        ref.reflection_data = data

        if xray_mode is None:
            xray_mode = getattr(ref, "target_mode", "ml")

        ref.scaler = Scaler(
            model, data,
            nbins=1 if xray_mode == "ls_wunit_k1" else getattr(ref, "nbins", 20),
            verbose=ref.verbose,
            device=ref.device,
        )
        ref.scaler.initialize()
        ref._init_targets(xray_mode=xray_mode)

        ref.reset_loss_state()
        # Clear cached LBFGS optimizers (they were built over the old model's
        # parameters and would now point at stale leaves).
        if hasattr(ref, "_persistent_optimizers"):
            ref._persistent_optimizers.clear()
        model.reset_cache()

    def _run_one_cutoff(self, d_min: float):
        ref = self.refinement
        rigid_model = ref.model

        state = ref.complete_loss_state()

        # Snapshot weights so we can restore after the step.
        original_weights = dict(state.weights)
        try:
            # Active targets during rigid-body refinement: xray only. Internal
            # bonded geometry is rigid by construction; ADP / occupancy are
            # frozen. Inter-chain vdW is intentionally off — Phenix runs
            # rigid-body without atomistic restraints and we've observed vdW
            # adds no signal here and destabilizes the coarsest cutoff.
            #
            # We DROP non-xray targets from state entirely (not just zero
            # their weight) so their maintenance() hooks don't fire during
            # the rigid-body LBFGS. In particular ``NonBondedTarget``
            # rebuilds its VDW pair list whenever atoms drift >1 Å, a
            # multi-second recomputation that's pure waste when the
            # target weight is 0. State is rebuilt fresh on the next
            # cutoff via _rebind_for_data → reset_loss_state →
            # _init_targets, so this drop is local to this cutoff.
            keep_names = {"xray"}
            for name in list(state.targets.keys()):
                if name not in keep_names:
                    state.targets.pop(name, None)
                    original_weights.pop(name, None)

            rigid_params = [
                rigid_model.xyz.euler_angles,
                rigid_model.xyz.translations,
            ]

            # Decide whether to use the inner-cycle (mask-refresh) loop.
            # Triggered when the scaler has a bulk-solvent component whose
            # mask depends on atom positions (ls_wunit_k1 path here). For
            # the ml path the scaler is fully refit between cutoffs
            # and co-optimized with rigid params in a single LBFGS.
            use_inner_cycles = (
                ref.scaler is not None
                and getattr(ref.scaler, "solvent", None) is not None
                and ref.scaler.log_scale.requires_grad is False
            )

            if use_inner_cycles:
                self._run_inner_cycles(d_min, state, rigid_params, n_inner=5)
            else:
                # Single-shot path: rigid params + scaler params co-optimized.
                if ref.scaler is not None:
                    opt_params = rigid_params + list(ref.scaler.parameters())
                else:
                    opt_params = rigid_params

                rigid_model.reset_cache()
                opt = torch.optim.LBFGS(
                    opt_params,
                    max_iter=self.iterations_per_step,
                    **self.DEFAULT_LBFGS_KWARGS,
                )
                state.step(
                    opt,
                    context=f"rigid_body[d_min={d_min:.2f}]",
                )
            if ref.verbose > 0:
                try:
                    rwork, rfree = ref.get_rfactor()
                    print(
                        f"  rigid-body d_min={d_min:.2f} "
                        f"(lbfgs, iters={self.iterations_per_step}): "
                        f"Rwork={rwork:.4f} Rfree={rfree:.4f}"
                    )
                except Exception:
                    pass
        finally:
            # Restore weights.
            for name, w in original_weights.items():
                state.set_weight(name, w)
        return state

    def _run_inner_cycles(self, d_min, state, rigid_params, n_inner: int = 5):
        """Phenix-style inner cycle: refresh mask + refit solvent + rigid LBFGS.

        Used at coarse cutoffs with a solvent-only scaler. Each inner cycle:

        1. Rebuild the bulk-solvent mask FFT at current atom positions
           (``SolventModel.update_solvent`` + invalidate ``_f_sol_raw``).
        2. Run a short LBFGS over the solvent parameters (``log_k_solvent``,
           ``b_solvent``, and ``phase_offset`` if it's a Parameter) to refit
           the solvent contribution against the current F_calc.
        3. Run the rigid-body LBFGS for ``iterations_per_step`` iterations
           over rigid params only (mask frozen here, matching Phenix's
           per-inner-cycle locality assumption). The total rigid LBFGS work
           per cutoff is therefore ``n_inner * iterations_per_step``.
        4. ``bake()`` the current rigid transform into ``original_xyz`` and
           zero the euler / translation params — matching Phenix's
           per-macro-cycle reset (mmtbx/refinement/rigid_body.py:344-370).
        """
        ref = self.refinement
        rigid_model = ref.model
        solvent = ref.scaler.solvent

        # Build the solvent parameter list — only the ones that are
        # actually refinable.
        solvent_params = []
        for name in ("log_k_solvent", "b_solvent"):
            p = getattr(solvent, name, None)
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                solvent_params.append(p)
        phase = getattr(solvent, "phase_offset", None)
        if isinstance(phase, torch.nn.Parameter) and phase.requires_grad:
            solvent_params.append(phase)

        # In the inner-cycle path, iterations_per_step is the per-INNER
        # LBFGS max_iter (Phenix uses 25). Total rigid LBFGS work per
        # cutoff = n_inner × iterations_per_step.
        iters_per_inner = self.iterations_per_step

        for inner in range(n_inner):
            # (a) Refresh the bulk-solvent mask at current positions.
            solvent.update_solvent()
            ref.scaler._f_sol_raw = None
            rigid_model.reset_cache()

            # (b) Refit solvent params (k_sol, B_sol[, phase]) against F_obs
            # — only if any solvent params are actually refinable.
            if solvent_params:
                sol_opt = torch.optim.LBFGS(
                    solvent_params,
                    max_iter=20,
                    **self.DEFAULT_LBFGS_KWARGS,
                )
                state.step(
                    sol_opt,
                    context=f"rigid_body[d_min={d_min:.2f},inner={inner + 1},solvent]",
                )

            # (c) Rigid LBFGS — mask + solvent params frozen here.
            rigid_model.reset_cache()
            rb_opt = torch.optim.LBFGS(
                rigid_params,
                max_iter=iters_per_inner,
                **self.DEFAULT_LBFGS_KWARGS,
            )
            state.step(
                rb_opt,
                context=f"rigid_body[d_min={d_min:.2f},inner={inner + 1},rigid]",
            )

            # (d) Bake the current rigid transform into original_xyz and
            # zero euler+translation, matching Phenix's per-macro-cycle
            # reset (mmtbx/refinement/rigid_body.py:344-370). Keeps the
            # next solve in the small-angle regime with a clean Hessian.
            rigid_model.xyz.bake()
            rigid_model.reset_cache()
