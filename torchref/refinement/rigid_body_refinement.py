"""Multi-resolution rigid-body refinement strategy.

Wraps an :class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement`, swaps the
model's ``xyz`` for a :class:`~torchref.model.rigid_xyz.RigidXYZTensor` via
:meth:`~torchref.model.model.Model.use_rigid_xyz`, and runs one LBFGS step per cutoff
coarse to fine. Above 6 Å the xray target is Phenix-style ``ls_wunit_k1`` (unit weights,
per-bin optimal scale recomputed every gradient call, no external scaler); at 6 Å and
below it switches to ``ml`` with the normal Scaler.
"""

from typing import List, Optional

import torch

from torchref.scaling.scaler import Scaler


class RigidBodyRefinementStep:
    """Multi-resolution rigid-body refinement.

    Parameters
    ----------
    refinement : LBFGSRefinement
        Its ``model`` is swapped for a rigid model for the duration of the run.
    cutoffs : list of float, optional
        High-resolution cutoffs (Å), coarse to fine. ``None`` generates a schedule from the
        native data resolution via :meth:`default_cutoffs`.
    iterations_per_step : int, optional
        ``max_iter`` per cutoff. The default 30 **under-converges** in practice (9RTS needs
        >= 100); raise it for production. Under the solvent-only (``ls_wunit_k1``)
        inner-cycle path this is per *inner* cycle, so total rigid-body iterations are
        ``n_inner * iterations_per_step``.
    commit : bool, optional
        If True (default), bake the final coordinates into a plain ``ModelFT`` so later
        refinement sees normal per-atom xyz; False leaves the rigid model installed.
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
        """Step through every cutoff coarse to fine and return
        ``[(d_min, LossState), ...]``.

        Restores the original ``reflection_data`` on exit, and bakes the final transform
        back
        into a plain ``ModelFT`` unless ``commit=False``.
        """
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
        """Point scaler, targets and loss state at ``data``.

        For ``ls_wunit_k1`` cutoffs the Scaler is built with ``nbins=1`` -- the mask-based
        bulk-solvent term is added to F_calc and the LS target's closed-form ``c[bins]``
        owns
        the overall scaling. Other modes get a fresh full Scaler with ``ref.nbins`` bins.
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
            # T_0 == 1, so a single coefficient IS one global K. ``ls_wunit_k1`` owns its
            # own closed-form scale, and any further isotropic freedom here double-scales
            # against it -- silently, since both fits succeed.
            n_iso_coeff=1 if xray_mode == "ls_wunit_k1" else getattr(
                ref, "n_iso_coeff", 6),
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
                and getattr(ref.scaler, "c_iso", None) is not None
                and ref.scaler.c_iso.requires_grad is False
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
        """Phenix-style inner cycle: refresh mask, refit solvent, then rigid LBFGS.

        Used at coarse cutoffs with a solvent-only scaler. Each cycle rebuilds the
        bulk-solvent mask FFT at the current positions, runs a short LBFGS over the solvent
        parameters, runs the rigid-body LBFGS for ``iterations_per_step`` iterations
        with the
        mask frozen, then ``bake()``s the transform into ``original_xyz`` and zeroes the
        euler/translation params (matching Phenix's per-macro-cycle reset). Total rigid
        LBFGS
        work per cutoff is therefore ``n_inner * iterations_per_step``.
        """
        ref = self.refinement
        rigid_model = ref.model
        solvent = ref.scaler.solvent

        # Build the solvent parameter list — only the ones that are
        # actually refinable.
        solvent_params = []
        for name in ("log_k_solvent", "log_ss_half", "log_n_exp"):
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
            ref.scaler.update_solvent()
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
